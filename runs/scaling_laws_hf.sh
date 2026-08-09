#!/usr/bin/env bash

set -euo pipefail

UV_VERSION="0.10.3"
HF_NAMESPACE="${HF_NAMESPACE:-}"
HF_BUCKET="${HF_BUCKET:-}"
GIT_REPO_URL="${GIT_REPO_URL:-}"
GIT_REF="${GIT_REF:-}"
RUN_LABEL="${RUN_LABEL:-}"
TRACKIO_SPACE_ID="${TRACKIO_SPACE_ID:-}"
TRACKIO_BUCKET="${TRACKIO_BUCKET:-}"
TRAIN_IMAGE="${TRAIN_IMAGE:-pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel}"
TRAIN_FLAVOR="h200x8"
NPROC_PER_NODE=8
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-24h}"
NANOCHAT_MOUNT="${NANOCHAT_MOUNT:-/mnt/nanochat}"
DRY_RUN="${DRY_RUN:-0}"

RESULTS_HEADER="run_name,flops_budget,actual_flops,depth,model_dim,params_wte,params_value_embeds,params_lm_head,params_transformer,params_scalars,params_total,num_iterations,tokens_trained,val_bpb,core_score,throughput_tok_per_sec,mfu,train_time_sec"

log() {
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: RUN_LABEL=<label> bash runs/scaling_laws_hf.sh [--worker]

Submits one detached h200x8 Job for the complete 24-point sweep.

Example:
  RUN_LABEL=august-2026 bash runs/scaling_laws_hf.sh

--worker is reserved for the remote Job.
EOF
}

repo_root() {
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

resolve_git_sha() {
    local root ref
    root="$(repo_root)"
    ref="${GIT_REF:-HEAD}"
    git -C "$root" rev-parse "${ref}^{commit}"
}

assert_pushed_sha() {
    local root sha
    root="$(repo_root)"
    sha="$1"
    if ! git -C "$root" branch -r --contains "$sha" | sed 's/^[[:space:]]*//' | grep -q '^origin/'; then
        die "Git SHA $sha is not present in any local origin/* ref. Push it before launching."
    fi
}

hf_cli() {
    uv run --frozen hf "$@"
}

resolve_hf_namespace() {
    local namespace
    if ! namespace="$(hf_cli auth whoami --quiet)" || [[ -z "$namespace" ]]; then
        die "Unable to determine the logged-in Hugging Face user; run 'uv run --frozen hf auth login' or set HF_NAMESPACE"
    fi
    printf '%s\n' "$namespace"
}

resolve_git_repo_url() {
    local root url
    root="$(repo_root)"
    if ! url="$(git -C "$root" remote get-url origin 2>/dev/null)"; then
        die "Unable to determine GIT_REPO_URL from the origin remote; set GIT_REPO_URL"
    fi
    case "$url" in
        git@github.com:*)
            printf 'https://github.com/%s\n' "${url#git@github.com:}"
            ;;
        ssh://git@github.com/*)
            printf 'https://github.com/%s\n' "${url#ssh://git@github.com/}"
            ;;
        http://*|https://*)
            printf '%s\n' "$url"
            ;;
        *)
            die "Cannot derive a public clone URL from origin '$url'; set GIT_REPO_URL"
            ;;
    esac
}

resolve_launcher_defaults() {
    if [[ -z "$HF_NAMESPACE" ]]; then
        HF_NAMESPACE="$(resolve_hf_namespace)"
    fi
    HF_BUCKET="${HF_BUCKET:-${HF_NAMESPACE}/nanochat-scaling-laws}"
    TRACKIO_SPACE_ID="${TRACKIO_SPACE_ID:-${HF_NAMESPACE}/nanochat-scaling-laws}"
    TRACKIO_BUCKET="${TRACKIO_BUCKET:-$HF_BUCKET}"
    if [[ -z "$GIT_REPO_URL" ]]; then
        GIT_REPO_URL="$(resolve_git_repo_url)"
    fi
}

run_python() {
    if command -v python >/dev/null 2>&1; then
        command python "$@"
    else
        uv run --frozen python "$@"
    fi
}

parse_job_id() {
    awk '$1 == "id:" && NF == 2 { print $2; found = 1; exit } END { if (!found) exit 1 }'
}

print_command() {
    printf 'Request: '
    printf '%q ' "$@"
    printf '\n'
}

validate_settings() {
    [[ -n "$RUN_LABEL" ]] || die "RUN_LABEL is required"
    [[ "$RUN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "RUN_LABEL may contain only letters, numbers, dot, underscore, and hyphen"
    [[ -n "$HF_NAMESPACE" && "$HF_NAMESPACE" != */* && "$HF_NAMESPACE" != *[[:space:]]* ]] || die "HF_NAMESPACE must be a single Hub namespace"
    [[ "$HF_BUCKET" == */* ]] || die "HF_BUCKET must be namespace/name"
    [[ "$TRACKIO_BUCKET" == */* ]] || die "TRACKIO_BUCKET must be namespace/name"
    [[ "$TRACKIO_SPACE_ID" == */* ]] || die "TRACKIO_SPACE_ID must be namespace/name"
    [[ "$TRAIN_FLAVOR" == "h200x8" && "$NPROC_PER_NODE" == "8" ]] || die "Scaling requires h200x8 with eight processes"
    [[ "$NANOCHAT_MOUNT" == "/mnt/nanochat" ]] || die "NANOCHAT_MOUNT is fixed to /mnt/nanochat"
}

remote_bootstrap_command() {
    cat <<'EOF'
set -euo pipefail
if ! command -v git >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y git ca-certificates
fi
if ! command -v uv >/dev/null 2>&1 || [[ "$(uv --version)" != "uv ${UV_VERSION} "* && "$(uv --version)" != "uv ${UV_VERSION}" ]]; then
  python -m pip install --no-cache-dir "uv==${UV_VERSION}"
fi
git clone "${GIT_REPO_URL}" /workspace/nanochat
cd /workspace/nanochat
git checkout --detach "${GIT_SHA}"
test "$(git rev-parse HEAD)" = "${GIT_SHA}"
uv sync --frozen --extra gpu
exec uv run --frozen --extra gpu bash runs/scaling_laws_hf.sh --worker
EOF
}

validate_data_manifest_stream() {
    run_python -c '
import json, sys
manifest = json.load(sys.stdin)
expected = {
    "schema_version": 1,
    "status": "COMPLETE",
    "dataset": "karpathy/climbmix-400b-shuffle",
    "num_train_shards": 170,
    "validation_shard": 6542,
    "parquet_count": 171,
    "tokenizer_vocab_size": 32768,
    "tokenizer_max_chars": 2_000_000_000,
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(f"Invalid data manifest: {key}={manifest.get(key)!r}, expected {value!r}")
files = manifest.get("parquet_files", [])
if len(files) != 171 or len(set(files)) != 171:
    raise SystemExit("Invalid data manifest parquet inventory")
print(manifest.get("git_sha", "unknown"))
'
}

bootstrap_trackio_space() {
    local space_id="$1" bucket_id="$2"
    uv run --frozen python - "$space_id" "$bucket_id" <<'PY'
import sys

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError
from trackio import deploy

space_id, bucket_id = sys.argv[1:]
api = HfApi()
try:
    info = api.space_info(space_id)
except RepositoryNotFoundError:
    deploy.create_space_if_not_exists(space_id, bucket_id=bucket_id, private=True)
else:
    files = {item.rfilename for item in info.siblings}
    if not ({"app.py", "ui/main.py"} & files):
        deploy.deploy_as_space(space_id, bucket_id=bucket_id, private=True)
PY
    hf_cli spaces volumes set "$space_id" --volume "hf://buckets/${bucket_id}:/data" >/dev/null
    hf_cli spaces wait "$space_id" --timeout 10m >/dev/null
}

verify_private_resources() {
    local space_id="$1" bucket_id="$2"
    hf_cli buckets info "$bucket_id" --format json | run_python -c 'import json,sys; info=json.load(sys.stdin); assert info.get("private") is True, "Trackio bucket must be private"'
    hf_cli spaces info "$space_id" --format json | run_python -c 'import json,sys; info=json.load(sys.stdin); assert info.get("private") is True, "Trackio Space must be private"; files={x["rfilename"] for x in info.get("siblings", [])}; assert {"app.py", "ui/main.py"} & files, "Trackio UI is missing"'
    hf_cli spaces volumes list "$space_id" --format json | run_python -c '
import json, sys
payload = json.load(sys.stdin)
text = json.dumps(payload)
bucket, mount = sys.argv[1:]
assert bucket in text and mount in text, f"Expected {bucket} mounted at {mount}: {payload}"
' "$bucket_id" /data
}

parse_training_log() {
    local log_file="$1" flops_budget="$2" depth="$3" run_name="$4" train_time="$5"
    run_python - "$log_file" "$flops_budget" "$depth" "$run_name" "$train_time" <<'PY'
import csv
import math
import re
import sys

log_path, flops_budget, depth_text, run_name, train_time_text = sys.argv[1:]
text = open(log_path, encoding="utf-8", errors="replace").read()

def last(pattern, label, flags=re.MULTILINE):
    matches = re.findall(pattern, text, flags)
    if not matches:
        raise SystemExit(f"Missing {label} in {log_path}")
    return matches[-1]

def integer(pattern, label):
    return int(last(pattern, label).replace(",", ""))

def finite(pattern, label, positive=False):
    value = float(last(pattern, label))
    if not math.isfinite(value) or (positive and value <= 0):
        raise SystemExit(f"Invalid {label} in {log_path}: {value}")
    return value

params = [
    integer(r"^wte\s+:\s+([\d,]+)\s*$", "wte parameters"),
    integer(r"^value_embeds\s+:\s+([\d,]+)\s*$", "value embedding parameters"),
    integer(r"^lm_head\s+:\s+([\d,]+)\s*$", "lm_head parameters"),
    integer(r"^transformer_matrices\s+:\s+([\d,]+)\s*$", "transformer parameters"),
    integer(r"^scalars\s+:\s+([\d,]+)\s*$", "scalar parameters"),
    integer(r"^total\s+:\s+([\d,]+)\s*$", "total parameters"),
]
num_iterations = integer(
    r"(?:Calculated number of iterations from target FLOPs|Using user-provided number of iterations):\s+([\d,]+)",
    "iteration count",
)
batch_size = integer(r"Total batch size\s+([\d,]+)\s+=>", "total batch size")
actual_flops = finite(r"Total training FLOPs estimate:\s+([0-9.eE+-]+)", "actual FLOPs", positive=True)
val_bpb = finite(r"Validation bpb:\s+([0-9.eE+-]+)", "validation BPB", positive=True)
core_score = finite(r"CORE metric:\s+([0-9.eE+-]+)", "CORE score")
throughput = integer(r"tok/sec:\s+([\d,]+)", "throughput")
mfu = finite(r"bf16_mfu:\s+([0-9.eE+-]+)", "MFU", positive=True)
depth = int(depth_text)
train_time = int(train_time_text)
if any(value <= 0 for value in params) or num_iterations <= 0 or batch_size <= 0 or throughput <= 0 or train_time < 0:
    raise SystemExit(f"Invalid non-positive parsed metric in {log_path}")

row = [
    run_name,
    flops_budget,
    f"{actual_flops:.12g}",
    depth,
    depth * 64,
    *params,
    num_iterations,
    num_iterations * batch_size,
    f"{val_bpb:.12g}",
    f"{core_score:.12g}",
    throughput,
    f"{mfu:.12g}",
    train_time,
]
csv.writer(sys.stdout, lineterminator="\n").writerow(row)
PY
}

write_status() {
    local state="$1" exit_code="${2:-0}"
    run_python - "$RESULTS_DIR" "$state" "$exit_code" <<'PY'
import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path

directory = Path(sys.argv[1])
state = sys.argv[2]
exit_code = int(sys.argv[3])
payload = {
    "state": state,
    "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "exit_code": exit_code,
    "job_id": os.environ.get("JOB_ID", "unknown"),
    "git_sha": os.environ["GIT_SHA"],
    "run_label": os.environ["RUN_LABEL"],
}
with tempfile.NamedTemporaryFile("w", dir=directory, prefix=f".{state}.", suffix=".tmp", delete=False) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    temporary = Path(handle.name)
os.replace(temporary, directory / state)
for other in {"RUNNING", "FAILED", "COMPLETE"} - {state}:
    (directory / other).unlink(missing_ok=True)
PY
}

write_provenance() {
    run_python - "$RESULTS_DIR/provenance.json" <<'PY'
import datetime as dt
import json
import os
import tempfile
from pathlib import Path

target = Path(__import__("sys").argv[1])
payload = {
    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "git_sha": os.environ["GIT_SHA"],
    "git_repo_url": os.environ["GIT_REPO_URL"],
    "job_id": os.environ.get("JOB_ID", "unknown"),
    "run_label": os.environ["RUN_LABEL"],
    "hardware": os.environ["TRAIN_FLAVOR"],
    "image": os.environ["TRAIN_IMAGE"],
    "data_bucket": os.environ["HF_BUCKET"],
    "data_manifest_git_sha": os.environ.get("DATA_MANIFEST_GIT_SHA", "unknown"),
    "trackio_space_id": os.environ["TRACKIO_SPACE_ID"],
    "trackio_bucket": os.environ["TRACKIO_BUCKET"],
    "nproc_per_node": int(os.environ["NPROC_PER_NODE"]),
}
with tempfile.NamedTemporaryFile("w", dir=target.parent, prefix=".provenance.", suffix=".tmp", delete=False) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    temporary = Path(handle.name)
os.replace(temporary, target)
PY
}

point_exists() {
    local results_file="$1" flops_budget="$2" depth="$3"
    run_python - "$results_file" "$flops_budget" "$depth" <<'PY'
import csv
import sys

path, flops, depth = sys.argv[1:]
with open(path, newline="") as handle:
    rows = csv.DictReader(handle)
    if any(row["flops_budget"] == flops and row["depth"] == depth for row in rows):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

assert_results_file() {
    if [[ ! -f "$RESULTS_FILE" ]]; then
        printf '%s\n' "$RESULTS_HEADER" > "$RESULTS_FILE"
    elif [[ "$(head -n 1 "$RESULTS_FILE")" != "$RESULTS_HEADER" ]]; then
        die "Existing results.csv has an incompatible schema"
    fi
}

acquire_writer_lock() {
    local lock_dir="$RESULTS_DIR/.writer-lock" previous_job inspect_json
    if [[ -d "$lock_dir" ]]; then
        previous_job="$(<"$lock_dir/job_id")"
        if ! inspect_json="$(hf_cli jobs inspect "$previous_job" --namespace "$HF_NAMESPACE" --format json 2>/dev/null)"; then
            die "Cannot verify previous writer $previous_job; refusing concurrent access"
        fi
        if printf '%s' "$inspect_json" | run_python -c '
import json, sys
payload = json.load(sys.stdin)
text = json.dumps(payload).upper()
raise SystemExit(0 if any(stage in text for stage in ("RUNNING", "SCHEDULING")) else 1)
'; then
            die "Job $previous_job is still active for RUN_LABEL=$RUN_LABEL"
        fi
        mv "$lock_dir" "$RESULTS_DIR/stale-writer-lock-$(date -u '+%Y%m%dT%H%M%SZ')"
    fi
    mkdir "$lock_dir"
    printf '%s\n' "${JOB_ID:-unknown}" > "$lock_dir/job_id"
    WRITER_LOCK_DIR="$lock_dir"
}

release_writer_lock() {
    if [[ -n "${WRITER_LOCK_DIR:-}" && -d "$WRITER_LOCK_DIR" ]]; then
        rm -f "$WRITER_LOCK_DIR/job_id"
        rmdir "$WRITER_LOCK_DIR"
    fi
}

cleanup_checkpoint() {
    local checkpoint_dir="$1" runtime_base="$2"
    run_python - "$checkpoint_dir" "$runtime_base" <<'PY'
import shutil
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1]).resolve()
base = Path(sys.argv[2]).resolve()
expected_parent = base / "base_checkpoints"
if checkpoint.parent != expected_parent:
    raise SystemExit(f"Refusing to remove unexpected checkpoint path: {checkpoint}")
shutil.rmtree(checkpoint, ignore_errors=True)
PY
}

stage_training_assets() {
    local runtime_base="$1"
    log "Staging training assets from the mounted bucket into local Job storage"
    run_python - "$NANOCHAT_MOUNT" "$runtime_base" <<'PY'
import os
import shutil
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

source = Path(sys.argv[1])
target = Path(sys.argv[2])
shard_names = [f"shard_{index:05d}.parquet" for index in range(170)] + ["shard_06542.parquet"]

def copy_with_retries(source_path, target_path, attempts=5):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_path.with_name(f".{target_path.name}.partial")
    for attempt in range(1, attempts + 1):
        try:
            temporary.unlink(missing_ok=True)
            with source_path.open("rb") as source_handle, temporary.open("wb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle, length=8 * 1024 * 1024)
                target_handle.flush()
                os.fsync(target_handle.fileno())
            if temporary.stat().st_size != source_path.stat().st_size:
                raise OSError(f"size mismatch for {source_path}")
            os.replace(temporary, target_path)
            return
        except OSError as error:
            temporary.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            delay = min(2**attempt, 30)
            print(f"Retrying mounted-bucket read for {source_path.name} after {error!r} ({attempt}/{attempts})")
            time.sleep(delay)

data_target = target / "base_data_climbmix"
for index, name in enumerate(shard_names, start=1):
    source_path = source / "base_data_climbmix" / name
    target_path = data_target / name
    copy_with_retries(source_path, target_path)
    parquet = pq.ParquetFile(target_path)
    if parquet.num_row_groups < 1 or "text" not in parquet.schema.names:
        raise SystemExit(f"Invalid staged Parquet file: {target_path}")
    print(f"Staged and validated {name} ({index}/{len(shard_names)})")

for name in ("tokenizer.pkl", "token_bytes.pt"):
    source_path = source / "tokenizer" / name
    target_path = target / "tokenizer" / name
    copy_with_retries(source_path, target_path)
    if target_path.stat().st_size == 0:
        raise SystemExit(f"Empty staged tokenizer artifact: {target_path}")
print(f"Staged {len(shard_names)} Parquet files and two tokenizer artifacts from {source}")
PY
}

run_point() {
    local flops_budget="$1" depth="$2" runtime_base="$3"
    local point_name run_name model_tag log_file device_batch start_time end_time train_time row train_rc
    local -a horizon_args train_args
    point_name="${flops_budget}_d${depth}"
    run_name="scaling_${RUN_LABEL}_${point_name}"
    model_tag="$run_name"
    log_file="$RESULTS_DIR/${point_name}_train.log"

    if point_exists "$RESULTS_FILE" "$flops_budget" "$depth"; then
        log "Skipping completed point $point_name"
        return
    fi

    if (( depth >= 28 )); then
        device_batch=8
    elif (( depth >= 20 )); then
        device_batch=16
    else
        device_batch=32
    fi

    horizon_args=(--target-flops="$flops_budget" --target-param-data-ratio=-1)
    train_args=(
        --eval-tokens=52428800
        --core-metric-every=999999
        --core-metric-max-per-task=-1
        --sample-every=-1
        --save-every=-1
    )

    log "Starting $run_name with $NPROC_PER_NODE-process torchrun"
    start_time="$(date +%s)"
    set +e
    NANOCHAT_BASE_DIR="$runtime_base" OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
        --depth="$depth" \
        "${horizon_args[@]}" \
        --run="$run_name" \
        --model-tag="$model_tag" \
        "${train_args[@]}" \
        --device-batch-size="$device_batch" \
        2>&1 | tee "$log_file"
    train_rc="${PIPESTATUS[0]}"
    set -e
    end_time="$(date +%s)"
    train_time=$((end_time - start_time))
    if [[ "$train_rc" -ne 0 ]]; then
        die "Training failed for $point_name with exit code $train_rc"
    fi

    row="$(parse_training_log "$log_file" "$flops_budget" "$depth" "$run_name" "$train_time")"
    printf '%s\n' "$row" >> "$RESULTS_FILE"
    cleanup_checkpoint "$runtime_base/base_checkpoints/$model_tag" "$runtime_base"
    log "Completed $point_name"
}

worker_main() {
    local runtime_base data_manifest_path row_count exit_code
    local -a flops_budgets depths
    validate_settings
    [[ -d "$NANOCHAT_MOUNT" ]] || die "Bucket is not mounted at $NANOCHAT_MOUNT"
    [[ -n "${GIT_SHA:-}" ]] || die "GIT_SHA is required in worker mode"
    [[ "$(git rev-parse HEAD)" == "$GIT_SHA" ]] || die "Worker checkout does not match GIT_SHA"
    log "HF accelerator metadata: ${ACCELERATOR:-unset}; requested flavor: $TRAIN_FLAVOR"

    run_python - "$NPROC_PER_NODE" <<'PY'
import sys
import torch

expected = int(sys.argv[1])
count = torch.cuda.device_count()
names = [torch.cuda.get_device_name(index) for index in range(count)]
print(f"CUDA_DEVICE_COUNT={count}")
print(f"CUDA_DEVICE_NAMES={names}")
if count != expected:
    raise SystemExit(f"Expected {expected} CUDA devices, found {count}")
PY

    data_manifest_path="$NANOCHAT_MOUNT/data_manifest.json"
    [[ -f "$data_manifest_path" ]] || die "Missing $data_manifest_path"
    DATA_MANIFEST_GIT_SHA="$(validate_data_manifest_stream < "$data_manifest_path")"
    export DATA_MANIFEST_GIT_SHA
    [[ -s "$NANOCHAT_MOUNT/tokenizer/tokenizer.pkl" ]] || die "Missing tokenizer.pkl"
    [[ -s "$NANOCHAT_MOUNT/tokenizer/token_bytes.pt" ]] || die "Missing token_bytes.pt"

    RESULTS_DIR="$NANOCHAT_MOUNT/scaling_laws/$RUN_LABEL"
    RESULTS_FILE="$RESULTS_DIR/results.csv"
    mkdir -p "$RESULTS_DIR"
    export RESULTS_DIR RESULTS_FILE
    acquire_writer_lock
    trap 'exit_code=$?; if [[ $exit_code -ne 0 ]]; then write_status FAILED "$exit_code" || true; fi; release_writer_lock || true' EXIT
    write_status RUNNING
    write_provenance
    assert_results_file

    runtime_base="/workspace/nanochat-runtime-${JOB_ID:-local}"
    mkdir -p "$runtime_base/base_checkpoints"
    stage_training_assets "$runtime_base"

    flops_budgets=(1e18 2.15e18 4.64e18 1e19)
    depths=(10 12 14 16 18 20)

    for flops_budget in "${flops_budgets[@]}"; do
        for depth in "${depths[@]}"; do
            run_point "$flops_budget" "$depth" "$runtime_base"
        done
    done

    row_count="$(run_python - "$RESULTS_FILE" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)"
    [[ "$row_count" -eq 24 ]] || die "Expected 24 result rows, found $row_count"
    write_status COMPLETE
    release_writer_lock
    trap - EXIT
    log "Pipeline complete: $RESULTS_FILE"
}

launcher_main() {
    local git_sha job_name remote_command active_jobs manifest_git_sha submit_output job_id
    local -a request
    command -v git >/dev/null 2>&1 || die "git is required"
    command -v uv >/dev/null 2>&1 || die "uv is required"
    resolve_launcher_defaults
    git_sha="$(resolve_git_sha)"
    validate_settings
    job_name="nanochat-scaling-${RUN_LABEL}"
    remote_command="$(remote_bootstrap_command)"
    request=(
        uv run --frozen hf jobs run
        --detach
        --namespace "$HF_NAMESPACE"
        --name "$job_name"
        --label pipeline=nanochat-scaling-laws
        --label "run_label=$RUN_LABEL"
        --label "git_sha=$git_sha"
        --flavor "$TRAIN_FLAVOR"
        --timeout "$TRAIN_TIMEOUT"
        --secrets HF_TOKEN
        --env "UV_VERSION=$UV_VERSION"
        --env "GIT_REPO_URL=$GIT_REPO_URL"
        --env "GIT_SHA=$git_sha"
        --env "HF_NAMESPACE=$HF_NAMESPACE"
        --env "HF_BUCKET=$HF_BUCKET"
        --env "RUN_LABEL=$RUN_LABEL"
        --env "TRACKIO_SPACE_ID=$TRACKIO_SPACE_ID"
        --env "TRACKIO_BUCKET=$TRACKIO_BUCKET"
        --env "TRAIN_IMAGE=$TRAIN_IMAGE"
        --env "TRAIN_FLAVOR=$TRAIN_FLAVOR"
        --env "NPROC_PER_NODE=$NPROC_PER_NODE"
        --env "NANOCHAT_MOUNT=$NANOCHAT_MOUNT"
        --volume "hf://buckets/${HF_BUCKET}:${NANOCHAT_MOUNT}:rw"
        --
        "$TRAIN_IMAGE" bash -lc "$remote_command"
    )

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'RUN_LABEL=%s\n' "$RUN_LABEL"
        printf 'TRACKIO_SPACE_ID=%s\n' "$TRACKIO_SPACE_ID"
        printf 'TRACKIO_BUCKET=%s\n' "$TRACKIO_BUCKET"
        print_command "${request[@]}"
        return
    fi

    [[ "$DRY_RUN" == "0" ]] || die "DRY_RUN must be 0 or 1"
    assert_pushed_sha "$git_sha"
    hf_cli auth whoami >/dev/null
    manifest_git_sha="$(hf_cli buckets cp "hf://buckets/${HF_BUCKET}/data_manifest.json" - | validate_data_manifest_stream)"
    log "Validated data manifest created by $manifest_git_sha"

    active_jobs="$(hf_cli jobs ps --namespace "$HF_NAMESPACE" --label pipeline=nanochat-scaling-laws --label "run_label=$RUN_LABEL" --format json)"
    if [[ "$active_jobs" != "[]" ]]; then
        die "A scaling Job for RUN_LABEL=$RUN_LABEL is already scheduling or running"
    fi

    hf_cli buckets create "$TRACKIO_BUCKET" --private --exist-ok >/dev/null
    bootstrap_trackio_space "$TRACKIO_SPACE_ID" "$TRACKIO_BUCKET"
    verify_private_resources "$TRACKIO_SPACE_ID" "$TRACKIO_BUCKET"

    log "Submitting $TRAIN_FLAVOR Job for RUN_LABEL=$RUN_LABEL"
    submit_output="$("${request[@]}")"
    printf '%s\n' "$submit_output"
    if ! job_id="$(printf '%s\n' "$submit_output" | parse_job_id)"; then
        die "HF CLI did not return a recognizable Job ID"
    fi
    printf 'Job ID: %s\n' "$job_id"
    printf 'Inspect: uv run --frozen hf jobs inspect %q --namespace %q\n' "$job_id" "$HF_NAMESPACE"
    printf 'Logs: uv run --frozen hf jobs logs -f %q --namespace %q\n' "$job_id" "$HF_NAMESPACE"
    printf 'Cancel: uv run --frozen hf jobs cancel %q --namespace %q\n' "$job_id" "$HF_NAMESPACE"
    printf 'Job: https://huggingface.co/jobs/%s/%s\n' "$HF_NAMESPACE" "${job_id##*/}"
    printf 'Trackio: https://huggingface.co/spaces/%s\n' "$TRACKIO_SPACE_ID"
    printf 'Results: https://huggingface.co/buckets/%s/tree/main/scaling_laws/%s\n' "$HF_BUCKET" "$RUN_LABEL"
}

case "${1:-}" in
    --worker)
        shift
        [[ $# -eq 0 ]] || die "Unexpected worker arguments: $*"
        worker_main
        ;;
    --parse-log)
        shift
        [[ $# -eq 5 ]] || die "--parse-log requires LOG FLOPS DEPTH RUN_NAME TRAIN_TIME"
        parse_training_log "$@"
        ;;
    --parse-job-id)
        shift
        [[ $# -eq 0 ]] || die "--parse-job-id reads HF CLI output from stdin"
        parse_job_id
        ;;
    -h|--help)
        usage
        ;;
    "")
        launcher_main
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
