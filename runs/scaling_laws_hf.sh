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
TRAIN_FLAVOR="${TRAIN_FLAVOR:-h200x8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
ASSET_PREFIX="${ASSET_PREFIX:-}"
if [[ "$ASSET_PREFIX" == "rh" ]]; then
    DATA_PREFIX="${DATA_PREFIX:-base_data}"
    MANIFEST_PATH="${MANIFEST_PATH:-data_manifest.json}"
    DOCUMENT_EOS="${DOCUMENT_EOS:-1}"
    HEAD_DIM="${HEAD_DIM:-64}"
    MIN_STEPS="${MIN_STEPS:-500}"
else
    DATA_PREFIX="${DATA_PREFIX:-base_data_climbmix}"
    MANIFEST_PATH="${MANIFEST_PATH:-${ASSET_PREFIX:+$ASSET_PREFIX/}data_manifest.json}"
    DOCUMENT_EOS="${DOCUMENT_EOS:-0}"
    HEAD_DIM="${HEAD_DIM:-128}"
    MIN_STEPS="${MIN_STEPS:-0}"
fi
export ASSET_PREFIX HEAD_DIM MIN_STEPS DOCUMENT_EOS MANIFEST_PATH
if [[ -z "${CORE_METRIC+x}" ]]; then
    if [[ "$DATA_PREFIX" == "base_data_climbmix" ]]; then CORE_METRIC=1; else CORE_METRIC=0; fi
fi
EVAL_TOKENS="${EVAL_TOKENS:-52428800}"
MIN_TOKEN_PARAM_RATIO="${MIN_TOKEN_PARAM_RATIO:-0.25}"
MAX_TOKEN_PARAM_RATIO="${MAX_TOKEN_PARAM_RATIO:-1000}"
export DATA_PREFIX EVAL_TOKENS CORE_METRIC MIN_TOKEN_PARAM_RATIO MAX_TOKEN_PARAM_RATIO NPROC_PER_NODE
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-24h}"
NANOCHAT_MOUNT="${NANOCHAT_MOUNT:-/mnt/nanochat}"

read -r -a FLOPS_BUDGETS <<< "${FLOPS_BUDGETS:-1e18 2.15e18 4.64e18 1e19}"
read -r -a DEPTHS <<< "${DEPTHS:-10 12 14 16 18 20}"

RESULTS_HEADER="run_name,flops_budget,actual_flops,depth,model_dim,head_dim,asset_prefix,document_eos,tokenizer_sha256,tokenizer_bucket_path,params_wte,params_value_embeds,params_lm_head,params_transformer,params_scalars,params_total,effective_params,num_iterations,total_batch_size,batch_lr_scale,weight_decay_scaled,validation_mode,eval_tokens,nproc_per_node,tokens_trained,text_tokens_trained,unique_tokens,passes,repeat_factor,val_bpb,core_score,throughput_tok_per_sec,mfu,train_time_sec"

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

Submits one detached Job. Defaults are the original ClimbMix h200x8 grid.
Set HF_BUCKET, ASSET_PREFIX, DATA_PREFIX, TRAIN_FLAVOR, NPROC_PER_NODE, FLOPS_BUDGETS,
DEPTHS, CORE_METRIC and EVAL_TOKENS to configure a sweep; budgets and depths
are space-separated. MIN_TOKEN_PARAM_RATIO and MAX_TOKEN_PARAM_RATIO bound
the estimated grid. Custom data disables CORE unless explicitly enabled.

Example:
  RUN_LABEL=lean ASSET_PREFIX=rh TRAIN_FLAVOR=h200 NPROC_PER_NODE=1 \
    FLOPS_BUDGETS="1e15" DEPTHS="2 4" bash runs/scaling_laws_hf.sh

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
    if [[ "$HF_BUCKET" != "${HF_NAMESPACE}/nanochat-scaling-laws" && -z "$TRACKIO_SPACE_ID" ]]; then
        die "Set TRACKIO_SPACE_ID explicitly when overriding HF_BUCKET"
    fi
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
    awk '$1 == "id:" && NF == 2 { print $2; found = 1; exit } $1 ~ /^id=/ { sub(/^id=/, "", $1); print $1; found = 1; exit } END { if (!found) exit 1 }'
}

validate_settings() {
    [[ -n "$RUN_LABEL" ]] || die "RUN_LABEL is required"
    [[ "$RUN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "RUN_LABEL may contain only letters, numbers, dot, underscore, and hyphen"
    [[ -n "$HF_NAMESPACE" && "$HF_NAMESPACE" != */* && "$HF_NAMESPACE" != *[[:space:]]* ]] || die "HF_NAMESPACE must be a single Hub namespace"
    [[ "$HF_BUCKET" == */* ]] || die "HF_BUCKET must be namespace/name"
    [[ "$TRACKIO_BUCKET" == */* ]] || die "TRACKIO_BUCKET must be namespace/name"
    [[ "$TRACKIO_SPACE_ID" == */* ]] || die "TRACKIO_SPACE_ID must be namespace/name"
    [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || die "NPROC_PER_NODE must be a positive integer"
    [[ "$DATA_PREFIX" =~ ^[A-Za-z0-9._-]+$ && "$DATA_PREFIX" != "." && "$DATA_PREFIX" != ".." ]] || die "Invalid DATA_PREFIX"
    [[ -z "$ASSET_PREFIX" || ( "$ASSET_PREFIX" =~ ^[A-Za-z0-9._-]+$ && "$ASSET_PREFIX" != "." && "$ASSET_PREFIX" != ".." ) ]] || die "Invalid ASSET_PREFIX"
    [[ "$HEAD_DIM" =~ ^[1-9][0-9]*$ && "$MIN_STEPS" =~ ^[0-9]+$ ]] || die "Invalid HEAD_DIM or MIN_STEPS"
    [[ "$DOCUMENT_EOS" == 0 || "$DOCUMENT_EOS" == 1 ]] || die "Invalid DOCUMENT_EOS"
    [[ "$ASSET_PREFIX" != rh || ( "$DOCUMENT_EOS" == 1 && "$DATA_PREFIX" == base_data && "$HEAD_DIM" == 64 ) ]] || die "RH requires canonical data, EOS, and 64-channel heads"
    [[ "$MANIFEST_PATH" =~ ^[A-Za-z0-9._/-]+$ && "$MANIFEST_PATH" != *..* && "$MANIFEST_PATH" != /* ]] || die "Invalid MANIFEST_PATH"
    [[ "$CORE_METRIC" == 0 || "$CORE_METRIC" == 1 ]] || die "CORE_METRIC must be 0 or 1"
    [[ "$EVAL_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "EVAL_TOKENS must be a positive integer"
    run_python - "$MIN_TOKEN_PARAM_RATIO" "$MAX_TOKEN_PARAM_RATIO" <<'PYWINDOW'
import math, sys
try:
    lo, hi = map(float, sys.argv[1:])
except ValueError:
    raise SystemExit("Token:parameter bounds must be numeric")
if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
    raise SystemExit("Token:parameter bounds must satisfy 0 < MIN <= MAX")
PYWINDOW
    [[ "$NANOCHAT_MOUNT" == "/mnt/nanochat" ]] || die "NANOCHAT_MOUNT is fixed to /mnt/nanochat"
    run_python - "${#FLOPS_BUDGETS[@]}" "${FLOPS_BUDGETS[@]}" "${DEPTHS[@]}" <<'PY'
import re
import sys
from decimal import Decimal, InvalidOperation

budget_count = int(sys.argv[1])
budgets = sys.argv[2 : 2 + budget_count]
depths = sys.argv[2 + budget_count :]

if not budgets:
    raise SystemExit("FLOPS_BUDGETS must not be empty")
if not depths:
    raise SystemExit("DEPTHS must not be empty")

parsed_budgets = []
for text in budgets:
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?", text):
        raise SystemExit(f"Invalid FLOPS_BUDGETS value: {text!r}")
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise SystemExit(f"Invalid FLOPS_BUDGETS value: {text!r}") from error
    if not value.is_finite() or value <= 0:
        raise SystemExit(f"FLOPS_BUDGETS values must be finite and positive: {text!r}")
    parsed_budgets.append(value)
if len(set(parsed_budgets)) != len(parsed_budgets):
    raise SystemExit("FLOPS_BUDGETS must not contain duplicates")

parsed_depths = []
for text in depths:
    if not text.isdigit() or text != str(int(text)) or int(text) <= 0:
        raise SystemExit(f"DEPTHS values must be positive integers: {text!r}")
    parsed_depths.append(int(text))
if len(set(parsed_depths)) != len(parsed_depths):
    raise SystemExit("DEPTHS must not contain duplicates")
PY
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
if manifest.get("dataset") == "alpo.ge/lean":
    assert manifest.get("schema_version") == 1 and manifest.get("status") == "COMPLETE"
    files = manifest.get("parquet_files", [])
    assert len(files) > 1 and len(files) == len(set(files))
    assert files == sorted(files) and manifest["validation_shard"] == files[-1]
    assert manifest["train"]["tokens"] > 0
    print(manifest.get("git_sha", "rh-preparation"))
    raise SystemExit
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
    run_python - "$log_file" "$flops_budget" "$depth" "$run_name" "$train_time" "${TRAIN_CORPUS_TOKENS:-0}" "$ASSET_PREFIX" "$HEAD_DIM" "$DOCUMENT_EOS" "${TOKENIZER_SHA256:-}" "${TOKENIZER_BUCKET_PATH:-}" "$EVAL_TOKENS" "$NPROC_PER_NODE" "$CORE_METRIC" <<'PY'
import csv
import math
import re
import sys

log_path, flops_budget, depth_text, run_name, train_time_text, corpus_tokens_text, asset_prefix, expected_head_dim, document_eos, tokenizer_sha256, tokenizer_bucket_path, eval_tokens, nproc, core_enabled = sys.argv[1:]
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
batch_lr_scale = finite(r"Scaling optimizer: batch_lr_scale=([0-9.eE+-]+)", "batch LR scale", positive=True)
weight_decay_scaled = finite(r"Scaling optimizer: batch_lr_scale=[0-9.eE+-]+ weight_decay=([0-9.eE+-]+)", "weight decay", positive=True)
actual_flops = finite(r"Total training FLOPs estimate:\s+([0-9.eE+-]+)", "actual FLOPs", positive=True)
val_bpb = finite(r"Validation bpb:\s+([0-9.eE+-]+)", "validation BPB", positive=True)
core_matches = re.findall(r"CORE metric:\s+([0-9.eE+-]+)", text)
if core_enabled == "1" and not core_matches:
    raise SystemExit(f"Missing CORE score in {log_path}")
core_score = float(core_matches[-1]) if core_matches else None
throughput = integer(r"tok/sec:\s+([\d,]+)", "throughput")
mfu = finite(r"bf16_mfu:\s+([0-9.eE+-]+)", "MFU")
if mfu < 0:
    raise SystemExit(f"Negative MFU in {log_path}")
depth = int(depth_text)
train_time = int(train_time_text)
if any(value <= 0 for value in params) or num_iterations <= 0 or batch_size <= 0 or throughput <= 0 or train_time < 0:
    raise SystemExit(f"Invalid non-positive parsed metric in {log_path}")

exposures = re.findall(r"Text exposure: unique_tokens=(\d+) text_tokens=(\d+)", text)
unique_tokens, text_tokens = map(int, exposures[-1]) if exposures else (0, 0)
if int(corpus_tokens_text) and not unique_tokens:
    raise SystemExit("Missing measured training data exposure")
width = integer(r'"n_embd"\s*:\s*(\d+)', "model width")
heads = integer(r'"n_head"\s*:\s*(\d+)', "head count")
if width % heads or width // heads != int(expected_head_dim):
    raise SystemExit(f"Unexpected head dimension in {log_path}")
row = [
    run_name,
    flops_budget,
    f"{actual_flops:.12g}",
    depth,
    width,
    width // heads,
    asset_prefix,
    document_eos,
    tokenizer_sha256,
    tokenizer_bucket_path,
    *params,
    params[2] + params[3],
    num_iterations,
    batch_size,
    f"{batch_lr_scale:.12g}",
    f"{weight_decay_scaled:.12g}",
    "full_documents" if document_eos == "1" else "fixed_tokens",
    "" if document_eos == "1" else int(eval_tokens),
    int(nproc),
    num_iterations * batch_size,
    text_tokens if unique_tokens else "",
    unique_tokens if unique_tokens else "",
    f"{text_tokens / int(corpus_tokens_text):.12g}" if int(corpus_tokens_text) else "",
    f"{text_tokens / unique_tokens:.12g}" if unique_tokens else "",
    f"{val_bpb:.12g}",
    f"{core_score:.12g}" if core_score is not None else "",
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
    run_python - "$RESULTS_DIR/provenance.json" "${#FLOPS_BUDGETS[@]}" "${FLOPS_BUDGETS[@]}" "${DEPTHS[@]}" <<'PY'
import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path

target = Path(sys.argv[1])
budget_count = int(sys.argv[2])
flops_budgets = sys.argv[3 : 3 + budget_count]
depths = [int(value) for value in sys.argv[3 + budget_count :]]
payload = {
    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "git_sha": os.environ["GIT_SHA"],
    "git_repo_url": os.environ["GIT_REPO_URL"],
    "job_id": os.environ.get("JOB_ID", "unknown"),
    "run_label": os.environ["RUN_LABEL"],
    "hardware": os.environ["TRAIN_FLAVOR"],
    "image": os.environ["TRAIN_IMAGE"],
    "data_bucket": os.environ["HF_BUCKET"],
    "asset_prefix": os.environ.get("ASSET_PREFIX", ""),
    "head_dim": int(os.environ["HEAD_DIM"]),
    "document_eos": os.environ["DOCUMENT_EOS"] == "1",
    "manifest_path": os.environ["MANIFEST_PATH"],
    "min_steps": int(os.environ["MIN_STEPS"]),
    "data_prefix": os.environ["DATA_PREFIX"],
    "eval_tokens": int(os.environ["EVAL_TOKENS"]),
    "core_metric": os.environ["CORE_METRIC"] == "1",
    "min_token_param_ratio": float(os.environ["MIN_TOKEN_PARAM_RATIO"]),
    "max_token_param_ratio": float(os.environ["MAX_TOKEN_PARAM_RATIO"]),
    "data_manifest_git_sha": os.environ.get("DATA_MANIFEST_GIT_SHA", "unknown"),
    "tokenizer_sha256": os.environ["TOKENIZER_SHA256"],
    "tokenizer_bucket_path": os.environ["TOKENIZER_BUCKET_PATH"],
    "trackio_space_id": os.environ["TRACKIO_SPACE_ID"],
    "trackio_bucket": os.environ["TRACKIO_BUCKET"],
    "nproc_per_node": int(os.environ["NPROC_PER_NODE"]),
    "flops_budgets": flops_budgets,
    "depths": depths,
    "sweep_size": len(flops_budgets) * len(depths),
}
if target.exists():
    old = json.loads(target.read_text())
    keys = ("asset_prefix", "data_prefix", "head_dim", "min_steps", "document_eos", "eval_tokens",
            "nproc_per_node", "tokenizer_sha256", "tokenizer_bucket_path", "core_metric", "git_sha")
    changed = [key for key in keys if old.get(key) != payload[key]]
    if changed:
        raise SystemExit(f"Existing label has incompatible provenance: {', '.join(changed)}; use a new label")
with tempfile.NamedTemporaryFile("w", dir=target.parent, prefix=".provenance.", suffix=".tmp", delete=False) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    temporary = Path(handle.name)
os.replace(temporary, target)
# Keep a provenance record for every job, including resumed labels.
job_key = payload["job_id"] if payload["job_id"] != "unknown" else payload["created_at"].replace(":", "-")
job_path = target.parent / "provenance_jobs" / (job_key + ".json")
job_path.parent.mkdir(exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=job_path.parent, prefix=".job.", suffix=".tmp", delete=False) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    job_temp = Path(handle.name)
os.replace(job_temp, job_path)
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
    run_python - "$NANOCHAT_MOUNT/${ASSET_PREFIX}" "$runtime_base" "$DATA_PREFIX" "$NANOCHAT_MOUNT/$MANIFEST_PATH" <<'PY'
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

source = Path(sys.argv[1])
target = Path(sys.argv[2])
manifest = json.loads(Path(sys.argv[4]).read_text())
shard_names = manifest["parquet_files"]
prefix = sys.argv[3]

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

data_target = target / prefix
for index, name in enumerate(shard_names, start=1):
    source_path = source / prefix / name
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

    local point_info ratio
    point_info="$(run_python -m scripts.scaling_point --depth "$depth" --flops "$flops_budget" --vocab-size "$TOKENIZER_VOCAB_SIZE" --head-dim "$HEAD_DIM")"
    ratio="$(run_python -c 'import json,sys; print(json.loads(sys.argv[1])["tokens_per_param"])' "$point_info")"
    local inside_window
    inside_window="$(run_python - "$ratio" "$MIN_TOKEN_PARAM_RATIO" "$MAX_TOKEN_PARAM_RATIO" <<'PYRATIO'
import math, sys
ratio, lower, upper = map(float, sys.argv[1:])
if not math.isfinite(ratio):
    raise SystemExit('Non-finite token-to-parameter ratio')
print('yes' if lower <= ratio <= upper else 'no')
PYRATIO
)"
    if [[ "$inside_window" == no ]]; then
        log "Skipping $point_name: estimated tokens:params $ratio outside [$MIN_TOKEN_PARAM_RATIO, $MAX_TOKEN_PARAM_RATIO]"
        printf '%s,%s,%s,%s,%s\n' "$flops_budget" "$depth" "$ratio" "$MIN_TOKEN_PARAM_RATIO" "$MAX_TOKEN_PARAM_RATIO" >> "$RESULTS_DIR/skipped.csv"
        skipped_count=$((skipped_count + 1))
        return
    fi
    if ! run_python - "$point_info" "$flops_budget" "$MIN_STEPS" "$NPROC_PER_NODE" <<'PYSTEPS'
import json, sys
info = json.loads(sys.argv[1]); budget = float(sys.argv[2]); min_steps = int(sys.argv[3]); ranks = int(sys.argv[4])
raise SystemExit(0 if not min_steps or budget / info['flops_per_token'] >= min_steps * 2048 * ranks else 1)
PYSTEPS
    then
        log "Skipping $point_name: budget cannot fit $MIN_STEPS steps at the minimum distributed batch"
        printf '%s,%s,%s,%s,%s\n' "$flops_budget" "$depth" "$ratio" "$MIN_TOKEN_PARAM_RATIO" "$MAX_TOKEN_PARAM_RATIO" >> "$RESULTS_DIR/skipped.csv"
        skipped_count=$((skipped_count + 1))
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
        --eval-tokens="$EVAL_TOKENS"
        --core-metric-every="$([[ "$CORE_METRIC" == 1 ]] && echo 999999 || echo -1)"
        --core-metric-max-per-task=-1
        --sample-every=-1
        --save-every=-1
    )

    if [[ "$DATA_PREFIX" != "base_data_climbmix" ]]; then
        train_args+=(--track-data-exposure)
    fi
    if [[ "$DOCUMENT_EOS" == 1 ]]; then
        train_args+=(--document-eos)
    fi
    log "Starting $run_name with $NPROC_PER_NODE-process torchrun"
    start_time="$(date +%s)"
    set +e
    NANOCHAT_BASE_DIR="$runtime_base" NANOCHAT_DATA_DIR="$runtime_base/$DATA_PREFIX" OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
        --depth="$depth" \
        --head-dim="$HEAD_DIM" \
        --min-steps="$MIN_STEPS" \
        --tokenizer-sha256="$TOKENIZER_SHA256" \
        --tokenizer-bucket-path="$TOKENIZER_BUCKET_PATH" \
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
    local checkpoint_source="$runtime_base/base_checkpoints/$model_tag"
    local checkpoint_dest="$RESULTS_DIR/checkpoints/$run_name"
    mkdir -p "$checkpoint_dest"
    local -a models=("$checkpoint_source"/model_*.pt)
    [[ -f "${models[0]}" && ${#models[@]} -eq 1 ]] || die "Expected one final model checkpoint"
    local model_file="${models[0]}" suffix="${models[0]##*/model_}"
    [[ -f "$checkpoint_source/meta_${suffix%.pt}.json" ]] || die "Missing final checkpoint metadata"
    local meta_file="$checkpoint_source/meta_${suffix%.pt}.json"
    local file target
    for file in "$model_file" "$meta_file"; do
        target="$checkpoint_dest/${file##*/}"
        cp "$file" "$target.partial"
        [[ "$(stat -c %s "$file")" == "$(stat -c %s "$target.partial")" ]] || die "Incomplete checkpoint copy: $target"
        mv -f "$target.partial" "$target"
    done
    for file in "$checkpoint_dest"/model_*.pt "$checkpoint_dest"/meta_*.json; do
        [[ "$file" == "$checkpoint_dest/${model_file##*/}" || "$file" == "$checkpoint_dest/${meta_file##*/}" ]] || rm -f "$file"
    done
    printf '%s\n' "$row" >> "$RESULTS_FILE"
    cleanup_checkpoint "$runtime_base/base_checkpoints/$model_tag" "$runtime_base"
    log "Completed $point_name"
}

worker_main() {
    local runtime_base data_manifest_path row_count expected_rows exit_code
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

    data_manifest_path="$NANOCHAT_MOUNT/$MANIFEST_PATH"
    [[ -f "$data_manifest_path" ]] || die "Missing $data_manifest_path"
    DATA_MANIFEST_GIT_SHA="$(validate_data_manifest_stream < "$data_manifest_path")"
    run_python - "$data_manifest_path" "$DATA_PREFIX" "$DOCUMENT_EOS" "$ASSET_PREFIX" <<'PYDATA'
import json, sys
manifest = json.load(open(sys.argv[1]))
if manifest.get("data_prefix", "base_data_climbmix") != sys.argv[2]:
    raise SystemExit("DATA_PREFIX does not match the manifest")
if bool(manifest.get("document_eos", False)) != (sys.argv[3] == '1'):
    raise SystemExit("DOCUMENT_EOS does not match the manifest")
if sys.argv[4] == 'rh' and (manifest.get('asset_prefix') != 'rh' or not manifest.get('tokenizer_sha256')):
    raise SystemExit('Canonical RH manifest is missing asset prefix or tokenizer fingerprint')
PYDATA
    TRAIN_CORPUS_TOKENS="$(run_python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("train", {}).get("tokens", 0))' "$data_manifest_path")"
    TOKENIZER_VOCAB_SIZE="$(run_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokenizer_vocab_size"])' "$data_manifest_path")"
    export DATA_MANIFEST_GIT_SHA TRAIN_CORPUS_TOKENS TOKENIZER_VOCAB_SIZE
    [[ -s "$NANOCHAT_MOUNT/${ASSET_PREFIX:+$ASSET_PREFIX/}tokenizer/tokenizer.pkl" ]] || die "Missing tokenizer.pkl"
    [[ -s "$NANOCHAT_MOUNT/${ASSET_PREFIX:+$ASSET_PREFIX/}tokenizer/token_bytes.pt" ]] || die "Missing token_bytes.pt"
    TOKENIZER_BUCKET_PATH="hf://buckets/${HF_BUCKET}/${ASSET_PREFIX:+$ASSET_PREFIX/}tokenizer/tokenizer.pkl"
    TOKENIZER_SHA256="$(run_python - "$data_manifest_path" "$NANOCHAT_MOUNT/${ASSET_PREFIX:+$ASSET_PREFIX/}tokenizer/tokenizer.pkl" "$TOKENIZER_BUCKET_PATH" <<'PYSHA'
import hashlib, json, sys
manifest = json.load(open(sys.argv[1]))
with open(sys.argv[2], 'rb') as handle:
    actual = hashlib.sha256(handle.read()).hexdigest()
expected = manifest.get('tokenizer_sha256')
if expected and actual != expected:
    raise SystemExit('Tokenizer hash does not match data manifest')
expected_path = manifest.get('tokenizer_bucket_path')
if expected_path and expected_path != sys.argv[3]:
    raise SystemExit('Tokenizer bucket path does not match data manifest')
print(actual)
PYSHA
)"
    export TOKENIZER_SHA256 TOKENIZER_BUCKET_PATH

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
    [[ "$(run_python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$runtime_base/tokenizer/tokenizer.pkl")" == "$TOKENIZER_SHA256" ]] || die "Staged tokenizer hash differs from bucket"
    if [[ "$DOCUMENT_EOS" == 1 ]]; then
        run_python - "$runtime_base/tokenizer" <<'PYEOS'
import pickle, sys, torch
from pathlib import Path
p = Path(sys.argv[1]); enc = pickle.load(open(p/'tokenizer.pkl', 'rb'))
special = enc._special_tokens
if '<|eos|>' not in special:
    raise SystemExit('Canonical tokenizer does not contain EOS')
byte_lengths = torch.load(p/'token_bytes.pt', map_location='cpu', weights_only=True)
if any(int(byte_lengths[i]) != 0 for i in special.values()):
    raise SystemExit('Special token has nonzero byte count')
PYEOS
    fi

    local skipped_count=0
    for flops_budget in "${FLOPS_BUDGETS[@]}"; do
        for depth in "${DEPTHS[@]}"; do
            run_point "$flops_budget" "$depth" "$runtime_base"
        done
    done

    row_count="$(run_python - "$RESULTS_FILE" "${#FLOPS_BUDGETS[@]}" "${FLOPS_BUDGETS[@]}" "${DEPTHS[@]}" <<'PY'
import csv
import sys
count = int(sys.argv[2])
budgets = set(map(float, sys.argv[3:3+count]))
depths = set(map(int, sys.argv[3+count:]))
with open(sys.argv[1], newline="") as handle:
    points = [(float(row['flops_budget']), int(row['depth'])) for row in csv.DictReader(handle)]
points = [p for p in points if p[0] in budgets and p[1] in depths]
if len(points) != len(set(points)):
    raise SystemExit('Duplicate results in requested grid')
print(len(points))
PY
)"
    expected_rows=$((${#FLOPS_BUDGETS[@]} * ${#DEPTHS[@]}))
    [[ $((row_count + skipped_count)) -eq "$expected_rows" ]] || die "Expected $expected_rows points, found $row_count completed and $skipped_count skipped"
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
        --env "DATA_PREFIX=$DATA_PREFIX"
        --env "ASSET_PREFIX=$ASSET_PREFIX"
        --env "MANIFEST_PATH=$MANIFEST_PATH"
        --env "DOCUMENT_EOS=$DOCUMENT_EOS"
        --env "HEAD_DIM=$HEAD_DIM"
        --env "MIN_STEPS=$MIN_STEPS"
        --env "CORE_METRIC=$CORE_METRIC"
        --env "EVAL_TOKENS=$EVAL_TOKENS"
        --env "MIN_TOKEN_PARAM_RATIO=$MIN_TOKEN_PARAM_RATIO"
        --env "MAX_TOKEN_PARAM_RATIO=$MAX_TOKEN_PARAM_RATIO"
        --env "FLOPS_BUDGETS=${FLOPS_BUDGETS[*]}"
        --env "DEPTHS=${DEPTHS[*]}"
        --volume "hf://buckets/${HF_BUCKET}:${NANOCHAT_MOUNT}:rw"
        --
        "$TRAIN_IMAGE" bash -lc "$remote_command"
    )

    assert_pushed_sha "$git_sha"
    hf_cli auth whoami >/dev/null
    manifest_git_sha="$(hf_cli buckets cp "hf://buckets/${HF_BUCKET}/${MANIFEST_PATH}" - | validate_data_manifest_stream)"
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
