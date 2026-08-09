#!/usr/bin/env bash

set -euo pipefail

UV_VERSION="0.10.3"
HF_NAMESPACE="${HF_NAMESPACE:-}"
HF_BUCKET="${HF_BUCKET:-}"
GIT_REPO_URL="${GIT_REPO_URL:-}"
GIT_REF="${GIT_REF:-}"
NUM_SHARDS="${NUM_SHARDS:-170}"
PREP_FLAVOR="${PREP_FLAVOR:-cpu-xl}"
PREP_TIMEOUT="${PREP_TIMEOUT:-4h}"
NANOCHAT_MOUNT="${NANOCHAT_MOUNT:-/mnt/nanochat}"

log() {
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: bash runs/prepare_data_hf.sh [--worker]

Without arguments, creates/checks the private bucket and submits a detached
Hugging Face CPU Job. --worker is used inside that Job to prepare and validate
the mounted data before atomically publishing data_manifest.json.
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

validate_settings() {
    [[ -n "$HF_NAMESPACE" && "$HF_NAMESPACE" != */* && "$HF_NAMESPACE" != *[[:space:]]* ]] || die "HF_NAMESPACE must be a single Hub namespace"
    [[ "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]] || die "NUM_SHARDS must be a positive integer"
    (( NUM_SHARDS <= 6542 )) || die "NUM_SHARDS must not exceed 6542"
    [[ "$HF_BUCKET" == */* ]] || die "HF_BUCKET must be namespace/name"
    [[ "$PREP_FLAVOR" == "cpu-xl" ]] || die "PREP_FLAVOR must be cpu-xl for this pipeline"
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
uv sync --frozen --extra cpu
exec uv run --frozen --extra cpu bash runs/prepare_data_hf.sh --worker
EOF
}

manifest_is_compatible() {
    run_python - "$NANOCHAT_MOUNT/data_manifest.json" "$NUM_SHARDS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_shards = int(sys.argv[2])
if not path.exists():
    raise SystemExit(1)
manifest = json.loads(path.read_text())
expected = {
    "schema_version": 1,
    "status": "COMPLETE",
    "dataset": "karpathy/climbmix-400b-shuffle",
    "num_train_shards": expected_shards,
    "validation_shard": 6542,
    "tokenizer_vocab_size": 32768,
    "tokenizer_max_chars": 2_000_000_000,
}
for key, value in expected.items():
    if manifest.get(key) != value:
        print(f"Incompatible data manifest: {key}={manifest.get(key)!r}, expected {value!r}", file=sys.stderr)
        raise SystemExit(2)
PY
}

validate_assets() {
    local write_manifest="${1:-0}"
    run_python - "$NANOCHAT_MOUNT" "$NUM_SHARDS" "$write_manifest" <<'PY'
import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

base = Path(sys.argv[1])
num_shards = int(sys.argv[2])
write_manifest = sys.argv[3] == "1"
data_dir = base / "base_data_climbmix"
tokenizer_dir = base / "tokenizer"
expected_names = {f"shard_{index:05d}.parquet" for index in range(num_shards)}
expected_names.add("shard_06542.parquet")

actual_names = {path.name for path in data_dir.glob("*.parquet")}
missing = sorted(expected_names - actual_names)
unexpected = sorted(actual_names - expected_names)
partials = sorted(str(path.relative_to(base)) for path in base.rglob("*.tmp"))
if missing or unexpected or partials:
    raise SystemExit(
        f"Asset validation failed: missing={missing}, unexpected={unexpected}, partials={partials}"
    )

for name in sorted(expected_names):
    path = data_dir / name
    if path.stat().st_size == 0:
        raise SystemExit(f"Empty parquet file: {path}")
    parquet = pq.ParquetFile(path)
    if parquet.num_row_groups < 1 or "text" not in parquet.schema.names:
        raise SystemExit(f"Invalid parquet structure: {path}")

tokenizer_files = [tokenizer_dir / "tokenizer.pkl", tokenizer_dir / "token_bytes.pt"]
for path in tokenizer_files:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Missing or empty tokenizer artifact: {path}")

if not write_manifest:
    print(f"Validated {len(expected_names)} parquet files and {len(tokenizer_files)} tokenizer files")
    raise SystemExit(0)

manifest = {
    "schema_version": 1,
    "status": "COMPLETE",
    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "job_id": os.environ.get("JOB_ID", "unknown"),
    "git_sha": os.environ["GIT_SHA"],
    "git_repo_url": os.environ["GIT_REPO_URL"],
    "dataset": "karpathy/climbmix-400b-shuffle",
    "dataset_base_url": "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main",
    "num_train_shards": num_shards,
    "validation_shard": 6542,
    "parquet_count": len(expected_names),
    "parquet_files": sorted(expected_names),
    "tokenizer_vocab_size": 32768,
    "tokenizer_max_chars": 2_000_000_000,
    "tokenizer_files": [
        {"path": str(path.relative_to(base)), "size": path.stat().st_size}
        for path in tokenizer_files
    ],
}
target = base / "data_manifest.json"
with tempfile.NamedTemporaryFile("w", dir=base, prefix=".data_manifest.", suffix=".tmp", delete=False) as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    temporary = Path(handle.name)
os.replace(temporary, target)
print(f"Published {target} after validating {len(expected_names)} parquet files")
PY
}

worker_main() {
    validate_settings
    [[ "$NANOCHAT_MOUNT" == "/mnt/nanochat" ]] || die "Worker mount must be /mnt/nanochat"
    [[ -d "$NANOCHAT_MOUNT" ]] || die "Bucket is not mounted at $NANOCHAT_MOUNT"
    [[ -n "${GIT_SHA:-}" ]] || die "GIT_SHA is required in worker mode"
    [[ "$(git rev-parse HEAD)" == "$GIT_SHA" ]] || die "Worker checkout does not match GIT_SHA"
    export NANOCHAT_BASE_DIR="$NANOCHAT_MOUNT"

    if [[ -f "$NANOCHAT_MOUNT/data_manifest.json" ]]; then
        if ! manifest_is_compatible; then
            die "Existing manifest is incompatible; refusing to alter bucket contents"
        fi
        validate_assets 0
        log "Preparation is already complete and compatible"
        return
    fi

    run_python - "$NANOCHAT_MOUNT/base_data_climbmix" "$NUM_SHARDS" <<'PY'
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])
num_shards = int(sys.argv[2])
if not data_dir.exists():
    raise SystemExit(0)
expected = {f"shard_{index:05d}.parquet" for index in range(num_shards)}
expected.add("shard_06542.parquet")
unexpected = sorted(path.name for path in data_dir.glob("*.parquet") if path.name not in expected)
if unexpected:
    raise SystemExit(f"Unexpected parquet shards present; refusing to delete them: {unexpected}")
PY

    log "Downloading $NUM_SHARDS training shards plus validation shard 06542"
    run_python -m nanochat.dataset --num-files "$NUM_SHARDS" --num-workers "${PREP_WORKERS:-16}"
    log "Training the 32K tokenizer on 2B characters"
    run_python -m scripts.tok_train --max-chars 2000000000 --vocab-size 32768
    validate_assets 1
}

launcher_main() {
    local git_sha short_sha job_name remote_command submit_output job_id
    local -a request
    command -v git >/dev/null 2>&1 || die "git is required"
    command -v uv >/dev/null 2>&1 || die "uv is required"
    resolve_launcher_defaults
    validate_settings
    git_sha="$(resolve_git_sha)"
    short_sha="${git_sha:0:8}"
    job_name="nanochat-prepare-data-${short_sha}"
    remote_command="$(remote_bootstrap_command)"
    request=(
        uv run --frozen hf jobs run
        --detach
        --namespace "$HF_NAMESPACE"
        --name "$job_name"
        --label pipeline=nanochat-data-prep
        --label "git_sha=$git_sha"
        --flavor "$PREP_FLAVOR"
        --timeout "$PREP_TIMEOUT"
        --secrets HF_TOKEN
        --env "UV_VERSION=$UV_VERSION"
        --env "GIT_REPO_URL=$GIT_REPO_URL"
        --env "GIT_SHA=$git_sha"
        --env "HF_NAMESPACE=$HF_NAMESPACE"
        --env "HF_BUCKET=$HF_BUCKET"
        --env "NUM_SHARDS=$NUM_SHARDS"
        --env "NANOCHAT_MOUNT=$NANOCHAT_MOUNT"
        --volume "hf://buckets/${HF_BUCKET}:${NANOCHAT_MOUNT}:rw"
        --
        python:3.12-bookworm bash -lc "$remote_command"
    )

    assert_pushed_sha "$git_sha"
    hf_cli auth whoami >/dev/null
    hf_cli buckets create "$HF_BUCKET" --private --exist-ok >/dev/null
    hf_cli buckets info "$HF_BUCKET" --format json | run_python -c 'import json,sys; info=json.load(sys.stdin); assert info.get("private") is True, "bucket must be private"'

    if [[ "$(hf_cli jobs ps --namespace "$HF_NAMESPACE" --label pipeline=nanochat-data-prep --label "git_sha=$git_sha" --format json)" != "[]" ]]; then
        die "A preparation Job for $git_sha is already scheduling or running"
    fi

    log "Submitting data preparation Job for $git_sha"
    submit_output="$("${request[@]}")"
    printf '%s\n' "$submit_output"
    if ! job_id="$(printf '%s\n' "$submit_output" | parse_job_id)"; then
        die "HF CLI did not return a recognizable Job ID"
    fi
    printf 'Job ID: %s\n' "$job_id"
    printf 'Inspect: uv run --frozen hf jobs inspect %q --namespace %q\n' "$job_id" "$HF_NAMESPACE"
    printf 'Logs: uv run --frozen hf jobs logs -f %q --namespace %q\n' "$job_id" "$HF_NAMESPACE"
    printf 'Cancel: uv run --frozen hf jobs cancel %q --namespace %q\n' "$job_id" "$HF_NAMESPACE"
    printf 'Web: https://huggingface.co/jobs/%s/%s\n' "$HF_NAMESPACE" "${job_id##*/}"
}

case "${1:-}" in
    --worker)
        shift
        [[ $# -eq 0 ]] || die "Unexpected worker arguments: $*"
        worker_main
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
