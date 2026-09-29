#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run LIBERO or LIBERO-Plus evaluation with one policy server and one simulator
worker per GPU. All workers share a resumable SQLite job queue.

Required arguments:
  --benchmark NAME              libero or libero-plus
  --checkpoint PATH             ActionUNet checkpoint directory
  --gpus LIST                   Comma-separated physical GPU IDs, e.g. 0,1,2,3
  --sim-python PATH             Python executable in the matching simulator venv
  --libero-config-path PATH     LIBERO_CONFIG_PATH for that simulator checkout

Required for LIBERO-Plus:
  --classification PATH         task_classification.json from LIBERO-Plus

Optional arguments:
  --actionunet-root PATH        Repository root; defaults to this script's repo
  --output-dir PATH             Queue, logs, and summary directory
  --base-port PORT              First policy server port (default: 8000)
  --server-timeout SECONDS      Startup timeout per server (default: 300)
  --retry-errors                Requeue jobs previously marked error
  -h, --help                    Show this help

Example:
  bash examples/libero/run_parallel_eval.sh \
    --benchmark libero \
    --checkpoint /checkpoints/pi05_libero__actionunet/libero_actionunet/40000 \
    --gpus 0,1,2,3,4,5,6,7 \
    --sim-python /venvs/libero/bin/python \
    --libero-config-path /opt/LIBERO/.libero \
    --output-dir eval/libero
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

BENCHMARK=""
CHECKPOINT=""
GPU_CSV=""
SIM_PYTHON=""
LIBERO_CONFIG=""
CLASSIFICATION=""
ACTIONUNET_ROOT="$DEFAULT_ROOT"
OUTPUT_DIR=""
BASE_PORT=8000
SERVER_TIMEOUT=300
RETRY_ERRORS=0

while (($#)); do
  case "$1" in
    --benchmark)
      (($# >= 2)) || die "--benchmark requires a value"
      BENCHMARK="$2"
      shift 2
      ;;
    --checkpoint)
      (($# >= 2)) || die "--checkpoint requires a value"
      CHECKPOINT="$2"
      shift 2
      ;;
    --gpus)
      (($# >= 2)) || die "--gpus requires a value"
      GPU_CSV="$2"
      shift 2
      ;;
    --sim-python)
      (($# >= 2)) || die "--sim-python requires a value"
      SIM_PYTHON="$2"
      shift 2
      ;;
    --libero-config-path)
      (($# >= 2)) || die "--libero-config-path requires a value"
      LIBERO_CONFIG="$2"
      shift 2
      ;;
    --classification)
      (($# >= 2)) || die "--classification requires a value"
      CLASSIFICATION="$2"
      shift 2
      ;;
    --actionunet-root)
      (($# >= 2)) || die "--actionunet-root requires a value"
      ACTIONUNET_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      (($# >= 2)) || die "--output-dir requires a value"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --base-port)
      (($# >= 2)) || die "--base-port requires a value"
      BASE_PORT="$2"
      shift 2
      ;;
    --server-timeout)
      (($# >= 2)) || die "--server-timeout requires a value"
      SERVER_TIMEOUT="$2"
      shift 2
      ;;
    --retry-errors)
      RETRY_ERRORS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ "$BENCHMARK" == "libero" || "$BENCHMARK" == "libero-plus" ]] || \
  die "--benchmark must be libero or libero-plus"
[[ -n "$CHECKPOINT" ]] || die "--checkpoint is required"
[[ -n "$GPU_CSV" ]] || die "--gpus is required"
[[ -n "$SIM_PYTHON" ]] || die "--sim-python is required"
[[ -n "$LIBERO_CONFIG" ]] || die "--libero-config-path is required"
[[ -d "$ACTIONUNET_ROOT" ]] || die "repository not found: $ACTIONUNET_ROOT"
[[ -x "$SIM_PYTHON" ]] || die "simulator Python is not executable: $SIM_PYTHON"
[[ -d "$LIBERO_CONFIG" ]] || die "LIBERO config directory not found: $LIBERO_CONFIG"
[[ -f "$CHECKPOINT/model.safetensors" ]] || die "missing $CHECKPOINT/model.safetensors"
[[ -f "$CHECKPOINT/metadata.pt" ]] || die "missing $CHECKPOINT/metadata.pt"
[[ "$BASE_PORT" =~ ^[0-9]+$ ]] || die "--base-port must be an integer"
[[ "$SERVER_TIMEOUT" =~ ^[0-9]+$ ]] || die "--server-timeout must be an integer"
command -v uv >/dev/null 2>&1 || die "uv is not on PATH"

if [[ "$BENCHMARK" == "libero-plus" ]]; then
  [[ -f "$CLASSIFICATION" ]] || \
    die "--classification must point to LIBERO-Plus task_classification.json"
fi

IFS=',' read -r -a RAW_GPUS <<<"$GPU_CSV"
GPU_LIST=()
for value in "${RAW_GPUS[@]}"; do
  value="${value// /}"
  [[ -n "$value" ]] || die "--gpus contains an empty GPU ID"
  GPU_LIST+=("$value")
done
((${#GPU_LIST[@]} > 0)) || die "--gpus must contain at least one GPU"
((BASE_PORT + ${#GPU_LIST[@]} - 1 <= 65535)) || die "requested port range exceeds 65535"

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$ACTIONUNET_ROOT/eval/$BENCHMARK"
fi
mkdir -p "$OUTPUT_DIR/logs"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd)"
ACTIONUNET_ROOT="$(cd -- "$ACTIONUNET_ROOT" && pwd)"
CHECKPOINT="$(cd -- "$CHECKPOINT" && pwd)"
LIBERO_CONFIG="$(cd -- "$LIBERO_CONFIG" && pwd)"
if [[ -n "$CLASSIFICATION" ]]; then
  CLASSIFICATION="$(cd -- "$(dirname -- "$CLASSIFICATION")" && pwd)/$(basename -- "$CLASSIFICATION")"
fi

QUEUE_SCRIPT="$ACTIONUNET_ROOT/examples/libero/eval_queue.py"
WORKER_SCRIPT="$ACTIONUNET_ROOT/examples/libero/main.py"
DATABASE="$OUTPUT_DIR/jobs.sqlite3"
SUMMARY="$OUTPUT_DIR/summary.tsv"

[[ -f "$QUEUE_SCRIPT" ]] || die "queue script not found: $QUEUE_SCRIPT"
[[ -f "$WORKER_SCRIPT" ]] || die "worker script not found: $WORKER_SCRIPT"

QUEUE_ENV=("LIBERO_CONFIG_PATH=$LIBERO_CONFIG")
if [[ "$BENCHMARK" == "libero-plus" ]]; then
  QUEUE_ENV+=("LIBERO_EVAL_CATEGORY_CLASSIFICATION=$CLASSIFICATION")
fi

env "${QUEUE_ENV[@]}" "$SIM_PYTHON" "$QUEUE_SCRIPT" init \
  --db "$DATABASE" \
  --benchmark "$BENCHMARK"

if ((RETRY_ERRORS)); then
  "$SIM_PYTHON" "$QUEUE_SCRIPT" requeue-errors --db "$DATABASE"
fi

SERVER_PIDS=()
WORKER_PIDS=()

cleanup() {
  local status=$?
  trap - EXIT
  for pid in "${WORKER_PIDS[@]}" "${SERVER_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${WORKER_PIDS[@]}" "${SERVER_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  exit "$status"
}

interrupted() {
  echo "evaluation interrupted; the SQLite queue can be resumed with the same command" >&2
  exit 130
}

trap cleanup EXIT
trap interrupted INT TERM

wait_for_server() {
  local pid=$1
  local port=$2
  local log_file=$3
  local started=$SECONDS
  while ((SECONDS - started < SERVER_TIMEOUT)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "policy server on port $port exited during startup" >&2
      tail -n 80 "$log_file" >&2 || true
      return 1
    fi
    if "$SIM_PYTHON" - "$port" <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/healthz", timeout=2) as response:
        raise SystemExit(0 if response.status == 200 and response.read() == b"OK\n" else 1)
except Exception:
    raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep 2
  done
  echo "policy server on port $port was not ready after ${SERVER_TIMEOUT}s" >&2
  tail -n 80 "$log_file" >&2 || true
  return 1
}

echo "starting ${#GPU_LIST[@]} policy servers"
for index in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$index]}"
  port=$((BASE_PORT + index))
  safe_gpu="${gpu//[^[:alnum:]_.-]/_}"
  server_log="$OUTPUT_DIR/logs/server_gpu_${safe_gpu}_port_${port}.log"
  (
    cd "$ACTIONUNET_ROOT"
    exec env CUDA_VISIBLE_DEVICES="$gpu" uv run python -m actionunet.serve \
      --base-config pi05_libero \
      --checkpoint "$CHECKPOINT" \
      --device cuda:0 \
      --num-steps 10 \
      --host 127.0.0.1 \
      --port "$port"
  ) >"$server_log" 2>&1 &
  SERVER_PIDS+=("$!")
done

for index in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$index]}"
  port=$((BASE_PORT + index))
  safe_gpu="${gpu//[^[:alnum:]_.-]/_}"
  server_log="$OUTPUT_DIR/logs/server_gpu_${safe_gpu}_port_${port}.log"
  wait_for_server "${SERVER_PIDS[$index]}" "$port" "$server_log"
  echo "server ready: gpu=$gpu port=$port"
done

echo "starting ${#GPU_LIST[@]} simulator workers"
for index in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$index]}"
  port=$((BASE_PORT + index))
  safe_gpu="${gpu//[^[:alnum:]_.-]/_}"
  worker_id="${BENCHMARK}-gpu-${safe_gpu}-port-${port}"
  worker_log="$OUTPUT_DIR/logs/worker_gpu_${safe_gpu}_port_${port}.log"
  env \
    "${QUEUE_ENV[@]}" \
    NO_COLOR=1 \
    TQDM_DISABLE=1 \
    MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID=0 \
    CUDA_VISIBLE_DEVICES="$gpu" \
    "$SIM_PYTHON" "$WORKER_SCRIPT" \
      --args.host 127.0.0.1 \
      --args.port "$port" \
      --args.job-db "$DATABASE" \
      --args.worker-id "$worker_id" \
      --args.no-save-videos \
      >"$worker_log" 2>&1 &
  WORKER_PIDS+=("$!")
  echo "worker started: id=$worker_id gpu=$gpu port=$port"
done

worker_status=0
for index in "${!WORKER_PIDS[@]}"; do
  if ! wait "${WORKER_PIDS[$index]}"; then
    worker_status=1
    gpu="${GPU_LIST[$index]}"
    port=$((BASE_PORT + index))
    echo "worker failed: gpu=$gpu port=$port; inspect $OUTPUT_DIR/logs" >&2
  fi
done

progress="$($SIM_PYTHON "$QUEUE_SCRIPT" progress --db "$DATABASE" --shell)"
echo "$progress"
"$SIM_PYTHON" "$QUEUE_SCRIPT" summary --db "$DATABASE" --output "$SUMMARY"
echo "summary: $SUMMARY"
echo "logs: $OUTPUT_DIR/logs"

pending=-1
running=-1
errors=-1
for field in $progress; do
  case "$field" in
    pending=*) pending="${field#*=}" ;;
    running=*) running="${field#*=}" ;;
    error=*) errors="${field#*=}" ;;
  esac
done

if ((worker_status != 0 || pending != 0 || running != 0 || errors != 0)); then
  echo "evaluation is incomplete or contains failed jobs; inspect the logs first" >&2
  echo "after fixing the cause, rerun with --retry-errors to retry failed episodes" >&2
  exit 1
fi

echo "evaluation completed successfully"
