#!/bin/bash

set -euo pipefail

WORKSPACE=${WORKSPACE:-/home/fangzibang}
PROJECT_DIR=${PROJECT_DIR:-$WORKSPACE/SelfEvolvingPrivacyRL}

BASE_MODEL=${BASE_MODEL:-/home/fangzibang/data_32T/fzb_data/ModelScope/Qwen3-4B-Instruct-2507}
GUARD_MODEL=${GUARD_MODEL:-$BASE_MODEL}
ATTACKER_MODEL=${ATTACKER_MODEL:-$BASE_MODEL}

# ---- PrivacyPeek ----
PRIVACYPEEK_ROOT=${PRIVACYPEEK_ROOT:-/home/fangzibang/PrivacyPeek-Resource}

# ---- PrivacyLens sandbox external paths (needed at runtime by reward wrapper) ----
export PRIVACYLENS_HOME=${PRIVACYLENS_HOME:-/home/fangzibang/PrivacyLens}
export PROCODER_HOME=${PROCODER_HOME:-/home/fangzibang/PromptCoder}

# ---- GPU layout ----
# Defender training GPUs (FSDP + colocated rollout vLLM, tensor-parallel across all listed GPUs).
# Default: GPU 0,1,3,4 (3×40GB + 1×80GB); GUARD_GPU=2 is auto-excluded if listed.
TRAIN_GPUS=${TRAIN_GPUS:-0,1,3,4}
# guard vLLM reward service on a separate 40GB card
GUARD_GPU=${GUARD_GPU:-2}
ATTACKER_GPU=${ATTACKER_GPU:-0}
GUARD_PORT=${GUARD_PORT:-5000}
ATTACKER_VLLM_PORT=${ATTACKER_VLLM_PORT:-5001}

# Guard vLLM config — PrivacyLens procoder prompts need lots of context.
# Keep GPU memory util LOW so the 4B model + 8K KV cache fits in 40GB.
export VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.35}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}

# PrivacyLens style data ratio (0.0 = all standard, 1.0 = all PL-style)
PL_STYLE_RATIO=${PL_STYLE_RATIO:-0.0}

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}

# Kill leftover vLLM / Ray workers before a new run.
cleanup_stale_gpu_processes() {
  echo "[CLEANUP] Stopping stale vLLM / Ray processes..."
  ray stop --force 2>/dev/null || true
  pkill -9 -f "start_vllm_server.py" 2>/dev/null || true
  pkill -9 -f "vllm_service_init/start.sh" 2>/dev/null || true
  pkill -9 -f "verl.trainer.main" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  pkill -9 -f "ray::WorkerDict" 2>/dev/null || true
  pkill -9 -f "ray::Runner" 2>/dev/null || true
  pkill -9 -f "ray::AutoRewardManager" 2>/dev/null || true
  for port in 5000 5001 5002 5003 8265; do
    fuser -k "${port}/tcp" 2>/dev/null || true
  done
  sleep 3
}

# Block until guard vLLM listens on GUARD_PORT (model loads before Flask binds).
wait_for_guard_service() {
  local port="${GUARD_PORT:-5000}"
  local timeout="${GUARD_STARTUP_TIMEOUT:-600}"
  local interval="${GUARD_STARTUP_POLL_INTERVAL:-5}"
  local elapsed=0
  local guard_pid="${GUARD_PID:-}"

  echo "[GUARD] Waiting for vLLM on 127.0.0.1:${port} (timeout ${timeout}s)..."

  while (( elapsed < timeout )); do
    if python3 - "$port" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    s.close()
sys.exit(0)
PY
    then
      echo "[GUARD] vLLM ready on port ${port} (${elapsed}s)"
      return 0
    fi

    if [[ -n "$guard_pid" ]] && ! kill -0 "$guard_pid" 2>/dev/null; then
      echo "[GUARD] ERROR: guard process (pid=${guard_pid}) exited before ready." >&2
      if [[ -n "${GUARD_LOG:-}" && -f "$GUARD_LOG" ]]; then
        echo "[GUARD] Last 40 log lines from ${GUARD_LOG}:" >&2
        tail -40 "$GUARD_LOG" >&2 || true
      fi
      return 1
    fi

    sleep "$interval"
    elapsed=$((elapsed + interval))
    if (( elapsed % 30 == 0 )); then
      echo "[GUARD] Still waiting... ${elapsed}s"
    fi
  done

  echo "[GUARD] ERROR: timeout after ${timeout}s waiting for port ${port}." >&2
  if [[ -n "${GUARD_LOG:-}" && -f "$GUARD_LOG" ]]; then
    echo "[GUARD] Last 40 log lines from ${GUARD_LOG}:" >&2
    tail -40 "$GUARD_LOG" >&2 || true
  fi
  return 1
}

# Remove guard GPU from training set if overlapping (prevents vLLM + FSDP OOM on same card)
resolve_train_gpus() {
  local train_gpus="$1"
  local guard_gpu="$2"
  local resolved=()
  local gpu
  IFS=',' read -ra _gpus <<< "$train_gpus"
  for gpu in "${_gpus[@]}"; do
    gpu="${gpu// /}"
    if [[ -n "$gpu" && "$gpu" != "$guard_gpu" ]]; then
      resolved+=("$gpu")
    fi
  done
  if [[ ${#resolved[@]} -eq 0 ]]; then
    echo "[WARN] TRAIN_GPUS empty after excluding GUARD_GPU=$guard_gpu; falling back to GPU 4" >&2
    resolved=(4)
  fi
  local IFS=,
  echo "${resolved[*]}"
}

# Compute rollout/global/val batch sizes that fit the dataset (drop_last=True requires >=1 batch).
resolve_batch_sizes() {
  local train_file="$1"
  local val_file="$2"
  local num_gpus="$3"
  python3 - "$train_file" "$val_file" "$num_gpus" <<'PY'
import json, os, sys
from pathlib import Path

train_file, val_file, num_gpus = sys.argv[1:4]
num_gpus = max(int(num_gpus), 1)

train_size = len(json.loads(Path(train_file).read_text(encoding="utf-8")))
val_size = len(json.loads(Path(val_file).read_text(encoding="utf-8")))

# Single-GPU 80GB training: keep batches small to avoid actor+rollout OOM.
per_gpu_rollout = 4 if num_gpus == 1 else 8
default_rollout = num_gpus * per_gpu_rollout
default_val = num_gpus * 2 if num_gpus == 1 else num_gpus * 4

rollout_cap = int(os.environ.get("ROLLOUT_BATCH_CAP", "0"))
rollout = min(default_rollout, train_size) if train_size > 0 else 1
val_batch = min(default_val, val_size) if val_size > 0 else 1
if rollout_cap > 0:
    rollout = min(rollout, rollout_cap)
rollout = max(rollout, 1)
val_batch = max(val_batch, 1)

print(f"{rollout} {val_batch} {train_size} {val_size}")
PY
}

# Pick rollout tensor-parallel width and per-GPU vLLM memory budget.
resolve_rollout_settings() {
  local num_gpus="$1"
  local tp="${ROLLOUT_TP:-$num_gpus}"
  local mem_util="${ROLLOUT_GPU_MEM_UTIL:-}"

  if [[ -z "$mem_util" ]]; then
    if (( num_gpus <= 1 )); then
      mem_util=0.30
    else
      mem_util=0.45
    fi
  fi

  echo "$tp $mem_util"
}

# ---- Original (non-PrivacyPeek) guard / attacker service helpers ----

function start_guard_service() {
  local model_path=$1
  local run_id=${2:-mock_run}
  bash "$PROJECT_DIR/scripts/guard_train.sh" "$model_path" "$run_id" "$GUARD_GPU" "$GUARD_PORT" &
}

function start_attacker_service() {
  local model_path=$1
  local run_id=${2:-attacker_run}
  bash "$PROJECT_DIR/scripts/attacker_train.sh" "$model_path" "$run_id" "$ATTACKER_GPU" "$ATTACKER_VLLM_PORT"
}

function stop_guard_service() {
  cleanup_stale_gpu_processes
  echo "[CLEANUP] Killing all vLLM server and related python processes..."
  pkill -f "start_vllm_server.py" || true
  pkill -f "vllm_service_init/start.sh" || true
  pkill -f "vllm" || true
  pkill -f "vllm_service_init" || true
  pkill -f "python.*vllm" || true
  for port in 5000 5001 5002 5003; do
    fuser -k ${port}/tcp 2>/dev/null || true
  done
}
