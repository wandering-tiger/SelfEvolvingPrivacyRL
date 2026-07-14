#!/bin/bash

set -euo pipefail

WORKSPACE=${WORKSPACE:-/home/fangzibang}
PROJECT_DIR=${PROJECT_DIR:-$WORKSPACE/SelfEvolvingPrivacyRL}

BASE_MODEL=${BASE_MODEL:-/home/fangzibang/data_32T/fzb_data/ModelScope/Qwen3-4B-Instruct-2507}
GUARD_MODEL=${GUARD_MODEL:-$BASE_MODEL}
ATTACKER_MODEL=${ATTACKER_MODEL:-$BASE_MODEL}

ATTACKER_GPU=${ATTACKER_GPU:-0}
GUARD_GPU=${GUARD_GPU:-2}
GUARD_PORT=${GUARD_PORT:-5000}
ATTACKER_VLLM_PORT=${ATTACKER_VLLM_PORT:-5001}

# PrivacyLens sandbox external paths (needed at runtime by reward wrapper)
export PRIVACYLENS_HOME=${PRIVACYLENS_HOME:-/home/fangzibang/PrivacyLens}
export PROCODER_HOME=${PROCODER_HOME:-/home/fangzibang/PromptCoder}

# Guard vLLM config — PrivacyLens procoder prompts need lots of context.
# Keep GPU memory util LOW so the 4B model + 8K KV cache fits in 40GB.
export VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.35}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}

# PrivacyLens style data ratio (0.0 = all standard, 1.0 = all PL-style)
PL_STYLE_RATIO=${PL_STYLE_RATIO:-0.0}

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
  echo "[CLEANUP] Killing all vLLM server and related python processes..."
  pkill -f "start_vllm_server.py" || true
  pkill -f "vllm_service_init/start.sh" || true
  pkill -f "vllm" || true
  pkill -f "vllm_service_init" || true
  pkill -f "python.*vllm" || true
  # 强制杀掉占用 victim 端口的进程（如 5000/5001/5002/5003）
  for port in 5000 5001 5002 5003; do
    fuser -k ${port}/tcp 2>/dev/null || true
  done
}
