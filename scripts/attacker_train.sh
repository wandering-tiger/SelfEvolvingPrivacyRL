#!/bin/bash
# Run attacker training directly with verl (no attacker vLLM service).

set -euo pipefail

WORKSPACE=${WORKSPACE:-/home/fangzibang}
PROJECT_DIR=${PROJECT_DIR:-$WORKSPACE/SelfEvolvingPrivacyRL}

source "$PROJECT_DIR/scripts/common.sh"
cleanup() {
	echo "[CLEANUP] Stopping vLLM services..."
	stop_guard_service
}
trap cleanup EXIT INT TERM

BASE_MODEL=${BASE_MODEL:-/home/fangzibang/data_32T/fzb_data/ModelScope/Qwen3-4B-Instruct-2507}
ATTACKER_MODEL=${ATTACKER_MODEL:-$BASE_MODEL}

ATTACKER_GPU=${ATTACKER_GPU:-0}
ATTACKER_VLLM_PORT=${ATTACKER_VLLM_PORT:-5001}
GUARD_MODEL=${GUARD_MODEL:-$ATTACKER_MODEL}
GUARD_GPU=${GUARD_GPU:-2}
GUARD_PORT=${GUARD_PORT:-5000}

MODEL_PATH=${1:-$ATTACKER_MODEL}
RUN_ID=${2:-attacker_run}
GPU_ID=${3:-$ATTACKER_GPU}

export BASE_MODEL
export ATTACKER_MODEL
export GUARD_MODEL

export REWRITE_BACKEND=${REWRITE_BACKEND:-vllm}


# 自动设置 GUARD_OPENAI_URL，复用已有参数
export GUARD_OPENAI_URL="http://127.0.0.1:${GUARD_PORT}/v1/chat/completions"
echo "[VLLM][GUARD] model=$GUARD_MODEL gpu=$GUARD_GPU port=$GUARD_PORT run_id=$RUN_ID url=$GUARD_OPENAI_URL"

bash "$PROJECT_DIR/vllm_service_init/start.sh" "$GUARD_MODEL" "$RUN_ID" "$GUARD_GPU" "$GUARD_PORT"

sleep 45

echo "[TRAIN][ATTACKER] model=$MODEL_PATH gpu=$GPU_ID run_id=$RUN_ID"

BASE_MODEL="$MODEL_PATH" \
GUARD_MODEL="$GUARD_MODEL" \
CUDA_VISIBLE_DEVICES=$GPU_ID \
	bash "$PROJECT_DIR/training/iterative_train_with_verl.sh"
