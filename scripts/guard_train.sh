#!/bin/bash
# Train the guard model by running a background attacker vLLM service.

set -euo pipefail

WORKSPACE=${WORKSPACE:-/home/fangzibang}
PROJECT_DIR=${PROJECT_DIR:-$WORKSPACE/SelfEvolvingPrivacyRL}

BASE_MODEL=${BASE_MODEL:-/home/fangzibang/data_32T/fzb_data/ModelScope/Qwen3-4B-Base}
GUARD_MODEL=${GUARD_MODEL:-$BASE_MODEL}
ATTACKER_MODEL=${ATTACKER_MODEL:-$BASE_MODEL}

GUARD_GPU=${GUARD_GPU:-2}
GUARD_PORT=${GUARD_PORT:-5000}
ATTACKER_GPU=${ATTACKER_GPU:-0}
ATTACKER_VLLM_PORT=${ATTACKER_VLLM_PORT:-5001}

MODEL_PATH=${1:-$GUARD_MODEL}
RUN_ID=${2:-guard_run}
GPU_ID=${3:-$GUARD_GPU}

export BASE_MODEL
export GUARD_MODEL
export ATTACKER_MODEL

export REWRITE_BACKEND=${REWRITE_BACKEND:-vllm}

echo "[VLLM][ATTACKER] model=$ATTACKER_MODEL gpu=$ATTACKER_GPU port=$ATTACKER_VLLM_PORT run_id=$RUN_ID"

bash "$PROJECT_DIR/vllm_service_init/start.sh" "$ATTACKER_MODEL" "$RUN_ID" "$ATTACKER_GPU" "$ATTACKER_VLLM_PORT"

sleep 45

echo "[TRAIN][GUARD] model=$MODEL_PATH gpu=$GPU_ID run_id=$RUN_ID"

BASE_MODEL="$MODEL_PATH" \
GUARD_MODEL="$ATTACKER_MODEL" \
ATTACKER_GPU="$GPU_ID" \
CUDA_VISIBLE_DEVICES=$GPU_ID \
	bash "$PROJECT_DIR/training/iterative_train_with_verl.sh"
