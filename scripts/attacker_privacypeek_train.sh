#!/bin/bash
# Train the attacker on PrivacyPeek with CER rewards against a frozen guard.
#
# GPU layout (same as defender Path B):
#   TRAIN_GPUS=0,1,3,4  → veRL FSDP + rollout vLLM
#   GUARD_GPU=2           → frozen victim/guard vLLM for reward
#
# Usage:
#   bash scripts/attacker_privacypeek_train.sh [MODEL_PATH] [RUN_ID]

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"

BASE_MODEL=${BASE_MODEL:-/home/fangzibang/data_32T/fzb_data/ModelScope/Qwen3-4B-Instruct-2507}
ATTACKER_MODEL=${ATTACKER_MODEL:-$BASE_MODEL}
GUARD_MODEL=${GUARD_MODEL:-$BASE_MODEL}

MODEL_PATH=${1:-$ATTACKER_MODEL}
RUN_ID=${2:-attacker_privacypeek_run}

if [[ $# -ge 3 && -n "${3:-}" ]]; then
  GUARD_GPU=$3
fi

EFFECTIVE_TRAIN_GPUS=$(resolve_train_gpus "$TRAIN_GPUS" "$GUARD_GPU")
IFS=',' read -ra TRAIN_GPU_LIST <<< "$EFFECTIVE_TRAIN_GPUS"
NUM_TRAIN_GPUS=${#TRAIN_GPU_LIST[@]}

export WORKSPACE
export PROJECT_DIR
export BASE_MODEL
export ATTACKER_MODEL
export GUARD_MODEL
export GUARD_GPU
export TRAIN_GPUS=$EFFECTIVE_TRAIN_GPUS
export GUARD_PORT=${GUARD_PORT:-5000}
export PRIVACYPEEK_ROOT=${PRIVACYPEEK_ROOT:-/home/fangzibang/PrivacyPeek-Resource}
export GUARD_OPENAI_URL=${GUARD_OPENAI_URL:-http://127.0.0.1:${GUARD_PORT}/v1/chat/completions}
export GUARD_VLLM_GPU_MEM_UTIL=${GUARD_VLLM_GPU_MEM_UTIL:-0.85}
export GUARD_VLLM_MAX_MODEL_LEN=${GUARD_VLLM_MAX_MODEL_LEN:-2048}
export DISABLE_KL=${DISABLE_KL:-true}
export ROLLOUT_N=${ROLLOUT_N:-2}
export ROLLOUT_BATCH_CAP=${ROLLOUT_BATCH_CAP:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}

read -r _DEFAULT_ROLLOUT_TP _DEFAULT_ROLLOUT_MEM <<< "$(resolve_rollout_settings "$NUM_TRAIN_GPUS")"
export ROLLOUT_TP=${ROLLOUT_TP:-$_DEFAULT_ROLLOUT_TP}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-$_DEFAULT_ROLLOUT_MEM}

echo "[GPU] guard(vLLM)=$GUARD_GPU | train(FSDP+rollout)=$EFFECTIVE_TRAIN_GPUS (${NUM_TRAIN_GPUS} GPU(s), TP=$ROLLOUT_TP)"
echo "[MEM] guard_vllm_util=$GUARD_VLLM_GPU_MEM_UTIL DISABLE_KL=$DISABLE_KL ROLLOUT_N=$ROLLOUT_N"

cleanup_stale_gpu_processes

GUARD_LOG="${WORKSPACE}/storage/logs/guard_vllm_${RUN_ID}_gpu${GUARD_GPU}.log"
export GUARD_LOG

echo "[VLLM][GUARD] model=$GUARD_MODEL gpu=$GUARD_GPU port=$GUARD_PORT run_id=$RUN_ID"
bash "$PROJECT_DIR/vllm_service_init/start.sh" "$GUARD_MODEL" "$RUN_ID" "$GUARD_GPU" "$GUARD_PORT"

GUARD_PID=$(cat "${WORKSPACE}/storage/logs/guard_vllm_${RUN_ID}.pid")
export GUARD_PID
wait_for_guard_service

echo "[TRAIN][ATTACKER] model=$MODEL_PATH train_gpus=$EFFECTIVE_TRAIN_GPUS run_id=$RUN_ID"
BASE_MODEL="$MODEL_PATH" \
GUARD_MODEL="$GUARD_MODEL" \
DISABLE_KL="$DISABLE_KL" \
ROLLOUT_GPU_MEM_UTIL="$ROLLOUT_GPU_MEM_UTIL" \
ROLLOUT_TP="$ROLLOUT_TP" \
ROLLOUT_N="$ROLLOUT_N" \
ROLLOUT_BATCH_CAP="$ROLLOUT_BATCH_CAP" \
bash "$PROJECT_DIR/training/iterative_train_attacker_privacypeek_verl.sh"
