#!/bin/bash
# Standalone launcher for native-AgentDojo attacker training; old launchers unchanged.

set -euo pipefail

WORKSPACE=${WORKSPACE:-/home/fangzibang}
PROJECT_DIR=${PROJECT_DIR:-$WORKSPACE/SelfEvolvingPrivacyRL}
source "$PROJECT_DIR/scripts/common.sh"

cleanup() {
  stop_guard_service
}
trap cleanup EXIT INT TERM

BASE_MODEL=${BASE_MODEL:-/home/fangzibang/data_32T/fzb_data/ModelScope/Qwen3-4B-Instruct-2507}
ATTACKER_MODEL=${ATTACKER_MODEL:-$BASE_MODEL}
ATTACKER_GPU=${ATTACKER_GPU:-0}
GUARD_GPU=${GUARD_GPU:-2}
GUARD_PORT=${GUARD_PORT:-5000}
RUN_ID=${1:-native_agentdojo_attacker}

export BASE_MODEL="$ATTACKER_MODEL"
export ATTACKER_MODEL
export GUARD_MODEL=${GUARD_MODEL:-$ATTACKER_MODEL}
export GUARD_PORT
export GUARD_OPENAI_URL="http://127.0.0.1:${GUARD_PORT}/v1/chat/completions"
export RAY_TMPDIR=${RAY_TMPDIR:-/mnt/32T/data/fzb_data/ray_tmp}
mkdir -p "$RAY_TMPDIR"

echo "[VLLM][GUARD] model=$GUARD_MODEL gpu=$GUARD_GPU port=$GUARD_PORT run_id=$RUN_ID"
bash "$PROJECT_DIR/vllm_service_init/start.sh" "$GUARD_MODEL" "$RUN_ID" "$GUARD_GPU" "$GUARD_PORT"
wait_for_guard_service

echo "[TRAIN][NATIVE-AGENTDOJO] attacker_gpus=$ATTACKER_GPU guard_gpu=$GUARD_GPU"
bash "$PROJECT_DIR/training/iterative_train_native_agentdojo_with_agentr1.sh"
