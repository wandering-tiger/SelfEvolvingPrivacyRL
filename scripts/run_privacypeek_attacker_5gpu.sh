#!/bin/bash
# PrivacyPeek attacker smoke/full train — uses all 5 GPUs.
#
# Layout:
#   GPU 0,1,3,4 → veRL FSDP actor + rollout vLLM TP=4
#   GPU 2       → frozen guard vLLM (PrivacyPeek victim for CER reward)
#
# Usage:
#   bash scripts/run_privacypeek_attacker_5gpu.sh                 # smoke, 20 cases
#   bash scripts/run_privacypeek_attacker_5gpu.sh full             # full run
#   bash scripts/run_privacypeek_attacker_5gpu.sh smoke 50         # smoke with limit=50

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/.."

MODE=${1:-smoke}
LIMIT_ARG=${2:-}

unset TRAIN_GPUS GUARD_GPU ROLLOUT_TP ROLLOUT_GPU_MEM_UTIL ROLLOUT_BATCH_CAP 2>/dev/null || true
unset ROLLOUT_N DISABLE_KL GUARD_VLLM_GPU_MEM_UTIL GUARD_VLLM_MAX_MODEL_LEN 2>/dev/null || true

export WORKSPACE=${WORKSPACE:-/home/fangzibang}
export PROJECT_DIR=${PROJECT_DIR:-$WORKSPACE/SelfEvolvingPrivacyRL}
export PRIVACYPEEK_ROOT=${PRIVACYPEEK_ROOT:-/home/fangzibang/PrivacyPeek-Resource}
export BASE_MODEL=${BASE_MODEL:-/home/fangzibang/data_32T/fzb_data/ModelScope/Qwen3-4B-Instruct-2507}
export ATTACKER_MODEL=${ATTACKER_MODEL:-$BASE_MODEL}
export GUARD_MODEL=${GUARD_MODEL:-$BASE_MODEL}
export PYTHONPATH=$PROJECT_DIR:${PYTHONPATH:-}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TRAIN_GPUS=0,1,3,4
export GUARD_GPU=2
export ROLLOUT_TP=4
export ROLLOUT_GPU_MEM_UTIL=0.45
export ROLLOUT_N=2
export ROLLOUT_BATCH_CAP=4
export DISABLE_KL=true
export GUARD_VLLM_GPU_MEM_UTIL=0.85
export GUARD_VLLM_MAX_MODEL_LEN=2048
export GUARD_PORT=5000

if [[ "$MODE" == "full" ]]; then
  export PRIVACYPEEK_LIMIT=${LIMIT_ARG:-}
  export STEPS_PER_ROUND=${STEPS_PER_ROUND:-50}
  export ROUNDS=${ROUNDS:-3}
  RUN_ID=attacker_5gpu_full
else
  export PRIVACYPEEK_LIMIT=${LIMIT_ARG:-20}
  export STEPS_PER_ROUND=${STEPS_PER_ROUND:-10}
  export ROUNDS=${ROUNDS:-1}
  RUN_ID=attacker_5gpu_smoke
fi

echo "============================================"
echo " PrivacyPeek 5-GPU attacker train"
echo " mode=$MODE"
echo " TRAIN_GPUS=$TRAIN_GPUS (FSDP+TP=$ROLLOUT_TP)"
echo " GUARD_GPU=$GUARD_GPU"
echo " PRIVACYPEEK_LIMIT=${PRIVACYPEEK_LIMIT:-all}"
echo " STEPS_PER_ROUND=$STEPS_PER_ROUND ROUNDS=$ROUNDS"
echo "============================================"

bash "$PROJECT_DIR/scripts/attacker_privacypeek_train.sh" "$BASE_MODEL" "$RUN_ID"
