#!/bin/bash
# Iterative challenger-style training: generate fresh data after each round.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../scripts/common.sh"

cleanup() {
  echo "[CLEANUP] Stopping guard service..."
  stop_guard_service
}

trap cleanup EXIT INT TERM

SAVE_PREFIX="privacy_attacker_iterative"
ROUNDS=${ROUNDS:-3}
STEPS_PER_ROUND=${STEPS_PER_ROUND:-50}
SAMPLES_PER_ROUND=${SAMPLES_PER_ROUND:-500}
REWRITE_BACKEND=${REWRITE_BACKEND:-vllm}

ATTACKER_GPU=${ATTACKER_GPU:-0}
IFS=',' read -ra ATTACKER_GPU_LIST <<< "$ATTACKER_GPU"
NUM_GPUS=${#ATTACKER_GPU_LIST[@]}
if [[ $NUM_GPUS -lt 1 ]]; then
  NUM_GPUS=1
fi


echo "[INFO] vLLM services are started by scripts/main.sh"

for ((round=0; round<ROUNDS; round++)); do
  echo "===== Iteration $round/$ROUNDS ====="

  if [[ $round -eq 0 ]]; then
    MODEL_FOR_DATA="$BASE_MODEL"
    LOAD_CKPT=""
  else
    MODEL_FOR_DATA="$WORKSPACE/data_32T/fzb_data/storage/models/$SAVE_PREFIX/round_$((round-1))"
    LOAD_CKPT="$MODEL_FOR_DATA"
  fi

  echo "Generating fresh dataset from model: $MODEL_FOR_DATA"
  BUILD_DATA_ARGS=(
    --num_samples $SAMPLES_PER_ROUND
    --model_path "$MODEL_FOR_DATA"
    --rewrite_backend $REWRITE_BACKEND
    --output_dir $PROJECT_DIR/data
  )
  if [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
    BUILD_DATA_ARGS+=(--vllm_max_model_len "$VLLM_MAX_MODEL_LEN")
  fi

  CUDA_VISIBLE_DEVICES=$ATTACKER_GPU \
    python3 $PROJECT_DIR/build_dataset.py "${BUILD_DATA_ARGS[@]}"

  SAVE_PATH="$WORKSPACE/data_32T/fzb_data/storage/models/$SAVE_PREFIX/round_${round}"

  TRAIN_ARGS=(
  config=$PROJECT_DIR/examples/config.yaml
    algorithm.adv_estimator=grpo
    data.max_response_length=512
    worker.actor.model.model_path=$BASE_MODEL
    trainer.experiment_name=${SAVE_PREFIX}_round_${round}
    trainer.save_checkpoint_path=$SAVE_PATH
    trainer.total_epochs=1
    trainer.max_steps=$STEPS_PER_ROUND
    worker.reward.reward_function=$PROJECT_DIR/reward/verl_reward_wrapper.py:compute_score
    trainer.val_freq=-1
    trainer.n_gpus_per_node=$NUM_GPUS
    worker.rollout.n=4
    worker.actor.global_batch_size=4
    worker.actor.micro_batch_size_per_device_for_update=1
    worker.actor.micro_batch_size_per_device_for_experience=1
  )

  if [[ -n "$LOAD_CKPT" ]]; then
    TRAIN_ARGS+=(trainer.load_checkpoint_path=$LOAD_CKPT)
  fi

  CUDA_VISIBLE_DEVICES=$ATTACKER_GPU python3 -m verl.trainer.main "${TRAIN_ARGS[@]}"

  echo "[DEBUG] ATTACKER_GPU=$ATTACKER_GPU (used for attacker training)"

done

echo "Iterative training completed!"

echo "[INFO] Cleanup handled by scripts/main.sh"
