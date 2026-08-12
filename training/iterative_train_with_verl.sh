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

RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SAVE_PREFIX="privacy_attacker_iterative_${RUN_TIMESTAMP}"
ROUNDS=${ROUNDS:-3}
STEPS_PER_ROUND=${STEPS_PER_ROUND:-50}
SAMPLES_PER_ROUND=${SAMPLES_PER_ROUND:-500}
REWRITE_BACKEND=${REWRITE_BACKEND:-vllm}
PL_STYLE_RATIO=${PL_STYLE_RATIO:-0.0}
AGENTDOJO_SIMPLE_RATIO=${AGENTDOJO_SIMPLE_RATIO:-0.0}

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
    PREV_ROUND_DIR="$WORKSPACE/data_32T/fzb_data/storage/models/$SAVE_PREFIX/round_$((round-1))"
    MODEL_FOR_DATA="$PREV_ROUND_DIR"
    # Load the latest (most trained) checkpoint from the previous round.
    # We don't trust the buggy checkpoint_tracker.json (which may attribute
    # the val_before_train score to the first saved checkpoint step).
    # With cumulative max_steps, the latest checkpoint always has training
    # budget remaining for the current round.
    LOAD_CKPT=$(ls -d "$PREV_ROUND_DIR"/global_step_* 2>/dev/null | sort -V | tail -1)
    if [[ -z "$LOAD_CKPT" ]]; then
      echo "[WARN] No global_step_* found in $PREV_ROUND_DIR, starting fresh"
      LOAD_CKPT=""
    else
      echo "[INFO] Loading latest checkpoint from previous round: $LOAD_CKPT"
    fi
  fi

  # Only use seed oversampling in first 2 rounds
  if [[ $round -ge 2 ]]; then
    export SKIP_SEED=1
  fi
  echo "Generating fresh dataset from model: $MODEL_FOR_DATA"
  BUILD_DATA_ARGS=(
    --num_samples $SAMPLES_PER_ROUND
    --model_path "$MODEL_FOR_DATA"
    --rewrite_backend $REWRITE_BACKEND
    --output_dir $PROJECT_DIR/data
    --attack_strategy direct,indirect,roleplay,reasoning,multi_turn \
  --pl_style_ratio $PL_STYLE_RATIO \
  --agentdojo_simple_ratio $AGENTDOJO_SIMPLE_RATIO
  )
  if [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
    BUILD_DATA_ARGS+=(--vllm_max_model_len "$VLLM_MAX_MODEL_LEN")
  fi

  CUDA_VISIBLE_DEVICES=$ATTACKER_GPU \
    python3 $PROJECT_DIR/build_dataset.py "${BUILD_DATA_ARGS[@]}"

  SAVE_PATH="$WORKSPACE/data_32T/fzb_data/storage/models/$SAVE_PREFIX/round_${round}"

  # Cumulative max_steps: round R has (R+1)*STEPS_PER_ROUND total steps so each round
  # adds STEPS_PER_ROUND new training steps regardless of the loaded checkpoint step.
  CUMULATIVE_MAX_STEPS=$(( (round + 1) * STEPS_PER_ROUND ))

  TRAIN_ARGS=(
  config=$PROJECT_DIR/examples/config.yaml
    algorithm.adv_estimator=grpo
    data.max_response_length=512
    worker.actor.model.model_path=$BASE_MODEL
    trainer.experiment_name=${SAVE_PREFIX}_round_${round}
    trainer.save_checkpoint_path=$SAVE_PATH
    trainer.total_epochs=1
    trainer.max_steps=$CUMULATIVE_MAX_STEPS
    worker.reward.reward_function=$PROJECT_DIR/reward/verl_reward_wrapper.py:compute_score
    trainer.val_freq=-1
    trainer.n_gpus_per_node=$NUM_GPUS
    worker.rollout.n=4
    worker.actor.global_batch_size=32
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
