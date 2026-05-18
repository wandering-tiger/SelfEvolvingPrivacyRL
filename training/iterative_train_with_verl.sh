#!/bin/bash
# Iterative challenger-style training: generate fresh data after each round.

set -euo pipefail

BASE_MODEL="/home/fangzibang/data_32T/fzb_data/ModelScope/Qwen3-4B-Base"
VICTIM_MODEL="/home/fangzibang/data_32T/fzb_data/ModelScope/Qwen3-4B-Base"
SAVE_PREFIX="privacy_attacker_iterative"
WORKSPACE="/home/fangzibang"
PROJECT_DIR="$WORKSPACE/SelfEvolvingPrivacyRL"
ROUNDS=${ROUNDS:-3}
STEPS_PER_ROUND=${STEPS_PER_ROUND:-50}
SAMPLES_PER_ROUND=${SAMPLES_PER_ROUND:-500}


echo "Starting Victim (Defender) Model API service in the background..."
(cd $PROJECT_DIR && bash vllm_service_init/start.sh $VICTIM_MODEL mock_run) &
sleep 45

for ((round=0; round<ROUNDS; round++)); do
  echo "===== Iteration $round/$ROUNDS ====="

  if [[ $round -eq 0 ]]; then
    MODEL_FOR_DATA="$BASE_MODEL"
    LOAD_CKPT=""
  else
    MODEL_FOR_DATA="$WORKSPACE/storage/models/$SAVE_PREFIX/round_$((round-1))"
    LOAD_CKPT="$MODEL_FOR_DATA"
  fi

  echo "Generating fresh dataset from model: $MODEL_FOR_DATA"
  python3 $PROJECT_DIR/build_dataset.py \
    --num_samples $SAMPLES_PER_ROUND \
    --model_path "$MODEL_FOR_DATA" \
  --output_dir $PROJECT_DIR/data

  SAVE_PATH="$WORKSPACE/storage/models/$SAVE_PREFIX/round_${round}"

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
    trainer.n_gpus_per_node=2
    worker.rollout.n=4
    worker.actor.global_batch_size=4
    worker.actor.micro_batch_size_per_device_for_update=1
    worker.actor.micro_batch_size_per_device_for_experience=1
  )

  if [[ -n "$LOAD_CKPT" ]]; then
    TRAIN_ARGS+=(trainer.load_checkpoint_path=$LOAD_CKPT)
  fi

  CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main "${TRAIN_ARGS[@]}"

done

echo "Iterative training completed!"

echo "Cleaning up background Victim API services..."
pkill -f "start_vllm_server.py"
sleep 2
