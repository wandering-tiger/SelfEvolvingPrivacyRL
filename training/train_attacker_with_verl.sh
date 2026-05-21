#!/bin/bash
# Train the privacy attacker by reusing R-Zero's `verl` framework.

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../scripts/common.sh"

cleanup() {
    echo "[CLEANUP] Stopping victim service..."
    stop_victim_service
}

trap cleanup EXIT INT TERM

# 1. Provide paths
SAVE_PATH="privacy_attacker_v1"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

echo "Starting Victim (Defender) Model API service in the background..."
# Booting up the background inference engine to simulate the Victim interaction
start_victim_service "$VICTIM_MODEL" "mock_run"
sleep 45  # Wait for vLLM models to fully load into memory before training starts

# 2. Run verl trainer on our privacy prompts and reward functions
CUDA_VISIBLE_DEVICES=$ATTACKER_GPU python3 -m verl.trainer.main \
    config=$PROJECT_DIR/examples/config.yaml \
    algorithm.adv_estimator=grpo \
    data.max_response_length=512 \
    worker.actor.model.model_path=$BASE_MODEL \
    trainer.experiment_name=$SAVE_PATH \
    trainer.save_checkpoint_path=$WORKSPACE/storage/models/$SAVE_PATH \
    trainer.total_epochs=10 \
    worker.reward.reward_function=${PROJECT_DIR}/reward/verl_reward_wrapper.py:compute_score \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=1 \
    worker.rollout.n=4 \
    worker.actor.global_batch_size=4 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    trainer.max_steps=50

echo "Privacy Attacker RL training completed!"

echo "Cleaning up background Victim API services..."
stop_victim_service
sleep 2
