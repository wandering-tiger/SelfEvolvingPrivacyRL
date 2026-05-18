#!/bin/bash
# Train the privacy attacker by reusing R-Zero's `verl` framework.

# 1. Provide paths
BASE_MODEL="/home/fangzibang/data_32T/fzb_data/ModelScope/Qwen3-4B-Base" # Base model to be fine-tuned as attacker
VICTIM_MODEL="/home/fangzibang/data_32T/fzb_data/ModelScope/Qwen3-4B-Base" # Base model used as the defending victim agent
SAVE_PATH="privacy_attacker_v1"
WORKSPACE="/home/fangzibang"
PROJECT_DIR="$WORKSPACE/SelfEvolvingPrivacyRL"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

echo "Starting Victim (Defender) Model API service in the background..."
# Booting up the background inference engine to simulate the Victim interaction
(cd $PROJECT_DIR && bash vllm_service_init/start.sh $VICTIM_MODEL mock_run) &
sleep 45  # Wait for vLLM models to fully load into memory before training starts

# 2. Run verl trainer on our privacy prompts and reward functions
CUDA_VISIBLE_DEVICES=1 python3 -m verl.trainer.main \
    config=$PROJECT_DIR/examples/config.yaml \
    algorithm.adv_estimator=grpo \
    data.max_response_length=512 \
    worker.actor.model.model_path=$BASE_MODEL \
    trainer.experiment_name=$SAVE_PATH \
    trainer.save_checkpoint_path=$WORKSPACE/storage/models/$SAVE_PATH \
    trainer.total_epochs=10 \
    worker.reward.reward_function=${PROJECT_DIR}/reward/verl_reward_wrapper.py:compute_score \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=2 \
    worker.rollout.n=4 \
    worker.actor.global_batch_size=4 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    trainer.max_steps=50

echo "Privacy Attacker RL training completed!"

echo "Cleaning up background Victim API services..."
pkill -f "start_vllm_server.py"
sleep 2
