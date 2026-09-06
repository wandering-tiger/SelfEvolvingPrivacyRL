#!/bin/bash
# Iterative challenger-style training with Agent-R1 (verl 0.7.0):
#   round 0  : build dataset from BASE_MODEL, train with agent_r1
#   round N>0: rebuild dataset from previous round's latest checkpoint, resume training
# Reference: docs/verl_to_agent_r1_migration.md

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
SAMPLES_PER_ROUND=${SAMPLES_PER_ROUND:-128}
REWRITE_BACKEND=${REWRITE_BACKEND:-vllm}
EVOLVE_RATIO=${EVOLVE_RATIO:-0.25}
EVOLVE_GPU_MEM_UTIL=${EVOLVE_GPU_MEM_UTIL:-0.5}
EVOLVE_MAX_MODEL_LEN=${EVOLVE_MAX_MODEL_LEN:-1536}
PL_STYLE_RATIO=${PL_STYLE_RATIO:-0.0}
AGENTDOJO_SIMPLE_RATIO=${AGENTDOJO_SIMPLE_RATIO:-0.0}
# Real AgentDojo suites (banking/slack/travel/workspace)
AGENTDOJO_RATIO=${AGENTDOJO_RATIO:-0.0}
AGENTDOJO_SUITES=${AGENTDOJO_SUITES:-workspace,travel,banking,slack}

# Keep defender trajectories bounded.  Most included tasks need at most one
# lookup, one action, and a final answer.
export AGENTDOJO_MAX_STEPS=${AGENTDOJO_MAX_STEPS:-3}
export GUARD_MAX_TOKENS=${GUARD_MAX_TOKENS:-256}

ATTACKER_GPU=${ATTACKER_GPU:-0}
IFS=',' read -ra ATTACKER_GPU_LIST <<< "$ATTACKER_GPU"
NUM_GPUS=${#ATTACKER_GPU_LIST[@]}
if [[ $NUM_GPUS -lt 1 ]]; then
  NUM_GPUS=1
fi

# Python interpreter: use the AgentPrivacy conda env by default (verl==0.7.0 installed there).
PYTHON=${PYTHON:-/home/fangzibang/.conda/envs/AgentPrivacy/bin/python}
# Agent-R1 repo root: training MUST be launched from this directory so the
# pip-installed verl 0.7.0 is not shadowed by the vendored verl/ in this repo.
AGENT_R1_ROOT=${AGENT_R1_ROOT:-/home/fangzibang/Agent-R1}

echo "[INFO] vLLM services are started by scripts/main.sh"

for ((round=0; round<ROUNDS; round++)); do
  echo "===== Iteration $round/$ROUNDS ====="

  if [[ $round -eq 0 ]]; then
    MODEL_FOR_DATA="$BASE_MODEL"
    LOAD_CKPT=""
    ROUND_EVOLVE_RATIO=0.0
  else
    PREV_ROUND_DIR="$WORKSPACE/data_32T/fzb_data/storage/models/$SAVE_PREFIX/round_$((round-1))"
    # Load the latest (most trained) checkpoint from the previous round.
    # With cumulative max_steps, the latest checkpoint always has training
    # budget remaining for the current round.
    LOAD_CKPT=$(ls -d "$PREV_ROUND_DIR"/global_step_* 2>/dev/null | sort -V | tail -1)
    if [[ -z "$LOAD_CKPT" ]]; then
      echo "[WARN] No global_step_* found in $PREV_ROUND_DIR, starting fresh"
      LOAD_CKPT=""
      MODEL_FOR_DATA="$BASE_MODEL"
      ROUND_EVOLVE_RATIO=0.0
    else
      echo "[INFO] Loading latest checkpoint from previous round: $LOAD_CKPT"
      MODEL_FOR_DATA="$LOAD_CKPT/actor/huggingface"
      if [[ ! -f "$MODEL_FOR_DATA/config.json" ]]; then
        echo "[ERROR] Missing HF checkpoint for data evolution: $MODEL_FOR_DATA" >&2
        exit 1
      fi
      ROUND_EVOLVE_RATIO=$EVOLVE_RATIO
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
    --evolve_ratio $ROUND_EVOLVE_RATIO
    --vllm_gpu_mem_util $EVOLVE_GPU_MEM_UTIL
    --vllm_max_model_len $EVOLVE_MAX_MODEL_LEN
    --output_dir $PROJECT_DIR/data
    --attack_strategy direct,indirect,roleplay,reasoning,multi_turn
    --pl_style_ratio $PL_STYLE_RATIO
    --agentdojo_simple_ratio $AGENTDOJO_SIMPLE_RATIO
    --agentdojo_ratio $AGENTDOJO_RATIO
    --agentdojo_suites "$AGENTDOJO_SUITES"
  )

  CUDA_VISIBLE_DEVICES=$ATTACKER_GPU \
    $PYTHON $PROJECT_DIR/build_dataset.py "${BUILD_DATA_ARGS[@]}"

  # JSON -> parquet (Agent-R1 / verl 0.7.0 data contract)
  $PYTHON $PROJECT_DIR/scripts/convert_json_to_parquet.py \
    $PROJECT_DIR/data/train.json $PROJECT_DIR/data/train.parquet
  $PYTHON $PROJECT_DIR/scripts/convert_json_to_parquet.py \
    $PROJECT_DIR/data/val.json $PROJECT_DIR/data/val.parquet

  SAVE_PATH="$WORKSPACE/data_32T/fzb_data/storage/models/$SAVE_PREFIX/round_${round}"

  # Cumulative max_steps: round R has (R+1)*STEPS_PER_ROUND total steps so each round
  # adds STEPS_PER_ROUND new training steps regardless of the loaded checkpoint step.
  CUMULATIVE_MAX_STEPS=$(( (round + 1) * STEPS_PER_ROUND ))

  TRAIN_ARGS=(
    data.train_files=$PROJECT_DIR/data/train.parquet
    data.val_files=$PROJECT_DIR/data/val.parquet
    actor_rollout_ref.model.path=$BASE_MODEL
    trainer.experiment_name=${SAVE_PREFIX}_round_${round}
    trainer.default_local_dir=$SAVE_PATH
    # verl 0.7.0: step count is capped by total_epochs × dataloader len, and
    # stops early when global_steps reaches total_training_steps. So keep
    # total_epochs large and let total_training_steps control the real budget.
    trainer.total_training_steps=$CUMULATIVE_MAX_STEPS
    trainer.n_gpus_per_node=$NUM_GPUS
  )

  # Stable defaults live in examples/config.yaml. Preserve the existing
  # environment-variable knobs as opt-in overrides for experiments.
  append_env_override() {
    local env_name="$1"
    local config_key="$2"
    local value="${!env_name-}"
    if [[ -n "$value" ]]; then
      TRAIN_ARGS+=("$config_key=$value")
    fi
  }
  append_env_override TRAIN_BATCH_SIZE data.train_batch_size
  append_env_override MAX_PROMPT_LENGTH data.max_prompt_length
  append_env_override MAX_RESPONSE_LENGTH data.max_response_length
  append_env_override ROLLOUT_MAX_MODEL_LEN actor_rollout_ref.rollout.max_model_len
  append_env_override PPO_MINI_BATCH_SIZE actor_rollout_ref.actor.ppo_mini_batch_size
  append_env_override KL_LOSS_COEF actor_rollout_ref.actor.kl_loss_coef
  append_env_override ENTROPY_COEFF actor_rollout_ref.actor.entropy_coeff
  append_env_override ROLLOUT_TP actor_rollout_ref.rollout.tensor_model_parallel_size
  append_env_override ROLLOUT_GPU_MEM_UTIL actor_rollout_ref.rollout.gpu_memory_utilization
  append_env_override ROLLOUT_N actor_rollout_ref.rollout.n
  append_env_override ROLLOUT_TEMPERATURE actor_rollout_ref.rollout.temperature
  append_env_override ROLLOUT_TOP_P actor_rollout_ref.rollout.top_p
  append_env_override AGENT_FLOW_WORKERS actor_rollout_ref.rollout.agent.num_workers
  append_env_override LOG_VAL_GENERATIONS trainer.log_val_generations
  append_env_override TOTAL_EPOCHS trainer.total_epochs
  append_env_override VAL_BEFORE_TRAIN trainer.val_before_train
  append_env_override SAVE_FREQ trainer.save_freq
  append_env_override SAVE_LIMIT trainer.max_actor_ckpt_to_keep

  if [[ -n "$LOAD_CKPT" ]]; then
    TRAIN_ARGS+=(trainer.resume_mode=resume_path)
    TRAIN_ARGS+=(trainer.resume_from_path=$LOAD_CKPT)
  fi

  echo "[TRAIN][AGENT-R1] round=$round gpus=$ATTACKER_GPU steps=$CUMULATIVE_MAX_STEPS"
  (cd "$AGENT_R1_ROOT" && CUDA_VISIBLE_DEVICES=$ATTACKER_GPU \
    $PYTHON -m agent_r1.trainer.main_agent_ppo \
      --config-dir="$PROJECT_DIR/examples" \
      --config-name=config \
      "${TRAIN_ARGS[@]}")

  echo "[DEBUG] ATTACKER_GPU=$ATTACKER_GPU (used for attacker training)"

done

echo "Iterative training completed!"

echo "[INFO] Cleanup handled by scripts/main.sh"
