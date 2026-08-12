#!/bin/bash
# Attacker GRPO training on PrivacyPeek cases (Path A).
# Actor rewrites benign tasks; frozen guard runs in PrivacyPeek env; reward = CER (+ forbidden bonus).

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../scripts/common.sh"

cleanup() {
  echo "[CLEANUP] Stopping guard service..."
  stop_guard_service
}

trap cleanup EXIT INT TERM

RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SAVE_PREFIX=${SAVE_PREFIX:-privacy_attacker_privacypeek_${RUN_TIMESTAMP}}
ROUNDS=${ROUNDS:-1}
STEPS_PER_ROUND=${STEPS_PER_ROUND:-50}
PRIVACYPEEK_CASES_DIR=${PRIVACYPEEK_CASES_DIR:-${PRIVACYPEEK_ROOT:-/home/fangzibang/PrivacyPeek-Resource}/data/cases}
PRIVACYPEEK_DIMS=${PRIVACYPEEK_DIMS:-}
PRIVACYPEEK_LIMIT=${PRIVACYPEEK_LIMIT:-}
DATA_OUTPUT_DIR=${DATA_OUTPUT_DIR:-$PROJECT_DIR/data/privacypeek_attacker}
CONFIG_PATH=${CONFIG_PATH:-$PROJECT_DIR/examples/config_privacypeek_attacker.yaml}
REWARD_FN=${REWARD_FN:-$PROJECT_DIR/reward/privacypeek_attacker_reward_wrapper.py:compute_score}

TRAIN_GPUS=${TRAIN_GPUS:-0,1,3,4}
GUARD_GPU=${GUARD_GPU:-2}
EFFECTIVE_TRAIN_GPUS=$(resolve_train_gpus "$TRAIN_GPUS" "$GUARD_GPU")
IFS=',' read -ra TRAIN_GPU_LIST <<< "$EFFECTIVE_TRAIN_GPUS"
NUM_GPUS=${#TRAIN_GPU_LIST[@]}
if [[ $NUM_GPUS -lt 1 ]]; then
  NUM_GPUS=1
fi
TRAIN_GPUS=$EFFECTIVE_TRAIN_GPUS

read -r ROLLOUT_TP ROLLOUT_GPU_MEM_UTIL <<< "$(resolve_rollout_settings "$NUM_GPUS")"
export ROLLOUT_TP
export ROLLOUT_GPU_MEM_UTIL

export PRIVACYPEEK_ROOT=${PRIVACYPEEK_ROOT:-/home/fangzibang/PrivacyPeek-Resource}
export PRIVACYPEEK_CASES_DIR
export GUARD_OPENAI_URL=${GUARD_OPENAI_URL:-http://127.0.0.1:${GUARD_PORT}/v1/chat/completions}
export GUARD_MODEL=${GUARD_MODEL:-$BASE_MODEL}
export DISABLE_KL=${DISABLE_KL:-true}
export ROLLOUT_BATCH_CAP=${ROLLOUT_BATCH_CAP:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

echo "[INFO] PrivacyPeek attacker training"
echo "  cases_dir=$PRIVACYPEEK_CASES_DIR"
echo "  config=$CONFIG_PATH"
echo "  guard_url=$GUARD_OPENAI_URL"
echo "  guard_gpu=$GUARD_GPU"
echo "  train_gpus=$TRAIN_GPUS ($NUM_GPUS GPUs, rollout_tp=$ROLLOUT_TP)"
echo "  disable_kl=$DISABLE_KL rollout_gpu_mem=$ROLLOUT_GPU_MEM_UTIL rollout_batch_cap=$ROLLOUT_BATCH_CAP"

for ((round=0; round<ROUNDS; round++)); do
  echo "===== Attacker round $round/$ROUNDS ====="

  if [[ $round -eq 0 ]]; then
    LOAD_CKPT=""
  else
    PREV_ROUND_DIR="$WORKSPACE/storage/models/$SAVE_PREFIX/round_$((round-1))"
    LOAD_CKPT=$(ls -d "$PREV_ROUND_DIR"/global_step_* 2>/dev/null | sort -V | tail -1)
    if [[ -z "$LOAD_CKPT" ]]; then
      echo "[WARN] No checkpoint found in $PREV_ROUND_DIR, starting fresh"
      LOAD_CKPT=""
    else
      echo "[INFO] Loading checkpoint: $LOAD_CKPT"
    fi
  fi

  BUILD_ARGS=(
    --cases-dir "$PRIVACYPEEK_CASES_DIR"
    --output-dir "$DATA_OUTPUT_DIR"
    --val-ratio 0.1
  )
  if [[ -n "$PRIVACYPEEK_DIMS" ]]; then
    BUILD_ARGS+=(--dims "$PRIVACYPEEK_DIMS")
  fi
  if [[ -n "$PRIVACYPEEK_LIMIT" ]]; then
    BUILD_ARGS+=(--limit "$PRIVACYPEEK_LIMIT")
  fi

  echo "[DATA] Building PrivacyPeek attacker dataset..."
  python3 "$PROJECT_DIR/build_privacypeek_attacker_dataset.py" "${BUILD_ARGS[@]}"

  TRAIN_JSON="$DATA_OUTPUT_DIR/train.json"
  VAL_JSON="$DATA_OUTPUT_DIR/val.json"
  read -r ROLLOUT_BATCH VAL_BATCH TRAIN_SIZE VAL_SIZE <<< "$(resolve_batch_sizes "$TRAIN_JSON" "$VAL_JSON" "$NUM_GPUS")"
  GLOBAL_BATCH=$ROLLOUT_BATCH

  echo "[DATA] train=$TRAIN_SIZE val=$VAL_SIZE rollout_batch=$ROLLOUT_BATCH val_batch=$VAL_BATCH"

  SAVE_PATH="$WORKSPACE/storage/models/$SAVE_PREFIX/round_${round}"
  CUMULATIVE_MAX_STEPS=$(( (round + 1) * STEPS_PER_ROUND ))

  TRAIN_ARGS=(
    config="$CONFIG_PATH"
    algorithm.adv_estimator=grpo
    data.train_files="$TRAIN_JSON"
    data.val_files="$VAL_JSON"
    data.prompt_key=prompt
    data.answer_key=case_dir
    data.max_prompt_length=1024
    data.max_response_length=256
    worker.actor.model.model_path="$BASE_MODEL"
    trainer.experiment_name="${SAVE_PREFIX}_round_${round}"
    trainer.save_checkpoint_path="$SAVE_PATH"
    trainer.total_epochs=1
    trainer.max_steps=$CUMULATIVE_MAX_STEPS
    worker.reward.reward_function="$REWARD_FN"
    trainer.val_freq=-1
    trainer.n_gpus_per_node=$NUM_GPUS
    worker.rollout.n=${ROLLOUT_N:-2}
    worker.rollout.tensor_parallel_size=$ROLLOUT_TP
    worker.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL
    worker.rollout.max_model_len=1280
    worker.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS:-2048}
    worker.rollout.enforce_eager=true
    worker.actor.use_torch_compile=false
    worker.ref.offload.offload_params=true
    worker.actor.offload.offload_params=true
    worker.actor.offload.offload_optimizer=true
    worker.actor.global_batch_size=$GLOBAL_BATCH
    worker.actor.micro_batch_size_per_device_for_update=1
    worker.actor.micro_batch_size_per_device_for_experience=1
    data.rollout_batch_size=$ROLLOUT_BATCH
    data.val_batch_size=$VAL_BATCH
  )

  if [[ "${DISABLE_KL:-true}" == "true" ]]; then
    TRAIN_ARGS+=(
      algorithm.disable_kl=true
      algorithm.use_kl_loss=false
    )
  fi

  if [[ -n "$LOAD_CKPT" ]]; then
    TRAIN_ARGS+=(trainer.load_checkpoint_path="$LOAD_CKPT")
  fi

  echo "[TRAIN] CUDA_VISIBLE_DEVICES=$TRAIN_GPUS"
  CUDA_VISIBLE_DEVICES=$TRAIN_GPUS python3 -m verl.trainer.main "${TRAIN_ARGS[@]}"
done

echo "PrivacyPeek attacker training completed!"
