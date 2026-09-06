#!/bin/bash
# Iterative full-AgentDojo attacker training using native AgentDojo attack names.
# This is intentionally separate from iterative_train_with_agentr1.sh.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE=${WORKSPACE:-/home/fangzibang}
PYTHON=${PYTHON:-/home/fangzibang/.conda/envs/AgentPrivacy/bin/python}
AGENT_R1_ROOT=${AGENT_R1_ROOT:-/home/fangzibang/Agent-R1}

RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SAVE_PREFIX=${SAVE_PREFIX:-privacy_attacker_native_agentdojo_${RUN_TIMESTAMP}}
ROUNDS=${ROUNDS:-3}
STEPS_PER_ROUND=${STEPS_PER_ROUND:-50}
SAMPLES_PER_ROUND=${SAMPLES_PER_ROUND:-128}
AGENTDOJO_SUITES=${AGENTDOJO_SUITES:-workspace,travel,banking,slack}
AGENTDOJO_NATIVE_ATTACK_TYPES=${AGENTDOJO_NATIVE_ATTACK_TYPES:-direct,ignore_previous,system_message,injecagent,important_instructions,tool_knowledge}
ATTACKER_GPU=${ATTACKER_GPU:-0}
BASE_MODEL=${BASE_MODEL:-/home/fangzibang/data_32T/fzb_data/ModelScope/Qwen3-4B-Instruct-2507}
AGENTDOJO_SRC=${AGENTDOJO_SRC:-/home/fangzibang/agentdojo/src}

IFS=',' read -ra ATTACKER_GPU_LIST <<< "$ATTACKER_GPU"
NUM_GPUS=${#ATTACKER_GPU_LIST[@]}
export AGENTDOJO_MAX_STEPS=${AGENTDOJO_MAX_STEPS:-3}
export GUARD_MAX_TOKENS=${GUARD_MAX_TOKENS:-256}

for ((round=0; round<ROUNDS; round++)); do
  if [[ $round -eq 0 ]]; then
    LOAD_CKPT=""
  else
    PREV_ROUND_DIR="$WORKSPACE/data_32T/fzb_data/storage/models/$SAVE_PREFIX/round_$((round - 1))"
    LOAD_CKPT=$(ls -d "$PREV_ROUND_DIR"/global_step_* 2>/dev/null | sort -V | tail -1 || true)
    if [[ -z "$LOAD_CKPT" ]]; then
      echo "[ERROR] Missing previous checkpoint under $PREV_ROUND_DIR" >&2
      exit 1
    fi
  fi

  echo "[NATIVE-AGENTDOJO] round=$round types=$AGENTDOJO_NATIVE_ATTACK_TYPES"
  PYTHONPATH="$AGENTDOJO_SRC" "$PYTHON" "$PROJECT_DIR/scripts/build_native_agentdojo_dataset.py" \
    --num-samples "$SAMPLES_PER_ROUND" \
    --output-dir "$PROJECT_DIR/data" \
    --suites "$AGENTDOJO_SUITES" \
    --attack-types "$AGENTDOJO_NATIVE_ATTACK_TYPES"
  PYTHONPATH="$AGENTDOJO_SRC" "$PYTHON" "$PROJECT_DIR/scripts/convert_json_to_parquet.py" "$PROJECT_DIR/data/train.json" "$PROJECT_DIR/data/train.parquet"
  PYTHONPATH="$AGENTDOJO_SRC" "$PYTHON" "$PROJECT_DIR/scripts/convert_json_to_parquet.py" "$PROJECT_DIR/data/val.json" "$PROJECT_DIR/data/val.parquet"

  SAVE_PATH="$WORKSPACE/data_32T/fzb_data/storage/models/$SAVE_PREFIX/round_${round}"
  CUMULATIVE_MAX_STEPS=$(((round + 1) * STEPS_PER_ROUND))
  TRAIN_ARGS=(
    data.train_files="$PROJECT_DIR/data/train.parquet"
    data.val_files="$PROJECT_DIR/data/val.parquet"
    actor_rollout_ref.model.path="$BASE_MODEL"
    trainer.experiment_name="${SAVE_PREFIX}_round_${round}"
    trainer.default_local_dir="$SAVE_PATH"
    trainer.total_training_steps="$CUMULATIVE_MAX_STEPS"
    trainer.n_gpus_per_node="$NUM_GPUS"
  )
  if [[ -n "$LOAD_CKPT" ]]; then
    TRAIN_ARGS+=(trainer.resume_mode=resume_path trainer.resume_from_path="$LOAD_CKPT")
  fi

  # Do not add PROJECT_DIR to PYTHONPATH here: it would shadow Agent-R1's
  # installed verl with this repository's legacy verl directory.
  (cd "$AGENT_R1_ROOT" && CUDA_VISIBLE_DEVICES="$ATTACKER_GPU" PYTHONPATH="$AGENTDOJO_SRC" \
    "$PYTHON" -m agent_r1.trainer.main_agent_ppo \
      --config-dir="$PROJECT_DIR/examples" --config-name=config "${TRAIN_ARGS[@]}")
done
