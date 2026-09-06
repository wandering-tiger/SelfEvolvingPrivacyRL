#!/bin/bash
# Direct AgentDojo defender training.  No external guard vLLM is started.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../scripts/common.sh"
# Dataset builders are project Python modules; make this launcher independent
# of the shell directory from which it is invoked.
cd "$PROJECT_DIR"

ROUNDS=${ROUNDS:-1}
STEPS_PER_ROUND=${STEPS_PER_ROUND:-10}
SAMPLES_PER_ROUND=${SAMPLES_PER_ROUND:-32}
DEFENDER_CLEAN_RATIO=${DEFENDER_CLEAN_RATIO:-0.2}
DEFENDER_SUITES=${DEFENDER_SUITES:-workspace}
# Offline attacker generations.  These are injected directly into AgentDojo
# and avoid an online attacker/vLLM during defender PPO rollout.
DEFENDER_FROZEN_ATTACKS=${DEFENDER_FROZEN_ATTACKS:-}
DEFENDER_REQUIRE_FROZEN_ATTACKS=${DEFENDER_REQUIRE_FROZEN_ATTACKS:-0}
# Require frozen samples to have been rendered from the same attacker action
# schema used during the current native_augmented attacker training.
DEFENDER_ATTACK_TEMPLATE=${DEFENDER_ATTACK_TEMPLATE:-native_augmented}
DEFENDER_LORA_RANK=${DEFENDER_LORA_RANK:-0}
DEFENDER_LORA_ALPHA=${DEFENDER_LORA_ALPHA:-64}
DEFENDER_GPU=${DEFENDER_GPU:-1,2,3}
PYTHON=${PYTHON:-/home/fangzibang/.conda/envs/AgentPrivacy/bin/python}
AGENT_R1_ROOT=${AGENT_R1_ROOT:-/home/fangzibang/Agent-R1}
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SAVE_PREFIX=${SAVE_PREFIX:-privacy_defender_iterative_${RUN_TIMESTAMP}}
IFS=',' read -ra GPU_LIST <<< "$DEFENDER_GPU"
NUM_GPUS=${#GPU_LIST[@]}

for ((round=0; round<ROUNDS; round++)); do
  "$PYTHON" -c "from defender.agentdojo_defender_dataset import build_agentdojo_defender_records; import json; from pathlib import Path; r=build_agentdojo_defender_records($SAMPLES_PER_ROUND, '$DEFENDER_SUITES'.split(','), $DEFENDER_CLEAN_RATIO, frozen_attacks_path='$DEFENDER_FROZEN_ATTACKS' or None, require_frozen_attacks=bool(int('$DEFENDER_REQUIRE_FROZEN_ATTACKS')), expected_attack_template='$DEFENDER_ATTACK_TEMPLATE' if '$DEFENDER_FROZEN_ATTACKS' else None); Path('$PROJECT_DIR/data').mkdir(exist_ok=True); Path('$PROJECT_DIR/data/defender_train.json').write_text(json.dumps(r, ensure_ascii=False), encoding='utf-8'); Path('$PROJECT_DIR/data/defender_val.json').write_text(json.dumps(r[:max(1, len(r)//8)], ensure_ascii=False), encoding='utf-8')"
  "$PYTHON" "$PROJECT_DIR/scripts/convert_json_to_parquet.py" "$PROJECT_DIR/data/defender_train.json" "$PROJECT_DIR/data/defender_train.parquet" privacy_defender
  "$PYTHON" "$PROJECT_DIR/scripts/convert_json_to_parquet.py" "$PROJECT_DIR/data/defender_val.json" "$PROJECT_DIR/data/defender_val.parquet" privacy_defender
  LOAD_ARGS=()
  if [[ $round -gt 0 ]]; then
    PREV_ROUND_DIR="$WORKSPACE/data_32T/fzb_data/storage/models/$SAVE_PREFIX/round_$((round - 1))"
    PREV_CKPT=$(find "$PREV_ROUND_DIR" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -n 1)
    if [[ -z "$PREV_CKPT" ]]; then
      echo "[ERROR] defender resume requested but no checkpoint found in $PREV_ROUND_DIR" >&2
      exit 1
    fi
    echo "[INFO] Resuming defender from $PREV_CKPT"
    LOAD_ARGS=(trainer.resume_mode=resume_path trainer.resume_from_path="$PREV_CKPT")
  fi
  # Do not put PROJECT_DIR itself on PYTHONPATH: this repository contains a
  # legacy top-level ``verl`` package that would shadow Agent R1's installed
  # verl 0.7.  The two source subdirectories expose only our custom modules.
  (cd "$AGENT_R1_ROOT" && PYTHONPATH="$PROJECT_DIR/defender:$PROJECT_DIR/reward:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="$DEFENDER_GPU" "$PYTHON" -m agent_r1.trainer.main_agent_ppo \
    --config-dir="$PROJECT_DIR/examples" --config-name=config_defender \
    data.train_files="$PROJECT_DIR/data/defender_train.parquet" data.val_files="$PROJECT_DIR/data/defender_val.parquet" \
    actor_rollout_ref.model.path="$BASE_MODEL" trainer.experiment_name="${SAVE_PREFIX}_round_${round}" \
    trainer.default_local_dir="$WORKSPACE/data_32T/fzb_data/storage/models/$SAVE_PREFIX/round_${round}" \
    trainer.total_training_steps=$(( (round + 1) * STEPS_PER_ROUND )) trainer.n_gpus_per_node="$NUM_GPUS" \
    data.train_batch_size="$NUM_GPUS" actor_rollout_ref.actor.ppo_mini_batch_size="$NUM_GPUS" \
    actor_rollout_ref.model.lora_rank="$DEFENDER_LORA_RANK" actor_rollout_ref.model.lora_alpha="$DEFENDER_LORA_ALPHA" \
    "${LOAD_ARGS[@]}")
done
