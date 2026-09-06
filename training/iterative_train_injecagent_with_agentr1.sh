#!/bin/bash
# Iterative modular-InjecAgent attacker curriculum.  Separate from AgentDojo.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE=${WORKSPACE:-/home/fangzibang}
PYTHON=${PYTHON:-/home/fangzibang/.conda/envs/AgentPrivacy/bin/python}
AGENT_R1_ROOT=${AGENT_R1_ROOT:-/home/fangzibang/Agent-R1}
BASE_MODEL=${BASE_MODEL:-/home/fangzibang/data_32T/fzb_data/ModelScope/Qwen3-4B-Instruct-2507}
ATTACKER_GPU=${ATTACKER_GPU:-1,2}
ROUNDS=${ROUNDS:-3}
STEPS_PER_ROUND=${STEPS_PER_ROUND:-50}
SAMPLES_PER_ROUND=${SAMPLES_PER_ROUND:-256}
INJECAGENT_SETTING=${INJECAGENT_SETTING:-enhanced}
INJECAGENT_ATTACK_KINDS=${INJECAGENT_ATTACK_KINDS:-dh,ds}
SAVE_PREFIX=${SAVE_PREFIX:-privacy_attacker_injecagent_$(date +%Y%m%d_%H%M%S)}

IFS=',' read -ra GPU_LIST <<< "$ATTACKER_GPU"
NUM_GPUS=${#GPU_LIST[@]}
# FSDP ranks require Ray's normal per-rank CUDA visibility mapping.  Do not
# set RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES here: it would leave rank 1
# seeing a one-device mask and cause an invalid-device-ordinal failure.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
for ((round=0; round<ROUNDS; round++)); do
  LOAD_CKPT=""
  if [[ $round -gt 0 ]]; then
    previous="$WORKSPACE/data_32T/fzb_data/storage/models/$SAVE_PREFIX/round_$((round-1))"
    LOAD_CKPT=$(ls -d "$previous"/global_step_* 2>/dev/null | sort -V | tail -1 || true)
    [[ -n "$LOAD_CKPT" ]] || { echo "Missing previous checkpoint: $previous" >&2; exit 1; }
  fi
  "$PYTHON" "$PROJECT_DIR/scripts/build_injecagent_dataset.py" \
    --num-samples "$SAMPLES_PER_ROUND" --output-dir "$PROJECT_DIR/data/injecagent" \
    --setting "$INJECAGENT_SETTING" --attack-kinds "$INJECAGENT_ATTACK_KINDS"
  "$PYTHON" "$PROJECT_DIR/scripts/convert_json_to_parquet.py" "$PROJECT_DIR/data/injecagent/train.json" "$PROJECT_DIR/data/injecagent/train.parquet"
  "$PYTHON" "$PROJECT_DIR/scripts/convert_json_to_parquet.py" "$PROJECT_DIR/data/injecagent/val.json" "$PROJECT_DIR/data/injecagent/val.parquet"
  save_path="$WORKSPACE/data_32T/fzb_data/storage/models/$SAVE_PREFIX/round_$round"
  args=(
    data.train_files="$PROJECT_DIR/data/injecagent/train.parquet"
    data.val_files="$PROJECT_DIR/data/injecagent/val.parquet"
    actor_rollout_ref.model.path="$BASE_MODEL"
    trainer.experiment_name="${SAVE_PREFIX}_round_${round}"
    trainer.default_local_dir="$save_path"
    trainer.total_training_steps=$(((round+1)*STEPS_PER_ROUND))
    trainer.n_gpus_per_node="$NUM_GPUS"
    +ray_kwargs.ray_init.num_gpus="$NUM_GPUS"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.8}"
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN:-1024}"
  )
  [[ -n "$LOAD_CKPT" ]] && args+=(trainer.resume_mode=resume_path trainer.resume_from_path="$LOAD_CKPT")
  (cd "$AGENT_R1_ROOT" && CUDA_VISIBLE_DEVICES="$ATTACKER_GPU" \
    "$PYTHON" -m agent_r1.trainer.main_agent_ppo --config-dir="$PROJECT_DIR/examples" --config-name=config "${args[@]}")
done
