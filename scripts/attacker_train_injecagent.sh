#!/bin/bash
# Launch an InjecAgent attacker curriculum with an isolated local victim.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/fangzibang/SelfEvolvingPrivacyRL}
PYTHON=${PYTHON:-/home/fangzibang/.conda/envs/AgentPrivacy/bin/python}
BASE_MODEL=${BASE_MODEL:-/home/fangzibang/data_32T/fzb_data/ModelScope/Qwen3-4B-Instruct-2507}
ATTACKER_GPU=${ATTACKER_GPU:-1,2}
GUARD_GPU=${GUARD_GPU:-3}
GUARD_PORT=${GUARD_PORT:-5021}
RUN_ID=${1:-injecagent_attacker}
export GUARD_MODEL=${GUARD_MODEL:-$BASE_MODEL}
export GUARD_OPENAI_URL="http://127.0.0.1:${GUARD_PORT}/v1/chat/completions"
export RAY_TMPDIR=${RAY_TMPDIR:-/mnt/32T/data/fzb_data/ray_tmp}
mkdir -p "$RAY_TMPDIR" "$PROJECT_DIR/runs"

CUDA_VISIBLE_DEVICES="$GUARD_GPU" "$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$GUARD_MODEL" --served-model-name "$GUARD_MODEL" --port "$GUARD_PORT" \
  --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL:-0.55}" --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}" \
  --enforce-eager --trust-remote-code --enable-auto-tool-choice --tool-call-parser hermes \
  > "$PROJECT_DIR/runs/${RUN_ID}_guard.log" 2>&1 &
guard_pid=$!
trap 'kill "$guard_pid" 2>/dev/null || true; wait "$guard_pid" 2>/dev/null || true' EXIT INT TERM
for _ in $(seq 1 90); do curl -fsS --max-time 2 "http://127.0.0.1:${GUARD_PORT}/v1/models" >/dev/null && break; sleep 5; done
curl -fsS --max-time 2 "http://127.0.0.1:${GUARD_PORT}/v1/models" >/dev/null || { tail -60 "$PROJECT_DIR/runs/${RUN_ID}_guard.log"; exit 1; }
ATTACKER_GPU="$ATTACKER_GPU" BASE_MODEL="$BASE_MODEL" \
  bash "$PROJECT_DIR/training/iterative_train_injecagent_with_agentr1.sh"
