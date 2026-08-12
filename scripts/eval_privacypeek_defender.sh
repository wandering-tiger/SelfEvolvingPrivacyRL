#!/bin/bash
# One-click PrivacyPeek evaluation for a trained defender checkpoint.
#
# Steps: merge FSDP → HF → start vLLM → run agent → CER judge → cleanup
#
# Usage:
#   bash scripts/eval_privacypeek_defender.sh              # latest ckpt, limit=50
#   bash scripts/eval_privacypeek_defender.sh latest 20     # latest ckpt, limit=20
#   bash scripts/eval_privacypeek_defender.sh /path/to/global_step_10 50
#   bash scripts/eval_privacypeek_defender.sh baseline 50   # base model, no merge
#
# Env overrides:
#   EVAL_GPU=4  EVAL_PORT=8000  WORLD_SIZE=4  HF_OUT=...

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

WORKSPACE=${WORKSPACE:-/home/fangzibang}
PRIVACYPEEK_ROOT=${PRIVACYPEEK_ROOT:-/home/fangzibang/PrivacyPeek-Resource}
BASE_MODEL=${BASE_MODEL:-/home/fangzibang/data_32T/fzb_data/ModelScope/Qwen3-4B-Instruct-2507}
EVAL_GPU=${EVAL_GPU:-4}
EVAL_PORT=${EVAL_PORT:-8000}
WORLD_SIZE=${WORLD_SIZE:-4}
SERVED_NAME=${SERVED_MODEL_NAME:-defender}
LIMIT=${2:-${EVAL_LIMIT:-50}}
CKPT_ARG=${1:-latest}
RUN_TAG=$(date +"%Y%m%d_%H%M%S")
RUNS_DIR=${RUNS_DIR:-$PROJECT_DIR/runs}
mkdir -p "$RUNS_DIR" "$WORKSPACE/storage/logs" "$WORKSPACE/storage/models"

VLLM_PID=""
cleanup() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[CLEANUP] Stopping vLLM pid=$VLLM_PID ..."
    kill "$VLLM_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$VLLM_PID" 2>/dev/null || true
  fi
  fuser -k "${EVAL_PORT}/tcp" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

resolve_ckpt() {
  local arg="$1"
  if [[ "$arg" == "baseline" ]]; then
    echo ""
    return
  fi
  if [[ "$arg" == "latest" ]]; then
    local found
    found=$(ls -d "$WORKSPACE"/storage/models/privacy_defender_privacypeek_*/round_*/global_step_* 2>/dev/null | sort -V | tail -1 || true)
    if [[ -z "$found" ]]; then
      echo "[ERROR] No checkpoint under $WORKSPACE/storage/models/privacy_defender_privacypeek_*/" >&2
      exit 1
    fi
    echo "$found"
    return
  fi
  if [[ -d "$arg" ]]; then
    echo "$arg"
    return
  fi
  echo "[ERROR] Checkpoint not found: $arg" >&2
  exit 1
}

wait_for_vllm() {
  local port="$1"
  local timeout="${VLLM_READY_TIMEOUT:-600}"
  local elapsed=0
  echo "[VLLM] Waiting for http://127.0.0.1:${port}/v1/models (timeout ${timeout}s)..."
  while (( elapsed < timeout )); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[VLLM] Ready (${elapsed}s)"
      return 0
    fi
    if [[ -n "$VLLM_PID" ]] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "[ERROR] vLLM exited early. Log: $VLLM_LOG" >&2
      tail -40 "$VLLM_LOG" >&2 || true
      exit 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo "[ERROR] vLLM not ready after ${timeout}s. Log: $VLLM_LOG" >&2
  tail -40 "$VLLM_LOG" >&2 || true
  exit 1
}

CKPT=$(resolve_ckpt "$CKPT_ARG")

if [[ "$CKPT_ARG" == "baseline" ]]; then
  MODEL_PATH=$BASE_MODEL
  SERVED_NAME=${SERVED_MODEL_NAME:-llama31-base}
  AGENT_OUT=$RUNS_DIR/baseline_${RUN_TAG}.json
  CER_OUT=$RUNS_DIR/cer_baseline_${RUN_TAG}.json
  echo "[EVAL] mode=baseline model=$MODEL_PATH"
else
  HF_OUT=${HF_OUT:-$WORKSPACE/storage/models/privacy_defender_hf_eval}
  echo "[EVAL] mode=trained ckpt=$CKPT"
  echo "[MERGE] FSDP -> HF at $HF_OUT (world_size=$WORLD_SIZE)"
  python3 "$PROJECT_DIR/scripts/merge_fsdp_checkpoint.py" \
    --ckpt "$CKPT" \
    --base-model "$BASE_MODEL" \
    --output "$HF_OUT" \
    --world-size "$WORLD_SIZE"
  MODEL_PATH=$HF_OUT
  AGENT_OUT=$RUNS_DIR/defender_${RUN_TAG}.json
  CER_OUT=$RUNS_DIR/cer_defender_${RUN_TAG}.json
fi

# Free eval port / leftover servers
fuser -k "${EVAL_PORT}/tcp" 2>/dev/null || true
sleep 1

VLLM_LOG=$WORKSPACE/storage/logs/eval_vllm_${RUN_TAG}.log
echo "[VLLM] Starting on GPU $EVAL_GPU port=$EVAL_PORT model=$MODEL_PATH"
echo "       log=$VLLM_LOG"
CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --port "$EVAL_PORT" \
  --gpu-memory-utilization "${EVAL_GPU_MEM_UTIL:-0.85}" \
  --max-model-len "${EVAL_MAX_MODEL_LEN:-8192}" \
  >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!
wait_for_vllm "$EVAL_PORT"

echo "[AGENT] Running PrivacyPeek agent (limit=$LIMIT) ..."
python "$PRIVACYPEEK_ROOT/evaluation/run_agent_opensource.py" \
  --model-name "$SERVED_NAME" \
  --vllm-endpoint "http://127.0.0.1:${EVAL_PORT}/v1" \
  --cases-dir "$PRIVACYPEEK_ROOT/data/cases" \
  --limit "$LIMIT" \
  --output "$AGENT_OUT"

echo "[CER] Scoring ..."
python "$PRIVACYPEEK_ROOT/evaluation/judges/cer_exact_match.py" \
  --input "$AGENT_OUT" \
  --output "$CER_OUT"

echo "============================================"
echo " Eval done"
echo " agent runs : $AGENT_OUT"
echo " CER output : $CER_OUT"
echo "============================================"
