#!/bin/bash
# One-click PrivacyPeek evaluation for a trained attacker checkpoint.
#
# Two-phase on one GPU by default:
#   1) merge attacker FSDP → serve attacker → generate attack prompts
#   2) stop attacker → serve frozen victim/guard → run PrivacyPeek env → CER
#
# Usage:
#   bash scripts/eval_privacypeek_attacker.sh                 # latest ckpt, limit=50
#   bash scripts/eval_privacypeek_attacker.sh latest 20
#   bash scripts/eval_privacypeek_attacker.sh /path/to/global_step_10 50
#   bash scripts/eval_privacypeek_attacker.sh baseline 50      # base model as attacker
#
# Env overrides:
#   EVAL_GPU=4  EVAL_PORT=8000  WORLD_SIZE=4
#   VICTIM_MODEL=...          # frozen guard (default: BASE_MODEL)
#   HF_OUT=...  ATTACKS_OUT=...

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

WORKSPACE=${WORKSPACE:-/home/fangzibang}
PRIVACYPEEK_ROOT=${PRIVACYPEEK_ROOT:-/home/fangzibang/PrivacyPeek-Resource}
BASE_MODEL=${BASE_MODEL:-/home/fangzibang/data_32T/fzb_data/ModelScope/Qwen3-4B-Instruct-2507}
VICTIM_MODEL=${VICTIM_MODEL:-$BASE_MODEL}
EVAL_GPU=${EVAL_GPU:-4}
EVAL_PORT=${EVAL_PORT:-8000}
WORLD_SIZE=${WORLD_SIZE:-4}
LIMIT=${2:-${EVAL_LIMIT:-50}}
CKPT_ARG=${1:-latest}
RUN_TAG=$(date +"%Y%m%d_%H%M%S")
RUNS_DIR=${RUNS_DIR:-$PROJECT_DIR/runs}
mkdir -p "$RUNS_DIR" "$WORKSPACE/storage/logs" "$WORKSPACE/storage/models"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PRIVACYPEEK_ROOT
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}

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
    found=$(ls -d "$WORKSPACE"/storage/models/privacy_attacker_privacypeek_*/round_*/global_step_* 2>/dev/null | sort -V | tail -1 || true)
    if [[ -z "$found" ]]; then
      echo "[ERROR] No checkpoint under $WORKSPACE/storage/models/privacy_attacker_privacypeek_*/" >&2
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

start_vllm() {
  local model_path="$1"
  local served_name="$2"
  local log_tag="$3"

  fuser -k "${EVAL_PORT}/tcp" 2>/dev/null || true
  sleep 1
  VLLM_LOG=$WORKSPACE/storage/logs/eval_attacker_vllm_${log_tag}_${RUN_TAG}.log
  echo "[VLLM] Starting on GPU $EVAL_GPU port=$EVAL_PORT model=$model_path name=$served_name"
  echo "       log=$VLLM_LOG"
  CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --served-model-name "$served_name" \
    --port "$EVAL_PORT" \
    --gpu-memory-utilization "${EVAL_GPU_MEM_UTIL:-0.85}" \
    --max-model-len "${EVAL_MAX_MODEL_LEN:-8192}" \
    >"$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
  wait_for_vllm "$EVAL_PORT"
}

stop_vllm() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[VLLM] Stopping pid=$VLLM_PID ..."
    kill "$VLLM_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$VLLM_PID" 2>/dev/null || true
  fi
  VLLM_PID=""
  fuser -k "${EVAL_PORT}/tcp" 2>/dev/null || true
  sleep 1
}

CKPT=$(resolve_ckpt "$CKPT_ARG")

if [[ "$CKPT_ARG" == "baseline" ]]; then
  ATTACKER_MODEL_PATH=$BASE_MODEL
  ATTACKER_SERVED=${SERVED_ATTACKER_NAME:-llama31-attacker-base}
  TAG=baseline
  echo "[EVAL] mode=baseline attacker=$ATTACKER_MODEL_PATH"
else
  HF_OUT=${HF_OUT:-$WORKSPACE/storage/models/privacy_attacker_hf_eval}
  echo "[EVAL] mode=trained ckpt=$CKPT"
  echo "[MERGE] FSDP -> HF at $HF_OUT (world_size=$WORLD_SIZE)"
  python3 "$PROJECT_DIR/scripts/merge_fsdp_checkpoint.py" \
    --ckpt "$CKPT" \
    --base-model "$BASE_MODEL" \
    --output "$HF_OUT" \
    --world-size "$WORLD_SIZE"
  ATTACKER_MODEL_PATH=$HF_OUT
  ATTACKER_SERVED=${SERVED_ATTACKER_NAME:-attacker}
  TAG=attacker
fi

VICTIM_SERVED=${SERVED_VICTIM_NAME:-victim}
ATTACKS_OUT=${ATTACKS_OUT:-$RUNS_DIR/attacks_${TAG}_${RUN_TAG}.json}
AGENT_OUT=$RUNS_DIR/${TAG}_${RUN_TAG}.json
CER_OUT=$RUNS_DIR/cer_${TAG}_${RUN_TAG}.json
SUMMARY_OUT=$RUNS_DIR/summary_${TAG}_${RUN_TAG}.json

# ---------- Phase 1: generate attacks ----------
start_vllm "$ATTACKER_MODEL_PATH" "$ATTACKER_SERVED" "attacker"

echo "[PHASE1] Generating attacks (limit=$LIMIT) ..."
python3 "$PROJECT_DIR/scripts/eval_privacypeek_attacker.py" \
  --attacker-endpoint "http://127.0.0.1:${EVAL_PORT}/v1" \
  --attacker-model "$ATTACKER_SERVED" \
  --cases-dir "$PRIVACYPEEK_ROOT/data/cases" \
  --limit "$LIMIT" \
  --attacks-only \
  --attacks-output "$ATTACKS_OUT" \
  --output "$ATTACKS_OUT"

stop_vllm

# ---------- Phase 2: frozen victim + CER ----------
start_vllm "$VICTIM_MODEL" "$VICTIM_SERVED" "victim"

echo "[PHASE2] Running victim on attacks ..."
python3 "$PROJECT_DIR/scripts/eval_privacypeek_attacker.py" \
  --victim-endpoint "http://127.0.0.1:${EVAL_PORT}/v1" \
  --victim-model "$VICTIM_SERVED" \
  --attacks-input "$ATTACKS_OUT" \
  --cases-dir "$PRIVACYPEEK_ROOT/data/cases" \
  --output "$AGENT_OUT" \
  --cer-output "$CER_OUT" \
  --summary-output "$SUMMARY_OUT"

echo "============================================"
echo " Eval done"
echo " attacks   : $ATTACKS_OUT"
echo " agent runs: $AGENT_OUT"
echo " CER output: $CER_OUT"
echo " summary   : $SUMMARY_OUT"
echo "============================================"
