#!/bin/bash
# Unified entrypoint for SelfEvolvingPrivacyRL (mirrors R-Zero style main script).

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"

BASE_MODEL=${1:-$BASE_MODEL}
GUARD_MODEL=${2:-$GUARD_MODEL}
ATTACKER_MODEL=${3:-$ATTACKER_MODEL}

export BASE_MODEL
export GUARD_MODEL
export ATTACKER_MODEL

# vLLM is required for this pipeline.
export REWRITE_BACKEND=vllm

cleanup() {
	echo "[CLEANUP] Stopping vLLM services..."
	stop_guard_service
}

trap cleanup EXIT INT TERM

MAIN_ROUNDS=${MAIN_ROUNDS:-1}

echo "[MAIN] Round 1/$MAIN_ROUNDS"
start_attacker_service "$ATTACKER_MODEL" "attacker_run_1"
start_guard_service "$GUARD_MODEL" "guard_run_1"

if [[ $MAIN_ROUNDS -gt 1 ]]; then
	for ((round=2; round<=MAIN_ROUNDS; round++)); do
		echo "[MAIN] Round $round/$MAIN_ROUNDS"
		start_attacker_service "$ATTACKER_MODEL" "attacker_run_${round}"
		start_guard_service "$GUARD_MODEL" "guard_run_${round}"
	done
fi
