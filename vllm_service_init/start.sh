model_path=$1
run_id=$2
gpu_id=${3:-2}
port=${4:-5000}
export VLLM_DISABLE_COMPILE_CACHE=1
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.35}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
VLLM_TOOL_CALL_PARSER=${VLLM_TOOL_CALL_PARSER:-hermes}

tool_call_args=()
if [[ -n "$VLLM_TOOL_CALL_PARSER" ]]; then
	tool_call_args=(--enable-auto-tool-choice --tool-call-parser "$VLLM_TOOL_CALL_PARSER")
fi

# Use the official vLLM OpenAI-compatible server instead of the legacy
# start_vllm_server.py, which deadlocks under vllm 0.11 (EngineCoreRequestType IPC error).
# --enforce-eager skips torch.compile / cudagraph capture for fast startup.
PYTHON=${PYTHON:-/home/fangzibang/.conda/envs/AgentPrivacy/bin/python}

CUDA_VISIBLE_DEVICES=$gpu_id nohup $PYTHON -m vllm.entrypoints.openai.api_server \
	--model "$model_path" \
	--port "$port" \
	--gpu-memory-utilization "$VLLM_GPU_MEM_UTIL" \
	--max-model-len "$VLLM_MAX_MODEL_LEN" \
	--served-model-name "$model_path" \
	--enforce-eager \
	--trust-remote-code \
	"${tool_call_args[@]}" \
	> "$(dirname "$0")/guard_${run_id}_port${port}.log" 2>&1 &
