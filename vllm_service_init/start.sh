model_path=$1
run_id=$2
gpu_id=${3:-2}
port=${4:-5000}
export VLLM_DISABLE_COMPILE_CACHE=1
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.8}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}

extra_args=()
if [[ -n "$VLLM_GPU_MEM_UTIL" ]]; then
	extra_args+=(--gpu_mem_util "$VLLM_GPU_MEM_UTIL")
fi
if [[ -n "$VLLM_MAX_MODEL_LEN" ]]; then
	extra_args+=(--max_model_len "$VLLM_MAX_MODEL_LEN")
fi

CUDA_VISIBLE_DEVICES=$gpu_id python vllm_service_init/start_vllm_server.py --port $port --model_path $model_path "${extra_args[@]}" &