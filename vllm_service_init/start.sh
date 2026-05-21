model_path=$1
run_id=$2
gpu_id=${3:-2}
port=${4:-5000}
export VLLM_DISABLE_COMPILE_CACHE=1
CUDA_VISIBLE_DEVICES=$gpu_id python vllm_service_init/start_vllm_server.py --port $port --model_path $model_path &