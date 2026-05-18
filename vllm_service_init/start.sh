model_path=$1
run_id=$2
export VLLM_DISABLE_COMPILE_CACHE=1
CUDA_VISIBLE_DEVICES=2 python vllm_service_init/start_vllm_server.py --port 5000 --model_path $model_path &