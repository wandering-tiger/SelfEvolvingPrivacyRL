"""Quick test helper: generate a small dataset using only template prompts."""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from build_dataset import build_attack_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/fangzibang/SelfEvolvingPrivacyRL/data/template_test",
    )
    parser.add_argument("--model_path", type=str, default="/home/fangzibang/data_32T/fzb_data/ModelScope/Qwen3-4B-Instruct-2507")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rewrite_backend", type=str, default="transformers")
    parser.add_argument("--vllm_gpu_mem_util", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    build_attack_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        rewrite_backend=args.rewrite_backend,
        vllm_gpu_mem_util=args.vllm_gpu_mem_util,
    )


if __name__ == "__main__":
    main()
