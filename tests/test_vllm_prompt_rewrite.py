"""Generate base prompts, then rewrite them with vLLM for inspection."""
import argparse
import json
import os
import sys
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from attacker.dataset_builder import build_attack_dataset
from attacker.state import AttackState
from build_dataset import _generate_prompts_with_vllm


def _load_prompts(output_dir: str, limit: int) -> List[str]:
    train_path = os.path.join(output_dir, "train.json")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing dataset file: {train_path}")
    with open(train_path, "r") as f:
        data = json.load(f)
    prompts = [item["prompt"] for item in data]
    if limit:
        prompts = prompts[:limit]
    return prompts

def _save_rewrites(
    output_dir: str,
    strategy: str,
    base_prompts: List[str],
    rewritten_prompts: List[str],
) -> str:
    rewrite_dir = os.path.join(output_dir, "rewrite")
    os.makedirs(rewrite_dir, exist_ok=True)

    strategy_name = strategy or "all"

    save_path = os.path.join(
        rewrite_dir,
        f"{strategy_name}_rewrite.json"
    )

    payload = [
        {
            "base_prompt": base,
            "rewritten_prompt": rewrite,
        }
        for base, rewrite in zip(
            base_prompts,
            rewritten_prompts,
        )
    ]

    with open(save_path, "w") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return save_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/fangzibang/SelfEvolvingPrivacyRL/data/template_test",
    )
    parser.add_argument("--model_path", type=str, default="/home/fangzibang/data_32T/fzb_data/ModelScope/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--vllm_gpu_mem_util", type=float, default=0.5)
    parser.add_argument("--vllm_max_model_len", type=int, default=8192)
    parser.add_argument(
        "--strategy",
        type=str,
        choices=AttackState.STRATEGIES,
        default=None,
        help="Generate dataset using a single attack strategy.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=5,
        help="Number of samples to print.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    build_attack_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        model_path=args.model_path,
        attack_strategy=args.strategy,
    )

    prompts = _load_prompts(args.output_dir, limit=None)

    rewritten = _generate_prompts_with_vllm(
        prompts=prompts,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_mem_util=args.vllm_gpu_mem_util,
        max_model_len=args.vllm_max_model_len,
    )

    for idx, (base, rewrite) in enumerate(zip(prompts[:args.show], rewritten[:args.show]), start=1):
        print(f"\n=== Sample {idx} ===")
        print("BASE PROMPT:")
        print(base)
        print("\nREWRITTEN PROMPT:")
        print(rewrite)

    save_path = _save_rewrites(
        output_dir=args.output_dir,
        strategy=args.strategy,
        base_prompts=prompts,
        rewritten_prompts=rewritten,
    )

    print(f"\nSaved rewritten prompts to:\n{save_path}")


if __name__ == "__main__":
    main()
