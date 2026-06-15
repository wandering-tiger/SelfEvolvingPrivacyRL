"""Build attack dataset for RL training.

Stores short task prompts in train.json / val.json.
The attacker model receives these prompts during RL training and generates
actual attack messages (with write_document + final for indirect attacks).
The reward wrapper processes the model's output through the attacker agent
pipeline before feeding it to the defender.

Usage:
    python build_dataset.py --num_samples 2000 --output_dir ./data
    python build_dataset.py --num_samples 2000 --output_dir ./data --attack_strategy direct
"""
import argparse

from attacker.dataset_builder import build_attack_dataset


def main():
    parser = argparse.ArgumentParser(description="Build attack dataset")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--output_dir", type=str,
                        default="/home/fangzibang/SelfEvolvingPrivacyRL/data")
    parser.add_argument("--attack_strategy", type=str, default=None,
                        help="Only generate this strategy (direct, indirect, roleplay, etc.)")

    # Deprecated args — accepted for backward compatibility with training scripts
    parser.add_argument("--model_path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rewrite_backend", type=str, default="transformers", help=argparse.SUPPRESS)
    parser.add_argument("--batch_size", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--max_new_tokens", type=int, default=128, help=argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, default=0.7, help=argparse.SUPPRESS)
    parser.add_argument("--top_p", type=float, default=0.95, help=argparse.SUPPRESS)
    parser.add_argument("--device", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--vllm_gpu_mem_util", type=float, default=0.8, help=argparse.SUPPRESS)
    parser.add_argument("--vllm_max_model_len", type=int, default=8192, help=argparse.SUPPRESS)

    args = parser.parse_args()

    build_attack_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        attack_strategy=args.attack_strategy,
    )


if __name__ == "__main__":
    main()
