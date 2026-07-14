"""Build attack dataset for RL training.

Stores short task prompts in train.json / val.json.
The attacker model receives these prompts during RL training and generates
actual attack messages.

Supports two modes mixed in one dataset:
  - Standard: file-based defender (agent_type absent or empty)
  - PrivacyLens: ToolEmu sandbox defender (agent_type="privacylens")

Usage:
    python build_dataset.py --num_samples 2000 --output_dir ./data
    python build_dataset.py --num_samples 2000 --pl_style_ratio 0.5  # 50% PL-style
"""
import argparse
import json
import os
import random

from attacker.dataset_builder import build_attack_dataset, build_pl_style_records


def main():
    parser = argparse.ArgumentParser(description="Build attack dataset")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--output_dir", type=str,
                        default="/home/fangzibang/SelfEvolvingPrivacyRL/data")
    parser.add_argument("--attack_strategy", type=str, default=None,
                        help="Comma-separated strategies (e.g., direct,indirect)")
    parser.add_argument("--pl_style_ratio", type=float, default=0.0,
                        help="Fraction of samples that are PrivacyLens-style (0.0-1.0)")

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

    # ------------------------------------------------------------------
    # PrivacyLens-style records
    # ------------------------------------------------------------------
    pl_ratio = max(0.0, min(1.0, args.pl_style_ratio))
    pl_samples = int(args.num_samples * pl_ratio)
    std_samples = args.num_samples - pl_samples

    all_records = []

    if std_samples > 0:
        # Standard mode: file-based defender
        std_dir = os.path.join(args.output_dir, "_std_temp")
        build_attack_dataset(
            num_samples=std_samples,
            output_dir=std_dir,
            attack_strategy=args.attack_strategy,
        )
        # Load generated records
        train_path = os.path.join(std_dir, "train.json")
        val_path = os.path.join(std_dir, "val.json")
        if os.path.exists(train_path):
            with open(train_path) as f:
                all_records.extend(json.load(f))
        if os.path.exists(val_path):
            with open(val_path) as f:
                all_records.extend(json.load(f))
        # Cleanup temp files
        import shutil
        if os.path.exists(std_dir):
            shutil.rmtree(std_dir)

    if pl_samples > 0:
        pl_records = build_pl_style_records(
            num_samples=pl_samples,
            attack_strategies=(
                args.attack_strategy.split(",")
                if args.attack_strategy
                else None
            ),
        )
        all_records.extend(pl_records)

    # ------------------------------------------------------------------
    # Shuffle and split into train/val
    # ------------------------------------------------------------------
    random.shuffle(all_records)
    n_val = max(1, int(len(all_records) * 0.1))
    val_records = all_records[:n_val]
    train_records = all_records[n_val:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.json")
    val_path = os.path.join(args.output_dir, "val.json")
    with open(train_path, "w") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)
    with open(val_path, "w") as f:
        json.dump(val_records, f, ensure_ascii=False, indent=2)

    n_pl = sum(1 for r in train_records if r.get("agent_type") == "privacylens")
    n_pl += sum(1 for r in val_records if r.get("agent_type") == "privacylens")
    print(f"Generated {len(train_records)} train + {len(val_records)} val samples "
          f"({n_pl} PrivacyLens-style) -> {args.output_dir}")


if __name__ == "__main__":
    main()
