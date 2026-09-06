#!/usr/bin/env python3
"""Build modular InjecAgent attacker data without changing legacy builders."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from attacker.injecagent_dataset import build_injecagent_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--setting",
        choices=("base", "enhanced", "native_augmented"),
        default="enhanced",
    )
    parser.add_argument("--attack-kinds", default="dh,ds")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    kinds = tuple(value.strip() for value in args.attack_kinds.split(",") if value.strip())
    records = build_injecagent_records(args.num_samples, args.setting, kinds, seed=args.seed)
    random.Random(args.seed).shuffle(records)
    n_val = max(1, len(records) // 10)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "train.json").write_text(json.dumps(records[n_val:], ensure_ascii=False, indent=2))
    (output / "val.json").write_text(json.dumps(records[:n_val], ensure_ascii=False, indent=2))
    print(f"[injecagent] wrote {len(records)-n_val} train + {n_val} val records to {output}")


if __name__ == "__main__":
    main()
