#!/usr/bin/env python3
"""Build a train/validation JSON dataset using native AgentDojo attack types."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacker.agentdojo_native_dataset import (
    NATIVE_AGENTDOJO_ATTACK_TYPES,
    build_native_agentdojo_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build full AgentDojo data labelled by native attack family"
    )
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suites", default="workspace,travel,banking,slack")
    parser.add_argument(
        "--attack-types",
        default=",".join(NATIVE_AGENTDOJO_ATTACK_TYPES),
        help="Comma-separated AgentDojo names: "
        + ",".join(NATIVE_AGENTDOJO_ATTACK_TYPES),
    )
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument(
        "--max-injection-tool-calls",
        type=int,
        default=None,
        help="Keep only injection tasks whose AgentDojo reference trajectory has at most this many tool calls.",
    )
    parser.add_argument(
        "--min-injection-tool-calls",
        type=int,
        default=None,
        help="Keep only injection tasks whose AgentDojo reference trajectory has at least this many tool calls.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_samples < 2:
        raise ValueError("--num-samples must be at least 2 for train/validation split")
    suites = [value.strip() for value in args.suites.split(",") if value.strip()]
    attack_types = [value.strip() for value in args.attack_types.split(",") if value.strip()]
    records = build_native_agentdojo_records(
        num_samples=args.num_samples,
        suite_names=suites,
        native_attack_types=attack_types,
        benchmark_version=args.benchmark_version,
        min_injection_tool_calls=args.min_injection_tool_calls,
        max_injection_tool_calls=args.max_injection_tool_calls,
    )
    random.Random(args.seed).shuffle(records)
    val_count = max(1, int(len(records) * 0.1))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.json").write_text(
        json.dumps(records[val_count:], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "val.json").write_text(
        json.dumps(records[:val_count], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    counts = {name: sum(r["native_attack_type"] == name for r in records) for name in attack_types}
    print(
        f"[native-agentdojo] wrote {len(records) - val_count} train + {val_count} val "
        f"samples to {output_dir}; attack types: {counts}; "
        f"injection call range: [{args.min_injection_tool_calls}, {args.max_injection_tool_calls}]"
    )


if __name__ == "__main__":
    main()
