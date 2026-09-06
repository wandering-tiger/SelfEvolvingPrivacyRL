"""Convert SelfEvolvingPrivacyRL JSON datasets to Agent-R1 (verl 0.7.0) parquet.

Usage:
    python3 scripts/convert_json_to_parquet.py data/train.json data/train.parquet
"""
import json
import sys

import pandas as pd


def convert(json_path: str, parquet_path: str, data_source: str = "privacy_attack") -> None:
    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)

    rows = []
    for r in records:
        row = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": r["prompt"]}],
            "reward_model": {"style": "rule", "ground_truth": r.get("sensitive", "")},
            # 其余字段全部进 extra_info，供 reward 函数读取
            "extra_info": {k: v for k, v in r.items() if k not in ("prompt", "sensitive")},
        }
        # Agent R1's AgentFlowWorker reads these fields directly from its
        # non-tensor batch; nesting them under extra_info silently loses the
        # defender environment configuration.
        for key in ("agent_name", "env_kwargs"):
            if key in r:
                row[key] = r[key]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path)
    print(f"[convert] {json_path} -> {parquet_path} ({len(rows)} rows)")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
