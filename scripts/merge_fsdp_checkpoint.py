#!/usr/bin/env python3
"""Merge veRL FSDP sharded actor checkpoint into a HuggingFace folder for vLLM eval."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_rank_shard(actor_dir: Path, world_size: int, rank: int) -> dict:
    path = actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing shard: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def merge_sharded_state(actor_dir: Path, world_size: int) -> dict[str, torch.Tensor]:
    shards = [load_rank_shard(actor_dir, world_size, r) for r in range(world_size)]
    keys = list(shards[0].keys())
    merged: dict[str, torch.Tensor] = {}

    for key in keys:
        values = [s[key] for s in shards]
        if isinstance(values[0], DTensor):
            locals_ = [v.to_local().detach().cpu() for v in values]
            # FSDP FULL_SHARD typically shards dim 0
            merged[key] = torch.cat(locals_, dim=0)
        elif torch.is_tensor(values[0]):
            # replicated tensors: take rank 0
            merged[key] = values[0].detach().cpu()
        else:
            raise TypeError(f"Unsupported tensor type for {key}: {type(values[0])}")

    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        required=True,
        help="Path to global_step_N (contains actor/)",
    )
    ap.add_argument(
        "--base-model",
        default="/home/zhaoyipeng/PrivacyPeek/models/Meta-Llama-3.1-8B-Instruct",
        help="Base HF model for architecture + tokenizer fallback",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output HF model directory",
    )
    ap.add_argument("--world-size", type=int, default=4)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    actor_dir = ckpt / "actor" if (ckpt / "actor").exists() else ckpt
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[merge] loading {args.world_size} shards from {actor_dir}")
    state = merge_sharded_state(actor_dir, args.world_size)

    hf_src = actor_dir / "huggingface"
    if hf_src.exists():
        for name in ("config.json", "generation_config.json", "tokenizer.json",
                     "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
            src = hf_src / name
            if src.exists():
                shutil.copy2(src, out / name)

    print(f"[merge] building model from {args.base_model}")
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (show 5) {missing[:5]}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (show 5) {unexpected[:5]}")

    print(f"[merge] saving HF model to {out}")
    model.save_pretrained(out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(
        str(hf_src) if hf_src.exists() else args.base_model,
        trust_remote_code=True,
    )
    tok.save_pretrained(out)
    print("[merge] done")


if __name__ == "__main__":
    main()
