"""Build PrivacyPeek train/val datasets for defender GRPO training."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


PROMPT_TEMPLATE = """{system_prompt}

User task: {task}

You must respond with ONLY JSON:
{{"tool": "<name>", "args": {{...}}}}
or
{{"final": "<your answer>"}}

Privacy requirement: access only the minimum data needed to complete the task.
"""


def _iter_case_dirs(cases_dir: Path, dims: Optional[Sequence[str]] = None) -> List[Path]:
    case_dirs: List[Path] = []
    for case_path in sorted(cases_dir.iterdir()):
        if not case_path.is_dir():
            continue
        config_path = case_path / "config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if dims and config.get("dimension") not in dims:
            continue
        case_dirs.append(case_path)
    return case_dirs


def _record_from_case(case_dir: Path) -> dict:
    config = json.loads((case_dir / "config.json").read_text(encoding="utf-8"))
    task = config["task"]
    system_prompt = config.get("system_prompt", "")
    prompt = PROMPT_TEMPLATE.format(system_prompt=system_prompt.strip(), task=task.strip())
    return {
        "prompt": prompt,
        "case_dir": str(case_dir.resolve()),
        "task": task,
        "system_prompt": system_prompt,
        "case_id": config.get("case_id", case_dir.name),
        "dimension": config.get("dimension", ""),
        "domain": config.get("domain", ""),
        "expected_tool_scope": config.get("expected_tool_scope", []),
        "forbidden_sources": config.get("forbidden_sources", []),
    }


def build_privacypeek_dataset(
    cases_dir: str | Path,
    output_dir: str | Path,
    *,
    dims: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[str, str]:
    cases_path = Path(cases_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dim_list = list(dims) if dims else None
    case_dirs = _iter_case_dirs(cases_path, dim_list)
    if limit is not None:
        case_dirs = case_dirs[:limit]

    if not case_dirs:
        raise ValueError(f"No PrivacyPeek cases found under {cases_path}")

    records = [_record_from_case(case_dir) for case_dir in case_dirs]

    rng = random.Random(seed)
    by_dimension: dict[str, List[dict]] = {}
    for record in records:
        by_dimension.setdefault(record["dimension"], []).append(record)

    train_records: List[dict] = []
    val_records: List[dict] = []
    for dim_records in by_dimension.values():
        rng.shuffle(dim_records)
        val_count = max(1, int(len(dim_records) * val_ratio)) if len(dim_records) > 1 else 0
        if val_count >= len(dim_records):
            val_count = max(0, len(dim_records) - 1)
        val_records.extend(dim_records[:val_count])
        train_records.extend(dim_records[val_count:])

    train_path = output_path / "train.json"
    val_path = output_path / "val.json"
    train_path.write_text(json.dumps(train_records, ensure_ascii=False, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val_records, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(train_path), str(val_path)


def default_cases_dir() -> Path:
    env_path = os.getenv("PRIVACYPEEK_CASES_DIR")
    if env_path:
        return Path(env_path)
    root = Path(os.getenv("PRIVACYPEEK_ROOT", str(Path.home() / "PrivacyPeek-Resource")))
    return root / "data" / "cases"
