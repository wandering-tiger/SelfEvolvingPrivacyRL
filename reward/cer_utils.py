"""CER utilities adapted from PrivacyPeek exact-match judge."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_privacypeek_judge():
    privacypeek_root = Path(
        os.getenv("PRIVACYPEEK_ROOT", str(Path.home() / "PrivacyPeek-Resource"))
    )
    judge_path = privacypeek_root / "evaluation" / "judges" / "cer_exact_match.py"
    if not judge_path.exists():
        raise FileNotFoundError(
            f"PrivacyPeek CER judge not found at {judge_path}. "
            "Set PRIVACYPEEK_ROOT to your PrivacyPeek checkout."
        )

    spec = importlib.util.spec_from_file_location("privacypeek_cer_judge", judge_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["privacypeek_cer_judge"] = module
    spec.loader.exec_module(module)
    return module


def evaluate_cer(agent_result: Dict[str, Any]) -> Dict[str, Any]:
    """Return CER verdict for one agent trajectory."""
    judge_module = _load_privacypeek_judge()
    case_payload = {
        "case_id": agent_result.get("case_id"),
        "dimension": agent_result.get("dimension", ""),
        "domain": agent_result.get("domain", ""),
        "sensitive_items": agent_result.get("sensitive_items", []),
        "sensitive_identifiers": agent_result.get("sensitive_identifiers", []),
        "tool_calls": agent_result.get("tool_calls", []),
    }
    return judge_module.judge_case(case_payload)


def cer_score_from_verdict(verdict: Dict[str, Any]) -> float:
    """Binary CER exposure score in [0, 1]."""
    return 1.0 if verdict.get("cer_total") else 0.0


def partial_cer_score(verdict: Dict[str, Any]) -> float:
    """Partial credit based on matched identifier groups."""
    matched = verdict.get("matched_group_indices") or []
    total_groups = max(len(verdict.get("matched_items") or []), 1)
    if not verdict.get("cer_total"):
        return 0.0
    return min(1.0, len(matched) / total_groups)
