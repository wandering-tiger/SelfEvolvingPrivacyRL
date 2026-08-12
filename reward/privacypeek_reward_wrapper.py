"""veRL batch reward wrapper for PrivacyPeek defender training."""

from __future__ import annotations

import os
from typing import Dict, List

from defender.privacypeek_env import run_privacypeek_agent, try_parse_action
from reward.privacypeek_reward import PrivacyPeekReward

REWARD_NAME = "privacypeek_defender"
REWARD_TYPE = "batch"


def _use_mock_guard() -> bool:
    return os.getenv("GUARD_USE_MOCK", "false").lower() in {"1", "true", "yes"}


def _mock_agent_result(case_dir: str) -> Dict:
    return {
        "case_id": os.path.basename(case_dir),
        "dimension": "",
        "domain": "",
        "task": "",
        "system_prompt": "",
        "agent_response": "Mock task completed.",
        "tool_calls": [],
        "n_tool_calls": 0,
        "sensitive_items": [],
        "sensitive_identifiers": [],
        "expected_tool_scope": [],
        "forbidden_sources": [],
        "format_error": False,
        "used_first_action": False,
    }


def compute_score(
    reward_inputs: List[Dict[str, str]],
    **kwargs,
) -> List[Dict[str, float]]:
    tcr_weight = float(os.getenv("PRIVACYPEEK_TCR_WEIGHT", "0.3"))
    format_penalty = float(os.getenv("PRIVACYPEEK_FORMAT_PENALTY", "0.1"))
    forbidden_penalty = float(os.getenv("PRIVACYPEEK_FORBIDDEN_PENALTY", "0.2"))
    max_steps = int(os.getenv("PRIVACYPEEK_MAX_STEPS", "6"))

    scorer = PrivacyPeekReward(
        tcr_weight=tcr_weight,
        format_penalty=format_penalty,
        forbidden_penalty=forbidden_penalty,
    )
    results: List[Dict[str, float]] = []

    case_dirs = kwargs.get("case_dir", [])
    for idx, reward_input in enumerate(reward_inputs):
        case_dir = reward_input.get("case_dir") or reward_input.get("ground_truth", "")
        if not case_dir and idx < len(case_dirs):
            case_dir = case_dirs[idx]

        actor_response = reward_input.get("response", "").strip()
        first_action = try_parse_action(actor_response)

        if _use_mock_guard():
            agent_result = _mock_agent_result(case_dir)
        else:
            try:
                agent_result = run_privacypeek_agent(
                    case_dir,
                    first_action=first_action,
                    max_steps=max_steps,
                    return_metadata=True,
                )
            except Exception as exc:
                agent_result = {
                    "case_id": os.path.basename(case_dir),
                    "dimension": "",
                    "domain": "",
                    "task": "",
                    "system_prompt": "",
                    "agent_response": f"[guard_error]{exc}",
                    "tool_calls": [],
                    "n_tool_calls": 0,
                    "sensitive_items": [],
                    "sensitive_identifiers": [],
                    "expected_tool_scope": [],
                    "forbidden_sources": [],
                    "format_error": True,
                    "used_first_action": False,
                }

        print("=" * 50)
        print("case_dir:", case_dir)
        print("actor_response:", actor_response[:300])
        print("used_first_action:", agent_result.get("used_first_action"))
        print("agent_response:", str(agent_result.get("agent_response", ""))[:300])
        print("n_tool_calls:", agent_result.get("n_tool_calls"))
        print("format_error:", agent_result.get("format_error"))

        results.append(scorer.compute_reward(agent_result))

    return results
