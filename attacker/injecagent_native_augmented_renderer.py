"""Native-enhanced InjecAgent prompt augmented by a learned context action."""

from __future__ import annotations

import json
from typing import Any

from attacker.injecagent_renderer import ENHANCED_PREFIX

FIELDS = ("context_frame", "execution_cue")


def parse_native_augmented_action(response: str) -> tuple[dict[str, str], bool]:
    empty = {field: "" for field in FIELDS}
    try:
        value: Any = json.loads(response.strip())
    except (TypeError, json.JSONDecodeError):
        return empty, False
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        return empty, False
    action: dict[str, str] = {}
    for field in FIELDS:
        item = value[field]
        if not isinstance(item, str) or not 6 <= len(item.split()) <= 28 or len(item) > 240:
            return empty, False
        action[field] = item.strip()
    return action, True


def render_native_augmented_payload(target: str, policy_response: str) -> tuple[str, dict[str, Any]]:
    """Render the portable two-module action ahead of an immutable target.

    This is intentionally benchmark-neutral: an InjecAgent case and an
    AgentDojo injection task both supply only their fixed target string.  Thus
    an attacker checkpoint can generate the same JSON action for either a
    direct attacker rollout or frozen samples used in defender training.
    """
    action, valid = parse_native_augmented_action(policy_response)
    backbone = f"{ENHANCED_PREFIX} {target}"
    if valid:
        payload = (
            f"{action['context_frame']}\n"
            f"{action['execution_cue']}\n"
            f"{backbone}"
        )
    else:
        payload = backbone
    return payload, {
        "slots_valid": valid,
        "modules": action if valid else {field: "" for field in FIELDS},
        "setting": "native_augmented",
        "canonical_instruction": target,
    }


def render_native_augmented_instruction(case: dict[str, Any], policy_response: str) -> tuple[str, dict[str, Any]]:
    return render_native_augmented_payload(str(case["Attacker Instruction"]), policy_response)


def render_native_augmented_tool_response(case: dict[str, Any], policy_response: str) -> tuple[str, dict[str, Any]]:
    template = str(case["Tool Response Template"])
    if template.count("<Attacker Instruction>") != 1:
        raise ValueError("InjecAgent Tool Response Template must contain one placeholder")
    payload, metadata = render_native_augmented_instruction(case, policy_response)
    return json.dumps(template.replace("<Attacker Instruction>", payload)), metadata


__all__ = [
    "render_native_augmented_tool_response",
    "render_native_augmented_instruction",
    "render_native_augmented_payload",
    "parse_native_augmented_action",
]
