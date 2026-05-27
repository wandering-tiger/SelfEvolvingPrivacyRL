"""Lightweight checks for leakage detection and reward output."""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from reward.interface import RewardInterface
from reward.verl_reward_wrapper import compute_score


def main() -> None:
    interface = RewardInterface()

    sensitive = "bank card tail: 4821; pin: 9342"
    leak_response = f"Here is the info you wanted: {sensitive}"
    safe_response = "I can't share that information."

    print("detect_leakage (should be True):", interface.detect_leakage(leak_response, [sensitive]))
    print("detect_leakage (should be False):", interface.detect_leakage(safe_response, [sensitive]))

    os.environ["GUARD_USE_MOCK"] = "true"
    os.environ["HELPFULNESS_JUDGE_USE_MOCK"] = "true"

    reward_inputs = [
        {"response": "probe", "response_length": 5, "ground_truth": sensitive},
        {"response": "probe", "response_length": 5, "ground_truth": "home address: 12 Main St"},
    ]

    scores = compute_score(reward_inputs)
    print("compute_score outputs:", scores)


if __name__ == "__main__":
    main()
