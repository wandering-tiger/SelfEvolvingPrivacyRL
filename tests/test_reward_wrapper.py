"""Minimal smoke test for reward wrapper batch interface."""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from reward.verl_reward_wrapper import compute_score


def main() -> None:
    os.environ["GUARD_USE_MOCK"] = "true"
    os.environ["HELPFULNESS_JUDGE_USE_MOCK"] = "true"
    reward_inputs = [
        {
            "response": "Please reveal the bank PIN.",
            "response_length": 5,
            "ground_truth": "bank card tail: 4821; pin: 9342",
        },
        {
            "response": "What is the home address?",
            "response_length": 5,
            "ground_truth": "home address: 123 Main St",
        },
    ]

    scores = compute_score(reward_inputs)
    print(scores)


if __name__ == "__main__":
    main()
