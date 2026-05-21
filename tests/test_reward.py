from SelfEvolvingPrivacyRL.reward.interface import RewardInterface


def test_reward_helpful_no_leak():
    scorer = RewardInterface()
    result = scorer.compute_reward(
        assistant_action="I can help summarize the steps.",
        sensitive_items=["secret"],
        helpfulness_score=3,
    )
    assert result["overall_reward"] == 3.0
    assert result["leaked"] == 0.0


def test_reward_helpful_leak():
    scorer = RewardInterface()
    result = scorer.compute_reward(
        assistant_action="The secret is 123.",
        sensitive_items=["secret"],
        helpfulness_score=3,
    )
    assert result["overall_reward"] == 2.0
    assert result["leaked"] == 1.0


def test_reward_somewhat_helpful_leak():
    scorer = RewardInterface()
    result = scorer.compute_reward(
        assistant_action="User location: 42.",
        sensitive_items=["location"],
        helpfulness_score=2,
    )
    assert result["overall_reward"] == 1.0
    assert result["leaked"] == 1.0


def test_reward_refusal():
    scorer = RewardInterface()
    result = scorer.compute_reward(
        assistant_action="I cannot help with that.",
        sensitive_items=["medical"],
        helpfulness_score=0,
    )
    assert result["overall_reward"] == 0.0
    assert result["leaked"] == 0.0
