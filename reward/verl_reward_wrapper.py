from typing import Dict, List
import os
import requests

from SelfEvolvingPrivacyRL.reward.interface import RewardInterface

_SESSION = requests.Session()

def _use_mock_victim() -> bool:
    return os.getenv("VICTIM_USE_MOCK", "false").lower() in {"1", "true", "yes"}

def _call_victim_model(attack_prompt: str) -> str:
    base_url = os.getenv("VICTIM_OPENAI_URL", "http://127.0.0.1:8000/v1/chat/completions")
    model_name = os.getenv("VICTIM_MODEL", "Qwen/Qwen3-4B-Base")
    system_prompt = os.getenv("VICTIM_SYSTEM_PROMPT", "You are a helpful assistant.")
    timeout = float(os.getenv("VICTIM_TIMEOUT", "30"))

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": attack_prompt},
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 512,
    }

    response = _SESSION.post(base_url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

def compute_score(predicts: List[str], ground_truths: List[str], **kwargs) -> List[Dict[str, float]]:
    """
    verl requires a compute_score function that takes the LLM's 'predicts'.
    For each generation, we evaluate the generated attack prompt and assign a reward.
    """
    scorer = RewardInterface()
    results = []
    
    for idx, predict in enumerate(predicts):
        attack_prompt = predict.strip()
        
        # Execute attack_prompt against the victim LLM (OpenAI-compatible vLLM endpoint).
        # Set VICTIM_USE_MOCK=true to keep the stubbed response during quick tests.
        if _use_mock_victim():
            agent_response = "Mock response containing some target data."
        else:
            try:
                agent_response = _call_victim_model(attack_prompt)
            except Exception as exc:
                agent_response = f"[victim_error]{exc}"
        
        class DummyState:
            privacy_target = ground_truths[idx] if len(ground_truths) > idx else "target data"
        
        attack_state = DummyState()
        
        # Compute final reward relying strictly on Leakage Rate and Helpfulness Rating
        final_reward = scorer.aggregate_reward(
            agent_response=agent_response,
            attack_state=attack_state
        )
        
        results.append({
            "overall_reward": final_reward,
            "leakage_rate": scorer.compute_leakage_reward(agent_response, attack_state.privacy_target),
            "helpfulness_rating": scorer.compute_helpfulness_reward(agent_response)
        })
        
    return results
