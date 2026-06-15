import os
import requests

_SESSION = requests.Session()


def call_guard_llm(messages):

    base_url = os.getenv(
        "GUARD_OPENAI_URL",
        "http://127.0.0.1:8000/v1/chat/completions"
    )

    model_name = os.getenv(
        "GUARD_MODEL",
        "Qwen/Qwen3-4B-Base"
    )

    timeout = float(
        os.getenv("GUARD_TIMEOUT", "30")
    )

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 512,
    }

    response = _SESSION.post(
        base_url,
        json=payload,
        timeout=timeout
    )

    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]["content"]