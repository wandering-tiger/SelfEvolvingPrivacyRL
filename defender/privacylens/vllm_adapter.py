"""LangChain-compatible vLLM adapter for PrivacyLens ToolEmu.

Provides ``VLLMChatModel``, a ``BaseLLM`` subclass that routes calls to our
local vLLM HTTP endpoint.  It extends ``BaseLLM`` (NOT ``BaseChatModel``) so
that PrivacyLens's ``ZeroShotAgentWithToolkit`` sets ``use_chat_format=False``
and produces a ``PromptTemplate`` (compatible with its Pydantic v1 validator).

The plain-string prompt is wrapped in a single ``user``-role message before
POST-ing to the vLLM endpoint.

Usage::

    from defender.privacylens.vllm_adapter import VLLMChatModel

    llm = VLLMChatModel(
        base_url="http://127.0.0.1:5000/v1/chat/completions",
        model_name="Qwen/Qwen3-32B",
        temperature=0.0,
        max_tokens=1024,
    )
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import BaseLLM
from langchain.schema import Generation, LLMResult

logger = logging.getLogger(__name__)


class VLLMChatModel(BaseLLM):
    """LangChain ``BaseLLM`` that talks to a vLLM OpenAI-compatible endpoint.

    Extends ``BaseLLM`` so PrivacyLens's agent builder treats it as a
    non-chat model, avoiding a Pydantic v1 validation error on the
    ``ChatPromptTemplate`` type.

    Parameters
    ----------
    base_url : str
        Full URL to the chat-completions endpoint.
    model_name : str
        Value for the ``model`` field in the JSON payload.
    temperature : float
        Sampling temperature (default 0.0).
    max_tokens : int
        Max tokens per completion (default 1024).
    top_p : float
        Nucleus sampling (default 1.0).
    timeout : int
        HTTP timeout in seconds (default 120).
    max_retries : int
        Number of retries on transient errors (default 3).
    stop : list[str] | None
        Stop sequences.
    """

    base_url: str = ""
    model_name: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    timeout: int = 120
    max_retries: int = 3
    stop: Optional[List[str]] = None

    streaming: bool = False  # required by ToolEmu's simulator

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        object.__setattr__(self, "_session", requests.Session())

    # ------------------------------------------------------------------
    # LangChain required properties
    # ------------------------------------------------------------------
    @property
    def _llm_type(self) -> str:
        return "vllm-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    # ------------------------------------------------------------------
    # Core generation  (BaseLLM interface — plain-string prompts)
    # ------------------------------------------------------------------
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Send *prompts* to the vLLM endpoint and return an ``LLMResult``."""
        generations: List[List[Generation]] = []

        for prompt in prompts:
            # Wrap the string prompt as a single user message
            payload: Dict[str, Any] = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            _stop = stop or self.stop
            if _stop:
                payload["stop"] = _stop

            # Retry loop
            last_error = None
            for attempt in range(self.max_retries + 1):
                try:
                    r = self._session.post(
                        self.base_url, json=payload, timeout=self.timeout,
                    )
                    r.raise_for_status()
                    data = r.json()
                    break
                except requests.exceptions.Timeout as e:
                    last_error = e
                    if attempt < self.max_retries:
                        time.sleep(2 ** attempt)
                except requests.exceptions.RequestException as e:
                    last_error = e
                    if attempt < self.max_retries:
                        time.sleep(2 ** attempt)
            else:
                raise RuntimeError(
                    f"vLLM request failed after {self.max_retries + 1} "
                    f"attempts: {last_error}"
                )

            # Extract response text
            choices = data.get("choices", [])
            text = ""
            if choices:
                text = choices[0].get("message", {}).get("content", "")

            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_env(
        cls,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 120,
        max_retries: int = 3,
        stop: Optional[List[str]] = None,
    ) -> "VLLMChatModel":
        """Build from ``GUARD_OPENAI_URL`` + ``GUARD_MODEL`` env vars."""
        base_url = os.getenv(
            "GUARD_OPENAI_URL",
            "http://127.0.0.1:8000/v1/chat/completions",
        )
        model_name = os.getenv("GUARD_MODEL", "Qwen/Qwen3-4B-Base")
        return cls(
            base_url=base_url,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            stop=stop,
        )

    @classmethod
    def from_config(
        cls,
        base_url: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 120,
        max_retries: int = 3,
        stop: Optional[List[str]] = None,
    ) -> "VLLMChatModel":
        """Build from explicit configuration."""
        return cls(
            base_url=base_url,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            stop=stop,
        )
