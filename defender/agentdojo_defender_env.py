"""Agent R1 environment for training a defender directly in AgentDojo.

Unlike :mod:`defender.agentdojo_env`, this environment never calls an external
guard service.  ``Action.text`` is the trainable actor's own response; parsed
tool calls are executed against a stateful AgentDojo ``FunctionsRuntime``.
"""

from __future__ import annotations

import json
from typing import Any

from agent_r1.env import AgentEnv, Observation
from agent_r1.env.base import Action
from agent_r1.env.tool_format import ToolFormatWrapper
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import get_text_content_as_str, text_content_block_from_string

try:  # Package import when invoked from the project root (tests/dataset tools).
    from reward.agentdojo_defender_reward import DefenderReward
    from defender.portable_defender_template import DEFENDER_SYSTEM_TEMPLATE
except ModuleNotFoundError:  # Top-level import on Agent R1 workers; see launcher PYTHONPATH.
    from agentdojo_defender_reward import DefenderReward
    from portable_defender_template import DEFENDER_SYSTEM_TEMPLATE

SYSTEM_PROMPT = DEFENDER_SYSTEM_TEMPLATE


def _escape_yaml_injections(injections: dict[str, str]) -> dict[str, str]:
    """Escape values substituted into AgentDojo's quoted YAML placeholders."""
    return {key: json.dumps(str(value), ensure_ascii=False)[1:-1] for key, value in injections.items()}


def _tool_schema(tool: Any) -> dict[str, Any]:
    """Convert an AgentDojo Function to the OpenAI tool schema chat templates use."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters.model_json_schema(),
        },
    }


@AgentEnv.register("agentdojo_defender")
class AgentDojoDefenderEnv(AgentEnv):
    """Stateful AgentDojo environment driven by a defender actor."""

    def __init__(
        self,
        suite_name: str,
        user_task_id: str,
        injection_task_id: str | None = None,
        injections: dict[str, str] | str | None = None,
        expected_tool_scope: list[str] | None = None,
        forbidden_sources: list[str] | None = None,
        benchmark_version: str = "v1.2.2",
        tool_format: str = "hermes",
        max_tool_calls_per_turn: int = 1,
        **_: Any,
    ) -> None:
        self.suite_name = suite_name
        self.user_task_id = user_task_id
        self.injection_task_id = injection_task_id
        self.injections = json.loads(injections) if isinstance(injections, str) else (injections or {})
        # Arrow/Pandas may restore nested list columns as numpy arrays.
        self.expected_tool_scope = list(expected_tool_scope) if expected_tool_scope is not None else []
        self.forbidden_sources = list(forbidden_sources) if forbidden_sources is not None else []
        self.benchmark_version = benchmark_version
        self.max_tool_calls_per_turn = max(1, max_tool_calls_per_turn)
        self.format_wrapper = ToolFormatWrapper.from_name(tool_format)
        self.rewarder = DefenderReward()
        self._messages: list[dict[str, Any]] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._format_errors = 0
        self._done = False

    def reset(self, **_: Any) -> Observation:
        self.suite = get_suite(self.benchmark_version, self.suite_name)
        self.user_task = self.suite.get_user_task_by_id(self.user_task_id)
        self.injection_task = (
            self.suite.get_injection_task_by_id(self.injection_task_id)
            if self.injection_task_id else None
        )
        # Parquet represents a dict column as a shared Arrow struct.  Mixed
        # suite datasets therefore restore absent keys as ``key: None`` from
        # other suites.  AgentDojo correctly rejects those foreign keys, so
        # retain only non-empty placeholders valid for this suite.
        valid_vectors = set(self.suite.get_injection_vector_defaults())
        suite_injections = {
            key: value for key, value in self.injections.items()
            if key in valid_vectors and value is not None
        }
        base_environment = self.suite.load_and_inject_default_environment(
            _escape_yaml_injections(suite_injections)
        )
        self.pre_environment = self.user_task.init_environment(base_environment)
        self.environment = self.pre_environment.model_copy(deep=True)
        self.runtime = FunctionsRuntime(self.suite.tools)
        self._tool_calls = []
        self._format_errors = 0
        self._done = False
        self._messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.user_task.PROMPT},
        ]
        return Observation(messages=list(self._messages))

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return [_tool_schema(tool) for tool in self.suite.tools]

    def _terminal_result(self, model_output: str, error: str | None = None) -> dict[str, Any]:
        output_blocks = [text_content_block_from_string(model_output)] if model_output else []
        stack = [
            # AgentDojo's checker only consumes .function/.args, so its native
            # FunctionCall is intentionally avoided here to keep call ids out.
            type("Call", (), {"function": c["function"], "args": c["args"]})()
            for c in self._tool_calls
        ]
        utility = self.suite._check_task_result(
            self.user_task, output_blocks, self.pre_environment, self.environment, stack
        )
        injection_success = bool(self.injection_task) and self.suite._check_task_result(
            self.injection_task, output_blocks, self.pre_environment, self.environment, stack
        )
        privacy_leak = any(
            call["function"] in set(self.forbidden_sources)
            for call in self._tool_calls
        )
        return {
            "utility": bool(utility),
            "task_completed": bool(utility),
            "security": not bool(injection_success),
            "injection_success": bool(injection_success),
            "attack_success": bool(injection_success),
            "privacy_leak": bool(privacy_leak),
            "tool_calls": list(self._tool_calls),
            "model_output": model_output,
            "error": error,
            "format_errors": self._format_errors,
            "expected_tool_scope": self.expected_tool_scope,
            "forbidden_sources": self.forbidden_sources,
            "n_tool_calls": len(self._tool_calls),
            "case_id": f"{self.suite_name}_{self.user_task_id}_{self.injection_task_id or 'clean'}",
        }

    async def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self._done:
            return Observation(messages=list(self._messages)), 0.0, True, {"already_done": True}
        if not action.text:
            self._format_errors += 1
            self._done = True
            result = self._terminal_result("", "empty_actor_response")
            score = self.rewarder.compute(result)
            return Observation(messages=list(self._messages)), score["overall"], True, score

        content, calls = self.format_wrapper.parse_response(action.text)
        self._messages.append({"role": "assistant", "content": action.text})
        # A malformed XML-looking response is distinct from a legitimate final answer.
        if "<tool_call>" in action.text and not calls:
            self._format_errors += 1

        if not calls:
            self._done = True
            result = self._terminal_result(content.strip() or action.text)
            score = self.rewarder.compute(result)
            return Observation(messages=list(self._messages)), score["overall"], True, score

        if len(calls) > self.max_tool_calls_per_turn:
            calls = calls[: self.max_tool_calls_per_turn]
            self._format_errors += 1

        tool_messages: list[str] = []
        # The primary signal is terminal and consists solely of the three
        # shared metrics.  Do not add shaping that would change its meaning.
        immediate_reward = 0.0
        for call in calls:
            result, error = self.runtime.run_function(self.environment, call.name, call.arguments)
            self._tool_calls.append({"function": call.name, "args": dict(call.arguments)})
            if error:
                rendered = f"Tool error: {error}"
                self._format_errors += 1
            elif hasattr(result, "model_dump"):
                rendered = json.dumps(result.model_dump(), ensure_ascii=False, default=str)
            else:
                rendered = json.dumps(result, ensure_ascii=False, default=str)
            tool_messages.append(self.format_wrapper.format_observation(rendered))

        self._messages.append({"role": "user", "content": "\n".join(tool_messages)})
        return Observation(messages=list(self._messages)), immediate_reward, False, {
            "n_tool_calls": len(self._tool_calls), "format_errors": self._format_errors
        }


__all__ = ["AgentDojoDefenderEnv"]
