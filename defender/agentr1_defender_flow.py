"""Agent R1 flow for direct AgentDojo defender training."""

from uuid import uuid4

from agent_r1.agent_flow.agent_env_loop import AgentEnvLoop
from agent_r1.agent_flow.agent_flow import AgentFlowOutput, AgentFlowStep
from agent_r1.env.base import Action
from verl.utils.profiler import simple_timer

# Import for the AgentEnv.register side effect.  This module is the Hydra
# target in agentdojo_defender_flow.yaml, so it is imported on every worker.
try:
    from defender.agentdojo_defender_env import AgentDojoDefenderEnv  # noqa: F401
except ModuleNotFoundError:
    from agentdojo_defender_env import AgentDojoDefenderEnv  # noqa: F401


class AgentDojoDefenderFlow(AgentEnvLoop):
    """AgentEnvLoop that also exports the three reward components as metrics."""

    async def run(self, sampling_params: dict, **kwargs) -> AgentFlowOutput:
        env = self._create_env(**kwargs)
        obs = env.reset(**kwargs)
        tools = getattr(env, "tool_schemas", None)
        steps, metrics = [], {}

        for step_idx in range(self.max_steps):
            prompt_ids = await self._obs_to_prompt(obs, tools=tools)
            if len(prompt_ids) > self.prompt_length:
                break
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=uuid4().hex, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
            response_ids = output.token_ids[: self.response_length]
            response_text = await self.loop.run_in_executor(
                None,
                lambda ids=response_ids: self.tokenizer.decode(ids, skip_special_tokens=self.skip_special_tokens),
            )
            with simple_timer("tool_calls", metrics):
                next_obs, reward, done, info = await env.step(Action(text=response_text, token_ids=response_ids))
            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=(output.log_probs[: self.response_length] if output.log_probs else None),
                routed_experts=(
                    output.routed_experts[: len(prompt_ids) + self.response_length]
                    if output.routed_experts is not None else None
                ),
                reward_score=reward,
            )
            # Terminal env info contains task_completed/privacy_leak/
            # attack_success; AgentFlowWorker exposes this dict as reward/*.
            metric_keys = {
                "task_completed", "privacy_leak", "attack_success", "format_error", "out_of_scope_tool",
                "defense_policy_quality", "defense_format_valid", "defense_boundary_respected",
                "defense_scope_respected",
            }
            step.extra_fields["reward_extra_info"] = {
                key: float(info.get(key, 0.0)) for key in metric_keys
            }
            step = await self._postprocess(step, **kwargs)
            steps.append(step)
            if done:
                break
            obs = next_obs
        return AgentFlowOutput(steps=steps, metrics=metrics)
