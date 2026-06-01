from attacker.strategies import default_strategies


class PromptBuilder:
    def __init__(self, llm_generator=None, strategies=None):
        self.llm_generator = llm_generator
        self.strategies = strategies or default_strategies()

    def build(self, attack_state):
        strategy = self.strategies.get(attack_state.attack_strategy)
        if strategy is None:
            strategy = self.strategies.get("direct")
        prompt = strategy.build_prompt(attack_state)

        if self.llm_generator:
            prompt = self.llm_generator.refine_prompt(prompt, attack_state)

        return prompt
