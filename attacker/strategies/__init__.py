from attacker.strategies.base import AttackStrategy
from attacker.strategies.direct import DirectPromptInjectionStrategy
from attacker.strategies.indirect import IndirectPromptInjectionStrategy
from attacker.strategies.roleplay import RoleplayStrategy
from attacker.strategies.reasoning import ReasoningBasedStrategy
from attacker.strategies.multi_turn import MultiTurnStrategy


def default_strategies():
    return {
        strategy.name: strategy
        for strategy in [
            DirectPromptInjectionStrategy(),
            IndirectPromptInjectionStrategy(),
            RoleplayStrategy(),
            ReasoningBasedStrategy(),
            MultiTurnStrategy(),
        ]
    }


__all__ = [
    "AttackStrategy",
    "DirectPromptInjectionStrategy",
    "IndirectPromptInjectionStrategy",
    "RoleplayStrategy",
    "ReasoningBasedStrategy",
    "MultiTurnStrategy",
    "default_strategies",
]
