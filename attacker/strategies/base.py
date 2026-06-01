from abc import ABC, abstractmethod


class AttackStrategy(ABC):
    name: str

    @abstractmethod
    def build_prompt(self, attack_state) -> str:
        raise NotImplementedError
1