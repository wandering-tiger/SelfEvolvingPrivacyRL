import random
import torch
import torch.nn as nn

from attacker.state import AttackState

class StateGenerator:
    def __init__(self, policy_model=None):
        self.policy_model = policy_model

    def random_initialization(
        self,
        privacy_target: str = None,
        attack_strategy: str = None,
        attack_path: str = None,
    ):
        return AttackState(
            privacy_target=privacy_target or random.choice(AttackState.TARGETS),
            attack_strategy=attack_strategy or random.choice(AttackState.STRATEGIES),
            attack_path=attack_path or random.choice(AttackState.PATHS),
        )

    def generate_from_policy(self, context_vector):
        if self.policy_model is None:
            return self.random_initialization()
        
        with torch.no_grad():
            logits = self.policy_model(context_vector)
            # Assuming logits contain concatenated probabilities for target, strategy, path
            # Simplified selection:
            target_idx = torch.argmax(logits[:len(AttackState.TARGETS)]).item()
            strategy_idx = torch.argmax(logits[len(AttackState.TARGETS):len(AttackState.TARGETS)+len(AttackState.STRATEGIES)]).item()
            path_idx = torch.argmax(logits[-len(AttackState.PATHS):]).item()
            
        return AttackState(
            privacy_target=AttackState.TARGETS[target_idx],
            attack_strategy=AttackState.STRATEGIES[strategy_idx],
            attack_path=AttackState.PATHS[path_idx]
        )
