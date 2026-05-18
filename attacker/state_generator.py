import random
import torch
import torch.nn as nn

class AttackState:
    TARGETS = ['profile', 'financial', 'medical', 'location', 'memory']
    STRATEGIES = ['direct', 'indirect', 'roleplay', 'reasoning', 'multi_turn']
    PATHS = ['memory', 'tool', 'reasoning_path', 'trajectory_path']

    def __init__(self, privacy_target, attack_strategy, attack_path):
        self.privacy_target = privacy_target
        self.attack_strategy = attack_strategy
        self.attack_path = attack_path

    def to_dict(self):
        return {
            "privacy_target": self.privacy_target,
            "attack_strategy": self.attack_strategy,
            "attack_path": self.attack_path
        }

class StateGenerator:
    def __init__(self, policy_model=None):
        self.policy_model = policy_model

    def random_initialization(self):
        return AttackState(
            privacy_target=random.choice(AttackState.TARGETS),
            attack_strategy=random.choice(AttackState.STRATEGIES),
            attack_path=random.choice(AttackState.PATHS)
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
