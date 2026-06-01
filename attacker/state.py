class AttackState:
    TARGETS = ["profile", "financial", "medical", "location", "memory"]
    STRATEGIES = ["direct", "indirect", "roleplay", "reasoning", "multi_turn"]
    PATHS = ["memory", "tool", "reasoning_path", "trajectory_path"]

    def __init__(self, privacy_target, attack_strategy, attack_path):
        self.privacy_target = privacy_target
        self.attack_strategy = attack_strategy
        self.attack_path = attack_path

    def to_dict(self):
        return {
            "privacy_target": self.privacy_target,
            "attack_strategy": self.attack_strategy,
            "attack_path": self.attack_path,
        }
