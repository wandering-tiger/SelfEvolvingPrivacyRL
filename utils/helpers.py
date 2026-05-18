def print_attack_summary(attack_state, prompt, reward):
    print("=== Attack Summary ===")
    print(f"Target:   {attack_state.privacy_target}")
    print(f"Strategy: {attack_state.attack_strategy}")
    print(f"Path:     {attack_state.attack_path}")
    print(f"Prompt:\n{prompt}")
    print(f"Reward:   {reward:.2f}")
    print("======================")
