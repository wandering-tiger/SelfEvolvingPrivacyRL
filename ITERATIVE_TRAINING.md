# Iterative Challenger-Style Training

This project supports iterative training where each round regenerates the attack dataset using the latest model parameters. This mirrors the “challenger → solver → update → regenerate” loop from the GRPO framework diagram.

## What changes each round

1. Generate a fresh dataset with the **latest attacker checkpoint**.
2. Train the attacker for a small number of steps.
3. Repeat for the next round.

## One-command iterative run

```bash
bash /home/fangzibang/SelfEvolvingPrivacyRL/scripts/main.sh
```

### GPU assignment (attacker vs guard)

```bash
ATTACKER_GPU=0 GUARD_GPU=1 bash /home/fangzibang/SelfEvolvingPrivacyRL/scripts/main.sh
```

### Optional environment overrides

```bash
ROUNDS=5 STEPS_PER_ROUND=100 SAMPLES_PER_ROUND=4000 \
  bash /home/fangzibang/SelfEvolvingPrivacyRL/scripts/main.sh
```

## Notes

- The script uses `/home/fangzibang/SelfEvolvingPrivacyRL/build_dataset.py --model_path <checkpoint>`
  with `--rewrite_backend vllm` by default to generate updated attacker prompts.
  to generate a new `train.json`/`val.json` each round.
- The vLLM guard service is started once at the beginning and stopped after all rounds finish.
- Checkpoints are written to:
  `/home/fangzibang/storage/models/privacy_attacker_iterative/round_<N>`

## Troubleshooting

- If dataset generation is slow, reduce `SAMPLES_PER_ROUND` or `max_new_tokens` in `build_dataset.py`.
- If a round fails to load a checkpoint, ensure the previous round completed and wrote a checkpoint directory.
