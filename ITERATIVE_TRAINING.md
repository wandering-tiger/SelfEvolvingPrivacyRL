# Iterative Challenger-Style Training

This project supports iterative training where each round regenerates the attack dataset using the latest model parameters. This mirrors the “challenger → solver → update → regenerate” loop from the GRPO framework diagram.

## What changes each round

1. Generate a fresh dataset with the **latest attacker checkpoint**.
2. Train the attacker for a small number of steps.
3. Repeat for the next round.

## One-command iterative run

```bash
bash /home/fangzibang/SelfEvolvingPrivacyRL/training/iterative_train_with_verl.sh
```

### Optional environment overrides

```bash
ROUNDS=5 STEPS_PER_ROUND=100 SAMPLES_PER_ROUND=4000 \
  bash /home/fangzibang/SelfEvolvingPrivacyRL/training/iterative_train_with_verl.sh
```

## Notes

- The script uses `/home/fangzibang/SelfEvolvingPrivacyRL/build_dataset.py --model_path <checkpoint>`
  to generate a new `train.parquet`/`val.parquet` each round.
- The vLLM victim service is started once at the beginning and stopped after all rounds finish.
- Checkpoints are written to:
  `/home/fangzibang/storage/models/privacy_attacker_iterative/round_<N>`

## Troubleshooting

- If dataset generation is slow, reduce `SAMPLES_PER_ROUND` or `max_new_tokens` in `build_dataset.py`.
- If a round fails to load a checkpoint, ensure the previous round completed and wrote a checkpoint directory.
