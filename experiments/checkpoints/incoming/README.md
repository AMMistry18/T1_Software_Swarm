# Drop Kaggle checkpoints here

Copy your downloaded `best_model.zip` into this folder (meta.json optional):

```text
experiments/checkpoints/incoming/best_model.zip
```

Then tell the agent to evaluate, or run:

```bash
./scripts/eval_incoming_checkpoint.sh
```

Outputs go to `experiments/results/incoming_eval/`.
