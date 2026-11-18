# Shared / dropped checkpoints

## MVP checkpoint (tracked in git)

`best_model.zip` is the current shared MaskablePPO MVP weights (promoted from
the anti-thrash Kaggle run). Teammates can use it directly:

```bash
python -m tar_swarm.mission_planning.cli plan \
  --scenario multi_mapper \
  --checkpoint experiments/checkpoints/incoming/best_model.zip \
  --output /tmp/plan.json

python scripts/play_mission_on_network.py \
  --scenario multi_mapper \
  --checkpoint experiments/checkpoints/incoming/best_model.zip \
  --animate \
  --output experiments/results/demo_network_live
```

## Dropping a new Kaggle download

Copy a fresh download here as e.g. `best_model (4).zip` (numbered zips stay
gitignored). To promote it for the team, overwrite `best_model.zip` and commit.

```bash
./scripts/eval_incoming_checkpoint.sh
```

Eval outputs go to `experiments/results/incoming_eval/`.
