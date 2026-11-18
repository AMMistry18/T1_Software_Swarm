# Mission-planning MVP demos

Side-by-side GIFs for the shared checkpoint
`experiments/checkpoints/incoming/best_model.zip`
(lawnmower baseline | RL planner | network playback).

| Scenario | Demo |
|---|---|
| `one_mapper_small` | [one_mapper_small.gif](one_mapper_small.gif) |
| `multi_mapper` | [multi_mapper.gif](multi_mapper.gif) |
| `concave_region` | [concave_region.gif](concave_region.gif) |

Run it yourself (branch `network`):

```bash
python -m pip install -r requirements-planning.txt -r requirements-network.txt
python -m pip install -e .
export PYTHONPATH=src

python scripts/play_mission_on_network.py \
  --scenario multi_mapper \
  --checkpoint experiments/checkpoints/incoming/best_model.zip \
  --animate \
  --output experiments/results/demo_network_live
```

More detail: [docs/mission_planning.md](../../mission_planning.md),
[docs/mission_network_merge.md](../../mission_network_merge.md).
