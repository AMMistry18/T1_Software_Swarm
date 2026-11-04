# Mission-planning examples

Scenario YAML lives in `configs/scenarios/planning/`:

| File | Purpose |
|---|---|
| `one_mapper_small.yaml` | Feasible 1 mapper + 2 relays |
| `multi_mapper.yaml` | 2 mappers + 3 relays |
| `concave.yaml` | L-shaped concave region |
| `failure_case.yaml` | Difficult under the distance model |

```bash
python -m tar_swarm.mission_planning.cli preprocess \
  --request configs/scenarios/planning/one_mapper_small.yaml \
  --output /tmp/prepared.json

python -m tar_swarm.mission_planning.cli plan \
  --request configs/scenarios/planning/one_mapper_small.yaml \
  --fallback static \
  --output /tmp/plan_result.json
```

See [docs/mission_planning.md](../../docs/mission_planning.md).
