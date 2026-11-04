# Offline RL mission planning

Pre-deployment planning for mapping drones and airborne relays. The trained
policy runs inside a **planning simulator** to construct a mission; real drones
later execute the **exported waypoint mission** with onboard flight control.
This is separate from MuJoCo Step 0 physics and from the toy network demo.

## Architecture

```text
MissionRequest (validated schema)
        │
        ▼
 deterministic preprocess  →  PreparedScenario
   (partition, lawnmower,   (segments, candidates,
    candidate relays)        initial inventory)
        │
        ▼
 Gymnasium MaskablePPO env  ↔  kinematic PlanningSimulator
   (masked discrete cmds)        + distance link model
        │
        ▼
 MissionPlan export  →  independent validator / replay
        │
        ▼
 adapter boundary → existing Reference/State (no auto-upload)
```

Separately callable stages: `prepare_scenario`, `PlanningSimulator`, policy
inference / baselines, `compile_from_sim`, `validate_mission`, `plan_mission`.

## Coordinates and altitude

- Internal frame: local ENU meters, **x east, y north, z up** (same as network /
  MuJoCo world conventions).
- Survey altitude is absolute **z** above the local datum. Flat-field planning
  treats this as AGL when the datum is local ground. Terrain-relative AGL is
  not modeled.
- Optional WGS84 inputs convert via a local tangent-plane about a declared
  origin (`coords.LocalFrame`). Intended for few-kilometer surveys.

## Supported geometry

| Feature | Status |
|---|---|
| Simple convex polygons | Supported |
| Simple concave polygons | Supported (not replaced by bounding boxes) |
| Polygon holes | **Rejected** explicitly |
| Obstacles / terrain | **Rejected** explicitly |
| Variable altitude survey | Not supported (fixed survey altitude) |

Coverage gaps from clipping, strip partitioning, or discretization are measured
and reported on `PreparedScenario.coverage_gap_area_m2` with warnings.

## Communication assumption

Default model: **conservative Euclidean range** with `connection_margin_m`.
Permitted forwarding nodes: ground station + relays (mappers optional via
`allow_mapper_forwarding`). Region shape does not reveal real RF. Graph
connectivity does **not** imply LiDAR/point-cloud throughput.

## RL formulation (summary)

See `spaces.py` for the full table of observation layout, action indexing,
transitions, rewards, and episode endings.

**Limitation:** mapper survey **segment order is deterministic**; RL learns
**relay placement** and **mapper release timing** only.

Capacities are fixed and padded (`max_mappers`, `max_relays`, `max_candidates`,
`max_segments_per_mapper`). Checkpoints declare capacities and reject larger
scenarios.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-planning.txt -r requirements-dev.txt
python -m pip install -e .
```

Editable install still declares `requires-python >=3.12,<3.13` for the full
package. The planning modules themselves run on 3.10+ if dependencies are
present and `PYTHONPATH=src` is set.

## Commands

```bash
# Preprocess
python -m tar_swarm.mission_planning.cli preprocess \
  --scenario one_mapper_small --output /tmp/prepared.json

# Smoke train (pipeline only — does not prove learning)
python -m tar_swarm.mission_planning.cli smoke \
  --output experiments/results/plan_smoke --timesteps 512

# Substantial training
python -m tar_swarm.mission_planning.cli train \
  --scenario one_mapper_small \
  --output experiments/results/plan_train \
  --timesteps 200000 --n-envs 4

# Resume
python -m tar_swarm.mission_planning.cli train \
  --scenario one_mapper_small \
  --output experiments/results/plan_train_resume \
  --resume experiments/results/plan_train/checkpoints/final_model.zip \
  --timesteps 100000

# Evaluate policy + baselines
python -m tar_swarm.mission_planning.cli evaluate \
  --scenario one_mapper_small \
  --checkpoint experiments/results/plan_train/checkpoints/final_model.zip \
  --output experiments/results/plan_eval.json

# Plan (RL) or explicit baseline fallback
python -m tar_swarm.mission_planning.cli plan \
  --scenario one_mapper_small \
  --checkpoint experiments/results/plan_train/checkpoints/final_model.zip \
  --output /tmp/planning_result.json

python -m tar_swarm.mission_planning.cli plan \
  --scenario one_mapper_small --fallback search \
  --output /tmp/baseline_result.json

# Validate / replay exported plan
python -m tar_swarm.mission_planning.cli validate \
  --scenario one_mapper_small --plan /tmp/plan.json

python -m tar_swarm.mission_planning.cli replay \
  --scenario one_mapper_small --baseline search \
  --output experiments/results/plan_replay

# Animated GIF (portable) and/or live Matplotlib window
python -m tar_swarm.mission_planning.cli replay \
  --scenario one_mapper_small --baseline static \
  --gif --output experiments/results/plan_replay_gif

python -m tar_swarm.mission_planning.cli replay \
  --scenario one_mapper_small --baseline static \
  --animate --output experiments/results/plan_replay_live
```

On WSL, `--gif` is the reliable path (open the `.gif` in Windows). `--animate`
needs a GUI backend / WSLg Matplotlib window.


Library API:

```python
from tar_swarm.mission_planning import plan_mission, MissionRequest
result = plan_mission(request, checkpoint=".../final_model.zip")
```

## Hardware integration

1. Produce a validated `MissionPlan` via `plan_mission`.
2. Consume actions with readiness dependencies (see `execution_contract`).
3. Optionally convert actions to `Reference` streams via
   `mission_planning.adapter.iter_references_from_action` for the existing
   MuJoCo/control boundary.
4. On missed dependencies or degraded links: **pause safely**, report, replan.
5. **Never** auto-upload to aircraft from this package. No MAVLink/PX4 protocol
   is invented here.

## Validator scope

The independent validator establishes geometric coverage under the configured
swath, distance-model connectivity, kinematic timing/dependencies, separation,
and flight-time budgets. It does **not** establish real RF throughput, battery
chemistry, wind, GNSS integrity, or tracking error.

## Example scenarios

| Name | Intent |
|---|---|
| `one_mapper_small` | Feasible 1 mapper + 2 relays |
| `multi_mapper` | 2 mappers + 3 relays within capacity |
| `concave_region` | L-shaped concave polygon |
| `failure_case` | Tiny radio range vs large region (difficult / often infeasible under the model) |

Failure to find a plan does not prove physical infeasibility.
