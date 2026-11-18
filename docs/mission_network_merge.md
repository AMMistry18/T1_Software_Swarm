# Merging mission planning ↔ packet/network simulator

This note captures the intended integration so we do not forget the longer-term
plan. **Phase 1 is implemented now:** planned missions can be *played back*
inside the packet/network simulator (links, ranges, routing, packets) instead of
relying only on the planning GIF.

## Goals

- Keep **fast RL training** in the mission-planning kinematic env.
- Use the **network/packet sim as radio-truth playback + visualization**.
- Eventually share link rules and success criteria without forcing every PPO
  step through full packet simulation.

## What exists today

| Piece | Owner | Notes |
|---|---|---|
| Regions, lawnmower, MaskablePPO, coverage GIF | `mission_planning/` | Fast; distance-graph connectivity |
| Role radio ranges, obstacles/LOS, packets, latency, live UI | `network/` | Higher fidelity RF + transport |
| Shared kinematics / graph helpers | both | `move_towards`, `build_graph`, thin link adapter |
| **Phase 1 bridge** | `mission_planning/network_playback.py` | `MissionPlan` → network scene + `MissionPlanPolicy` |

### Phase 1 commands

```bash
# Live window: watch a baseline plan with real network viz
python scripts/play_mission_on_network.py \
  --scenario one_mapper_small --fallback search --animate \
  --output experiments/results/network_play_search

# Or via module CLI
python -m tar_swarm.mission_planning.cli network-play \
  --scenario one_mapper_small --fallback search --animate \
  --output experiments/results/network_play_search

# RL checkpoint (zip) → headless packet logs + trajectory_connectivity.png
python -m tar_swarm.mission_planning.cli network-play \
  --scenario one_mapper_small \
  --checkpoint experiments/checkpoints/incoming/best_model.zip \
  --output experiments/results/network_play_rl
```

Outputs include `network_scene.yaml`, `mission_plan.json`, network `summary.json`
(with packet stats), and `trajectory_connectivity.png`.

GIF replay (`cli replay --gif`) remains available as a lightweight offline
artifact; network-play is the preferred interactive / RF-aware viewer.

## Recommended future phases

### Phase 2 — Shared radio validation (no training slowdown)

1. Map `CommunicationConfig` ↔ network `radio_range_m` (and optional obstacles).
2. After `plan_mission`, always run a **headless network playback** and attach
   packet/connectivity KPIs to `PlanningResult.metrics`.
3. Gate “mission accepted” on both:
   - planning coverage / returns, and
   - network metrics (e.g. mapper reachability fraction, keyframe delivery,
     command RTT bounds).
4. Keep RL rewards on the fast distance model; use network metrics for
   **accept/reject** and dashboards.

### Phase 3 — Shared link semantics

1. Replace or wrap planning’s `ConservativeDistanceLink` with network
   `RangeLink` (per-role ranges + obstacle LOS) behind `PlanningLinkModel`.
2. Unify forwarding rules (who may relay: ground/relays/optional mappers).
3. Align clocks: document resampling between `sim_dt_s` / `decision_dt_s` and
   network `dt_s` / `graph_interval_s`.
4. Align IDs: `ground` station, mapper/relay names, altitude conventions
   (network currently requires fixed altitude per agent).

### Phase 4 — Optional closed loop

1. Occasional RL eval episodes in the network sim (expensive).
2. Or distill network failures into planning scenarios / curriculum.
3. Obstacle-aware planning (planning currently rejects `obstacles_m`).
4. Combined UI: region coverage overlay + network link/packet panel on one
   figure or shared live window.

## Design rules (do not forget)

1. **Do not train PPO inside the packet loop** by default — too slow / noisy.
2. Treat planning connectivity as an **estimate**; network playback is
   **radio-truth for visualization and acceptance**.
3. Graph “connected” ≠ LiDAR throughput — packet stats matter for mapping QoS.
4. Prefer adapters (`MissionPlan` → network config/policy) over merging the two
   simulators into one class.

## Open questions

- Which packet KPIs are hard requirements vs soft diagnostics?
- Should playback use planner `usable_range_m` or measured VoCore2 tables?
- How to show survey coverage footprint inside the network live window?
- When (if ever) to put obstacles into `MissionRequest` for joint planning?

## Related docs

- `docs/mission_planning.md` — planner architecture and rewards
- `docs/network_simulator.md` — packet/network sim usage and live controls
- `docs/algorithm_stack.md` — broader stack milestones
