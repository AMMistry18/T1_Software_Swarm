# Step 0 delivery and file manifest

Work is on branch `step0-flight-simulation`. No commits or remote writes were
performed. Earlier manual-flight and swarm placeholder code was preserved.

## Exact modified files

- `README.md` — installation, launch commands, frames, controls, results, limitations.
- `pyproject.toml` — narrow Ruff exception for existing manual-flight formatting.
- `assets/mujoco/mapper.xml` — standalone generated mapping quadrotor model.
- `assets/mujoco/relay.xml` — standalone generated relay quadrotor model.
- `scripts/run_mujoco.py` — Step 0 mission CLI entry point.
- `scripts/plot_results.py` — plots from saved simulation logs.
- `src/tar_swarm/common/config.py` — YAML loading and runtime validation.
- `src/tar_swarm/mujoco/renderer.py` — live overlays and offline snapshots.

## Exact created files

- `configs/vehicles.yaml`
- `configs/hardware_assumptions.yaml`
- `configs/simulation.yaml`
- `configs/worlds/empty.yaml`
- `configs/worlds/outdoor.yaml`
- `configs/missions/step0.yaml`
- `src/tar_swarm/vehicles/__init__.py`
- `src/tar_swarm/vehicles/model.py`
- `src/tar_swarm/control/__init__.py`
- `src/tar_swarm/control/interfaces.py`
- `src/tar_swarm/control/cascaded.py`
- `src/tar_swarm/control/mission.py`
- `src/tar_swarm/mujoco/model.py`
- `src/tar_swarm/mujoco/flight.py`
- `src/tar_swarm/mujoco/runner.py`
- `src/tar_swarm/evaluation/__init__.py`
- `src/tar_swarm/evaluation/logging.py`
- `src/tar_swarm/evaluation/plots.py`
- `scripts/export_models.py`
- `tests/test_step0.py`
- `environments/step0-windows-py312.txt`
- `docs/step0_engineering.md`
- `docs/step0_delivery.md`

## Validation

On Windows 11 with project-local Python 3.12.10 and MuJoCo 3.13.0:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
git diff --check
```

Result: **27 tests passed, 0 failed**, repository Ruff checks passed, and Git
whitespace checks passed. Test runtime was 7.23 seconds. Tests include both
model classes, both worlds, independent mass/inertia verification, initialization,
actuator identities/bounds, unpowered gravity, four wrench-axis directions,
disturbed hover recovery, waypoint visits, deterministic repeated execution,
timed logging, quintic continuity, configuration rejection, agent-count changes,
standalone MJCF loading, zero-net-force handling, and output protection.

## Reproducible demonstrations and generated files

```powershell
.\.venv\Scripts\python.exe scripts/run_mujoco.py --viewer --snapshot
.\.venv\Scripts\python.exe scripts/run_mujoco.py --headless --world configs/worlds/empty.yaml --snapshot
.\.venv\Scripts\python.exe scripts/plot_results.py experiments/results/step0_outdoor_final
```

The final full outdoor viewer demonstration and the empty-world headless run
each execute the 36-second mission. Saved output folders are:

- `experiments/results/step0_outdoor_final/` — full viewer mission and render.
- `experiments/results/step0_empty/` — full empty-world headless experiment.
- `experiments/results/step0_outdoor/` — initial complete headless outdoor run.
- `experiments/results/step0_viewer_check/` — preliminary 3-second viewer check.
- `experiments/results/step0_first/` — first headless run, without plots.

In each complete final run, generated files are:

- `states.csv`
- `controls.csv`
- `metrics.json`
- `resolved_config.yaml`
- `environment.json`
- `scene.xml`
- `scene.png`
- `plots/mapper_0_control.png`
- `plots/relay_0_control.png`
- `plots/relay_1_control.png`
- `plots/relay_2_control.png`
- `plots/trajectories_3d.png`

These local experiment files are intentionally ignored by Git. They are actual
MuJoCo simulation data, not synthetic idealized trajectories.

## Quantitative result and assumptions

Mapper whole-mission position RMSE is 0.02699 m, peak error 0.05274 m, settled
hover altitude RMSE 0.0001242 m, and final hover position RMSE 0.0007389 m.
Relays have position RMSE 0.000922–0.001206 m and hover altitude RMSE
0.0000581–0.0000760 m. Each full run logs 1,801 state samples per vehicle and
14,404 total per-vehicle controller updates. No logged contacts or motor
saturation occurred. Outdoor and empty trajectories agree because the manually
selected paths do not interact with obstacles.

Mapper mass is provisionally 3.615 kg; relay mass is 0.1352 kg. Rotor force limits
are provisionally 22 N and 0.8 N per rotor, implying thrust-to-weight ratios
2.48 and 2.41. Inertias come from component distributions, not arbitrary point
masses. Exact airframes, Jetson model, propulsion curves, ESC response, battery
behavior, wiring, sensor mounts, and antenna properties require measurements.
The relay CAD link was unavailable; its primitive frame is explicitly estimated.

No wind, sensing noise, estimator error, motor lag, energy depletion, SLAM,
networking, online planning, collision avoidance, or MPC is modeled. Ground-truth
feedback and instantaneous force actuators explain the very small settled hover
errors. Electrical and hardware flight readiness are not established. Further
equations, hardware uncertainties, and acceptance thresholds are documented in
`docs/step0_engineering.md`. Work stops at Step 0.
