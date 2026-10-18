# Basic network simulator

This is a standalone **kinematic simulator with a toy communication graph**.
It runs independently of the MuJoCo flight simulator and requires only NumPy,
Matplotlib, and PyYAML. It is a foundation for network experiments, not the full
Step 1 link-quality milestone in `algorithm_stack.md`.

## Run

From the repository root, use your installed Python to create a project-local
environment with just the network dependencies. The standalone simulator works
with Python 3.10 (including WSL's Python 3.10.12); it does not need a separate
Python 3.12 installation. The full package's editable install still requires
Python 3.12. A venv isolates this project's packages while reusing your installed
Python interpreter.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-network.txt

# Animated top-down display; space pauses/resumes.
.venv/bin/python scripts/run_network.py --animate

# Unattended execution, logs, and final PNG (no GUI/OpenGL required).
.venv/bin/python scripts/run_network.py --headless --output experiments/results/network_trial
```

For subsequent runs in your WSL terminal:

```bash
cd /home/rschnurr/T1_Software_Swarm
source .venv/bin/activate
python scripts/run_network.py --animate
```

Press space to pause/resume. Close the window to save logs and the final plot.
Run `deactivate` to leave the environment. If WSL's Python lacks `ensurepip`
but `python3 -m pip` is available, bootstrap the venv with that existing pip:

```bash
python3 -m venv --without-pip .venv
python3 -m pip --python .venv/bin/python install pip -r requirements-network.txt
```

On Windows, use `py -3.12` to create the environment and
`.venv\Scripts\python.exe` in place of `.venv/bin/python`.
The script works directly from a checkout; installing the full package is
unnecessary. Installing `requirements.txt` still supports both simulators.
Omitting a display option defaults to headless execution. Every run uses a new
output directory; an existing directory is rejected. The animation holds the
final frame at the configured duration; close the window to save outputs.
Closing it early saves a partial run with
`completed: false` and metrics clipped to elapsed simulation time.

Edit [configs/network.yaml](../configs/network.yaml), or pass `--config PATH`.
Agent counts come from the `agents` list: exactly one active ground station,
at least one mapper, and any number of relays. IDs must be unique. Each drone
has its own initial 3D position, speed limit, waypoint sequence, and optional
`active: false` flag. An inactive node stays visible/logged, has no links, and
does not move. Empty waypoint sequences hold the initial position.

## Model and clocks

- Positions use meters, x east, y north, z up. Each drone holds its own starting
  altitude; waypoint altitude changes are rejected. The display is top-down.
- Each step moves toward one waypoint by at most `speed_m_s * dt_s`. A short
  remaining segment snaps exactly to its target without overshoot. Remaining
  movement budget is discarded for that step; the next segment starts next
  step. Final waypoints hold indefinitely. Ground position never changes.
- This is kinematics: no acceleration, attitude, dynamics, collision avoidance,
  obstacle model, sensor simulation, or physical flight claims.
- Two active nodes have a **physical edge** iff their Euclidean **3D** distance
  is `<= communication_radius_m`. This is an uncalibrated, bidirectional toy
  model; it does not represent VoCore2 range, link quality, or routing software.
- Ground reachability traverses physical edges through any number of graph
  nodes. All active nodes may be intermediate vertices in this abstraction.
  Reachability never inserts a direct ground-to-mapper edge. No forwarding
  behavior or packet transport is simulated.
- Tick zero records initial positions and graph. Each subsequent tick moves all
  drones using targets from the previous snapshot, advances the integer clock,
  updates the graph if due, then samples state and next targets. Motion updates
  are simultaneous in the sense that policies see a complete position snapshot.
- Duration and graph interval must be positive integer multiples of `dt_s`.
  Graph edges/reachability are held between graph updates; their observation time
  is logged and displayed. Lines connect current displayed endpoints, even if a
  held edge has become stale before the next graph update.
- Drawing never steps the engine. Animation timer callbacks request one fixed
  step; slower rendering only changes wall-clock playback. Pause advances no
  simulation time. Headless and animated stepping use the same engine.
- There are no random inputs in the baseline. The config records a seed, and
  `Simulator.rng` provides a local seeded NumPy generator for future extensions.
  Reproducibility applies within the same numerical environment, whose versions
  are saved with the run.

## Outputs and metric convention

Each run saves:

| File | Contents |
|---|---|
| `states.csv` | Every tick including t=0 and terminal state: ID, role, active flag, 3D position, target, observed graph timestamp, ground reachability |
| `graphs.jsonl` | Initial graph and each scheduled graph update, including empty edge sets, active IDs, physical edge pairs, and reachable IDs |
| `summary.json` | Completion flag, elapsed time, model labels, seed, per-mapper connected seconds/fraction and outage intervals |
| `resolved_config.yaml` | Validated config including default agent fields |
| `environment.json` | Python and dependency versions |
| `trajectory_connectivity.png` | Final positions/links, complete trails, and each mapper's ground reachability over time |

Sample k's observed reachability applies to **[t[k], t[k+1])**. Connected
duration is the sum of those interval lengths while connected; fraction divides
by actual elapsed time. The terminal sample has zero duration. Outages are
half-open intervals clipped to the run; an outage still open at shutdown ends
at the run's final timestamp. A zero-duration partial run has fraction `null`.
The observation interval limits outage timing resolution. These are graph
connectivity metrics, not packet delivery, latency, or throughput measurements.

## Demonstrated result

The default 24-second run begins with the chain
`ground — relay_1 — relay_2 — mapper_1`. The ground is at (0, 0, 0), relays
start at x=25 and x=50 at 10 m altitude, and the mapper starts at (70, 0, 10).
The 30 m radius supports the chain, with no direct ground-to-mapper edge.
The mapper travels out to x=110 and back at 4 m/s. Relays independently move
8 m sideways and return at 1 m/s; this is entirely scripted motion.

Actual headless output in `experiments/results/network_demo_verified/`:

- Mapper outage: **[2.5, 17.5) seconds**; reconnects at 17.5 seconds.
- Mapper connected time: **9 / 24 seconds = 37.5%**.
- **241** state snapshots (964 agent rows) and **49** graph updates.
- Final connectivity is restored. CSV/JSON/config output was identical across
  repeated runs in the tested environment.

Generated runs are ignored by Git, following the existing results convention.
Rerun with a new output directory to reproduce the demonstration.

## Where to add algorithms

| Location/interface | Team extension |
|---|---|
| `network/motion.py`: `TargetPolicy.target(id, time_s, positions)` | Supply movement targets, including future relay placement; pass an instance to `Simulator(config, policy=...)` |
| `network/radio.py`: `LinkModel.usable(first, second)` | Replace the radius predicate with a different bidirectional availability model; pass `link_model=...` |
| `network/graph.py`: `build_graph(positions, active, time_s, link_model)` | Feed actual MuJoCo position snapshots into the same graph function, independently of the kinematic runner |
| `network/simulator.py`: `step()` / returned `Frame` | Use timestamped state and graph observations for future message generation and queues |
| Future `network/transport.py` | Add packet events/transport when authorized; graph observations currently say nothing about packet delivery |
| `metrics/network.py`, `network/results.py`, `network/visualization.py` | Define metrics once, then log and display them |

Policies currently receive global ground-truth positions: this is a centralized
experiment interface. Future distributed policies will need explicit local
observations and delivered-message caches. Target policies must treat position
snapshots as read-only and return finite fixed-altitude 3D targets. Policy calls
occur once per agent per recorded tick; rendering never calls policies.

Do not infer realistic flight or radio performance from this baseline. It has
no autonomous relay placement, outage recovery, routing protocols, measured
link quality, obstacle effects, queues, or packet exchange.

## Verification

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_network.py
.venv/bin/python -m ruff check .
```

For PowerShell, set `$env:PYTHONPATH = 'src'` before the pytest command. Focused
checks cover speed/waypoint behavior and final holding, stationary ground,
direct/multi-hop/disconnected/inactive nodes, 3D range boundaries, graph timing,
configurable counts, replaceable target policies, time-weighted metrics,
repeatability, CLI output, existing-output protection, and animation controls.

Local verification: **16 network checks and all 43 repository tests passed**;
repo-wide Ruff checks passed. The available interpreter was Python 3.10.12
(the project's supported environment remains Python 3.12, configured in CI).
Animation drawing, pause/resume, and identical stepping were checked through
callbacks with the Agg backend. Follow-up WSL setup repaired the project-local
`.venv` with Python 3.10.12, installed the network dependencies there, and verified
both a complete 24-second headless demo and a real TkAgg animation window through
WSLg. The venv demo produced the same [2.5, 17.5) outage and 37.5% connected fraction.
CI includes a separate network-only job that installs no MuJoCo or other physics
dependencies.
