# Basic network simulator

This is a standalone **kinematic simulator with a toy communication graph** and
a live window where you can edit the run with the mouse while it plays.
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

Close the window to save logs and the final plot. Run `deactivate` to leave the
environment. If WSL's Python lacks `ensurepip`
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
output directory; an existing directory is rejected. The live window's clock
keeps running until you close it, even past `duration_s`, so you can keep
editing; `duration_s` ends headless runs. Closing the window saves the run up to
that moment; `completed` says whether it reached `duration_s`.

### Live window controls (`--animate`)

| Action | Effect |
|---|---|
| Click a drone, then click the map | Selects the drone (gold ring), then sends it flying there at its own speed; it holds on arrival. Click the selected drone again, or press Esc, to deselect. |
| Drag a drone | Puts it there at once; it then holds that spot (its remaining waypoints are dropped). The ground station never moves. |
| Drag on an empty part of the map | Draws an obstacle box. Boxes block radio links that would pass through them (drones still fly through them for now). |
| Right-click an obstacle | Removes it |
| Right-click an empty spot | Adds a network drone there, at `new_relay_altitude_m` and `new_relay_speed_m_s`, holding still |
| Shift + right-click an empty spot | Adds a mapping drone there, at `new_mapper_altitude_m` and `new_mapper_speed_m_s`, holding still |
| Right-click a drone | Removes it (never the ground station or the last mapper) |
| "Show radio range" checkboxes | Show or hide the range circles of network drones, the ground station, and mappers |
| "Radio range (m)" sliders | Change the range of network drones, the ground station, and mappers during the run (5% of the map's side up to the full side) |
| "auto optimal positions" checkbox | While checked, every step re-plans the best spots (one straight chain from the ground station to each mapper) for where the mappers are now, and the network drones fly toward them, so the chains follow moving mappers (see `strategies/fixed_relay.py`). The title shows how stretched the planned hops are. Hand-set goals for network drones are ignored while it's checked; unchecking it makes them hold where they are. |
| "save results when closed" checkbox | Unchecked: closing the window saves nothing |
| Space | Pause / resume |
| `+` (or `=`) / `-` | Faster / slower: 1x, 2x, 5x, 10x, 20x, 50x, or 100x real time (starts at 5x) |

The ring around each drone is green when it can reach the ground station and red
when it can't; a dashed line and an `x` show where a drone is flying. Moving green
dashes trace each mapper's route to the ground station (`network/routing.py`),
flowing toward the ground station like the mapper's messages. The
toolbar's zoom and pan tools work as usual; mouse edits are ignored while one of
them is on.

**Optimal positions.** For n network drones there are n + 1 hops. Each hop's
limit is the smaller range of its two ends. The plan puts the drones on the
straight line from the ground station to the mapper, with every hop stretched to
the same fraction of its limit: hop length = distance x limit / (sum of limits).
No other placement makes the most-stretched hop less stretched. Example: a mapper
750 m away, all ranges 300 m, 2 drones: hops of 250 m, each at 83% of its limit.
Ranges are 3D but the plan is on the flat map, so a hop from the ground station
that also climbs (10 m in the scenes) gets sqrt(limit^2 - climb^2) on the map.
Above 100% even the best spacing can't connect: then each hop takes its full
limit from the ground station, so every drone stays linked to it and the only
gap is at the mapper's end. With several mappers, each gets its own chain; the
network drones are handed out one at a time to whichever chain is most
stretched, and the Hungarian method (`scipy.optimize.linear_sum_assignment`)
pairs drones with spots so the total flying distance is as small as possible.
The plan ignores obstacles for now.

### Scenes

The default scene, [configs/network.yaml](../configs/network.yaml), is a 1 km x
1 km map with the ground station at its bottom-left corner (0, 0, 0), 300 m
placeholder radio ranges, and grid squares of `grid_step_m` (100 m).
The original 24-second, 30 m demo lives in `tests/test_network.py` (`SMALL_DEMO`),
because many tests check its exact numbers. `radio_range_m` gives one
range per role (`ground_station`, `relay`, `mapper`). `obstacles_m` (optional)
lists boxes `[x_min, y_min, x_max, y_max]` that block radio links. Five settings
exist for the live window: `grid_step_m` (drawing only), and
`new_relay_altitude_m`, `new_relay_speed_m_s`, `new_mapper_altitude_m`, and
`new_mapper_speed_m_s` (for drones added with a right-click).

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
  is at most the **smaller** of their two roles' ranges in `radio_range_m`: a link
  carries messages both ways, so each end must reach the other. A larger ground
  station range alone therefore doesn't lengthen ground-to-relay links. An
  obstacle box blocks a link whose straight line passes through it on the map
  (touching an edge or corner doesn't count; `common/world.py`). A 1e-6 m slack
  keeps rounding from breaking a hop placed exactly at its limit. This is
  an uncalibrated toy model; it does not represent VoCore2 range, link quality,
  or routing software.
- Ground reachability traverses physical edges through any number of graph
  nodes. All active nodes may be intermediate vertices in this abstraction.
  Reachability never inserts a direct ground-to-mapper edge. Routing
  (`network/routing.py`, a stand-in for batman-adv) finds each drone's next hop
  toward a destination by breadth-first search: the fewest-hops path, with ties
  broken the same way every time. It re-routes instantly when links change
  (batman takes a few seconds). No packet transport is simulated yet.
- Tick zero records initial positions and graph. Each subsequent tick moves all
  drones using targets from the previous snapshot, advances the integer clock,
  updates the graph if due, then samples state and next targets. Motion updates
  are simultaneous in the sense that policies see a complete position snapshot.
- Duration and graph interval must be positive integer multiples of `dt_s`.
  Graph edges/reachability are held between graph updates; their observation time
  is logged and displayed. Lines connect current displayed endpoints, even if a
  held edge has become stale before the next graph update.
- Drawing never steps the engine. Each live-window frame takes as many fixed
  steps as real elapsed time times the speed requires (at 5x, 0.05 real seconds
  owe 0.25 s: 2 steps of 0.1 s, carrying the leftover forward; one frame never
  counts more than 0.25 real seconds). Steps are identical however fast the
  window draws; only the number of steps between frames changes. Pause advances
  no simulation time. Headless and animated stepping use the same engine.
- The live window redraws only what moves ("blitting"): after each full redraw
  it keeps a picture of the still parts (grid, numbers, controls, help) and each
  frame pastes it back, then draws the drones, links, circles, and title on top.
- Automatic placement: `Simulator.set_relay_planner(planner)` runs
  `planner(simulator, positions)` in every snapshot; network drones in its
  result fly toward their planned spots (`None` switches it off and makes them
  hold). The live window's checkbox uses `optimal_chain_plan`.
- Hand edits (`Simulator.add_drone`, `remove_drone`, `move_agent`, `set_goal`,
  `set_range`, `add_obstacle`, `remove_obstacle`; the live window calls these) never advance the clock. Each one
  re-checks the links at once and replaces the latest snapshot, so the edit shows
  immediately and the next step starts from it. A moved or added drone holds its
  spot; a drone with a goal flies there and then holds. Removed IDs are never
  reused within a run, and `Simulator.roles` remembers every drone that took part,
  for logging.
- There are no random inputs in the baseline. The config records a seed, and
  `Simulator.rng` provides a local seeded NumPy generator for future extensions.
  Reproducibility applies within the same numerical environment, whose versions
  are saved with the run.

## Outputs and metric convention

Each run saves:

| File | Contents |
|---|---|
| `states.csv` | Every tick including t=0 and terminal state: ID, role, active flag, 3D position, target, observed graph timestamp, ground reachability. A drone added or removed by hand has rows only while it was in the run. |
| `graphs.jsonl` | Initial graph, each scheduled graph update, and each hand edit's re-check, including empty edge sets, active IDs, physical edge pairs, and reachable IDs |
| `summary.json` | Completion flag, whether anything was changed by hand (`edited_by_hand`), the radio ranges in use at the end (`radio_range_m_at_end`), elapsed time, model labels, seed, per-mapper connected seconds/fraction and outage intervals |
| `resolved_config.yaml` | Validated config including default agent fields: how the run started (hand edits are not in it) |
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

**Default 1 km scene** (`configs/network.yaml`, no hand edits): the chain
`ground — relay_1 — relay_2 — mapper_1` runs diagonally from the corner, with
relays at (200, 150) and (400, 300) and the mapper at (600, 450), all 10 m up;
each hop is about 250 m against the 300 m placeholder range. The mapper flies
out to (900, 675) and back at 4 m/s. Actual headless output:

- Mapper outage: **[12.5, 175.5) seconds**. At 12.5 s the mapper has flown 50 m,
  putting it right at the 300 m edge of relay_2's range.
- Mapper connected time: **137 / 300 seconds = 45.7%**.

**Small demo** (`SMALL_DEMO` in `tests/test_network.py`): the original 24-second run
begins with the same chain on a 135 m strip. The ground is at (0, 0, 0), relays
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
| `network/radio.py`: `LinkModel.usable(first_role, second_role, first, second)` | Replace the range rule with a different availability model; pass `link_model=...` (the range sliders need `RangeLink`'s `range_m`) |
| `network/graph.py`: `build_graph(positions, active, time_s, link_model, roles)` | Feed actual MuJoCo position snapshots into the same graph function, independently of the kinematic runner |
| `network/simulator.py`: `step()` / returned `Frame` | Use timestamped state and graph observations for future message generation and queues |
| `network/simulator.py`: `add_drone`, `remove_drone`, `move_agent`, `set_goal`, `set_range`, `add_obstacle`, `remove_obstacle` | Script the same hand edits the live window makes (for example, a relay failure at a set time) |
| `network/routing.py`: `next_hops(graph, destination)`, `route(graph, source, destination)` | Fewest-hops routing; grow it into link-quality (ETX) routing once links have quality scores |
| `common/world.py`: `segment_crosses_box`, `line_of_sight` | Obstacle geometry for links now, and for path planning around obstacles later |
| `strategies/fixed_relay.py`: `hop_limits`, `chain_slots`, `split_relays`, `match_relays_to_slots`, `optimal_chain_plan` | The straight-chain placement behind "auto optimal positions"; compare other placement strategies against it through `Simulator.set_relay_planner` |
| Future `network/transport.py` | Add packet events/transport when authorized; graph observations currently say nothing about packet delivery |
| `metrics/network.py`, `network/results.py`, `network/visualization.py` | Define metrics once, then log and display them |

Policies currently receive global ground-truth positions: this is a centralized
experiment interface. Future distributed policies will need explicit local
observations and delivered-message caches. Target policies must treat position
snapshots as read-only and return finite fixed-altitude 3D targets. Policy calls
occur once per agent per recorded tick; rendering never calls policies.

Do not infer realistic flight or radio performance from this baseline. It has
only one placement strategy (straight chains, one per mapper, switched on by
hand, ignoring obstacles), no path planning around obstacles (drones fly
straight through them), no outage recovery, no measured link quality, no
routing delay, and no queues or packet exchange.

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
repeatability, CLI output, existing-output protection, animation controls, the
default 1 km scene, hand edits (add/move/remove, goals, ranges, unique IDs, logs
of removed drones), the two-way range rule, hand-computed optimal spacing,
chains that can't reach, the relay split and Hungarian pairing, two-mapper
plans, routing (chain, shortcut, tie, cut off), obstacle geometry and blocked
links, and the live window's mouse edits, goals, obstacles, mapping drones,
route dashes, checkboxes, sliders, "auto optimal positions" checkbox, and save
checkbox.

Interactive-window update (2026-10-18): **38 network checks and all 69
repository tests passed** in the project's Python 3.12 venv; Ruff passed. On a
WSLg screen (TkAgg, 1050 x 800 pixels, default scene) the live window ran at
about 22 frames per second with 4 drones (with the route dashes moving) and
about 11 with 34; per-drone name
tags are the slowest part to draw. Without blitting, a full redraw took about
56 ms with 4 drones (Agg, off-screen); with blitting, a frame took about 10 ms.

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
