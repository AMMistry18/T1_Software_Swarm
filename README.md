# TAR Swarm — Step 0 flight simulation

Texas Aerial Robotics is developing mapping UAVs supported by smaller airborne
wireless relays and a ground station. Step 0 implements **MuJoCo vehicle physics
and baseline flight control**. The default mission has one mapper, three relays,
and one ground station. Agent counts, vehicle parameters, gains, waypoints,
clocks, gravity, and world geometry are configurable.

The mapper reflects the supplied 18-inch MAD propulsion / 12S battery BOM, with
Jetson, Cube/carrier, downward Unitree L2, thermal camera, GNSS, and radio mass
allowances. Relays reflect 1303 motors, 3-inch props, 2S batteries, a custom
36 × 39 mm ESP32-S3 PCB, IMU, GNSS, TF-Luna, and VoCore2. Hardware values remain
provisional: see [engineering assumptions and results](docs/step0_engineering.md)
and [uncertainty registry](configs/hardware_assumptions.yaml).

## Install

Use Python 3.12 and a project-local virtual environment. The inspected Windows
environment already has Python 3.12.10 and MuJoCo 3.13.0 in `.venv`. System Python
3.14 is separate and does not have MuJoCo.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

On Linux/macOS use `python3.12 -m venv .venv` and substitute `.venv/bin/python`.
On macOS the passive viewer must run with
`.venv/bin/mjpython scripts/run_mujoco.py --viewer`.
To reproduce recorded dependency versions, install
`environments/step0-windows-py312.txt` before the editable package. That records
the tested environment; it is not a cross-platform binary lock. Runtime
requirements retain the existing NumPy, SciPy, Matplotlib, MuJoCo, PyYAML, and
NetworkX list; no additional robotics dependencies were introduced.
Step 0 does not require ROS, PX4, Gazebo, or Blender.

## Run from the repository root

Launch the live outdoor mission:

```powershell
.\.venv\Scripts\python.exe scripts/run_mujoco.py --viewer
```

Run unattended, generate plots, and render a scene PNG:

```powershell
.\.venv\Scripts\python.exe scripts/run_mujoco.py --headless --snapshot
```

Run the empty-world control test:

```powershell
.\.venv\Scripts\python.exe scripts/run_mujoco.py --headless --world configs/worlds/empty.yaml
```

Use `--config PATH`, `--duration SECONDS`, `--output NEW_DIRECTORY`, and
`--no-plots` as needed. Existing output directories are protected. Headless
physics and plotting do not require OpenGL; `--snapshot` and `--viewer` do.
The older falling-body smoke test remains available with
`python -m tar_swarm.mujoco.simulation`.

The mission runs for 36 simulation seconds, then closes the viewer and saves
results. Closing early saves a partial run with `completed: false`.
Blue mapping bodies and orange relay bodies have distinct dimensions and mass
distributions. Colored trails show actual motion, dashed lines show desired
paths, translucent markers show moving references and upcoming waypoints,
and RGB axes show body orientation. Floating labels identify vehicles.
Use MuJoCo mouse camera controls to inspect the smaller relays more closely.

## Outputs and analysis

Each run creates a new folder under `experiments/results/`, containing:

- `states.csv`: measured position, velocity, attitude, angular velocity, desired
  position/velocity, held rotor commands, limit flags, and scene contact count.
- `controls.csv`: every controller update and force command, including
  motor saturation and acceleration-limit flags.
- `metrics.json`: tracking RMSE, peak error, settled hover errors, saturation
  sample counts, completion status, and logged contact count.
- `resolved_config.yaml`, `scene.xml`, `environment.json`: reproducibility inputs.
- `plots/*_control.png`: desired/actual x/y/z, error, velocity, attitude, rotor
  force commands, and saturation events for each vehicle.
- `plots/trajectories_3d.png`: measured and reference 3D trajectories with IDs.
- `scene.png`: optional rendered MuJoCo scene with viewer overlays.

Regenerate figures from an existing run:

```powershell
.\.venv\Scripts\python.exe scripts/plot_results.py experiments/results/step0_outdoor
```

The validated outdoor run is in `experiments/results/step0_outdoor`. Mapper
position RMSE was **0.02699 m**, maximum error **0.05274 m**, and settled hover
altitude RMSE **0.0001242 m**. Relay mission RMSE was 0.000922–0.001206 m.
No contacts or motor saturation occurred. These ideal provisional model results
do not predict hardware performance. Generated outputs are ignored by Git;
the engineering note records acceptance results and limitations.

## Models, frames, and control

Each drone is a rigid free body built from component boxes and cylinders, with
four local rotor sites and force/torque actuators. Mass, COM, and full inertia
follow the component distribution. Both classes have their own limits and
gains. Maximum forces imply provisional thrust-to-weight ratios of 2.48
(3.615 kg mapper) and 2.41 (135.2 g relay). Rigid propeller disks represent
clearance and collisions; blade aerodynamics and rotor spin are not simulated.

World axes are x east, y north, z up. Body axes are forward, left, up. Quaternions
are MuJoCo wxyz, body-to-world. Linear velocity is world-frame; angular velocity
is body-frame. Future PX4 integration requires ENU→NED and FLU→FRD transforms
described in the engineering note.

The controller uses position PD and velocity damping with reference acceleration
and gravity compensation, followed by geometric attitude PD and body-rate
damping. A geometry-derived mixer converts collective force and moments into
four bounded rotor commands. This is a PD baseline with no integral term.
The controller drives actual MuJoCo dynamics; mission commands never set
position or velocity directly. Quintic waypoint segments provide smooth takeoff,
motion, and hover references.

Physics dt is 0.002 s, controller dt 0.01 s, mission/log dt 0.02 s, visualization
dt 0.04 s. Intervals must be integer multiples of physics dt. Derived state is
refreshed after integration so poses and timestamps describe the same instant.
Graphics pacing cannot alter integration clocks. There are no stochastic inputs
in Step 0; the seed is recorded for future sensor/environment noise.

## Architecture and extension interfaces

```text
configs/vehicles.yaml             component masses, geometry, gains, rotor limits
configs/hardware_assumptions.yaml uncertainty and provenance
configs/simulation.yaml           clocks and configuration paths
configs/worlds/                   empty and outdoor obstacle worlds
configs/missions/step0.yaml        agent instances and timed waypoints
src/tar_swarm/common/config.py    loading and parameter validation
src/tar_swarm/vehicles/           geometry and independent mass properties
src/tar_swarm/control/            State/Reference/FlightController; PD and missions
src/tar_swarm/mujoco/             MJCF generation, physics, runner, viewer
src/tar_swarm/evaluation/         logging, metrics, analysis figures
assets/mujoco/                   exported standalone models
models/mujoco/                   preserved earlier manual-flight example
scripts/                         launch, plot, and MJCF export entry points
tests/                           smoke, physics, flight, and determinism checks
docs/step0_engineering.md         equations, results, limitations
```

`State`, `Reference`, and `FlightController.update()` do not import MuJoCo.
Future planners can supply references; future networking can consume state
snapshots independently of physics internals. The physics adapter provides state,
bounded rotor commands, and deterministic single-step integration. Existing
`network`, `mapping`, `strategies`, and Gazebo modules remain untouched
placeholders; their filenames do not imply implemented algorithms.

Export standalone MJCF examples after editing vehicle YAML:

```powershell
.\.venv\Scripts\python.exe scripts/export_models.py
```

Runtime scenes regenerate from YAML. Exported models are derived inspection
assets, not a second source of configuration.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

Checks cover loading, initialization, mass/inertia, rotor identities and force
bounds, gravity, hover force, moment directions, disturbance recovery, waypoint
visits, smooth references, configurable counts, invalid settings, logging, and
determinism. Settled altitude RMSE must be <0.05 m, final-hover position RMSE
<0.05 m, mission position RMSE <0.15 m, maximum error <0.30 m, and waypoint
visit error <0.15 m. The engineering note defines the measurement windows.
The legacy manual-flight script retains its existing formatting through a
narrow Ruff exception; all new Step 0 code is fully linted.

## Limitations and next milestone

No SLAM, sensor outputs, communication graph, routing, dynamic relay placement,
MPC, collision avoidance, battery discharge, motor lag, wind, or hardware
flight-controller integration is implemented. Sensors have mass and mounting
geometry only. Routes are manually separated from obstacles and agents;
edited routes need checking. The controller assumes ground truth and modest
attitudes and does not handle inverted-flight recovery. Physical flight requires
measured propulsion, inertia, timing, sensing, and electrical validation.

The next separately authorized milestone is a dynamic wireless graph with
directional delivery probabilities, line-of-sight checks, and ETX link cost.
This implementation stops at Step 0.
