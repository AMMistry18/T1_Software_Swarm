# TAR Swarm

Shared Python development environment for a university aerial swarm robotics
project. Use MuJoCo for fast dynamics, control, optimization, swarm algorithms,
and learning experiments; use Gazebo Harmonic, ROS 2 Jazzy, and PX4 SITL for
robotics integration and end-to-end flight testing.

## Setup (Python 3.12)

Install Python **3.12**, clone this repository, and open a terminal in its root.
Each developer creates their own `.venv`; virtual environments are never committed.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

If PowerShell blocks activation, you can run these commands with
`.\.venv\Scripts\python.exe` in place of `python`, without activation.

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

The editable install makes `tar_swarm` importable while you develop in `src/`.
In each new terminal, repeat the activation command for your platform. Run
`deactivate` to leave the environment.

### Verify the environment

```bash
python -m pytest
python scripts/mujoco_smoke.py
python -m ruff check .
```

The smoke script prints the MuJoCo version, creates a falling sphere, and advances
ten simulation steps without a window or rendering. You can also run it as
`python -m tar_swarm.simulation.mujoco.smoke` from any directory after installation.
GitHub Actions installs Python 3.12 and runs lint and pytest without a GUI.

## Repository structure

```text
src/tar_swarm/
  common/              # Shared math, geometry, configuration, utilities
  estimation/          # Localization, fusion, degeneracy detection, SLAM algorithms
  swarm/               # Graphs, allocation, exploration, relays, connectivity
  simulation/
    mujoco/            # MuJoCo wrappers and simulation runners
    gazebo/            # Gazebo / ROS 2 / PX4 helpers
models/{mujoco,gazebo}/       # Robot and object models
environments/{mujoco,gazebo}/ # Simulation scenes and worlds
scripts/               # Developer entry points
tests/                 # Automated smoke tests
docs/                  # Experiment notes and integration documentation
```

Keep simulator-independent algorithms in `common/`, `estimation/`, or `swarm/`.
For example, relay placement belongs in `swarm/`; its MuJoCo adapter belongs in
`simulation/mujoco/`. The initial package contains no swarm algorithms.

## Dependencies and external tools

`requirements.txt` is the single runtime dependency list: `mujoco`, `numpy`,
`scipy`, `matplotlib`, `networkx`, and `pyyaml`. Package metadata reads that same
file. `requirements-dev.txt` adds only `pytest` and `ruff`.
Versions are initially unpinned; this is a lightweight starting environment,
not a locked experiment snapshot. Record versions for reproducible experiments
and agree on tested pins as the project develops.

Install **Gazebo Harmonic + ROS 2 Jazzy + PX4 SITL separately in Ubuntu / WSL2**
(Ubuntu 24.04 is the intended integration environment). These system-level
robotics dependencies are intentionally not managed by pip or installed by CI.
Their installation and flight-stack configuration are outside this initial setup.
See the official [ROS 2 Jazzy Ubuntu documentation](https://docs.ros.org/en/jazzy/Installation/Alternatives/Ubuntu-Install-Binary.html)
and [Gazebo Harmonic installation guide](https://gazebosim.org/docs/harmonic/install_ubuntu/).

Install **Blender separately** for environment and mesh creation, and export
assets into the appropriate `models/` or `environments/` directory.

## Which simulator should I use?

**MuJoCo:**

- Dynamics and controller prototyping
- Optimization and MPC experiments
- Rapid algorithm testing
- Repeated parameter sweeps / Monte Carlo experiments
- Reinforcement learning or learning-based control
- Simplified multi-drone simulations
- Swarm algorithm testing when realistic ROS/sensor behavior is unnecessary

**Gazebo + PX4 + ROS 2:**

- PX4 SITL
- ROS 2 nodes and message passing
- Realistic LiDAR, camera, optical-flow, IMU, and GPS/NavSat pipelines
- SLAM/localization integration
- Sensor failure and degradation testing
- Actual autonomy software tested against simulated sensors
- Multi-process / multi-node integration
- Final simulation before hardware testing

## Collaboration

Use a branch and pull request for changes; run the verification commands before
requesting review. Commit source, model assets, and documentation, keeping local
environments, generated simulator files, recordings, and ROS build output out of Git.
