# LiDAR localization: contributions (branch `dhruv`)

Work toward the project claim that a downward LiDAR over flat ground cannot
observe x, y and yaw (geometric degeneracy; Zhang, Kaess & Singh, 2016). This
file records what has been added for LiDAR localization and how to use it.

## Status

| Stage | Scope | Status |
|-------|-------|--------|
| A | Simulated Unitree L2 scans in MuJoCo | Done |
| B | Single-scan degeneracy analysis, test worlds, decision study | Not started |
| C | Scan-to-scan LiDAR odometry (point-to-plane ICP) | Not started |

## Stage A: LiDAR scan simulation

### What was added
- `src/tar_swarm/sensors/lidar.py`: scan pattern, noise and dropout, conversion
  to points, parameter validation. No MuJoCo imports.
- `configs/sensors.yaml`: `lidar:` section for `mapping_drone`. Placeholders
  carry `# TODO(source): ...` comments; none are measured values.
- `configs/simulation.yaml`: new `sensors_file` key.
- `src/tar_swarm/common/config.py`: loads `config["sensors"]`; validates LiDAR
  settings and that the scan interval is a multiple of the physics step.
- `src/tar_swarm/mujoco/flight.py`: `cast_rays` and `FlightSimulation.site_pose`
  / `FlightSimulation.cast_rays`. True ray hits only (distance, geom id, world
  normal); the scanning drone's own body is excluded. No noise here.
- `src/tar_swarm/mujoco/runner.py`: each mapper is scanned at `rate_hz`, using
  `np.random.default_rng(config["simulation"]["seed"])`.
- `src/tar_swarm/evaluation/logging.py`: `save_lidar`, called from `save_run`.
- `tests/test_lidar.py`: 20 known-answer tests (noise off unless stated).
- `scripts/check_lidar.py`: 3D world-frame plot of one scan plus hit statistics.

### Outputs of a run
- `lidar_scans.npz`: `time_s`, `vehicle_id`, `scan_start` (index of each scan's
  first point) and flat `points` (valid returns, in the sensor frame). These
  are the measurements.
- `lidar_truth.npz`: true distance, geom id and world normal for each point.
  Oracles for tests and checks only; estimators must not read them.

### Model and assumptions
- Mounted facing down; the sensor axis is +z of the `<agent>_lidar_down` site
  (that site's rotation is diag(1, -1, -1) relative to the body).
- Field of view is a spherical cap of half-angle `max_polar_deg` (90 = lower
  hemisphere, 96 = negative angle mode). Vendor range 0.05 to 30 m and 64,000
  points/s come from the Unitree L2 store page.
- **Assumption:** the real L2 scan pattern is unpublished. `random_uniform`
  draws fresh directions uniformly over the cap each scan.
- **Placeholders (TODO(source)):** frame rate, range noise, dropout.
- A scan is taken at one instant; motion distortion within a scan is ignored.

### Finding to remember
Rays that would land beyond the ground plane's `ground_half_size_m` return no
hit. Worlds used for degeneracy studies must be larger than the longest ray
footprint (the spec uses 100 m for `flat_open`).

### How to run
```
.\.venv\Scripts\python.exe -m pytest -q tests/test_lidar.py
.\.venv\Scripts\python.exe scripts/run_mujoco.py --headless --output experiments/results/<name>
.\.venv\Scripts\python.exe scripts/check_lidar.py experiments/results/<name> --scan 120
```

### Verification so far
- Full suite: 47 passed; `ruff check .` clean.
- Default mission run (`experiments/results/lidar_a`): scan at t = 12 s had
  5,938 valid points, ranges 2.9 to 29.9 m, most on the ground and the rest on
  buildings.

## Next
Stage B (normals, degeneracy matrix, `flat_open` / `sparse_field` worlds,
decision study) and Stage C (ICP), each reported before the next starts.
