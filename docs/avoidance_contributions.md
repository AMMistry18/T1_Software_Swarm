# Obstacle avoidance: contributions (branch `dhruv`)

Basic obstacle avoidance in MuJoCo: a simulated TF-Luna forward rangefinder on
the network (relay) drone, obstacle distances from simulated Unitree L2 scans,
and a speed-limit / stop-and-hold behavior that works with either sensor. This
file records what has been added for avoidance and how to use it.

## Status

| Stage | Scope | Status |
|-------|-------|--------|
| AV0 | Housekeeping: per-sensor RNG streams, truth-isolation test | Done |
| AV1 | TF-Luna simulation | Done |
| AV2 | L2 obstacle extraction (ground removal, sectors) | Done |
| AV3 | Avoidance behavior (speed limit, stop and hold) | Done |
| AV4 | Test worlds | Done (2 m/s matrix 10 of 10; 2 wire collisions) |
| AV5 | Metrics and study | Done (3 of 3 pass criteria; limits recorded) |

## AV0: Housekeeping

### What was added
- `src/tar_swarm/sensors/rng.py`: `sensor_rngs(seed)` gives one generator per
  sensor kind from `np.random.SeedSequence(seed).spawn(...)`. `SENSOR_STREAMS`
  is append-only; a kind's stream depends only on its index, so adding a
  sensor never changes another sensor's noise.
- `src/tar_swarm/mujoco/runner.py` (existing code, one line): the L2 now draws
  from `sensor_rngs(seed)["lidar"]` instead of `np.random.default_rng(seed)`.
- `tests/test_sensor_rng.py`: determinism, stream stability when kinds are
  added, distinct streams.
- `tests/test_truth_isolation.py`: fails if any module under `sensors/`,
  `perception/` or `control/` contains a `*_truth` / `truth.npz` string,
  imports `tar_swarm.evaluation` (where truth files are written), or imports a
  name containing `truth`. Includes a self-check of the detector.

### Outputs of a run
Unchanged file set. L2 scan and noise values differ from runs made before AV0,
because the L2 moved to a spawned stream (accepted change; distribution
unchanged).

### Model and assumptions
- One stream per sensor kind, shared by all agents of that kind. Adding an
  agent that carries the same sensor kind still shifts that kind's draws.

### Findings
Codebase inspection before starting the track, where it differs from the brief:
1. The network drone exists as vehicle type `relay_drone` (agents
   `relay_0..2` in `configs/missions/step0.yaml`, static hover). Both vehicle
   types already list a `tf_luna` component, so `<agent>_luna_fwd` can be
   placed from it, as `<agent>_lidar_down` is placed from `unitree_l2`.
2. The controller tracks a position `Reference` (position, velocity,
   acceleration feed-forward, yaw), not a velocity command. Clipping only the
   reference velocity would not stop the drone; the position error would pull
   it on. AV3 needs a reference shaper between `WaypointMission.sample()` and
   `CascadedController.update()` that advances a held reference position at
   the limited velocity, and passes the mission reference through unchanged
   when avoidance is off.
3. There is no truth loader; truth files are read with `np.load` (only by
   `scripts/check_lidar.py`), so the isolation test checks source text.
4. `mujoco/renderer.py:draw_overlays` already draws targets and trails into
   `user_scn`; the new sensor overlays must append to it, not reset it.

### How to run
```
.\.venv\Scripts\python.exe -m pytest -q tests/test_sensor_rng.py tests/test_truth_isolation.py
```

### Verification
- Full suite: 52 passed; `ruff check .` clean.
- `scripts/run_mujoco.py --headless --duration 2 --no-plots` wrote the same
  file set as before, including `lidar_scans.npz` and `lidar_truth.npz`.

## AV1: TF-Luna simulation

### What was added
- `src/tar_swarm/sensors/rangefinder.py` (no MuJoCo imports): parameters and
  validation, `beam_directions` (one axis ray, or a sunflower bundle over the
  beam cone), `nearest_hit` (bundle returns its nearest hit), `apply_noise`
  (range-dependent noise, dropout, min / effective-max limits).
- `configs/sensors.yaml`: `rangefinder:` section. `relay_drone` enabled;
  `mapping_drone` present with `enabled: false`.
- `src/tar_swarm/common/config.py`: validates rangefinder settings, the reading
  interval against the physics step, and that an enabled vehicle has a
  `tf_luna` component.
- `src/tar_swarm/mujoco/model.py` (existing code, additive): adds site
  `<agent>_luna_fwd` at the `tf_luna` component, site +z = body +x, only for
  vehicles with an enabled rangefinder. Sites carry no mass or collision.
- `src/tar_swarm/mujoco/runner.py` (existing code, additive): reads each
  rangefinder at `rate_hz` with its own RNG stream; logs readings and oracles;
  new `overlay` argument and `--overlay` flag (default off).
- `src/tar_swarm/evaluation/logging.py`: `save_rangefinder`, called from
  `save_run` through a new optional argument.
- `src/tar_swarm/mujoco/overlays.py` and `configs/overlays.yaml`: TF-Luna beam
  overlay, loaded only with `--overlay`. Line from sensor to hit point; green
  when clear, red inside `d_stop_m`, faint grey to the effective max range on
  no return.
- `tests/test_rangefinder.py`: 24 tests.

### Outputs of a run
- `rangefinder.npz`: `time_s`, `vehicle_id`, `distance_m` (NaN = no return).
  The measurements.
- `rangefinder_truth.npz`: true nearest distance (-1 = no hit), geom id, and
  sensor site position and rotation in the world, row-aligned with
  `rangefinder.npz`. Oracles; avoidance must not read them.

### Model and assumptions
- Source for datasheet values: Benewake TF-Luna product manual (ranging
  table). TODO(source): pin the manual revision.
- Datasheet: 100 Hz default rate, 0.2 to 8 m range (90% reflectivity, 0 klux),
  accuracy ±6 cm for 0.2 to 3 m and ±2% for 3 to 8 m, 2° field of view.
- **Assumption:** the datasheet accuracy bound is used as a Gaussian 1-sigma
  (conservative; the datasheet gives no distribution).
- **Placeholders (TODO(source)):** outdoor `effective_max_range_m` 4.0,
  dropout, and the overlay's `d_stop_m` (to be taken over by the AV3 avoidance
  config).
- `num_rays: 1` by default, so the beam is an ideal line; objects off the axis
  are missed. A bundle (`num_rays > 1`) approximates the 2° cone and returns
  its nearest hit.
- Reflectivity, sunlight and multi-target behavior inside the beam are not
  modeled.

### Findings
1. Readings below `min_range_m` (0.2 m) are reported as no return, the same as
   open space. An obstacle that gets that close is indistinguishable from a
   clear path; AV3 must not treat no return as clear without other evidence.
2. Default run, first 2 s: `relay_0` reads `relay_1` 2.95 m ahead while both
   sit on the ground (until they climb apart at about t = 0.95 s);
   `relay_2`'s beam hits `building_c` at 6.8 m, beyond the 4 m outdoor cap, so
   it reports no return. Both are as expected.
3. With `--overlay` off, the default run is unchanged: `states.csv`,
   `controls.csv`, `metrics.json` and `lidar_scans.npz` are byte-identical to
   the AV0 run. New files: `rangefinder.npz`, `rangefinder_truth.npz`; the
   `scene.xml` and `resolved_config.yaml` contents gain the new site and
   config section.

### How to run
```
.\.venv\Scripts\python.exe -m pytest -q tests/test_rangefinder.py
.\.venv\Scripts\python.exe scripts/run_mujoco.py --overlay --output experiments/results/<name>
```

### Verification
- Full suite: 76 passed; `ruff check .` clean.
- Regression vs AV0: identical outputs as listed in finding 3.
- Offscreen render with the overlay: beams follow each relay's yaw; green
  beam relay_0 to relay_1, grey no-return beams on the others.

## AV2: L2 obstacle extraction

### What was added
- `src/tar_swarm/perception/obstacles.py` (no MuJoCo imports): level-frame
  transform (roll/pitch removed, heading kept), RANSAC ground plane on the
  measured points (near-vertical normals only, SVD refit), ground removal,
  height band, 72 sectors with nearest horizontal distance (NaN = no return)
  and point counts.
- `configs/perception.yaml` (`l2_obstacles: mapping_drone:`, `enabled: false`)
  and `perception_file` in `configs/simulation.yaml`; validated in
  `common/config.py` (needs a LiDAR on the same vehicle).
- `sensors/rng.py`: new appended stream `l2_obstacles` for RANSAC.
- `mujoco/overlays.py`: `lidar_points` (obstacle vs removed-ground points) and
  `sector_ring` overlays, each toggled in `configs/overlays.yaml`. The runner
  computes obstacles for display when either is on with `--overlay`; outputs
  are unchanged.
- `scripts/check_l2_visibility.py`: the forward-wall check below.
- `tests/test_obstacles.py`: 19 tests (synthetic point sets, overlays, runner).

- `perception/obstacles.py`: `obstacles_enabled`. The runner computes
  obstacles when `enabled`, when an L2 overlay is on, or when avoidance uses
  the `l2` source; it logs them only when `enabled`.
- `evaluation/logging.py`: `save_l2_obstacles`, called from `save_run`.

### Outputs of a run (when `l2_obstacles.enabled: true`)
- `l2_obstacles.npz`: per processed scan `time_s`, `vehicle_id`, `scan_index`
  (into `lidar_scans.npz`), `sector_distance_m` (S x 72, NaN = no return),
  `sector_count`, `ground_found`, `ground_normal`, `ground_offset`, and a flat
  `ground_mask` aligned with that scan's points (`mask_start`). All
  measurement-side; there is no truth file for this stage.

### Model and assumptions
- **Assumption:** roll and pitch for the level frame come from the simulated
  attitude (the IMU on hardware). Heading is kept, so sector 0 is the heading.
- Ground = the near-level plane (normal within `ground_max_tilt_deg` of
  vertical) with the most RANSAC inliers, refit by SVD. The true ground height
  is never used. Points on or below the plane within `ransac_threshold_m` are
  removed. Risk: at low altitude a large flat roof below the drone could win.
- Sector distance is the horizontal distance from the body origin to the
  nearest kept point, within the height band ±0.5 m about the body origin.
- **Placeholders (TODO(source)):** band, RANSAC settings, minimum range.

### Forward-wall visibility check
Mapper held at 3 m, wall face straight ahead, 20 scans per case, L2 noise on.
Height band ±0.5 m about the body origin; L2 origin is 0.111 m below it.
Pitch +10° = nose down (as when accelerating forward), -10° = nose up (as
when braking). "Band seen" = height range of wall points seen in the forward
sectors (±12.5°).

| Wall | max_polar | Pitch | Forward return rate | Points / scan | Band seen (m) |
|------|-----------|-------|---------------------|---------------|---------------|
| 5 m | 90 | 0 | 0.99 | 31 | -0.50 .. -0.11 |
| 5 m | 90 | +10 | 0.00 | 0 | none |
| 5 m | 90 | -10 | 1.00 | 88 | -0.50 .. +0.50 |
| 5 m | 96 | 0 | 1.00 | 74 | -0.50 .. +0.42 |
| 5 m | 96 | +10 | 0.41 | 3 | -0.50 .. -0.45 |
| 5 m | 96 | -10 | 1.00 | 78 | -0.50 .. +0.50 |
| 2 m | 90 | 0 | 1.00 | 83 | -0.50 .. -0.11 |
| 2 m | 90 | +10 | 0.84 | 8 | -0.50 .. -0.46 |
| 2 m | 96 | 0 | 1.00 | 117 | -0.50 .. +0.10 |
| 2 m | 96 | +10 | 1.00 | 49 | -0.50 .. -0.25 |

Measured sector-0 distance: 4.97 m (5 m wall) and 1.96 m (2 m wall). The
nearest of the noisy points sits slightly short of the true distance.

Blind region (geometry, matches the table): the L2 sees only directions below
`max_polar_deg - 90` above its own horizontal. A level drone at horizontal
distance d sees up to `-0.111 + d * tan(e)` m relative to the body, with
e = 0° (90 mode) or 6° (96 mode); pitching nose-down by θ lowers the forward
limit by θ, and nose-up raises it by θ (the rear limit moves the other way).
- 90 mode: everything above the sensor is blind in every direction, so the
  upper half of the band is never seen.
- 96 mode: covers the full band ahead only beyond about 6 m; at 2 m it sees
  up to +0.10 m.
- Nose-down flight at 10°: forward is essentially blind at 5 m in both modes.

Self-occlusion (not modeled in the main simulation, which excludes the
drone's own body from ray casts): with the current mapper model, the
`landing_gear` box (0.26 × 0.24 m plate at z = -0.19 m, below the L2) blocks
every ray with polar angle up to ~66°, 95% at 66–72° and 20% at 72–78°.
Only 78–96° is clear. The plate is a mass allowance, not real gear geometry.

Outputs: `experiments/results/av2_visibility_5m/` (`visibility.json`,
`visibility.png`, `wall_polar90.png`, `wall_polar96.png`) and
`experiments/results/av2_visibility_2m/visibility.json`.

```
.\.venv\Scripts\python.exe scripts/check_l2_visibility.py --output experiments/results/<name> [--distance 2]
```

## AV3: Avoidance behavior

### What was added
- `src/tar_swarm/control/avoidance.py` (no MuJoCo imports):
  - `speed_limit`: for sectors within the cone around the horizontal command
    direction, `v = -a t + sqrt((a t)^2 + 2 a max(d - d_stop, 0))` with
    `t = sensor period + control delay + data age`. With `t = 0` this is
    `sqrt(2 a (d - d_stop))`. Below `d_stop` the limit is 0 (stop and hold).
    Data older than `stale_timeout_s` limits speed to `stale_speed_m_s`, and
    after a further `stale_hold_after_s` to 0.
  - `rangefinder_sector`: the TF-Luna as one forward sector, with a latch. A
    no-return right after a reading below `rangefinder_latch_m` counts as 0 m
    (AV1 finding 1), not as clear.
  - `ReferenceShaper`: holds its own reference position and advances it at
    the limited horizontal velocity, with the braking deceleration fed forward
    (bounded by `a_brake`). Altitude follows the mission. When the path is
    clear and the reference is within 0.1 m of the mission, it returns the
    mission reference unchanged.
- `configs/avoidance.yaml` and `avoidance_file` in `simulation.yaml`. Both
  vehicle types are `enabled: false`; relay uses `[rangefinder]`, mapper uses
  `[l2]`. `common/config.py` checks that each enabled source exists.
- `mujoco/runner.py` (existing code): when avoidance is enabled for an
  agent, the shaper sits between the mission sample and
  `CascadedController.update`. `states.csv` `desired_*` then logs the shaped
  reference (what the controller tracked). Disabled: unchanged.
- `evaluation/logging.py`: `save_avoidance`.
- `mujoco/overlays.py`: `velocity_arrows` (commanded blue; limited green when
  equal, orange when limited, red at stop).
- `src/tar_swarm/mujoco/replay.py` and `scripts/replay_run.py`: play a saved
  run in the viewer from `states.csv` and the measurement files, with all
  overlays. Never reads truth files.
- `tests/test_avoidance.py`: 20 tests.
- `--record` is deferred to AV5 (needs a video dependency).

### Outputs of a run (when avoidance is enabled)
- `avoidance.npz`: per controller tick `time_s`, `vehicle_id`,
  `commanded_velocity`, `limited_velocity` (world), `speed_limit_m_s` (inf =
  none), `status` (clear, limited, stop, stale, stale_hold),
  `reference_position`.

### Model and assumptions
- Only horizontal motion is limited; sectors are horizontal.
- **Assumption:** heading for the level frame comes from the simulated
  attitude (estimator on hardware).
- No path planning: after a stop the vehicle holds even if the mission goes
  on past the obstacle.
- **Placeholders (TODO(source)):** all of `avoidance.yaml`.

### Findings
1. The controller tracks a position reference. With the braking deceleration
   not fed forward, the drone overshot the stopped reference by about
   a/kp = 0.33 m (sensor 0.64 m from the wall with `d_stop` 1.0 m). Feeding it
   forward reduced this.
2. Relay into a wall at peak 2.6 m/s (`d_stop` 1.0, `a_brake` 2.0, TF-Luna
   noise on): no contact. Closest approach was 0.80 m sensor to wall, and the
   hold settled at about 0.88 m. The remaining 0.2 m comes from tracking lag
   during braking (pitch reached 21° nose up).
3. With noisy readings near `d_stop`, the limit flickers between 0 and a few
   cm/s, so the held reference creeps about 9 cm. Hysteresis on `d_stop` would
   remove this; not added yet.
4. The TF-Luna only sees the wall from 4 m (`effective_max_range_m`). At
   2.6 m/s that leaves about 1.4 m of margin above the stopping distance plus
   `d_stop`. Faster than about 3 m/s it cannot stop in time (AV5 will
   measure).
5. The first controller tick has no sensor data yet (status `stale_hold`);
   harmless because the vehicle is still on the ground.

### How to run
```
.\.venv\Scripts\python.exe -m pytest -q tests/test_avoidance.py
.\.venv\Scripts\python.exe scripts/replay_run.py experiments/results/<run_dir>
```
Enable avoidance by setting `enabled: true` in `configs/avoidance.yaml` (and
`l2_obstacles.enabled` in `configs/perception.yaml` for the mapper).

### Verification
- Full suite: 115 passed; `ruff check .` clean.
- Default 2 s run with everything disabled: `states.csv`, `controls.csv`,
  `metrics.json` and all npz files byte-identical to the AV2 run.
- Relay-wall run (`experiments/results/av3_relay_wall`) replayed offscreen at
  4.6, 5.1 and 7.0 s: beam and arrows show clear, limited and stop as logged.

## AV4: Test worlds

### What was added
- `configs/worlds/avoid_wall.yaml`, `avoid_pole.yaml`, `avoid_thin.yaml`,
  `avoid_offaxis.yaml`, `clear_path.yaml`. Lane y = 0, flight along +x at
  2 m; obstacle faces at x = 12; ground half-size 65 m.
  - wall: 0.2 x 6 x 6 m box.
  - pole: upright cylinder, 0.1 m diameter, 6 m tall.
  - thin: 2 cm square wire across the lane at 2 m height, free-floating
    (support posts near the lane would sit inside the avoidance cone and be
    detected instead of the wire).
  - offaxis: the same pole centred 0.3 m left of the lane.
  - clear_path: no obstacles.
- `common/config.py`, `mujoco/model.py` (existing code, additive): optional
  obstacle `shape: cylinder` (upright, diameter = `size_m[0]`). Obstacles
  without `shape` are boxes, as before.
- `src/tar_swarm/evaluation/scenarios.py`: `avoidance_config(world, sensor,
  peak_speed, seed, noise)` builds a single-drone scenario from the base
  config: `l2` = `mapper_0` with L2 obstacles and avoidance on;
  `rangefinder` = `relay_0` with TF-Luna avoidance on. Mission: take off to
  2 m, hover, one quintic segment to x = 30 (requested speed is the peak
  speed), settle. Also the ground-edge checks.
- `scripts/run_avoidance.py`: runs one scenario, optionally in the viewer
  with `--overlay`; writes `scenario.json` and fails (exit 1) on the edge
  check.
- `tests/test_scenarios.py`: 18 tests, including a relay run into
  `avoid_wall`.

### Ground-edge hard check
Ground half-size must be at least farthest horizontal position + longest
sensor range (L2, 30 m) + 1 m. Checked on the planned waypoints before a run
(raises) and on the flown trajectory after it (`scenario.json`
`ground_edge_margin_m`, exit 1 if negative). A ray cannot reach farther than
its range, so a positive margin guarantees no ray was lost off the edge.

### Model and assumptions
- Quintic lane: the obstacle (x = 12 of 30) is met near peak speed. Peak
  speed is rounded down slightly so durations are whole physics steps.
- `d_stop` is measured from the sensor (TF-Luna, at the nose) or from the
  body origin (L2 sectors). It is not measured from the airframe outline.

### Results (peak 2 m/s, noise on, seed 42)
Each run is done separately in the foreground; L2 runs take about 5.5 min each.
Clearance is the minimum airframe-to-obstacle distance (`mj_geomDistance` on
replayed logged poses; evaluation only). "First reaction" is the drone x at the
first `limited`/`stop` tick.

| World | Sensor | Stopped at x | First reaction x | Min clearance | Contacts |
|-------|--------|--------------|------------------|---------------|----------|
| avoid_wall | rangefinder | 11.20 | 10.00 | 0.71 m | 0 |
| avoid_wall | l2 | 11.18 | 9.85 | 0.25 m | 0 |
| avoid_pole | rangefinder | 11.20 | 10.00 | 0.71 m | 0 |
| avoid_pole | l2 | 11.20 | 9.92 | 0.34 m | 0 |
| avoid_thin | rangefinder | 12.42 (hit) | 11.02 | 0.00 m | 16 |
| avoid_thin | l2 | 11.94 (hit) | never | 0.00 m | 4 |
| avoid_offaxis | rangefinder | 30.00 (passed) | never | 0.16 m | 0 |
| avoid_offaxis | l2 | 11.20 | 9.89 | 0.23 m | 0 |
| clear_path | rangefinder | 30.00 | never | n/a | 0 |
| clear_path | l2 | 30.00 | never | n/a | 0 |

Approach speed was about 1.87 m/s in every run. Each run also has one
`stale_hold` tick at the start, before the first reading arrives (AV3 finding).

### Findings
1. Wall and poles: both sensors stop before the obstacle at about 1.9 m/s.
2. Mapper clearance is only 0.23 to 0.34 m. The L2 measures `d_stop` from the body
   centre, but the mapper's propeller tips reach about 0.57 m forward, and
   tracking lag adds about 0.2 m. For the mapper, `d_stop` must be set
   relative to the airframe outline (or the propeller radius added).
3. **Wire, L2: never detected; collision.** Not one of 2.2 M L2 points hit the
   wire. The L2 is mounted 0.11 m below the body origin and, with
   `max_polar_deg: 90`, sees only at or below its own height (AV2 finding).
   The wire is at flight height, so it is outside the field of view. The 96°
   mode or a nose-up attitude would cover it (AV2), but a 2 cm target is also
   sparse in a 6400-point scan.
4. **Wire, TF-Luna: detected too late; collision.** The model is a single ray
   (`num_rays: 1`). That ray first hit the 2 cm wire at 0.96 m, already inside
   `d_stop`, so the drone could not stop. The real 2° beam is 17 cm wide at
   5 m. `num_rays > 1` (beam bundle, already supported) would model that, but
   whether a 2 cm wire returns enough signal is unknown
   (`TODO(source)`).
5. **Off-axis pole, TF-Luna: never seen; near miss.** A single forward beam
   cannot see a pole 0.3 m to the side. The relay passed with 0.16 m
   clearance. The L2 saw it and stopped the mapper.
6. Clear path: no false stops with either sensor.

### How to run
```
.\.venv\Scripts\python.exe scripts/run_avoidance.py --world avoid_wall --sensor rangefinder --viewer --overlay
.\.venv\Scripts\python.exe scripts/run_avoidance.py --world avoid_pole --sensor l2 --speed 3 --output experiments/results/<name>
.\.venv\Scripts\python.exe scripts/replay_run.py experiments/results/<name>
```

### Verification
- Full suite: 133 passed; `ruff check .` clean.
- Matrix outputs in `experiments/results/av4_matrix/` (not committed).
- Default `outdoor` run unaffected (obstacle geoms stay boxes; tested).

## AV5: Metrics and study

### What was added
- `src/tar_swarm/evaluation/avoidance_metrics.py`: `run_metrics(run_dir)`. Metrics:
  - **Collision:** the airframe touches an obstacle (clearance ≤ 0) or MuJoCo
    reports contacts after takeoff.
  - **Minimum clearance:** airframe to obstacle, from `mj_geomDistance` on the
    logged poses replayed against `scene.xml`. Evaluation only.
  - **First reaction:** the first `limited`/`stop` tick, with its x and speed.
  - **Stopping distance:** farthest x minus the reaction x.
  - **Time to stop:** reaction until horizontal speed < 0.1 m/s.
  - **Intervention ticks:** non-clear ticks after takeoff starts, so the start-up
    tick before the first reading doesn't count.
  - **False stop:** in a world with no obstacles, any intervention or not reaching
    the lane end.
- `src/tar_swarm/evaluation/avoidance_study.py`: the sweep cases, `run_case`,
  `criteria`, `summarize`.
- `scripts/sweep_avoidance.py`:
  - Sweeps worlds × sensors × speeds (1, 2, 3, 5 m/s) × noise on/off × seeds
    (42, 43, 44), 240 runs by default.
  - Writes one directory per run, plus `sweep.csv` and `summary.json`, and exits
    1 on a failed criterion.
  - `--jobs N` runs cases in parallel. `--record-failures` saves `clip.gif` for
    every collision or false stop, rendered from the saved run (no re-simulation).
- `src/tar_swarm/evaluation/avoidance_plots.py` and `scripts/check_avoidance.py`:
  one figure per run with these panels:
  - Top and side trajectory views, with obstacles, first reaction and farthest point.
  - L2 sector distances over time, as a heatmap with the heading cone marked.
  - Forward distance against `d_stop`.
  - Commanded vs limited speed, with the speed limit and avoidance status bands.

  The figure reads measurement files and the config only.
- `src/tar_swarm/mujoco/recording.py`: `GifRecorder`, using MuJoCo's offscreen
  renderer and Pillow, plus `record_replay`. New `--record <clip.gif>` flags,
  default off:
  - `scripts/run_mujoco.py` (live, headless OK; add `--overlay` for sensor overlays).
  - `scripts/run_avoidance.py` (implies overlays).
  - `scripts/replay_run.py`, with `--start` and `--end`.

  It writes GIF, not mp4: the user chose no new dependency. The camera follows
  the first agent at 10 fps and 480×360. Pillow merges identical consecutive
  frames but keeps their total duration.
- Fixes found while building AV5:
  - `fit_ground_plane` used a full SVD, which builds an N×N matrix (about 300 MB
    and 0.6 s per L2 scan). This made L2 runs 5.5 min and probably caused the
    AV4 out-of-memory stop. The thin SVD gives byte-identical output, and a run
    now takes about 16 s.
  - `load_run` re-read (decompressed) `.npz` arrays inside its per-scan loop.
    Each array is now read once: an L2 clip took 91 s before and 12 s after.
- Overlay: removed ground points are hidden by default
  (`lidar_points.show_ground_points: false` in `configs/overlays.yaml`). Shown,
  they flickered over the whole floor and were mistaken for detections.
- `tests/test_avoidance_study.py` (10 tests):
  - Known-answer metric helpers and criteria scope.
  - The GIF recorder.
  - `--record` leaves the run unchanged.
  - A wall run end to end: metrics, check plot and replay clip.
  - No false stop on a clear path.
  - The plot and recording modules never read truth.

### Pass criteria (from the brief)
| Criterion | Runs | Result |
|-----------|------|--------|
| Zero collisions on `avoid_wall` and `avoid_pole` at ≤ 2 m/s, both sensors | 48 | PASS |
| Zero false stops on `clear_path` (all speeds, both sensors) | 48 | PASS |
| Every run completed, ground-edge margin ≥ 0 | 240 | PASS |

### Results (sweep of 240 runs)
Each cell gives collisions out of 6 runs (3 seeds × noise on/off) and, when
there were none, the minimum clearance. A requested peak of 5 m/s is met at
about 4.4 m/s where the obstacle is.

| World | Sensor | 1 m/s | 2 m/s | 3 m/s | 5 m/s |
|-------|--------|-------|-------|-------|-------|
| avoid_wall | rangefinder | 0/6, 0.73 m | 0/6, 0.71 m | 0/6, 0.71 m | **6/6** |
| avoid_wall | l2 | 0/6, 0.31 m | 0/6, 0.25 m | 0/6, 0.35 m | **6/6** |
| avoid_pole | rangefinder | 0/6, 0.73 m | 0/6, 0.71 m | 0/6, 0.71 m | **6/6** |
| avoid_pole | l2 | 0/6, 0.40 m | 0/6, 0.34 m | 0/6, 0.14 m | **6/6** |
| avoid_thin (limit) | rangefinder | 0/6, 0.05 m | **6/6** | **6/6** | **6/6** |
| avoid_thin (limit) | l2 | **6/6** | **6/6** | **6/6** | **6/6** |
| avoid_offaxis (limit) | rangefinder | 0/6, 0.16 m (never seen) | same | same | same |
| avoid_offaxis (limit) | l2 | 0/6, 0.27 m | 0/6, 0.23 m | 0/6, 0.32 m | **6/6** |
| clear_path | both | 0 false stops | 0 | 0 | 0 |

Noise on vs off made no difference to any collision count. It moves clearance by
up to about 0.3 m (for example L2 pole at 3 m/s: 0.14 m with noise, 0.45 m without).
With the TF-Luna on the wall, mean stopping distance is 0.46, 1.17 and 2.09 m at
1, 2 and 3 m/s, and time to stop is 0.60, 0.88 and 1.30 s.

### Findings
1. **Both sensors are safe to about 3 m/s and fail at 5 m/s, for different reasons.**
   With `a_brake` 2 m/s² and `d_stop` 1 m, stopping from 4.4 m/s needs about
   5.8 m of warning.
   - **TF-Luna:** the 4 m effective range (`TODO(source)`) gives a first return at
     4.0 m. The highest speed that can stop within that is about
     sqrt(2·2·(4 − 1)) ≈ 3.5 m/s, less latency.
   - **L2:** it is blinded by its own pitch. With the 90° polar mode it has no rays
     above its horizontal plane, so pitching nose-down by p while accelerating
     only sees a wall within `band_below_m` / tan p of the sensor. At 5 m/s
     (p ≈ 7.8°) that is 3.6 m: the wall seen from 12 m in the hover disappears
     at t = 6.6 s and returns only at 3.9 m. At 3 m/s (p ≈ 2.8°) it returns at
     7.6 m, which is enough. The 96° mode or a larger `band_below_m` would
     extend this; the latter risks taking the ground for an obstacle.
2. **The wire is a limit for both sensors** (AV4 findings 3–4 hold at every speed).
   - **L2:** never sees it.
   - **TF-Luna:** its single ray first hits the wire about 1 m away. That only
     stops the drone at 1 m/s, with 0.05 m to spare.
3. **TF-Luna never sees a pole 0.3 m off its axis.** It passes at 0.16 m at every
   speed, which is pure geometry, not avoidance. The L2 stops for it up to 3 m/s.
4. **The mapper stops too close:** L2 clearance is 0.14–0.45 m against the
   TF-Luna's 0.71–0.86 m (AV4 finding 2). The mapper's `d_stop` is measured from
   the body centre, but its propeller tips stick out about 0.57 m in front.
5. **No false stops:** 48 clear-path runs had zero intervention ticks, with
   noise and dropout on or off.
6. **Creep after stop (AV3 finding) is visible in the check plot.** TF-Luna noise
   around `d_stop` re-opens the limit for short moments, roughly 1 s after the
   stop.

### How to run
```
.\.venv\Scripts\python.exe scripts/sweep_avoidance.py --jobs 4 --record-failures --output experiments/results/av5_sweep
.\.venv\Scripts\python.exe scripts/sweep_avoidance.py --worlds avoid_wall --sensors l2 --speeds 3 5 --seeds 42 --noise on
.\.venv\Scripts\python.exe scripts/check_avoidance.py experiments/results/av5_sweep/avoid_wall_l2_v5_clean_s42
.\.venv\Scripts\python.exe scripts/replay_run.py experiments/results/av5_sweep/avoid_wall_l2_v5_clean_s42
.\.venv\Scripts\python.exe scripts/run_avoidance.py --world avoid_pole --sensor rangefinder --record experiments/results/pole.gif
```
Failure clips: `experiments/results/av5_sweep/<run>/clip.gif` (72 clips, one per
collision; they start at lane start, t = 6 s).

### Verification
- Full suite: 144 passed; `ruff check .` clean.
- 2 s default run (`run_mujoco.py --headless --duration 2 --no-plots`):
  `states.csv`, `controls.csv`, `metrics.json`, `scene.xml`, the config, and
  all lidar and rangefinder npz files are byte-identical to `av3_smoke`.
- L2 obstacle output with the thin SVD is byte-identical to the AV4 run.
- Sweep: 240 runs in about 12 min with 4 processes. That is 5.1 GB on disk,
  mostly L2 scans and truth; it is not committed.

## Next
These decisions are open:
- Measure the mapper's `d_stop` from the airframe edge rather than the body centre.
- L2 forward coverage while pitched: 96° polar mode, or a larger `band_below_m`.
- The TF-Luna beam model (`num_rays`), and whether thin wires are in scope.
- A maximum speed per sensor: about 3 m/s with the current settings.
- Hysteresis to stop the creep after a stop.
