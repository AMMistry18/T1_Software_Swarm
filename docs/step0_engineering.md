# Step 0 engineering note

The simulation uses provisional vehicle models assembled from the supplied TAR
BOM. No vehicle mass, inertia, propulsion curve, radio performance, or electrical
design has been validated by measurement. Configuration provenance is recorded
in `configs/hardware_assumptions.yaml`; runnable values live in `vehicles.yaml`.

## Physical model

| Quantity | Mapping quadrotor | Relay quadrotor |
|---|---:|---:|
| Estimated takeoff mass | 3.615 kg | 0.1352 kg |
| Hub-to-motor distance | 0.480 m | 0.075 m |
| Opposite motor span | 0.960 m | 0.150 m |
| Propeller diameter | 0.4572 m (18 in) | 0.0762 m (3 in) |
| Adjacent disk clearance | 0.222 m | 0.030 m |
| Assumed maximum force per rotor | 22 N | 0.8 N |
| Assumed thrust / weight at 9.81 m/s² | 2.48 | 2.41 |
| Nominal force per rotor at hover | 8.866 N | 0.3316 N |
| Inertia diagonal in body axes [kg m²] | (0.10819, 0.11077, 0.18908) | (0.00012878, 0.00013915, 0.00021702) |

These thrust limits are plausible simulation allowances, **not claims about the
selected motors**. Mapping battery mass is the combined two-pack mass, at 12S
and 5000 mAh in series. Relay propulsion assumes a 2S supply only as metadata;
the simulation does not calculate RPM, voltage sag, or endurance.

Mapping frame allowance is 680 g: center 280 g, arms 320 g, landing gear 80 g.
Jetson/carrier allowance is 200 g and the full harness allowance is 110 g.
Explicit GNSS antenna allowance is 15 g. Sensor plate and custom carrier add
30 g and 20 g. The Cube/carrier allowance is 73 g. Both GNSS lists contain one
chip each. PDB, BEC, and PM02 masses are retained conservatively; their overlapping
functions require an electrical design review before the BOM is finalized.

Relay printed frame allowance is 25 g, including four 3 g arms. The 8 g PCB
allowance represents the bare board; chip, regulator/passive, antenna, and
harness allowances are separately included. This distinction is provisional
and must be reconciled with measured assembled-board mass. The supplied Google
Drive CAD link could not be retrieved; no CAD dimensions or mass were extracted.

Every mass element has a box or cylinder geometry. Box inertia is
`m/12 diag(y²+z², x²+z², x²+y²)` for full dimensions. Cylinder inertia uses the
standard uniform solid-cylinder expression. Rotated local inertias and the
parallel-axis terms `m ((d·d) I - d dᵀ)` are summed about the computed COM.
All geometry is translated so the body origin is the COM. MuJoCo independently
compiles the same component geometries; tests compare its principal-axis inertia
back in body axes against this independent calculation. Small off-diagonal terms
from asymmetric sensor mounting are retained rather than dropped.

Propellers are rigid swept-disk collision shapes. They do not spin, produce
downwash, or model blade aerodynamics. Four site motors apply local +z force
and alternating reaction torque `±k f`, with k in meters. Rotor arm moments arise
from the actual site offsets. Wrench allocation uses columns
`[1, y_i, -x_i, ±k]ᵀ` measured from COM. Both controller and MuJoCo enforce
nonnegative force bounds. ESC lag is deliberately absent until measured.

## Controller and clocks

World axes are x east, y north, z up; body axes are x forward, y left, z up.
MuJoCo free-joint quaternions are wxyz, mapping body to world. States expose
world linear velocity and body angular velocity. ENU to NED for future PX4 is
`[x_n, y_n, z_n] = [y_e, x_e, -z_e]`; FLU to FRD is `[x, -y, -z]`.
Orientation must transform both frames, not just swap quaternion entries.

The translational model is `m p̈ = R e_z Σf_i - m g e_z`.
The position controller uses feed-forward acceleration plus position PD and
velocity damping. Acceleration and tilt are limited before computing desired
body orientation and collective force projected onto the current thrust axis.
Attitude feedback uses the vee map of
`(R_desᵀ R - Rᵀ R_des)/2`, body rate damping, and gyroscopic compensation
`ω × Jω`. Gains describe desired angular acceleration, scaled by vehicle inertia.
The inverse wrench matrix produces individual rotor forces, then clips to the
configured actuator limits. Clipping does not preserve all requested moments;
these events are logged. No integral term means there is no integral windup,
but a persistent external disturbance can cause steady position error.

The geometric attitude error has a 180-degree singularity; this baseline is
intended for modest tilts and yaw changes, not inverted flight or global recovery.

Physics uses RK4 at 500 Hz; controller 100 Hz; mission and log 50 Hz; viewer
25 Hz. All intervals must be integer multiples of physics dt. Tick counts
schedule tasks; timestamps come from `MjData.time`. Graphics only pace wall
time, never change the number or duration of physics steps. There is no random
noise in Step 0: the configured seed is reserved for later stochastic models,
and identical resolved configurations reproduce identical numerical trajectories
on the tested environment. Cross-platform bitwise equivalence is not promised.

Quintic segment interpolation gives zero endpoint velocity and acceleration.
The mapper takes off, hovers, traverses the waypoints, returns, and hovers.
Relays take off to static positions. Spawns begin a few centimeters above the
ground, upright and with zero velocity; the controller is active immediately.
This avoids unmodeled arming/ground effect behavior. Flight occurs only through
rotor commands and physics stepping; no mission teleports are used.

## Acceptance and measured results

Tests require settled hover altitude RMSE below 0.05 m, final hover position RMSE
below 0.05 m, whole-mission position RMSE below 0.15 m, maximum position error
below 0.30 m, and mapper waypoint visits within 0.15 m. Hover intervals are
5–7 s and 33–36 s; the final-hover metric uses the last two seconds. Initial
takeoff is included in whole-mission error. Disturbance-recovery tests add a
position offset, velocity, roll/pitch error, and a yaw command to both classes,
then require position and velocity errors below 0.03 m and 0.03 m/s after 8 s.

The initial validated 36 s outdoor run (`experiments/results/step0_outdoor`)
produced the following values directly from simulation logs:

| Vehicle | Position RMSE [m] | Maximum error [m] | Settled hover altitude RMSE [m] |
|---|---:|---:|---:|
| mapper_0 | 0.02699 | 0.05274 | 0.0001242 |
| relay_0 | 0.000922 | 0.004301 | 0.0000581 |
| relay_1 | 0.001064 | 0.004963 | 0.0000670 |
| relay_2 | 0.001206 | 0.005624 | 0.0000760 |

There were zero contacts and zero motor saturation updates. Very small settled
relay errors reflect ideal ground truth, exact model compensation, no wind, and
unlimited simulation endurance. They do not predict physical flight accuracy.
Full per-run metrics, commands at 100 Hz, states at 50 Hz, generated MJCF,
resolved YAML, and runtime versions are saved alongside plots. Final plots show
actual measurements, desired references, attitude, velocity, forces, and limit
flags. The no-thrust gravity test supplies an uncontrolled physical baseline;
the old manual-hover model is preserved and has different mass assumptions, so
no apples-to-apples controller comparison is claimed.

## Measurements required before hardware flight

Measure total assembled mass, COM, inertia, actual motor spacing, sensor mounts,
propeller clearance, thrust/torque curves at the chosen battery voltages,
ESC response and dead zones, battery sag/discharge, controller timing, IMU
noise/bias, and actuator/flight-controller frame conventions. Confirm the Jetson
model, GNSS integration and antennas, complete harness, thermal mounting,
power wiring, and regulator capacity under radio forwarding load. Measure radio
power and RF performance separately; 2.5 W and 5 g are budget allowances only.
Firmware mesh support, aerial range, and multi-hop goodput remain unknown.

Mass and thrust curves dominate vertical margins; inertia, motor lag, arm
geometry, and reaction-torque coefficient dominate attitude dynamics. These
uncertainties are the highest-priority bench measurements. Sensor and radio
interfaces are mount/mass placeholders, with no measurement generation or
traffic simulation. Collision geometry exists, but neither collision avoidance
nor route planning is implemented. The supplied mission is manually separated
from buildings and other agents; edited routes must be checked by the operator.

Step 1 will add a simulator-independent communication graph and ETX link model
consuming snapshots of vehicle states and world geometry. Routing, SLAM, MPC,
coordination, PX4, ROS 2, and Gazebo integration remain future milestones.
