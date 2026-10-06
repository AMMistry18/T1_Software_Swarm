import mujoco
import mujoco.viewer
import numpy as np
import time

# Roughly calculated based on drone weight * 9.81 (F) / num motors (4)
HOVER_THRUST = 6.48 

# Controls received from keyboard
target_controls = {
    "throttle": HOVER_THRUST,
    "pitch": 0.0,
    "roll": 0.0,
    "yaw": 0.0,
    "reset_requested": False
}

# Controls currently acting on drone
current_controls = {
    "throttle": HOVER_THRUST,
    "pitch": 0.0,
    "roll": 0.0,
    "yaw": 0.0
}

#register keyboard commands and update target controls
def key_callback(keycode):
    global target_controls

    try:
        char = chr(keycode).upper()
    except ValueError:
        char = None

    # Pitch & Roll
    if keycode == 265:    # UP Arrow
        target_controls["pitch"] = 0.2
    elif keycode == 264:  # DOWN Arrow
        target_controls["pitch"] = -0.2
    elif keycode == 263:  # LEFT Arrow
        target_controls["roll"] = -0.2
    elif keycode == 262:  # RIGHT Arrow
        target_controls["roll"] = 0.2

    # Yaw
    elif char == 'A':
        target_controls["yaw"] = -0.4
    elif char == 'D':
        target_controls["yaw"] = 0.4

    # Throttle
    elif keycode == 32:    # Space  -> Ascend
        target_controls["throttle"] = min(12.0, target_controls["throttle"] + 0.3)
    elif keycode == 340:   # Left Shift -> Descend
        target_controls["throttle"] = max(0.0, target_controls["throttle"] - 0.3)

    # System commands
    elif char == 'R':
        target_controls["reset_requested"] = True


def apply_motor_mixing(data):
    global current_controls, target_controls

    # Changing current control based on recieved inputs

    alpha = 0.05 #moderates change for smoothness so that sim doesn't overreact/crash

    current_controls["throttle"] += alpha * (target_controls["throttle"] - current_controls["throttle"])
    current_controls["pitch"] += alpha * (target_controls["pitch"] - current_controls["pitch"])
    current_controls["roll"] += alpha * (target_controls["roll"] - current_controls["roll"])
    current_controls["yaw"] += alpha * (target_controls["yaw"] - current_controls["yaw"])

    # Return pitch/roll to center if keys aren't being held
    target_controls["pitch"] *= 0.995
    target_controls["roll"] *= 0.995
    target_controls["yaw"] *= 0.990

    t = current_controls["throttle"]
    p = current_controls["pitch"]
    r = current_controls["roll"]
    y = current_controls["yaw"]

    #coordinate motors accordingly but stay within 0-20N range
    data.ctrl[0] = np.clip(t - p - r + y, 0, 20)
    data.ctrl[1] = np.clip(t - p + r - y, 0, 20)
    data.ctrl[2] = np.clip(t + p + r + y, 0, 20)
    data.ctrl[3] = np.clip(t + p - r - y, 0, 20)

def reset_drone(model, data):
    mujoco.mj_resetData(model, data)
    target_controls["throttle"] = HOVER_THRUST
    current_controls["throttle"] = HOVER_THRUST
    for key in ["pitch", "roll", "yaw"]:
        target_controls[key] = 0.0
        current_controls[key] = 0.0
    target_controls["reset_requested"] = False
    print("\n[RESET] Drone reset to starting position.")



def main():
    model = mujoco.MjModel.from_xml_path("drone.xml")
    data = mujoco.MjData(model)

    drone_body_id = model.body("drone").id
    last_telemetry_time = time.time()

    print("=" * 60)
    print("  Pitch Forward/Back : UP / DOWN Arrows")
    print("  Roll Left/Right    : LEFT / RIGHT Arrows")
    print("  Yaw Left/Right     : A / D")
    print("  Throttle Up        : Space")
    print("  Throttle Down      : Left Shift")
    print("  Reset Position     : R")
    print("=" * 60)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:

	#tracking cam allows us to follow the drone as it flies
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING 
        viewer.cam.trackbodyid = drone_body_id
        viewer.cam.distance = 3.5

        while viewer.is_running():
            step_start = time.time()

            if target_controls["reset_requested"]:
                reset_drone(model, data)

            apply_motor_mixing(data)
            mujoco.mj_step(model, data)

            now = time.time()

            # Output data from sensors to console every 0.5 seconds

            if now - last_telemetry_time >= 0.5:
                gps_xy = data.sensor("gps_pos").data[0:2]
                single_point_fwd = data.sensor("single_point_dist").data[0]
                lidar_z = data.sensor("lidar_dist").data[0]

                print(f"Time: {data.time:.1f}s | "
                      f"GPS (X,Y): [{gps_xy[0]:.2f}, {gps_xy[1]:.2f}] m | "
                      f"LiDAR Height (Z): {lidar_z:.2f} m | "
                      f"Single-Point Forward Distance: {single_point_fwd:.2f} m")
                last_telemetry_time = now

            viewer.sync()

            # Physics model timing
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()