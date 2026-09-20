"""Run a tiny free-fall simulation without creating a GUI or renderer."""

import mujoco

_MODEL_XML = """
<mujoco model="smoke">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def run_smoke() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Construct a free body and advance it by ten physics steps."""
    model = mujoco.MjModel.from_xml_string(_MODEL_XML)
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
    return model, data


def main() -> None:
    """Print the installed version and confirm that headless stepping succeeds."""
    print(f"MuJoCo version: {mujoco.__version__}")
    _, data = run_smoke()
    print(f"Headless smoke test passed: 10 steps, simulation time {data.time:.3f} s")


if __name__ == "__main__":
    main()
