"""Export standalone MJCF examples derived from the authoritative YAML."""

from copy import deepcopy
from pathlib import Path

from tar_swarm.common.config import load_config
from tar_swarm.mujoco.model import build_scene


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs/simulation.yaml", root / "configs/worlds/empty.yaml"
    )
    for vehicle, filename in [
        ("mapping_drone", "mapper.xml"),
        ("relay_drone", "relay.xml"),
    ]:
        example = deepcopy(config)
        example["mission"]["agents"] = [
            next(a for a in config["mission"]["agents"] if a["vehicle"] == vehicle)
        ]
        path = root / "assets/mujoco" / filename
        path.write_text(build_scene(example), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
