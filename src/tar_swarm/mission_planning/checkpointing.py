"""Checkpoint metadata: capacities, versions, compatibility checks."""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
from typing import Any

from tar_swarm.mission_planning.schemas import CapacityLimits, MissionRequest
from tar_swarm.mission_planning.versions import PREPROCESS_VERSION, SCHEMA_VERSION


def dependency_versions() -> dict[str, str]:
    out = {}
    for name in (
        "numpy",
        "gymnasium",
        "stable-baselines3",
        "sb3-contrib",
        "shapely",
        "pydantic",
    ):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "missing"
    return out


def build_checkpoint_meta(
    request: MissionRequest,
    *,
    train_config: dict[str, Any],
    best_metric: float | None = None,
) -> dict[str, Any]:
    caps = request.planning.capacities
    return {
        "schema_version": SCHEMA_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
        "capacities": caps.model_dump(),
        "norm_scale_m": train_config.get("norm_scale_m", 200.0),
        "decision_dt_s": request.planning.decision_dt_s,
        "sim_dt_s": request.planning.sim_dt_s,
        "reward_weights": train_config.get("reward_weights", {}),
        "dependency_versions": dependency_versions(),
        "train_config": train_config,
        "best_metric": best_metric,
    }


def save_meta(path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def load_meta(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_compatible(meta: dict[str, Any], request: MissionRequest) -> None:
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint schema_version {meta.get('schema_version')} incompatible "
            f"with runtime {SCHEMA_VERSION}"
        )
    if meta.get("preprocess_version") != PREPROCESS_VERSION:
        raise ValueError(
            f"Checkpoint preprocess_version {meta.get('preprocess_version')} "
            f"incompatible with runtime {PREPROCESS_VERSION}"
        )
    caps = CapacityLimits.model_validate(meta["capacities"])
    req_caps = request.planning.capacities
    for field in (
        "max_mappers",
        "max_relays",
        "max_candidates",
        "max_segments_per_mapper",
    ):
        if getattr(req_caps, field) > getattr(caps, field):
            raise ValueError(
                f"Scenario {field}={getattr(req_caps, field)} exceeds checkpoint "
                f"capacity {getattr(caps, field)}"
            )
    n_m = len(request.mappers)
    n_r = len(request.relays)
    if n_m > caps.max_mappers or n_r > caps.max_relays:
        raise ValueError(
            f"Active inventory mappers={n_m}, relays={n_r} exceeds checkpoint "
            f"capacities max_mappers={caps.max_mappers}, max_relays={caps.max_relays}"
        )
