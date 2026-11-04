"""Replaceable communication models for planning-time connectivity.

ASSUMPTION: The default model is a conservative Euclidean distance threshold
with a configurable connection margin. Region geometry alone does not reveal
real RF conditions. Graph connectivity does NOT guarantee throughput for
point-cloud / LiDAR map traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from tar_swarm.network.connectivity import reachable_from
from tar_swarm.network.graph import GraphSnapshot, build_graph


class PlanningLinkModel(Protocol):
    """Planning-time link availability between two ENU positions."""

    def usable(self, first: np.ndarray, second: np.ndarray) -> bool: ...


@dataclass(frozen=True)
class ConservativeDistanceLink:
    """Distance-based model with explicit margin (not measured RF)."""

    max_range_m: float
    connection_margin_m: float = 2.0
    label: str = "conservative_distance_assumption"

    @property
    def usable_range_m(self) -> float:
        return max(0.0, self.max_range_m - self.connection_margin_m)

    def usable(self, first: np.ndarray, second: np.ndarray) -> bool:
        return bool(np.linalg.norm(first - second) <= self.usable_range_m)


@dataclass(frozen=True)
class MeasuredLinkHook:
    """Placeholder interface for future measured/obstacle-aware models.

    Hooks exist so callers can inject calibrated links, obstacle attenuation,
    or throughput constraints without rewriting the planner. Default behavior
    rejects usage until a concrete model is provided.
    """

    def usable(self, first: np.ndarray, second: np.ndarray) -> bool:
        raise NotImplementedError(
            "MeasuredLinkHook requires a concrete measured/obstacle model"
        )


@dataclass
class _NetworkAdapter:
    """Adapt PlanningLinkModel to the post-merge network LinkModel API.

    network.graph.build_graph now calls
    ``usable(first_role, second_role, first, second)``. Planning keeps a
    position-only model; roles are accepted and ignored here.
    """

    inner: PlanningLinkModel

    def usable(
        self,
        first_role: str,
        second_role: str,
        first: np.ndarray,
        second: np.ndarray,
    ) -> bool:
        _ = first_role, second_role
        return self.inner.usable(first, second)


def build_connectivity(
    positions: Mapping[str, np.ndarray],
    active: Mapping[str, bool],
    time_s: float,
    link_model: PlanningLinkModel,
    ground_id: str,
    roles: Mapping[str, str] | None = None,
) -> tuple[GraphSnapshot, frozenset[str]]:
    """Build undirected availability graph and ground-reachable node set."""
    if roles is None:
        roles = {node_id: "unknown" for node_id in positions}
        roles[ground_id] = "ground_station"
    adapter = _NetworkAdapter(link_model)
    graph = build_graph(dict(positions), dict(active), time_s, adapter, dict(roles))
    reachable = reachable_from(graph, ground_id)
    return graph, reachable


def filter_forwarding_nodes(
    roles: Mapping[str, str],
    allow_mapper_forwarding: bool,
) -> dict[str, bool]:
    """Return which node IDs may forward (roles: ground always, relays default)."""
    permitted: dict[str, bool] = {}
    for node_id, role in roles.items():
        if role == "ground_station":
            permitted[node_id] = True
        elif role == "relay":
            permitted[node_id] = True
        elif role == "mapper":
            permitted[node_id] = bool(allow_mapper_forwarding)
        else:
            permitted[node_id] = False
    return permitted


def mapper_connected_via_permitted(
    positions: Mapping[str, np.ndarray],
    active: Mapping[str, bool],
    roles: Mapping[str, str],
    time_s: float,
    link_model: PlanningLinkModel,
    ground_id: str,
    mapper_ids: list[str],
    allow_mapper_forwarding: bool,
) -> dict[str, bool]:
    """Connectivity where only permitted roles act as forwarding nodes.

    Non-permitted nodes remain graph endpoints for their own reachability check
    but are deactivated as intermediate vertices by excluding them from the
    active forwarding set when testing other mappers.
    """
    permitted = filter_forwarding_nodes(roles, allow_mapper_forwarding)
    result: dict[str, bool] = {}
    for mapper_id in mapper_ids:
        if not active.get(mapper_id, False):
            result[mapper_id] = False
            continue
        check_active = {
            nid: bool(
                active.get(nid, False)
                and (permitted.get(nid, False) or nid == mapper_id)
            )
            for nid in positions
        }
        _, reachable = build_connectivity(
            positions,
            check_active,
            time_s,
            link_model,
            ground_id,
            roles=roles,
        )
        result[mapper_id] = mapper_id in reachable
    return result
