"""Physical neighbor snapshots, independent of any motion backend."""

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import numpy as np

from tar_swarm.network.radio import LinkModel


@dataclass(frozen=True)
class GraphSnapshot:
    time_s: float
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]


def build_graph(
    positions: Mapping[str, np.ndarray],
    active: Mapping[str, bool],
    time_s: float,
    link_model: LinkModel,
    roles: Mapping[str, str],
) -> GraphSnapshot:
    """Only active nodes form edges; inactive IDs remain in the snapshot.

    roles maps each ID to its role, because a drone's radio range depends on it.
    """
    nodes = tuple(sorted(positions))
    edges = tuple(
        (a, b)
        for a, b in combinations(nodes, 2)
        if active[a]
        and active[b]
        and link_model.usable(roles[a], roles[b], positions[a], positions[b])
    )
    return GraphSnapshot(time_s, nodes, edges)
