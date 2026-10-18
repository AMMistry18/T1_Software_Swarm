"""Ground reachability without inserting virtual physical edges."""

from tar_swarm.network.graph import GraphSnapshot


def reachable_from(graph: GraphSnapshot, source: str) -> frozenset[str]:
    if source not in graph.nodes:
        raise ValueError(f"Unknown source: {source}")
    neighbors = {node: [] for node in graph.nodes}
    for a, b in graph.edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    visited = {source}
    pending = [source]
    while pending:
        for neighbor in neighbors[pending.pop()]:
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return frozenset(visited)
