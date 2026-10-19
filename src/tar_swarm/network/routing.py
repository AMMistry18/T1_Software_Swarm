"""Routing: which neighbor each drone hands a packet to, so it reaches its target.

On the real drones, batman-adv works this out from its "I'm here" messages. This
is a small stand-in that gives the same answers from the simulator's snapshot of
working links (a GraphSnapshot from graph.py).

The method is a breadth-first search, like ripples spreading out from a stone:
- the destination (say, the base) is ring 0
- drones linked to it are ring 1
- drones linked to a ring-1 drone, and not already in a ring, are ring 2, and so on
A drone's next hop is the neighbor it was reached from, one ring closer. Following
next hops from any drone walks a fewest-hops path to the destination.

Why fewest hops: links are simply on or off for now, so every working link is
equally good, and batman picks the fewest hops too. Ties are broken by the order
the IDs sort in, so results repeat exactly. Once links have quality scores, the
same idea grows into "fewest expected tries" (ETX with Dijkstra, in the team plan).
"""

from collections import deque


def next_hops(graph, destination):
    """Each drone's next hop toward `destination`, as a dict: ID -> neighbor ID.

    graph: a GraphSnapshot (its nodes, and its edges = working links).
    Drones with no path to the destination are left out, and so is the
    destination itself.
    """
    if destination not in graph.nodes:
        raise ValueError(f"Unknown destination: {destination}")
    neighbors = {node: [] for node in graph.nodes}
    for a, b in graph.edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    hops = {}
    reached = {destination}
    # A deque is a waiting line: append adds to the back, popleft takes from the
    # front. Taking the oldest first is what makes the rings come out in order.
    waiting = deque([destination])
    while waiting:
        node = waiting.popleft()
        for neighbor in sorted(neighbors[node]):  # sorted: ties always break alike
            if neighbor not in reached:
                reached.add(neighbor)
                hops[neighbor] = node  # it was reached from `node`: its next hop
                waiting.append(neighbor)
    return hops


def route(graph, source, destination):
    """A fewest-hops path from source to destination, as a list of IDs, or None
    if there is no path. Example: ["mapper_1", "relay_2", "relay_1", "ground"]."""
    hops = next_hops(graph, destination)
    if source != destination and source not in hops:
        return None
    path = [source]
    while path[-1] != destination:
        path.append(hops[path[-1]])  # path[-1] is the last ID in the list so far
    return path
