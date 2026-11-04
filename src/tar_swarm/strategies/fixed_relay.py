"""Fixed relay placement: straight chains of network drones, one per mapping drone.

The problem: the base station is at one spot, each mapping drone at another, and
we have n network drones (relays). Where should they go so the radio links
base -> relay -> ... -> relay -> mapper are as strong as possible?

For ONE mapper, the answer (for the toy radio model, where a link works up to a
fixed distance):
1. Put the relays on the straight line from the base to the mapper. Any bend makes
   the chain longer, so its hops would have to be longer too.
2. Each hop has a limit: the longest it can be, which is the smaller radio range
   of its two ends. Stretch every hop to the SAME fraction of its limit.

If even that isn't enough (above 100%), the relays instead go out from the base
with every hop at its full limit: they all stay linked to the base, and the only
gap is the last hop, at the mapper's end, which closes as soon as the mapper comes
back within reach. 

For SEVERAL mappers ("option A"): one separate chain per mapper.
- Splitting the relays: hand them out one at a time, each to whichever chain is
  most stretched right now (split_relays).
- Which relay flies to which spot: the Hungarian method (match_relays_to_slots).
"""

import math

from scipy.optimize import linear_sum_assignment


def hop_limits(n_relays, range_m, base_rise_m=0.0):
    """The longest each hop of a base -> relays -> mapper chain may be ON THE MAP,
    in meters.

    n_relays: how many network drones sit between the base and the mapper.
    range_m: the radio range per role ("ground_station", "relay", "mapper").
    base_rise_m: how much higher the drones fly than the base.

    A chain with n relays has n + 1 hops. Each hop's limit is the smaller range of
    its two end drones. 
    """
    base = range_m["ground_station"]
    relay = range_m["relay"]
    mapper = range_m["mapper"]

    def flat(limit_m):
        """A base hop's limit on the map, after the climb (never below 0)."""
        return math.sqrt(max(limit_m**2 - base_rise_m**2, 0.0))  # ** is "to the power"

    if n_relays == 0:
        return [flat(min(base, mapper))]  # one direct hop, base to mapper
    # [relay] * 2 makes the list [relay, relay]: one limit per middle hop.
    return [flat(min(base, relay))] + [relay] * (n_relays - 1) + [min(relay, mapper)]


def chain_slots(base_xy, mapper_xy, limits_m):
    """Spots for one chain's relays, on the straight line from base to mapper.

    base_xy, mapper_xy: (x, y) map points, in meters.
    limits_m: each hop's limit from the base outward (see hop_limits).
    Returns (slots, stretch):
      slots: one (x, y) spot per relay, from the base outward.
      stretch: distance / (sum of limits). 0.83 means every hop uses 83% of its
        limit; above 1 means the chain can't reach, so the hops are at their full
        limits from the base and the gap is at the mapper's end.
    """
    distance_m = math.dist(base_xy, mapper_xy)
    total_m = sum(limits_m)
    # Reaching: shrink every hop by the same factor (total_m / distance_m of its
    # limit). Too far: use each hop's full limit, measured from the base.
    # max(...) picks whichever of those two cases applies.
    spread_m = max(total_m, distance_m)
    slots = []
    covered_m = 0.0  # how far along the chain's hop limits we are so far
    for limit_m in limits_m[:-1]:  # a relay at the far end of every hop but the last
        covered_m += limit_m
        fraction = covered_m / spread_m if spread_m > 0 else 0.0
        x = base_xy[0] + fraction * (mapper_xy[0] - base_xy[0])
        y = base_xy[1] + fraction * (mapper_xy[1] - base_xy[1])
        slots.append((x, y))
    return slots, distance_m / total_m


def split_relays(n_relays, distances_m, range_m, base_rise_m=0.0):
    """How many relays each mapper's chain gets: a list of counts.

    distances_m: each mapper's distance from the base on the map, in meters.
    range_m, base_rise_m: same as in hop_limits.

    Hand the relays out one at a time, each to the chain that is most stretched
    right nowm since the only way to make the worst chain better is to give it
    another relay.
    """
    counts = [0] * len(distances_m)  # [0] * 3 makes [0, 0, 0]

    def stretch(chain):
        """How stretched chain number `chain` is with its relays so far."""
        limits_m = hop_limits(counts[chain], range_m, base_rise_m)
        return distances_m[chain] / sum(limits_m)

    for _ in range(n_relays):
        # max(..., key=stretch) finds the chain number with the biggest stretch.
        worst = max(range(len(counts)), key=stretch)
        counts[worst] += 1
    return counts


def match_relays_to_slots(relay_xy, slots):
    """Pair every relay with a spot so the total flying distance is as small as
    possible: the Hungarian method.

    relay_xy: a dict of relay ID -> current (x, y). slots: a list of (x, y) spots,
    one per relay. Returns a dict of relay ID -> its spot.

    The Hungarian method looks at a table of every relay-to-spot distance and
    finds the pairing with the smallest total, without trying every possible
    pairing (10 relays already have 3,628,800 pairings). scipy's
    linear_sum_assignment does the work.
    """
    names = sorted(relay_xy)
    # cost[i][j]: how far relay i would fly to reach spot j, in meters.
    cost = [[math.dist(relay_xy[name], slot) for slot in slots] for name in names]
    rows, columns = linear_sum_assignment(cost)
    # zip pairs each chosen row (a relay) with its column (a spot).
    return {names[row]: slots[column] for row, column in zip(rows, columns)}


def optimal_chain_plan(simulator, positions):
    """Spots for every working network drone: one straight chain from the base to
    each working mapping drone, for where everyone is right now.

    simulator: the running Simulator (for each drone's role, whether it's on, the
      base's ID, and the radio ranges).
    positions: drone ID -> current 3D position, in meters.
    Returns ({relay ID: (x, y) spot}, worst stretch over all chains), or
    ({}, None) if there are no network drones or no mapping drones. While the live
    window's "auto optimal positions" box is checked, Simulator.set_relay_planner
    runs this every step.
    """
    names = sorted(positions)
    working = [name for name in names if simulator.active[name]]
    relays = [name for name in working if simulator.roles[name] == "relay"]
    mappers = [name for name in working if simulator.roles[name] == "mapper"]
    if not relays or not mappers:
        return {}, None
    ranges_m = simulator.link_model.range_m
    base = positions[simulator.ground_id]
    base_xy = base[:2]
    # How much higher than the base the drones fly; the biggest difference is the
    # safe one to plan with. (The plan assumes the drones share one altitude.)
    rise_m = max(abs(positions[name][2] - base[2]) for name in relays + mappers)
    distances_m = [math.dist(base_xy, positions[name][:2]) for name in mappers]
    counts = split_relays(len(relays), distances_m, ranges_m, rise_m)
    slots = []
    worst = 0.0
    for mapper, count in zip(mappers, counts):
        mapper_xy = positions[mapper][:2]
        limits_m = hop_limits(count, ranges_m, rise_m)
        chain, stretch = chain_slots(base_xy, mapper_xy, limits_m)
        slots += chain  # += adds this chain's spots to the end of the list
        worst = max(worst, stretch)
    relay_xy = {name: positions[name][:2] for name in relays}
    return match_relays_to_slots(relay_xy, slots), worst
