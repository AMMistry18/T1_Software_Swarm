"""Time-weighted connectivity metrics, using held graph observations."""


def connectivity_metrics(history, mapper_ids):
    """Sample k describes [t[k], t[k+1]); terminal samples have no weight."""
    end = history[-1].time_s
    result = {}
    for name in mapper_ids:
        connected_s = 0.0
        outages = []
        outage_start = None
        for first, second in zip(history, history[1:]):
            if name in first.reachable:
                connected_s += second.time_s - first.time_s
                if outage_start is not None:
                    outages.append([outage_start, first.time_s])
                    outage_start = None
            elif outage_start is None:
                outage_start = first.time_s
        if outage_start is not None:
            outages.append([outage_start, end])
        result[name] = {
            "connected_time_s": connected_s,
            "connected_time_fraction": connected_s / end if end else None,
            "outage_intervals_s": outages,
        }
    return result
