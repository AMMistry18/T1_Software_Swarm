"""Simulator-independent world and obstacle concepts.

Obstacles are rectangles on the map, given as boxes (x_min, y_min, x_max, y_max)
in meters. For now the world is 2D: a box blocks radio signals that would pass
through it, at any altitude.
"""


def segment_crosses_box(p, q, box):
    """True if the straight line from point p to point q passes through box.

    p, q: (x, y) points in meters; box: (x_min, y_min, x_max, y_max).

    The idea: walk along the line from p to q. Write each point on it as
    p + t * (q - p), where t goes from 0 (at p) to 1 (at q). The line is inside
    the box only while it is inside the box's x-range AND its y-range at the same
    time. So for each direction, find the stretch of t where the line is inside
    that range, and check whether the two stretches overlap. A line that only
    touches an edge or a corner does not count as passing through.
    """
    t_in, t_out = 0.0, 1.0  # the stretch of t that is inside every range so far
    for start, end, low, high in (
        (p[0], q[0], box[0], box[2]),  # the x-direction
        (p[1], q[1], box[1], box[3]),  # the y-direction
    ):
        change = end - start
        if abs(change) < 1e-12:
            # The line runs parallel to this pair of sides: it is either inside
            # this range the whole way or outside it the whole way.
            if not low < start < high:
                return False
            continue
        # The t values where the line crosses the two sides of this range.
        t_low = (low - start) / change
        t_high = (high - start) / change
        t_in = max(t_in, min(t_low, t_high))
        t_out = min(t_out, max(t_low, t_high))
    # A real overlap, longer than a rounding error (1e-9 of the line's length).
    return t_out - t_in > 1e-9


def line_of_sight(p, q, boxes):
    """True if no box blocks the straight line between p and q."""
    # any(...) is True if at least one item is True.
    return not any(segment_crosses_box(p, q, box) for box in boxes)
