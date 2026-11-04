"""The simulator's pictures: a live, editable window and a final saved plot.

The live window (`animate`) shows the map from above and lets you change the run
with the mouse while it plays:
- click a drone to select it, then click the map: the drone flies there
- drag a drone to put it somewhere at once (it then stays; the base can't move)
- drag on an empty part of the map to draw an obstacle box; right-click a box to
  remove it. Boxes block radio links (drones still fly through them for now).
- right-click an empty spot to add a network drone, or shift + right-click to add
  a mapping drone; right-click a drone to remove it
- sliders set the radio ranges; checkboxes show or hide them as circles
- "auto optimal positions" makes the network drones keep flying to the best
  spots, one straight chain from the base to each mapping drone, even as they
  move (the math is in strategies/fixed_relay.py)
- a checkbox chooses whether closing the window saves the results
- space pauses and resumes; + and - change the simulation speed
- moving green dashes show each mapping drone's route to the base (routing.py),
  flowing the way its messages travel

The clock keeps running until the window is closed (duration_s only ends headless
runs). Drawing never moves the clock; only `InteractiveWindow.update` does.
Everything on screen is an "artist" (matplotlib's word for anything drawn: a
marker, a line, a circle, a piece of text).
"""

import time

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Rectangle

from tar_swarm.network.routing import route
from tar_swarm.strategies.fixed_relay import optimal_chain_plan

# How each kind of drone looks: (marker shape, color, marker area, legend name).
# Marker codes: "s" = square, "^" = triangle, "o" = dot. Marker area is in
# points squared (a point is 1/72 inch), so an area of 121 is 11 points across.
STYLES = {
    "ground_station": ("s", "black", 140, "base station"),
    "mapper": ("^", "tab:blue", 170, "mapping drone"),
    "relay": ("o", "tab:orange", 110, "network drone"),
}

# The short names on the range checkboxes and sliders, one per kind of drone.
ROLE_LABELS = {"relay": "network", "ground_station": "base", "mapper": "mapper"}

WINDOW_SIZE_IN = (10.5, 8.0)  # inches; at 100 pixels per inch, 1050 x 800 pixels
# The timer waits this long AFTER each frame, so frames come as fast as drawing
# allows (on this laptop a frame takes about 25-45 ms, so about 20 frames a second).
FRAME_WAIT_MS = 10
MAX_FRAME_S = 0.25  # one frame never moves the clock more than this much real time
SPEEDS = (1, 2, 5, 10, 20, 50, 100)  # simulated seconds per real second
START_SPEED = 5
GRAB_PX = 15  # a click within this many screen pixels of a drone picks it
DRAG_START_PX = 5  # the mouse must move this far while held to count as a drag
LEFT_BUTTON, RIGHT_BUTTON = 1, 3  # matplotlib numbers the mouse buttons 1, 2, 3
# The route dashes: 4 units of line, then 3 of gap (a unit is the line's width).
# They slide DASH_SPEED units per real second toward the line's start.
ROUTE_DASHES = (4, 3)
DASH_SPEED = 14

HELP_TEXT = (
    "click a drone, then the map: fly it there\n"
    "drag a drone: put it there now\n"
    "drag on empty map: draw an obstacle\n"
    "right-click: add / remove a network drone\n"
    "shift + right-click: add a mapping drone\n"
    "right-click an obstacle: remove it\n"
    "space: pause   +/-: speed   esc: deselect\n"
    "ring: green = reaches base, red = cut off\n"
    "green dashes: messages to the base"
)


def setup_axes(ax, config):
    """Draw the empty map: a square grid, the map's outline, and axis labels."""
    (xmin, xmax), (ymin, ymax) = config["plot_bounds_m"]
    step = config["grid_step_m"]
    # A little room around the map, so drones on its edge show in full.
    pad = 0.03 * max(xmax - xmin, ymax - ymin)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")  # 1 m east looks as long as 1 m north
    # A grid line every grid step, but a number only every second line: numbers
    # are slow to draw. np.arange(a, b, step) counts a, a + step, ... and stops
    # before b; the extra half step lets it reach xmax. minor=True makes ticks
    # without numbers.
    ax.set_xticks(np.arange(xmin, xmax + step / 2, 2 * step))
    ax.set_xticks(np.arange(xmin, xmax + step / 2, step), minor=True)
    ax.set_yticks(np.arange(ymin, ymax + step / 2, 2 * step))
    ax.set_yticks(np.arange(ymin, ymax + step / 2, step), minor=True)
    ax.grid(which="both", color="0.88", linewidth=0.8)  # both = every tick
    ax.set_axisbelow(True)  # grid lines go behind everything else
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor="0.45",
            linewidth=1.5,
            zorder=0.6,
        )
    )
    ax.set_xlabel("East x (m)")
    ax.set_ylabel("North y (m)")


class NetworkView:
    """Draws the simulator's latest snapshot on a map: drones, links, routes,
    ranges, and obstacles.

    It only reads simulator.history[-1] and never moves the clock. The live window
    and the saved plot both use it. zorder is the stacking order: higher numbers
    are drawn on top. live=True marks every moving artist as "animated", which
    tells matplotlib to leave it out of full redraws (see InteractiveWindow).
    """

    def __init__(self, ax, simulator, live=False):
        self.ax = ax
        self.simulator = simulator
        self.live = live
        self.selected = None  # the ID of the drone to highlight, if any
        self.route_offset = 0.0  # how far the route dashes have slid (see update)
        setup_axes(ax, simulator.config)
        # Which roles' range circles are shown (the checkboxes flip these).
        self.show_range = dict.fromkeys(STYLES, True)
        self.range_circles = {}  # drone ID -> its radio-range circle
        self.labels = {}  # drone ID -> its name tag
        self.obstacle_patches = {}  # obstacle box -> its gray rectangle
        # All links as ONE object (a LineCollection), not one line per pair.
        self.links = ax.add_collection(
            LineCollection([], colors="slategray", linewidths=2, zorder=1)
        )
        # Each mapper's route to the base: green dashes that slide along it.
        self.routes = ax.add_collection(
            LineCollection([], colors="tab:green", linewidths=3, zorder=1.2)
        )
        # A dashed line from each flying drone to its goal.
        self.goal_lines = ax.add_collection(
            LineCollection([], colors="0.35", linewidths=1.2, linestyles="--")
        )
        # A gold ring around the selected drone.
        self.selection = ax.scatter(
            [], [], s=800, facecolors="none", edgecolors="gold", linewidths=3
        )
        # A ring around every drone: green = can reach the base, red = cut off.
        self.rings = ax.scatter([], [], s=420, facecolors="none", linewidths=2)
        # One scatter per role: drones of one role share a shape and a color, so
        # one artist draws all of them at once.
        self.markers = {}
        for role, (marker, color, size, name) in STYLES.items():
            self.markers[role] = ax.scatter(
                [], [], marker=marker, s=size, color=color, zorder=3, label=name
            )
        # An x on each flying drone's goal (drones holding still get none).
        self.targets = ax.scatter(
            [], [], marker="x", s=70, color="0.25", zorder=3, label="goal"
        )
        # The outline of an obstacle while you're dragging to draw it.
        self.box_preview = ax.add_patch(
            Rectangle((0, 0), 0, 0, fill=False, edgecolor="0.2", linestyle="--")
        )
        self.box_preview.set_visible(False)
        # The artists that exist for the whole run, in drawing order (bottom
        # first). Circles, obstacles, and name tags come and go.
        self.fixed_artists = [
            self.links,
            self.routes,
            self.goal_lines,
            self.selection,
            self.rings,
            *self.markers.values(),  # *list spreads the list's items in place
            self.targets,
            self.box_preview,
            ax.title,
        ]
        for artist in self.fixed_artists:
            artist.set_animated(live)

    def moving_artists(self):
        """Everything that can change between frames, in drawing order."""
        return [
            *self.range_circles.values(),
            *self.obstacle_patches.values(),
            *self.fixed_artists[:-1],
            *self.labels.values(),
            self.ax.title,
        ]

    def draw(self):
        """Move every artist to match the simulator's latest snapshot."""
        sim = self.simulator
        frame = sim.history[-1]
        names = sorted(frame.positions)  # the drones in the run right now
        # Each drone's spot on the map: [:2] keeps x and y and drops the altitude.
        xy = {name: frame.positions[name][:2] for name in names}
        mappers = [name for name in names if sim.roles[name] == "mapper"]

        # Links: one (start, end) pair of map points per working link.
        self.links.set_segments(
            [(xy[a], xy[b]) for a, b in frame.graph.edges if a in xy and b in xy]
        )
        # Routes: each mapper's fewest-hops path to the base (routing.py). Each
        # line starts at the base, because the dashes slide toward a line's start:
        # that way they flow toward the base, like the mapper's messages.
        routes = []
        for mapper in mappers:
            path = route(frame.graph, mapper, sim.ground_id)
            if path is not None:
                routes.append([xy[name] for name in reversed(path)])
        self.routes.set_segments(routes)
        self.routes.set_linestyle((self.route_offset, ROUTE_DASHES))
        # Drones flying somewhere: a dashed line to the goal and an x on it.
        flying = [
            name
            for name in names
            if not np.array_equal(frame.targets[name], frame.positions[name])
        ]
        self.goal_lines.set_segments(
            [(xy[name], frame.targets[name][:2]) for name in flying]
        )
        # np.reshape(points, (-1, 2)) means "as many rows as needed, 2 columns",
        # which keeps the shape right even when a list is empty.
        self.targets.set_offsets(
            np.reshape([frame.targets[name][:2] for name in flying], (-1, 2))
        )
        for role, marker in self.markers.items():
            points = [xy[name] for name in names if sim.roles[name] == role]
            marker.set_offsets(np.reshape(points, (-1, 2)))
        self.rings.set_offsets(np.reshape([xy[name] for name in names], (-1, 2)))
        self.rings.set_edgecolors(
            ["green" if name in frame.reachable else "red" for name in names]
        )
        chosen = [xy[self.selected]] if self.selected in xy else []
        self.selection.set_offsets(np.reshape(chosen, (-1, 2)))
        self._update_circles_and_labels(names, xy, frame)
        self._update_obstacles()

        cut_off = [name for name in mappers if name not in frame.reachable]
        if len(mappers) == 1:
            status = f"{mappers[0]}: {'OUTAGE' if cut_off else 'connected'}"
        else:  # with several mappers, a count keeps the title short
            connected = len(mappers) - len(cut_off)
            status = f"{connected} of {len(mappers)} mappers connected"
        if sim.plan_stretch is not None:  # automatic placement is on
            # f"{x:.0%}" writes 0.833 as "83%".
            status += f"      planned hops up to {sim.plan_stretch:.0%}"
        self.ax.set_title(
            f"t = {frame.time_s:.1f} s      {status}",
            fontsize=12,
            color="darkred" if cut_off else "darkgreen",
        )
        return []

    def _update_circles_and_labels(self, names, xy, frame):
        """Give each drone a range circle and a name tag; drop removed drones'."""
        sim = self.simulator
        ranges_m = sim.link_model.range_m  # role -> range; sliders change these
        for name in list(self.range_circles):  # list(...) copies, so we can delete
            if name not in xy:  # this drone was removed by hand
                self.range_circles.pop(name).remove()
                self.labels.pop(name).remove()
        for name in names:
            role = sim.roles[name]
            if name not in self.range_circles:  # a new drone: make its artists
                color = STYLES[role][1]
                # to_rgba(color, alpha) adds see-through-ness: 0 invisible, 1 solid.
                circle = Circle(
                    xy[name],
                    ranges_m[role],
                    facecolor=to_rgba(color, 0.06),
                    edgecolor=to_rgba(color, 0.45),
                    linewidth=1.2,
                    zorder=0.8,
                )
                self.range_circles[name] = self.ax.add_patch(circle)
                # annotate puts the tag 8 points up and right of the drone, so it
                # stays the same distance away at any zoom.
                self.labels[name] = self.ax.annotate(
                    name,
                    xy[name],
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=9,
                    zorder=4,
                )
                self.range_circles[name].set_animated(self.live)
                self.labels[name].set_animated(self.live)
            circle = self.range_circles[name]
            circle.set_center(xy[name])
            circle.set_radius(ranges_m[role])
            # A turned-off (inactive) drone has no working radio, so no circle.
            circle.set_visible(self.show_range[role] and frame.active[name])
            self.labels[name].xy = xy[name]

    def _update_obstacles(self):
        """Give each obstacle box a gray rectangle; drop removed boxes' ones."""
        boxes = set(self.simulator.obstacles)  # a set holds each box only once
        for box in list(self.obstacle_patches):
            if box not in boxes:
                self.obstacle_patches.pop(box).remove()
        for box in boxes:
            if box not in self.obstacle_patches:
                x_min, y_min, x_max, y_max = box
                patch = Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    facecolor="0.6",
                    edgecolor="0.25",
                    alpha=0.75,
                    zorder=0.9,
                )
                patch.set_animated(self.live)
                self.obstacle_patches[box] = self.ax.add_patch(patch)


class InteractiveWindow:
    """The live window: the map, the controls, mouse editing, and keys.

    matplotlib calls the on_... methods when you use the mouse, the keyboard, or
    a control, and a timer calls update() again FRAME_WAIT_MS after each frame.

    Speed trick ("blitting"): text is slow to draw, and most of the window (grid,
    numbers, controls, help) rarely changes. So after every full redraw, we take
    a snapshot of those still parts (on_full_draw). Each frame then pastes the
    snapshot back and draws only the moving parts on top (draw_moving_parts).
    """

    def __init__(self, simulator):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import CheckButtons, Slider

        self.simulator = simulator
        # The live window's clock runs until the window is closed, so you can keep
        # editing and playing past the config's duration_s.
        simulator.endless = True
        self.paused = False
        self.speed = START_SPEED
        self.owed_s = 0.0  # simulated seconds the clock still owes (see update)
        self.selected = None  # the drone clicked last: a map click sends it there
        # While the left button is down: (drone ID or None, x and y in pixels, the
        # map point). None means the press was on empty map.
        self.pressed = None
        self.dragging = False  # True once the held mouse has moved DRAG_START_PX
        self.box_corner = None  # the far corner of an obstacle being drawn
        self.message = ""  # what the last action did, shown in the side panel
        self.save_on_close = True
        self.background = None  # the snapshot of the still parts (see above)
        # The real-time clock, in seconds. Tests swap in a pretend clock.
        self.clock = time.perf_counter
        self.last_frame_s = self.clock()

        self.fig = plt.figure(figsize=WINDOW_SIZE_IN)
        # fig.add_axes([left, bottom, width, height]) makes a drawing area; the
        # numbers are fractions of the window, 0 to 1. The map comes first, so
        # it's fig.axes[0].
        map_ax = self.fig.add_axes([0.08, 0.07, 0.62, 0.86])
        self.view = NetworkView(map_ax, simulator, live=True)
        # useblit=False on every control: when one changes, it asks for a full
        # redraw, which also refreshes our snapshot of the still parts.
        toggle_ax = self.fig.add_axes([0.725, 0.80, 0.26, 0.12])
        toggle_ax.set_title("Show radio range", fontsize=10, loc="left")
        self.toggles = CheckButtons(
            toggle_ax, list(ROLE_LABELS.values()), [True] * 3, useblit=False
        )
        self.toggles.on_clicked(self.on_toggle)
        self.fig.text(0.725, 0.765, "Radio range (m)", fontsize=10)
        # Slider limits follow the map's size: 5% of its side up to the full side.
        (xmin, xmax), (ymin, ymax) = simulator.config["plot_bounds_m"]
        side_m = max(xmax - xmin, ymax - ymin)
        self.sliders = {}
        for row, (role, label) in enumerate(ROLE_LABELS.items()):
            box = self.fig.add_axes([0.80, 0.725 - 0.035 * row, 0.12, 0.022])
            start_m = simulator.link_model.range_m[role]
            self.sliders[role] = Slider(
                box,
                label,
                0.05 * side_m,
                max(side_m, start_m),
                valinit=start_m,
                valstep=side_m / 100,
                valfmt="%.0f",
                initcolor="none",  # no line marking the starting value
            )
            self.sliders[role].on_changed(self.on_range_slider)
        auto_ax = self.fig.add_axes([0.725, 0.555, 0.26, 0.05])
        self.auto_box = CheckButtons(
            auto_ax, ["auto optimal positions"], [False], useblit=False
        )
        self.auto_box.on_clicked(self.on_auto_toggle)
        save_ax = self.fig.add_axes([0.725, 0.49, 0.26, 0.05])
        self.save_box = CheckButtons(
            save_ax, ["save results when closed"], [True], useblit=False
        )
        self.save_box.on_clicked(self.on_save_toggle)
        self.status = self.fig.text(
            0.725, 0.465, "", fontsize=10, va="top", linespacing=1.4
        )
        self.fig.text(0.725, 0.33, HELP_TEXT, fontsize=9, va="top", linespacing=1.5)

        # Callbacks: "whenever this happens, run that method".
        canvas = self.fig.canvas
        canvas.mpl_connect("draw_event", self.on_full_draw)
        canvas.mpl_connect("button_press_event", self.on_press)
        canvas.mpl_connect("motion_notify_event", self.on_motion)
        canvas.mpl_connect("button_release_event", self.on_release)
        canvas.mpl_connect("key_press_event", self.on_key)
        # The frame timer; animate() starts it.
        self.timer = canvas.new_timer(interval=FRAME_WAIT_MS)
        self.timer.add_callback(self.update)
        self.show_status()

    # Drawing

    def on_full_draw(self, _event):
        """matplotlib just redrew the whole window (first showing, a resize, a
        zoom, a control changing): snapshot the still parts, add the moving ones."""
        if self.fig.canvas.is_saving():
            # Saving a picture (the toolbar's save button) draws everything,
            # moving parts included, so adding them again would draw them twice.
            return
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.draw_moving_parts()

    def draw_moving_parts(self):
        """Paste the snapshot of the still parts and draw only the moving parts."""
        canvas = self.fig.canvas
        if self.background is None:  # no snapshot yet: one full redraw makes it
            canvas.draw_idle()
            return
        canvas.restore_region(self.background)
        for artist in self.view.moving_artists():
            self.fig.draw_artist(artist)
        canvas.blit(self.fig.bbox)

    def redraw(self):
        """Show the latest snapshot now, even while paused (used after edits)."""
        self.view.selected = self.selected
        self.view.draw()
        self.draw_moving_parts()

    def show_status(self):
        """Write the speed, the selection, and the last action in the side panel.

        This text is one of the still parts, so changing it needs a full redraw.
        """
        state = "PAUSED" if self.paused else "running"
        lines = [f"Speed: {self.speed}x      ({state})"]
        if self.selected is not None:
            lines.append(f"Selected: {self.selected} (click the map)")
        if self.message:
            lines.append(self.message)
        self.status.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()

    # The clock

    def update(self):
        """Called by the timer: keep the simulation clock in step with real time
        (times the speed) unless paused, then draw.

        Example: at 5x, if 0.05 real seconds passed since the last frame, the clock
        owes 5 x 0.05 = 0.25 simulated seconds. It only moves in whole steps of dt_s
        (0.1 s), so it takes 2 steps and keeps the leftover 0.05 s for next time.
        Every step is the same fixed step, so the run itself doesn't depend on how
        fast the window draws; only how many steps fall between frames does.
        """
        sim = self.simulator
        now = self.clock()
        # min(...) caps a long stall (say, the window was being dragged), so the
        # clock doesn't try to catch up with hundreds of steps at once.
        real_s = min(now - self.last_frame_s, MAX_FRAME_S)
        self.last_frame_s = now
        if not self.paused:
            self.owed_s += self.speed * real_s
            # The 1e-9 forgives tiny rounding errors: 0.1 isn't exact in binary.
            while self.owed_s >= sim.dt_s - 1e-9:
                sim.step()
                self.owed_s -= sim.dt_s
            # Slide the route dashes. % (remainder) wraps the slide after one full
            # dash + gap, which looks the same as no slide at all.
            slid = self.view.route_offset + DASH_SPEED * real_s
            self.view.route_offset = slid % sum(ROUTE_DASHES)
        self.redraw()

    # Mouse

    def drone_near(self, event):
        """The ID of the drone drawn within GRAB_PX pixels of the mouse, or None."""
        frame = self.simulator.history[-1]
        names = list(frame.positions)
        # transData turns map meters into screen pixels, so "near" means the same
        # on screen however far you zoom in or out.
        pixels = self.view.ax.transData.transform(
            [frame.positions[name][:2] for name in names]
        )
        gaps = np.hypot(pixels[:, 0] - event.x, pixels[:, 1] - event.y)
        nearest = int(np.argmin(gaps))  # argmin: the position of the smallest gap
        return names[nearest] if gaps[nearest] <= GRAB_PX else None

    def obstacle_at(self, point):
        """The obstacle box under map point (x, y), or None (the newest wins)."""
        x, y = point
        for box in reversed(self.simulator.obstacles):  # newest first
            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                return box
        return None

    def on_press(self, event):
        """A mouse button went down on the map."""
        # Ignore clicks off the map (on a control, say), and clicks while the
        # toolbar's zoom or pan tool is switched on.
        if event.inaxes is not self.view.ax or self.fig.canvas.widgetlock.locked():
            return
        name = self.drone_near(event)
        point = (event.xdata, event.ydata)  # where the click landed, in meters
        if event.button == LEFT_BUTTON:
            if name == self.simulator.ground_id:
                self.message = "The base station stays put."
                self.show_status()
                return
            # A click or the start of a drag: the next mouse moves decide, in
            # on_motion and on_release. On a drone, a drag moves it; on empty
            # map, a drag draws an obstacle.
            self.pressed = (name, event.x, event.y, point)
            self.dragging = False
        elif event.button == RIGHT_BUTTON:
            self.on_right_click(name, point, shift=event.key == "shift")

    def on_right_click(self, name, point, shift):
        """Right-click: add a drone, or remove a drone or an obstacle."""
        sim = self.simulator
        box = self.obstacle_at(point)
        mappers = [other for other in sim.positions if sim.roles[other] == "mapper"]
        if name is None and box is not None:
            sim.remove_obstacle(box)
            self.message = "removed an obstacle"
        elif name is None:
            role = "mapper" if shift else "relay"  # shift + right-click: mapper
            self.message = f"added {sim.add_drone(role, point)}"
        elif name == sim.ground_id:
            self.message = "The base station stays put."
        elif mappers == [name]:
            self.message = "The run needs at least one\nmapping drone."
        else:
            sim.remove_drone(name)
            if self.selected == name:
                self.selected = None
            self.message = f"removed {name}"
        self.show_status()
        self.redraw()

    def on_motion(self, event):
        """The mouse moved: drag the held drone, or stretch the obstacle outline."""
        if self.pressed is None or event.inaxes is not self.view.ax:
            return
        name, start_x, start_y, start = self.pressed
        if not self.dragging:
            if np.hypot(event.x - start_x, event.y - start_y) < DRAG_START_PX:
                return  # too small a wiggle to be a drag
            self.dragging = True
        point = (event.xdata, event.ydata)
        if name is None:  # drawing an obstacle: show its outline so far
            self.box_corner = point
            x0, y0 = start
            # set_bounds(x, y, width, height) moves and resizes the outline.
            self.view.box_preview.set_bounds(x0, y0, point[0] - x0, point[1] - y0)
            self.view.box_preview.set_visible(True)
        elif name in self.simulator.positions:  # dragging a drone
            self.simulator.move_agent(name, point)
        self.redraw()

    def on_release(self, _event):
        """The left button came up: finish a drag, or handle a click."""
        if self.pressed is None:
            return
        name, _, _, start = self.pressed
        sim = self.simulator
        if self.dragging and name is None and self.box_corner is not None:
            # Finished drawing an obstacle: its corners are the press point and the
            # last point the mouse reached.
            (x0, y0), (x1, y1) = start, self.box_corner
            self.view.box_preview.set_visible(False)
            if x0 != x1 and y0 != y1:
                sim.add_obstacle((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
                self.message = "added an obstacle"
        elif not self.dragging and name is not None:
            # Clicked a drone: select it, or deselect it if it was selected.
            self.selected = None if self.selected == name else name
            self.message = ""
        elif not self.dragging:
            # Clicked the empty map: send the selected drone there.
            if self.selected is None:
                self.message = "Click a drone first, then the map."
            elif sim.relay_planner is not None and sim.roles[self.selected] == "relay":
                self.message = "Network drones follow the plan\nwhile it's checked."
            else:
                sim.set_goal(self.selected, start)
                x, y = start
                self.message = f"{self.selected} flying to ({x:.0f}, {y:.0f})"
        self.pressed = None
        self.dragging = False
        self.box_corner = None
        self.show_status()
        self.redraw()

    # Keys and controls

    def on_key(self, event):
        """Space pauses or resumes; + (or =) and - change the speed; esc deselects."""
        if event.key == " ":
            self.paused = not self.paused
        elif event.key in ("+", "=") and self.speed != SPEEDS[-1]:
            self.speed = SPEEDS[SPEEDS.index(self.speed) + 1]
        elif event.key == "-" and self.speed != SPEEDS[0]:
            self.speed = SPEEDS[SPEEDS.index(self.speed) - 1]
        elif event.key == "escape":
            self.selected = None
        self.show_status()
        self.redraw()

    def on_toggle(self, label):
        """A range checkbox was clicked: show or hide that role's circles."""
        # Find the role whose short name is this label.
        role = next(role for role, name in ROLE_LABELS.items() if name == label)
        self.view.show_range[role] = not self.view.show_range[role]
        self.redraw()

    def on_range_slider(self, _value):
        """A range slider moved: give each kind of drone its slider's range."""
        for role, slider in self.sliders.items():
            if slider.val != self.simulator.link_model.range_m[role]:
                self.simulator.set_range(role, slider.val)
        self.redraw()

    def on_auto_toggle(self, _label):
        """The "auto optimal positions" box was clicked: switch the automatic
        placement on or off.

        While it's on, every step re-plans the best spots for where the mappers
        are right now: one straight chain per mapper, with the network drones
        split between the chains and paired with spots by the Hungarian method
        (strategies/fixed_relay.py). The network drones fly toward their spots, so
        the chains follow the mappers. The title shows how stretched the planned
        hops are. Switched off, the network drones hold wherever they are.
        """
        on = self.auto_box.get_status()[0]
        self.simulator.set_relay_planner(optimal_chain_plan if on else None)
        if on:
            self.message = (
                "Network drones now follow the\nbest spots as the mappers move."
            )
        else:
            self.message = "Network drones hold where they are."
        self.show_status()
        self.redraw()

    def on_save_toggle(self, _label):
        """The save checkbox was clicked: remember whether to save on close."""
        self.save_on_close = self.save_box.get_status()[0]
        self.message = "" if self.save_on_close else "Results won't be saved."
        self.show_status()


def animate(simulator):
    """Open the live window and wait until it's closed.

    Returns True if "save results when closed" was still checked at the end.
    """
    import matplotlib.pyplot as plt

    window = InteractiveWindow(simulator)
    window.timer.start()
    plt.show()
    return window.save_on_close


def save_plot(simulator, output):
    """Save the run as a picture: the final map with every drone's path, and a
    timeline of when each mapper could reach the base."""
    import matplotlib.pyplot as plt

    fig, (map_ax, status_ax) = plt.subplots(
        2,
        1,
        figsize=(10.8, 11),
        gridspec_kw={"height_ratios": [4, 1]},
        constrained_layout=True,
    )
    NetworkView(map_ax, simulator).draw()
    # The legend sits just outside the map's top-right corner.
    map_ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), borderaxespad=0)
    # Each drone's path. A drone added or removed by hand has a path only for
    # the part of the run it was in.
    for name, role in simulator.roles.items():
        path = np.reshape(
            [f.positions[name][:2] for f in simulator.history if name in f.positions],
            (-1, 2),
        )
        map_ax.plot(
            path[:, 0], path[:, 1], color=STYLES[role][1], alpha=0.5, zorder=1
        )
    times = [frame.time_s for frame in simulator.history]
    mappers = [name for name, role in simulator.roles.items() if role == "mapper"]
    for index, name in enumerate(mappers):
        values = [int(name in frame.reachable) for frame in simulator.history]
        status_ax.step(times, np.array(values) * 0.7 + index, where="post", label=name)
    status_ax.set_yticks(
        [value for i in range(len(mappers)) for value in (i, i + 0.7)],
        [value for name in mappers for value in (f"{name}: off", f"{name}: on")],
    )
    status_ax.set_xlim(0, max(simulator.time_s, simulator.dt_s))
    status_ax.set_xlabel("Simulation time (s); graph status held between updates")
    status_ax.grid(alpha=0.25)
    fig.savefig(output, dpi=160)
    plt.close(fig)
