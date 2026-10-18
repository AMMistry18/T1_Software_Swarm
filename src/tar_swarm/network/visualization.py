"""Top-down animation and final figures; plotting never sets simulation time."""

import numpy as np

STYLES = {
    "ground_station": ("s", "black"),
    "mapper": ("^", "tab:blue"),
    "relay": ("o", "tab:orange"),
}


def setup_axes(ax, config):
    ax.set_xlim(*config["plot_bounds_m"][0])
    ax.set_ylim(*config["plot_bounds_m"][1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East x (m)")
    ax.set_ylabel("North y (m)")
    ax.grid(alpha=0.25)


class NetworkView:
    def __init__(self, ax, simulator):
        self.ax = ax
        self.simulator = simulator
        setup_axes(ax, simulator.config)
        self.artists = {}
        for agent in simulator.agents:
            name = agent["id"]
            marker, color = STYLES[agent["role"]]
            (trail,) = ax.plot([], [], color=color, alpha=0.4, linewidth=1)
            (node,) = ax.plot(
                [],
                [],
                marker=marker,
                color=color,
                linestyle="none",
                markersize=9,
                label=f"{name} ({agent['role']})",
            )
            (status,) = ax.plot(
                [],
                [],
                marker="o",
                markerfacecolor="none",
                markersize=17,
                markeredgewidth=2,
                linestyle="none",
            )
            (target,) = ax.plot([], [], marker="x", color=color, linestyle="none")
            label = ax.text(0, 0, name, fontsize=8)
            self.artists[name] = (trail, node, status, target, label)
        # Allocate all pairs once so drawing doesn't change graph/state.
        names = sorted(self.artists)
        self.edges = {}
        for index, first in enumerate(names):
            for second in names[index + 1 :]:
                (line,) = ax.plot([], [], color="slategray", linewidth=1.5, zorder=0)
                self.edges[first, second] = line
        ax.legend(loc="upper left", fontsize=7)
        ax.text(
            0.99,
            0.02,
            "Ring: green = ground reachable; red = unreachable\n"
            "Lines = physical links; x = target; paths = trails",
            transform=ax.transAxes,
            ha="right",
            fontsize=7,
        )

    def draw(self):
        sim = self.simulator
        frame = sim.history[-1]
        for name, (trail, node, status, target, label) in self.artists.items():
            path = np.array([sample.positions[name] for sample in sim.history])
            position = frame.positions[name]
            trail.set_data(path[:, 0], path[:, 1])
            node.set_data([position[0]], [position[1]])
            status.set_data([position[0]], [position[1]])
            status.set_markeredgecolor("green" if name in frame.reachable else "red")
            target.set_data([frame.targets[name][0]], [frame.targets[name][1]])
            label.set_position((position[0] + 1.5, position[1] + 1.5))
        for pair, line in self.edges.items():
            if pair in frame.graph.edges:
                points = np.array([frame.positions[name] for name in pair])
                line.set_data(points[:, 0], points[:, 1])
            else:
                line.set_data([], [])
        statuses = ", ".join(
            f"{a['id']}: {'connected' if a['id'] in frame.reachable else 'OUTAGE'}"
            for a in sim.agents
            if a["role"] == "mapper"
        )
        self.ax.set_title(
            f"Toy radius graph | t={frame.time_s:.1f}s "
            f"(graph t={frame.graph.time_s:.1f}s)\n{statuses}",
            fontsize=10,
        )
        return []


def animate(simulator):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(11, 6))
    view = NetworkView(ax, simulator)
    paused = False

    def on_key(event):
        nonlocal paused
        if event.key == " ":
            paused = not paused
            fig.suptitle("Paused — space to resume" if paused else "Space to pause")
            fig.canvas.draw_idle()

    def update(_):
        if not paused and not simulator.done:
            simulator.step()
        view.draw()
        if simulator.done:
            animation.event_source.stop()
            fig.suptitle("Complete — close window to save results")
        return []

    fig.suptitle("Space to pause")
    fig.canvas.mpl_connect("key_press_event", on_key)
    animation = FuncAnimation(
        fig,
        update,
        init_func=view.draw,
        interval=max(1, simulator.dt_s * 1000),
        cache_frame_data=False,
    )
    plt.show()


def save_plot(simulator, output):
    import matplotlib.pyplot as plt

    fig, (map_ax, status_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    NetworkView(map_ax, simulator).draw()
    times = [frame.time_s for frame in simulator.history]
    mappers = [a["id"] for a in simulator.agents if a["role"] == "mapper"]
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
