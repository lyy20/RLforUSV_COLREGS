import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from multiagent.scenarios.usv_tracking_base import Scenario
from utilities.paths import scenario_generation_dir


def build_world(seed):
    np.random.seed(seed)
    scenario = Scenario()
    world = scenario.make_world(
        num_agents=1,
        num_landmarks=1,
        num_obstacles=5,
        num_static_obstacles=8,
        map_half_size=2.0,
        static_obstacle_min_size=0.04,
        static_obstacle_max_size=0.10,
        ob_range=0.8,
        num_ob=3,
        num_static_ob_slots=1,
        landmark_movable=True,
        obstacle_movable=True,
        landmark_vel=0.0003,
        max_vel=0.0003,
        random_vel=False,
        movement="escape",
        obstacle_movement="linear",
        pre_method="LS",
        rew_err_th=0.03,
        rew_dis_th=0.3,
        max_range=2.0,
        max_current_vel=0.0003,
        range_dropping=0.01,
        control_model="usv_3dof",
    )
    world.angle = [0.0 for _ in world.agents]
    return scenario, world


def entity_snapshot(world):
    return {
        "agent": np.array([world.agents[0].state.p_pos.copy()]),
        "target": np.array([world.landmarks[0].state.p_pos.copy()]),
        "dynamic": np.array([ob.state.p_pos.copy() for ob in world.obstacles[:world.num_obstacles]]),
        "static": np.array([ob.state.p_pos.copy() for ob in world.static_obstacles]),
    }


def simulate(scenario, world, steps):
    snapshots = [entity_snapshot(world)]
    fixed_agent_pos = world.agents[0].state.p_pos.copy()
    world.agents[0].action.u = np.zeros(world.agent_action_dim)
    world.agents[0].action.c = np.zeros(world.dim_c)
    for _ in range(steps):
        scenario.observation(world.agents[0], world)
        world.agents[0].action.u = np.zeros(world.agent_action_dim)
        world.step(False)
        world.agents[0].state.p_pos = fixed_agent_pos.copy()
        world.agents[0].state.p_vel = np.zeros(world.dim_p)
        world.agents[0].state.usv_3dof = None
        snapshots.append(entity_snapshot(world))
    return snapshots


def min_pairwise_distance(points_a, radii_a, points_b, radii_b):
    min_clearance = np.inf
    same_collection = points_a is points_b
    for i, point_a in enumerate(points_a):
        for j, point_b in enumerate(points_b):
            if same_collection and i == j:
                continue
            clearance = np.linalg.norm(point_a - point_b) - radii_a[i] - radii_b[j]
            min_clearance = min(min_clearance, float(clearance))
    return min_clearance if np.isfinite(min_clearance) else 0.0


def summarize(world, snapshots):
    dynamic_points = snapshots[0]["dynamic"]
    dynamic_radii = [ob.size for ob in world.obstacles[:world.num_obstacles]]
    static_points = snapshots[0]["static"]
    static_radii = [ob.size for ob in world.static_obstacles]
    target_agent_distance = np.linalg.norm(snapshots[0]["target"][0] - snapshots[0]["agent"][0])
    half_size = world.map_half_size
    outside_count = 0
    for snap in snapshots:
        for group in ("agent", "target", "dynamic", "static"):
            if np.any(np.abs(snap[group]) > half_size + 1.0e-9):
                outside_count += 1
    return {
        "target_agent_distance": float(target_agent_distance),
        "dynamic_static_min_clearance": min_pairwise_distance(
            dynamic_points,
            dynamic_radii,
            static_points,
            static_radii,
        ),
        "static_static_min_clearance": min_pairwise_distance(
            static_points,
            static_radii,
            static_points,
            static_radii,
        ),
        "outside_count": outside_count,
    }


def draw_scene(ax, world, snapshot, traces=None, title=None):
    half = world.map_half_size
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-half - 0.15, half + 0.15)
    ax.set_ylim(-half - 0.15, half + 0.15)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.grid(True, alpha=0.25)
    ax.add_patch(plt.Rectangle((-half, -half), 2 * half, 2 * half, fill=False, lw=1.8, color="black"))

    if title:
        ax.set_title(title)

    if traces is not None:
        ax.plot(traces["agent"][:, 0], traces["agent"][:, 1], color="#2563eb", lw=1.5, alpha=0.75)
        ax.plot(traces["target"][:, 0], traces["target"][:, 1], color="#dc2626", lw=1.5, alpha=0.75)
        for dyn_trace in traces["dynamic"]:
            ax.plot(dyn_trace[:, 0], dyn_trace[:, 1], color="#16a34a", lw=1.0, alpha=0.45)

    for obstacle in world.static_obstacles:
        ax.add_patch(plt.Circle(obstacle.state.p_pos, obstacle.size, color="#f59e0b", alpha=0.55))
    for i, obstacle in enumerate(world.obstacles[:world.num_obstacles]):
        ax.add_patch(plt.Circle(snapshot["dynamic"][i], obstacle.size, color="#16a34a", alpha=0.42))
        heading = getattr(obstacle, "ra", 0.0)
        ax.arrow(
            snapshot["dynamic"][i, 0],
            snapshot["dynamic"][i, 1],
            np.cos(heading) * 0.10,
            np.sin(heading) * 0.10,
            color="#166534",
            width=0.003,
            head_width=0.035,
            length_includes_head=True,
        )

    ax.scatter(snapshot["agent"][:, 0], snapshot["agent"][:, 1], s=70, c="#2563eb", label="USV")
    ax.scatter(snapshot["target"][:, 0], snapshot["target"][:, 1], s=70, c="#dc2626", label="target")
    ax.scatter([], [], s=70, c="#16a34a", alpha=0.55, label="dynamic vessel")
    ax.scatter([], [], s=70, c="#f59e0b", alpha=0.55, label="static point obstacle")
    ax.legend(loc="upper right", fontsize=8)


def traces_from_snapshots(snapshots):
    traces = {
        "agent": np.vstack([snap["agent"][0] for snap in snapshots]),
        "target": np.vstack([snap["target"][0] for snap in snapshots]),
        "dynamic": [],
    }
    dynamic_count = snapshots[0]["dynamic"].shape[0]
    for i in range(dynamic_count):
        traces["dynamic"].append(np.vstack([snap["dynamic"][i] for snap in snapshots]))
    return traces


def save_static_preview(world, snapshots, output_path):
    traces = traces_from_snapshots(snapshots)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    draw_scene(ax, world, snapshots[-1], traces=traces, title="Scenario generation preview")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_gif(world, snapshots, output_path):
    traces = traces_from_snapshots(snapshots)
    fig, ax = plt.subplots(figsize=(7.0, 7.0))

    def update(frame):
        ax.clear()
        frame_traces = {
            "agent": traces["agent"][: frame + 1],
            "target": traces["target"][: frame + 1],
            "dynamic": [trace[: frame + 1] for trace in traces["dynamic"]],
        }
        draw_scene(ax, world, snapshots[frame], traces=frame_traces, title=f"Scenario step {frame}")

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=140)
    ani.save(output_path, writer=animation.PillowWriter(fps=7))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output-dir", default=str(scenario_generation_dir()))
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    scenario, world = build_world(args.seed)
    snapshots = simulate(scenario, world, args.steps)
    png_path = os.path.abspath(os.path.join(args.output_dir, "scenario_generation_preview.png"))
    gif_path = os.path.abspath(os.path.join(args.output_dir, "scenario_generation_preview.gif"))
    save_static_preview(world, snapshots, png_path)
    if not args.no_gif:
        save_gif(world, snapshots, gif_path)
    summary = summarize(world, snapshots)
    print("SCENARIO_PREVIEW")
    print(f"png={png_path}")
    if not args.no_gif:
        print(f"gif={gif_path}")
    for key, value in summary.items():
        print(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}")


if __name__ == "__main__":
    main()
