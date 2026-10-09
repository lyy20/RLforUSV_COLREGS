import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.preview_scenario_generation import build_world, simulate


def _clearance(point_a, radius_a, point_b, radius_b):
    return float(np.linalg.norm(point_a - point_b) - radius_a - radius_b)


def _min_clearance(points_a, radii_a, points_b, radii_b, same_collection=False):
    minimum = np.inf
    for i, point_a in enumerate(points_a):
        for j, point_b in enumerate(points_b):
            if same_collection and i == j:
                continue
            minimum = min(minimum, _clearance(point_a, radii_a[i], point_b, radii_b[j]))
    return minimum if np.isfinite(minimum) else 0.0


def _min_clearance_detail(points_a, radii_a, points_b, radii_b, same_collection=False):
    minimum = np.inf
    pair = (-1, -1)
    for i, point_a in enumerate(points_a):
        for j, point_b in enumerate(points_b):
            if same_collection and i >= j:
                continue
            clearance = _clearance(point_a, radii_a[i], point_b, radii_b[j])
            if clearance < minimum:
                minimum = clearance
                pair = (i, j)
    return (minimum if np.isfinite(minimum) else 0.0), pair


def _inside_count(snapshot, half_size):
    outside = 0
    for group in ("agent", "target", "dynamic", "static"):
        if np.any(np.abs(snapshot[group]) > half_size + 1.0e-9):
            outside += 1
    return outside


def check_one(seed, steps):
    scenario, world = build_world(seed)
    snapshots = simulate(scenario, world, steps)
    first = snapshots[0]

    agent_radius = [world.agents[0].size]
    target_radius = [world.landmarks[0].size]
    dynamic_radii = [obstacle.size for obstacle in world.obstacles[:world.num_obstacles]]
    static_radii = [obstacle.size for obstacle in world.static_obstacles]
    half_size = world.map_half_size

    target_agent_distance = float(np.linalg.norm(first["target"][0] - first["agent"][0]))
    map_width = half_size * 2.0
    min_required_target_distance = 0.45 * map_width

    min_values = {
        "static_static": _min_clearance(
            first["static"],
            static_radii,
            first["static"],
            static_radii,
            same_collection=True,
        ),
        "dynamic_static": np.inf,
        "dynamic_dynamic": np.inf,
        "target_static": np.inf,
        "target_dynamic": np.inf,
        "dynamic_agent": np.inf,
    }
    min_details = {name: {"step": 0, "pair": (-1, -1)} for name in min_values}
    outside_count = 0
    for step, snapshot in enumerate(snapshots):
        outside_count += _inside_count(snapshot, half_size)
        checks = {
            "dynamic_static": _min_clearance_detail(snapshot["dynamic"], dynamic_radii, snapshot["static"], static_radii),
            "dynamic_dynamic": _min_clearance_detail(
                snapshot["dynamic"],
                dynamic_radii,
                snapshot["dynamic"],
                dynamic_radii,
                same_collection=True,
            ),
            "target_static": _min_clearance_detail(snapshot["target"], target_radius, snapshot["static"], static_radii),
            "target_dynamic": _min_clearance_detail(snapshot["target"], target_radius, snapshot["dynamic"], dynamic_radii),
            "dynamic_agent": _min_clearance_detail(snapshot["dynamic"], dynamic_radii, snapshot["agent"], agent_radius),
        }
        for name, (value, pair) in checks.items():
            if value < min_values[name]:
                min_values[name] = value
                min_details[name] = {"step": step, "pair": pair}

    failures = []
    if len(world.static_obstacles) != 8:
        failures.append(f"static_count={len(world.static_obstacles)}")
    if world.num_obstacles != 5:
        failures.append(f"dynamic_count={world.num_obstacles}")
    if target_agent_distance < min_required_target_distance:
        failures.append(
            f"target_agent_distance={target_agent_distance:.6f}<required={min_required_target_distance:.6f}"
        )
    if outside_count:
        failures.append(f"outside_count={outside_count}")
    for name, value in min_values.items():
        if value < -1.0e-9:
            failures.append(f"{name}_clearance={value:.6f}")

    return {
        "seed": seed,
        "target_agent_distance": target_agent_distance,
        "outside_count": outside_count,
        "min_clearances": min_values,
        "min_details": min_details,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--start-seed", type=int, default=1)
    args = parser.parse_args()

    results = [check_one(args.start_seed + index, args.steps) for index in range(args.seeds)]
    failures = [result for result in results if result["failures"]]

    print("SCENARIO_GENERATION_CHECK")
    print(f"seeds={args.seeds}")
    print(f"steps={args.steps}")
    print(f"failures={len(failures)}")
    print(
        "target_agent_distance_min="
        f"{min(result['target_agent_distance'] for result in results):.6f}"
    )
    for key in ("static_static", "dynamic_static", "dynamic_dynamic", "target_static", "target_dynamic", "dynamic_agent"):
        print(
            f"{key}_clearance_min="
            f"{min(result['min_clearances'][key] for result in results):.6f}"
        )

    if failures:
        for result in failures[:10]:
            detail_text = []
            for name, detail in result["min_details"].items():
                if result["min_clearances"][name] < -1.0e-9:
                    detail_text.append(f"{name}@step{detail['step']}:pair{detail['pair']}")
            print(f"FAIL seed={result['seed']} {';'.join(result['failures'])} {';'.join(detail_text)}")
        raise SystemExit(1)

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
