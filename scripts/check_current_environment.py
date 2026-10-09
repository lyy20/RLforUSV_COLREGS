"""Smoke-check the current 600 m, 1 s USV environment configuration."""

from __future__ import annotations

import argparse
import configparser
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utilities import envs
from utilities.scenario_config import read_environment_config


def _reward_config(config: configparser.ConfigParser) -> dict:
    section = config["hyperparam"]
    return {
        "success": section.getfloat("REWARD_SUCCESS"),
        "collision": section.getfloat("REWARD_COLLISION"),
        "out_of_bounds": section.getfloat("REWARD_OUT_OF_BOUNDS"),
        "timeout": section.getfloat("REWARD_TIMEOUT"),
        "progress_weight": section.getfloat("REWARD_PROGRESS_WEIGHT"),
        "range_weight": section.getfloat("REWARD_RANGE_WEIGHT"),
        "safety_weight": section.getfloat("REWARD_SAFETY_WEIGHT"),
        "boundary_weight": section.getfloat("REWARD_BOUNDARY_WEIGHT"),
        "action_smooth_weight": section.getfloat("REWARD_ACTION_SMOOTH_WEIGHT"),
        "intervention_active_penalty": section.getfloat(
            "REWARD_INTERVENTION_ACTIVE_PENALTY"
        ),
        "intervention_magnitude_weight": section.getfloat(
            "REWARD_INTERVENTION_MAGNITUDE_WEIGHT"
        ),
        "time_penalty": section.getfloat("REWARD_TIME_PENALTY"),
        "safe_clearance": section.getfloat("REWARD_SAFE_CLEARANCE"),
        "boundary_safe_distance": section.getfloat("REWARD_BOUNDARY_SAFE_DISTANCE"),
    }


def make_environment(config: configparser.ConfigParser):
    section = config["hyperparam"]
    environment_config = read_environment_config(config)
    environment_config["max_episode_steps"] = section.getint("episode_length")
    return envs.make_env(
        section.get("SCENARIO"),
        num_agents=section.getint("num_agents"),
        num_landmarks=section.getint("num_landmarks"),
        num_obstacles=section.getint("num_obstacles"),
        num_static_obstacles=section.getint("num_static_obstacles"),
        map_half_size=section.getfloat("map_half_size"),
        static_obstacle_min_size=section.getfloat("static_obstacle_min_size"),
        static_obstacle_max_size=section.getfloat("static_obstacle_max_size"),
        ob_range=section.getfloat("ob_range"),
        num_ob=section.getint("num_ob"),
        num_static_ob_slots=section.getint("num_static_ob_slots"),
        landmark_depth=section.getfloat("landmark_depth"),
        landmark_movable=section.getboolean("landmark_movable"),
        obstacle_movable=section.getboolean("obstacle_movable"),
        landmark_vel=section.getfloat("landmark_vel"),
        max_vel=section.getfloat("max_vel"),
        random_vel=section.getboolean("random_vel"),
        movement=section.get("movement"),
        obstacle_movement=section.get("obstacle_movement"),
        pre_method=section.get("pre_method"),
        rew_err_th=section.getfloat("rew_err_th"),
        rew_dis_th=section.getfloat("rew_dis_th"),
        max_range=section.getfloat("max_range"),
        max_current_vel=section.getfloat("max_current_vel"),
        range_dropping=section.getfloat("range_dropping"),
        control_model=section.get("control_model"),
        target_observation_mode=section.get("target_observation_mode"),
        target_position_noise_std=section.getfloat("target_position_noise_std"),
        scenario_profile=section.get("SCENARIO_PROFILE"),
        dynamic_obstacle_min_speed=section.getfloat("dynamic_obstacle_min_speed"),
        dynamic_obstacle_max_speed=section.getfloat("dynamic_obstacle_max_speed"),
        reward_config=_reward_config(config),
        environment_config=environment_config,
        benchmark=True,
    )


def _minimum_hull_clearance(world) -> float:
    entities = (
        list(world.agents[: world.num_agents])
        + list(world.landmarks[: world.num_landmarks])
        + list(world.obstacles[: world.num_obstacles])
        + list(world.static_obstacles[: world.num_static_obstacles])
    )
    clearances = []
    for entity_i, entity in enumerate(entities):
        for other in entities[entity_i + 1 :]:
            distance = float(np.linalg.norm(entity.state.p_pos - other.state.p_pos))
            clearances.append(distance - float(entity.size) - float(other.size))
    return min(clearances)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        default="CTIPSAC_SMALL_STATIC_TARGET_1S_500M.txt",
    )
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = configparser.ConfigParser()
    if not config.read(config_path):
        raise FileNotFoundError(config_path)
    section = config["hyperparam"]
    expected_obs_dim = 7 + 7 * section.getint("num_ob") + 5 * section.getint(
        "num_static_ob_slots"
    )
    max_steps = section.getint("episode_length")

    env = make_environment(config)
    try:
        for seed in range(args.seeds):
            env.seed(seed)
            observations = env.reset()
            world = env.world
            assert len(observations) == section.getint("num_agents")
            assert len(observations[0]) == expected_obs_dim
            assert np.all(np.isfinite(observations[0]))
            assert np.isclose(world.dt, 1.0)
            assert np.isclose(world.usv_3dof_model.config.dt, 0.1)
            assert int(world._usv_3dof_substeps()) == 10
            assert len(world.obstacles[: world.num_obstacles]) == 2
            assert len(world.static_obstacles[: world.num_static_obstacles]) == 3
            assert all(0.010 <= obstacle.size <= 0.020 for obstacle in world.obstacles[:2])
            assert all(
                0.010 <= obstacle.size <= 0.020 for obstacle in world.static_obstacles[:3]
            )
            initial_clearance = _minimum_hull_clearance(world)
            assert initial_clearance > 0.0, initial_clearance

            terminated_at = max_steps
            for step_i in range(max_steps):
                action = np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
                next_observations, rewards, dones, info = env.step([action])
                assert np.all(np.isfinite(next_observations[0]))
                assert np.all(np.isfinite(rewards))
                assert np.all(np.isfinite(info))
                if bool(np.any(dones)):
                    terminated_at = step_i + 1
                    break

            print(
                "seed=%d obs_dim=%d initial_clearance_m=%.2f steps=%d agent_speed_mps=%.3f"
                % (
                    seed,
                    expected_obs_dim,
                    initial_clearance * 1000.0,
                    terminated_at,
                    np.linalg.norm(world.agents[0].state.p_vel) * 1000.0,
                )
            )
    finally:
        env.close()

    print("ENVIRONMENT_SMOKE_OK")


if __name__ == "__main__":
    main()
