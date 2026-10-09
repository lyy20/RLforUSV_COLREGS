"""Validate the 3-DOF/body-frame observation contract with known geometry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamics import USV3DOFState, body_to_world_vector, world_to_body_vector
from see_trained_once import make_eval_env, read_config
from utilities.observation_schema import (
    BASE_INDEX,
    BASE_OBS_DIM,
    DYNAMIC_OBS_FEATURE_DIM,
    STATIC_OBS_FEATURE_DIM,
    observation_dim,
)


def assert_close(name, actual, expected, atol=1.0e-6):
    if not np.allclose(actual, expected, rtol=0.0, atol=atol):
        raise AssertionError("%s mismatch: actual=%r expected=%r" % (name, actual, expected))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="SAC_3DOF_BODY_FRAME_SMOKE",
        help="Config stem or path used to construct the environment.",
    )
    args = parser.parse_args()

    config, _, _ = read_config(PROJECT_ROOT, args.config)
    env = make_eval_env(config, max_episode_steps=8)
    env.reset()
    world = env.world
    scenario = env.observation_callback.__self__
    scenario.target_position_noise_std = 0.0
    agent = world.agents[0]
    target = world.landmarks[0]
    dynamic = world.obstacles[0]
    static = world.static_obstacles[0]

    psi = np.pi / 2.0
    agent.state.p_pos = np.array([0.0, 0.0], dtype=float)
    agent.state.p_vel = np.zeros(2, dtype=float)
    agent.state.usv_3dof = USV3DOFState(
        x=0.0,
        y=0.0,
        psi=psi,
        u=1.0,
        v=0.2,
        r=np.deg2rad(5.0),
        thrust=20.0,
        rudder=np.deg2rad(10.0),
    )
    world.angle[0] = psi

    target_body_m = np.array([100.0, 20.0], dtype=float)
    target.state.p_pos = body_to_world_vector(psi, target_body_m) / world.usv_3dof_m_per_km
    target.state.p_vel = np.zeros(2, dtype=float)

    dynamic_body_m = np.array([50.0, -10.0], dtype=float)
    dynamic.state.p_pos = body_to_world_vector(psi, dynamic_body_m) / world.usv_3dof_m_per_km
    relative_velocity_body_mps = np.array([0.5, -0.2], dtype=float)
    agent_world_velocity_mps = body_to_world_vector(psi, [1.0, 0.2])
    dynamic_world_velocity_mps = agent_world_velocity_mps + body_to_world_vector(
        psi,
        relative_velocity_body_mps,
    )
    agent.state.p_vel = agent_world_velocity_mps / world.usv_3dof_m_per_km
    dynamic.state.p_vel = dynamic_world_velocity_mps / world.usv_3dof_m_per_km

    static_body_m = np.array([30.0, 40.0], dtype=float)
    static.state.p_pos = body_to_world_vector(psi, static_body_m) / world.usv_3dof_m_per_km
    static.state.p_vel = np.zeros(2, dtype=float)
    for entity in world.obstacles[1:world.num_obstacles]:
        entity.state.p_pos = np.array([0.25, 0.25], dtype=float)
        entity.state.p_vel = np.zeros(2, dtype=float)
    for entity in world.static_obstacles[1:world.num_static_obstacles]:
        entity.state.p_pos = np.array([-0.25, -0.25], dtype=float)
        entity.state.p_vel = np.zeros(2, dtype=float)

    observation = scenario.observation(agent, world)
    expected_dim = observation_dim(world.num_ob, world.num_static_ob_slots)
    if observation.shape != (expected_dim,):
        raise AssertionError(
            "Observation shape mismatch: actual=%r expected=(%d,)"
            % (observation.shape, expected_dim)
        )
    if not np.all(np.isfinite(observation)):
        raise AssertionError("Observation contains NaN or infinite values.")

    cfg = world.usv_3dof_model.config
    assert_close("surge_u", observation[BASE_INDEX["surge_u"]], 1.0 / cfg.max_surge_speed)
    assert_close("sway_v", observation[BASE_INDEX["sway_v"]], 0.2 / cfg.max_sway_speed)
    assert_close("yaw_rate_r", observation[BASE_INDEX["yaw_rate_r"]], np.deg2rad(5.0) / cfg.max_yaw_rate)
    assert_close("sin_heading", observation[BASE_INDEX["sin_heading"]], 1.0)
    assert_close("cos_heading", observation[BASE_INDEX["cos_heading"]], 0.0)
    assert_close("actual_rudder", observation[BASE_INDEX["actual_rudder"]], np.deg2rad(10.0) / cfg.max_rudder_angle)

    map_side_km = 2.0 * world.map_half_size
    assert_close(
        "target_body_xy",
        observation[BASE_INDEX["target_x_body"]:BASE_INDEX["target_y_body"] + 1],
        target_body_m / world.usv_3dof_m_per_km / map_side_km,
    )
    assert_close(
        "target_distance",
        observation[BASE_INDEX["target_distance"]],
        np.linalg.norm(target_body_m)
        / world.usv_3dof_m_per_km
        / (np.sqrt(2.0) * map_side_km),
    )

    dynamic_start = BASE_OBS_DIM
    assert_close(
        "dynamic_body_xy",
        observation[dynamic_start:dynamic_start + 2],
        dynamic_body_m / world.usv_3dof_m_per_km / world.ob_range,
    )
    assert_close(
        "dynamic_relative_velocity_body",
        observation[dynamic_start + 4:dynamic_start + 6],
        relative_velocity_body_mps
        / world.usv_3dof_m_per_km
        / scenario.observation_velocity_scale,
    )

    static_start = BASE_OBS_DIM + world.num_ob * DYNAMIC_OBS_FEATURE_DIM
    assert_close(
        "static_body_xy",
        observation[static_start:static_start + 2],
        static_body_m / world.usv_3dof_m_per_km / world.ob_range,
    )
    num_static_slots = int(getattr(world, 'num_static_ob_slots', 1))
    if static_start + num_static_slots * STATIC_OBS_FEATURE_DIM != expected_dim:
        raise AssertionError(
            "Static slot offset does not end at the observation boundary: "
            "expected_dim=%d, static_start=%d, num_static_slots=%d" % (expected_dim, static_start, num_static_slots)
        )

    probe = np.array([3.2, -1.7], dtype=float)
    assert_close(
        "rotation_inverse",
        world_to_body_vector(psi, body_to_world_vector(psi, probe)),
        probe,
        atol=1.0e-12,
    )

    print("body_frame_observation_check = PASS")
    print("observation_dim             = %d" % expected_dim)
    print("base_observation_dim        = %d" % BASE_OBS_DIM)
    print("target_body_normalized      = %s" % observation[6:9].tolist())
    print("dynamic_slot_0              = %s" % observation[dynamic_start:dynamic_start + 7].tolist())
    print("static_slot_0               = %s" % observation[static_start:static_start + 5].tolist())


if __name__ == "__main__":
    main()
