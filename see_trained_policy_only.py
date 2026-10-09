# -*- coding: utf-8 -*-
"""Policy-only visual evaluation for the clean USV MASAC project.

This script intentionally does not reuse the legacy viewer arrows. It records
the actual environment entities after each step and draws circles with long-tail
trajectories using matplotlib.

Unlike ``see_trained_once.py``, this evaluator disables only the trained USV's
COLREGs action-set filter. Dynamic obstacle vessels keep their normal scripted
COLREGs and collision-avoidance behavior, so the policy is evaluated in the
same physical scene without receiving rule-action assistance.

Terminal examples (run from the project directory):

    # One episode, last checkpoint, at most 200 environment steps.
    python see_trained_policy_only.py SAC_0815_1_STAGE_1_REWARD_ABLATION --checkpoint last --episodes 1 --max-steps 200

    # Five episodes with seeds 20..24; save PNG summaries but skip GIF output.
    python see_trained_policy_only.py SAC_0815_1_STAGE_1_REWARD_ABLATION --episodes 5 --seed 20 --no-save-gif

    # Hide the circular collision boundary and show only the environmental motif.
    python see_trained_policy_only.py SAC_0815_1_STAGE_1_REWARD_ABLATION --hide-collision-boundary

    # Best checkpoint on CUDA, custom output directory and GIF frame rate.
    python see_trained_policy_only.py SAC_0815_1_STAGE_1_REWARD_ABLATION --checkpoint best --device cuda --fps 15 --output-dir D:/USV/evaluation

Use ``python see_trained_policy_only.py --help`` to list every terminal option.
"""

from __future__ import annotations

import argparse
import configparser
import copy
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib

from see_trained_once import (
    DEFAULT_STATIC_OBSTACLE_VISUAL_RATIO,
    decode_step_info as decode_action_chain_info,
    draw_static_obstacle,
    format_static_obstacle_visual_ratio,
    parse_static_obstacle_visual_ratio,
    static_obstacle_ratio_arg,
    static_obstacle_theme_map,
)

from algorithms.sac.masac import MASAC
from evaluation.colregs_event_audit import COLREGsEventAudit
from evaluation.test_terminal_report import print_terminal_report
from utilities import envs
from utilities.paths import evaluation_output_dir, training_model_dir
from utilities.scenario_config import read_environment_config


ACTION_DIMS = {
    "heading_rate": 1,
    "usv_3dof": 2,
}


def torch_load_checkpoint(path: Path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def config_path_from_arg(project_root: Path, config_name: str) -> Path:
    raw = Path(config_name)
    if raw.suffix != ".txt":
        raw = raw.with_suffix(".txt")
    if raw.is_absolute():
        return raw
    return project_root / raw


def read_config(project_root: Path, config_name: str) -> tuple[configparser.ConfigParser, Path, str]:
    config_path = config_path_from_arg(project_root, config_name)
    config = configparser.ConfigParser()
    # UTF-8 是当前配置文件的统一编码；为兼容旧 GBK 配置保留回退读取。
    try:
        loaded_files = config.read(config_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        config = configparser.ConfigParser()
        loaded_files = config.read(config_path, encoding="gbk")
    if not loaded_files:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    return config, config_path, config_path.stem


def hp_get(config: configparser.ConfigParser, name: str, fallback=None):
    return config.get("hyperparam", name, fallback=fallback)


def hp_getint(config: configparser.ConfigParser, name: str, fallback=None) -> int:
    return config.getint("hyperparam", name, fallback=fallback)


def hp_getfloat(config: configparser.ConfigParser, name: str, fallback=None) -> float:
    return config.getfloat("hyperparam", name, fallback=fallback)


def hp_getbool(config: configparser.ConfigParser, name: str, fallback=None) -> bool:
    return config.getboolean("hyperparam", name, fallback=fallback)


def default_model_dir(config_stem: str) -> Path:
    return training_model_dir(config_stem)

def read_reward_config(config: configparser.ConfigParser) -> dict:
    """Read the same reward weights used by main.py for stage-matched evaluation."""
    return {
        "success": hp_getfloat(config, "REWARD_SUCCESS", fallback=100.0),
        "collision": hp_getfloat(config, "REWARD_COLLISION", fallback=120.0),
        "out_of_bounds": hp_getfloat(config, "REWARD_OUT_OF_BOUNDS", fallback=100.0),
        "timeout": hp_getfloat(config, "REWARD_TIMEOUT", fallback=25.0),
        "progress_weight": hp_getfloat(config, "REWARD_PROGRESS_WEIGHT", fallback=12.0),
        "range_weight": hp_getfloat(config, "REWARD_RANGE_WEIGHT", fallback=0.5),
        "safety_weight": hp_getfloat(config, "REWARD_SAFETY_WEIGHT", fallback=20.0),
        "boundary_weight": hp_getfloat(config, "REWARD_BOUNDARY_WEIGHT", fallback=8.0),
        "action_smooth_weight": hp_getfloat(
            config,
            "REWARD_ACTION_SMOOTH_WEIGHT",
            fallback=0.02,
        ),
        "intervention_active_penalty": hp_getfloat(
            config,
            "REWARD_INTERVENTION_ACTIVE_PENALTY",
            fallback=0.05,
        ),
        "intervention_magnitude_weight": hp_getfloat(
            config,
            "REWARD_INTERVENTION_MAGNITUDE_WEIGHT",
            fallback=0.25,
        ),
        "time_penalty": hp_getfloat(config, "REWARD_TIME_PENALTY", fallback=0.05),
        "safe_clearance": hp_getfloat(config, "REWARD_SAFE_CLEARANCE", fallback=0.15),
        "boundary_safe_distance": hp_getfloat(
            config,
            "REWARD_BOUNDARY_SAFE_DISTANCE",
            fallback=0.25,
        ),
    }


def validate_checkpoint_metadata(
    config: configparser.ConfigParser,
    model_dir: Path,
    obs_dim: int,
    action_dim: int,
    target_observation_mode: str,
) -> None:
    state_path = model_dir / "training_state_last.pt"
    if not state_path.exists():
        return
    state = torch_load_checkpoint(state_path, map_location="cpu")
    if not isinstance(state, dict):
        return

    saved_obs_dim = state.get("obs_dim")
    if saved_obs_dim is not None and int(saved_obs_dim) != int(obs_dim):
        raise ValueError(
            "Checkpoint obs_dim mismatch: saved %d, current %d. "
            "Please evaluate with a config that keeps the same observation/network contract as the checkpoint."
            % (int(saved_obs_dim), int(obs_dim))
        )

    saved_action_dim = state.get("action_dim")
    if saved_action_dim is not None and int(saved_action_dim) != int(action_dim):
        raise ValueError(
            "Checkpoint action_dim mismatch: saved %d, current %d."
            % (int(saved_action_dim), int(action_dim))
        )

    saved_contract = state.get("network_contract")
    if isinstance(saved_contract, dict):
        current_contract = {
            "policy_algorithm": hp_get(config, "DNN"),
            "control_model": hp_get(config, "control_model", fallback="usv_3dof"),
            "num_agents": hp_getint(config, "num_agents"),
            "num_landmarks": hp_getint(config, "num_landmarks"),
            "dynamic_observation_slots": hp_getint(config, "num_ob"),
            "static_observation_slots": hp_getint(
                config, "num_static_ob_slots", fallback=1
            ),
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "hidden_dim_1": hp_getint(config, "DIM_1"),
            "hidden_dim_2": hp_getint(config, "DIM_2"),
            "rnn": hp_getbool(config, "RNN"),
            "history_length": hp_getint(config, "HISTORY_LENGTH"),
            "automatic_entropy_tuning": hp_getbool(config, "AUTOMATIC_ENTROPY"),
            "entity_encoder_enabled": hp_getbool(
                config, "ENTITY_ENCODER_ENABLED", fallback=False
            ),
            "entity_encoder_latent_dim": hp_getint(
                config, "ENTITY_ENCODER_LATENT_DIM", fallback=32
            ),
            "ctip_active": hp_getbool(config, "CTIP_active", fallback=False),
            "icm_active": hp_getbool(config, "ICM", fallback=False),
        }
        contract_mismatches = []
        for key, current_value in current_contract.items():
            saved_value = saved_contract.get(key)
            if saved_value is not None and saved_value != current_value:
                contract_mismatches.append(
                    "%s: saved=%r current=%r" % (key, saved_value, current_value)
                )
        if contract_mismatches:
            raise ValueError(
                "Checkpoint network contract does not match the test config: "
                + "; ".join(contract_mismatches)
            )

    saved_mode = state.get("target_observation_mode")
    if saved_mode is not None and str(saved_mode).lower() != str(target_observation_mode).lower():
        raise ValueError(
            "Checkpoint target_observation_mode mismatch: saved=%s current=%s."
            % (saved_mode, target_observation_mode)
        )

    expected_values = {
        "scenario": hp_get(config, "SCENARIO"),
        "scenario_profile": hp_get(config, "SCENARIO_PROFILE", fallback="DEFAULT"),
        "curriculum_stage": hp_getint(config, "CURRICULUM_STAGE", fallback=0),
        "control_model": hp_get(config, "control_model", fallback="usv_3dof"),
        "num_agents": hp_getint(config, "num_agents"),
        "num_landmarks": hp_getint(config, "num_landmarks"),
        "num_obstacles": hp_getint(config, "num_obstacles"),
        "num_static_obstacles": hp_getint(config, "num_static_obstacles", fallback=0),
        "num_ob": hp_getint(config, "num_ob"),
        "num_static_ob_slots": hp_getint(config, "num_static_ob_slots", fallback=1),
    }
    for key, current_value in expected_values.items():
        saved_value = state.get(key)
        if saved_value is not None and saved_value != current_value:
            raise ValueError(
                "Checkpoint metadata mismatch for %s: saved=%r current=%r."
                % (key, saved_value, current_value)
            )

    expected_floats = {
        "map_half_size": hp_getfloat(config, "map_half_size", fallback=2.0),
        "ob_range": hp_getfloat(config, "ob_range"),
        "static_obstacle_min_size": hp_getfloat(
            config, "static_obstacle_min_size", fallback=0.04
        ),
        "static_obstacle_max_size": hp_getfloat(
            config, "static_obstacle_max_size", fallback=0.10
        ),
    }
    optional_float_fields = (
        "dynamic_obstacle_min_speed",
        "dynamic_obstacle_max_speed",
    )
    for key in optional_float_fields:
        if config.has_option("hyperparam", key):
            expected_floats[key] = hp_getfloat(config, key)
    for key, current_value in expected_floats.items():
        saved_value = state.get(key)
        if saved_value is not None and not np.isclose(
            float(saved_value), float(current_value), rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                "Checkpoint metadata mismatch for %s: saved=%r current=%r."
                % (key, saved_value, current_value)
            )

    current_environment = read_environment_config(config)
    current_environment["max_episode_steps"] = hp_getint(config, "episode_length")
    saved_environment = state.get("environment_config")
    if isinstance(saved_environment, dict):
        mismatches = []
        for key, current_value in current_environment.items():
            saved_value = saved_environment.get(key)
            if saved_value is None:
                continue
            if isinstance(current_value, bool):
                matches = bool(saved_value) == current_value
            elif isinstance(current_value, (int, float)):
                matches = np.isclose(
                    float(saved_value), float(current_value), rtol=0.0, atol=1.0e-12
                )
            else:
                matches = saved_value == current_value
            if not matches:
                mismatches.append("%s: saved=%r current=%r" % (key, saved_value, current_value))
        if mismatches:
            raise ValueError(
                "Checkpoint environment configuration does not match the test config: "
                + "; ".join(mismatches)
            )


def make_eval_env(
    config: configparser.ConfigParser,
    max_episode_steps: int | None = None,
):
    environment_config = read_environment_config(config)
    environment_config["max_episode_steps"] = int(
        max_episode_steps
        if max_episode_steps is not None
        else hp_getint(config, "episode_length")
    )
    return envs.make_env(
        hp_get(config, "SCENARIO"),
        num_agents=hp_getint(config, "num_agents"),
        num_landmarks=hp_getint(config, "num_landmarks"),
        num_obstacles=hp_getint(config, "num_obstacles"),
        num_static_obstacles=hp_getint(config, "num_static_obstacles", fallback=0),
        map_half_size=hp_getfloat(config, "map_half_size", fallback=2.0),
        static_obstacle_min_size=hp_getfloat(config, "static_obstacle_min_size", fallback=0.04),
        static_obstacle_max_size=hp_getfloat(config, "static_obstacle_max_size", fallback=0.10),
        ob_range=hp_getfloat(config, "ob_range"),
        num_ob=hp_getint(config, "num_ob"),
        num_static_ob_slots=hp_getint(config, "num_static_ob_slots", fallback=1),
        landmark_depth=hp_getfloat(config, "landmark_depth"),
        landmark_movable=hp_getbool(config, "landmark_movable"),
        obstacle_movable=hp_getbool(config, "obstacle_movable"),
        landmark_vel=hp_getfloat(config, "landmark_vel"),
        max_vel=hp_getfloat(config, "max_vel", fallback=0.0),
        random_vel=hp_getbool(config, "random_vel", fallback=False),
        movement=hp_get(config, "movement"),
        obstacle_movement=hp_get(config, "obstacle_movement"),
        pre_method=hp_get(config, "pre_method", fallback="DIRECT"),
        rew_err_th=hp_getfloat(config, "rew_err_th"),
        rew_dis_th=hp_getfloat(config, "rew_dis_th"),
        max_range=hp_getfloat(config, "max_range"),
        max_current_vel=hp_getfloat(config, "max_current_vel"),
        range_dropping=hp_getfloat(config, "range_dropping"),
        control_model=hp_get(config, "control_model", fallback="usv_3dof"),
        target_observation_mode=hp_get(config, "target_observation_mode", fallback="direct_noisy"),
        target_position_noise_std=hp_getfloat(config, "target_position_noise_std", fallback=0.005),
        scenario_profile=hp_get(config, "SCENARIO_PROFILE", fallback="DEFAULT"),
        dynamic_obstacle_min_speed=(
            hp_getfloat(config, "dynamic_obstacle_min_speed")
            if config.has_option("hyperparam", "dynamic_obstacle_min_speed")
            else None
        ),
        dynamic_obstacle_max_speed=(
            hp_getfloat(config, "dynamic_obstacle_max_speed")
            if config.has_option("hyperparam", "dynamic_obstacle_max_speed")
            else None
        ),
        reward_config=read_reward_config(config),
        environment_config=environment_config,
        benchmark=True,
    )


def disable_agent_colregs_intervention(env) -> None:
    """Disable rule-action replacement for the learned USV policy only."""
    world = env.world
    world.agent_colregs_action_filter_enabled = False
    for agent in world.agents:
        agent.action_filter_active = False
        agent.action_filter_mode = "policy_only"
        agent.action_filter_decision = None


def validate_runtime_environment(env, config: configparser.ConfigParser) -> None:
    """Verify that reset produced the physical scale requested by the config."""
    world = env.world
    environment_config = read_environment_config(config)
    expected_counts = {
        "agents": hp_getint(config, "num_agents"),
        "targets": hp_getint(config, "num_landmarks"),
        "dynamic obstacles": hp_getint(config, "num_obstacles"),
        "static obstacles": hp_getint(config, "num_static_obstacles", fallback=0),
    }
    actual_counts = {
        "agents": len(world.agents[: world.num_agents]),
        "targets": len(world.landmarks[: world.num_landmarks]),
        "dynamic obstacles": len(world.obstacles[: world.num_obstacles]),
        "static obstacles": len(world.static_obstacles[: world.num_static_obstacles]),
    }
    for name, expected in expected_counts.items():
        if actual_counts[name] != expected:
            raise ValueError(
                "Runtime environment %s count mismatch: expected=%d actual=%d."
                % (name, expected, actual_counts[name])
            )

    expected_map_half_size = hp_getfloat(config, "map_half_size", fallback=2.0)
    if not np.isclose(float(world.map_half_size), expected_map_half_size):
        raise ValueError(
            "Runtime map_half_size mismatch: expected=%r actual=%r."
            % (expected_map_half_size, world.map_half_size)
        )

    radius_checks = (
        ("agent", world.agents[: world.num_agents], environment_config["agent_radius"], environment_config["agent_radius"]),
        ("target", world.landmarks[: world.num_landmarks], environment_config["target_radius"], environment_config["target_radius"]),
        (
            "dynamic obstacle",
            world.obstacles[: world.num_obstacles],
            environment_config["dynamic_obstacle_min_size"],
            environment_config["dynamic_obstacle_max_size"],
        ),
        (
            "static obstacle",
            world.static_obstacles[: world.num_static_obstacles],
            hp_getfloat(config, "static_obstacle_min_size", fallback=0.04),
            hp_getfloat(config, "static_obstacle_max_size", fallback=0.10),
        ),
    )
    for entity_type, entities, minimum, maximum in radius_checks:
        for entity in entities:
            radius = float(entity.size)
            if radius < float(minimum) - 1.0e-12 or radius > float(maximum) + 1.0e-12:
                raise ValueError(
                    "Runtime %s radius is outside the configured range: "
                    "%s radius=%r expected=[%r, %r]."
                    % (entity_type, entity.name, radius, minimum, maximum)
                )

    if config.has_option("hyperparam", "dynamic_obstacle_min_speed"):
        minimum_speed = hp_getfloat(config, "dynamic_obstacle_min_speed")
        maximum_speed = hp_getfloat(config, "dynamic_obstacle_max_speed")
        for obstacle in world.obstacles[: world.num_obstacles]:
            command_speed = float(obstacle.obstacle_vel)
            if (
                command_speed < minimum_speed - 1.0e-12
                or command_speed > maximum_speed + 1.0e-12
            ):
                raise ValueError(
                    "Runtime dynamic obstacle speed command is outside the configured range: "
                    "%s speed=%r expected=[%r, %r]."
                    % (obstacle.name, command_speed, minimum_speed, maximum_speed)
                )

    expected_dt = environment_config["environment_dt"]
    expected_substeps = environment_config["usv_substeps"]
    actual_substeps = int(world._usv_3dof_substeps())
    if not np.isclose(float(world.dt), float(expected_dt)) or actual_substeps != expected_substeps:
        raise ValueError(
            "Runtime time-scale mismatch: dt=%r substeps=%r, expected dt=%r substeps=%r."
            % (world.dt, actual_substeps, expected_dt, expected_substeps)
        )


def runtime_environment_summary(env) -> dict:
    world = env.world
    dynamic = list(world.obstacles[: world.num_obstacles])
    static = list(world.static_obstacles[: world.num_static_obstacles])
    return {
        "map_size_m": float(world.map_half_size) * 2000.0,
        "environment_dt_s": float(world.dt),
        "usv_substeps": int(world._usv_3dof_substeps()),
        "agent_radii_m": [float(entity.size) * 1000.0 for entity in world.agents[: world.num_agents]],
        "target_radii_m": [float(entity.size) * 1000.0 for entity in world.landmarks[: world.num_landmarks]],
        "dynamic_radii_m": [float(entity.size) * 1000.0 for entity in dynamic],
        "static_radii_m": [float(entity.size) * 1000.0 for entity in static],
        "dynamic_speeds_mps": [float(np.linalg.norm(entity.state.p_vel)) * 1000.0 for entity in dynamic],
    }


def make_model(config: configparser.ConfigParser, action_dim: int, device: str) -> MASAC:
    dnn = hp_get(config, "DNN")
    if dnn != "MASAC":
        raise ValueError("This clean evaluator currently supports MASAC only, got %s." % dnn)
    return MASAC(
        num_agents=hp_getint(config, "num_agents"),
        num_landmarks=hp_getint(config, "num_landmarks"),
        landmark_depth=hp_getfloat(config, "landmark_depth"),
        num_ob=hp_getint(config, "num_ob"),
        discount_factor=hp_getfloat(config, "GAMMA"),
        tau=hp_getfloat(config, "TAU"),
        lr_actor=hp_getfloat(config, "LR_ACTOR"),
        lr_critic=hp_getfloat(config, "LR_CRITIC"),
        weight_decay=hp_getfloat(config, "WEIGHT_DECAY"),
        device=device,
        rnn=hp_getbool(config, "RNN"),
        alpha=hp_getfloat(config, "ALPHA"),
        automatic_entropy_tuning=hp_getbool(config, "AUTOMATIC_ENTROPY"),
        dim_1=hp_getint(config, "DIM_1"),
        dim_2=hp_getint(config, "DIM_2"),
        CTIP_active=hp_getbool(config, "CTIP_active", fallback=False),
        action_dim=action_dim,
        num_static_ob_slots=hp_getint(config, "num_static_ob_slots", fallback=1),
        entity_encoder_enabled=hp_getbool(
            config, "ENTITY_ENCODER_ENABLED", fallback=False
        ),
        entity_encoder_latent_dim=hp_getint(
            config, "ENTITY_ENCODER_LATENT_DIM", fallback=32
        ),
    )


def load_model_weights(
    model: MASAC,
    model_dir: Path,
    checkpoint_kind: str,
    automatic_entropy: bool,
    ctip_active: bool,
    device: str,
) -> Path:
    checkpoint_path = model_dir / ("episode_%s.pt" % checkpoint_kind)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch_load_checkpoint(checkpoint_path, map_location=device)
    if not isinstance(payload, (list, tuple)):
        raise ValueError(f"Unexpected checkpoint payload in {checkpoint_path}")

    for agent_i, sac_agent in enumerate(model.masac_agent):
        agent_state = payload[agent_i] if agent_i < len(payload) else payload[0]
        sac_agent.actor.load_state_dict(agent_state["actor_params"])
        if "critic_params" in agent_state:
            sac_agent.critic.load_state_dict(agent_state["critic_params"])
        if "target_critic_params" in agent_state:
            sac_agent.target_critic.load_state_dict(agent_state["target_critic_params"])
        if ctip_active and "icm_state_dict" in agent_state:
            sac_agent.icm.load_state_dict(agent_state["icm_state_dict"])
        sac_agent.actor.eval()
        sac_agent.critic.eval()
        sac_agent.target_critic.eval()

    if automatic_entropy:
        entropy_path = model_dir / ("episode_target_entropy_%s.file" % checkpoint_kind)
        log_alpha_path = model_dir / ("episode_log_alpha_%s.file" % checkpoint_kind)
        alpha_path = model_dir / ("episode_alpha_%s.file" % checkpoint_kind)
        if entropy_path.exists() and log_alpha_path.exists() and alpha_path.exists():
            with entropy_path.open("rb") as f:
                target_entropy_aux = pickle.load(f)
            with log_alpha_path.open("rb") as f:
                log_alpha_aux = pickle.load(f)
            with alpha_path.open("rb") as f:
                alpha_aux = pickle.load(f)
            for agent_i, sac_agent in enumerate(model.masac_agent):
                source_i = agent_i if agent_i < len(target_entropy_aux) else 0
                sac_agent.target_entropy = target_entropy_aux[source_i]
                sac_agent.log_alpha = log_alpha_aux[source_i]
                sac_agent.alpha = alpha_aux[source_i]

    return checkpoint_path


def entity_snapshot(entity) -> dict:
    pos = np.asarray(entity.state.p_pos, dtype=float)
    vel = np.asarray(entity.state.p_vel, dtype=float) if entity.state.p_vel is not None else np.zeros(2)
    return {
        "name": entity.name,
        "pos": pos.copy(),
        "vel": vel.copy(),
        "speed": float(np.linalg.norm(vel)),
        "size": float(entity.size),
        "color": np.asarray(entity.color if entity.color is not None else [0.2, 0.2, 0.2], dtype=float).copy(),
    }


def collect_snapshot(env) -> dict:
    world = env.world
    return {
        "agents": [entity_snapshot(entity) for entity in world.agents[: world.num_agents]],
        "targets": [entity_snapshot(entity) for entity in world.landmarks[: world.num_landmarks]],
        "dynamic_obstacles": [
            entity_snapshot(entity) for entity in world.obstacles[: world.num_obstacles]
        ],
        "static_obstacles": [
            entity_snapshot(entity)
            for entity in getattr(world, "static_obstacles", [])[: getattr(world, "num_static_obstacles", 0)]
        ],
    }


def append_history(history: dict, snapshot: dict) -> None:
    for group, entities in snapshot.items():
        history.setdefault(group, [])
        while len(history[group]) < len(entities):
            history[group].append({"name": "", "positions": [], "size": 0.0, "color": np.zeros(3)})
        for idx, entity in enumerate(entities):
            history[group][idx]["name"] = entity["name"]
            history[group][idx]["positions"].append(entity["pos"])
            history[group][idx]["size"] = entity["size"]
            history[group][idx]["color"] = entity["color"]


def make_policy_inputs(obs_n, history_obs, history_act, device: str):
    his_all = []
    obs_all = []
    for agent_i, obs in enumerate(obs_n):
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        his_arr = np.concatenate([history_obs[agent_i], history_act[agent_i]], axis=1)
        his_all.append(torch.as_tensor(his_arr.reshape(1, his_arr.shape[0], his_arr.shape[1]), dtype=torch.float32, device=device))
        obs_all.append(torch.as_tensor(obs_arr, dtype=torch.float32, device=device))
    return his_all, obs_all


def update_history(history_obs, history_act, obs_n, applied_actions) -> None:
    for agent_i, obs in enumerate(obs_n):
        history_obs[agent_i] = np.roll(history_obs[agent_i], shift=-1, axis=0)
        history_obs[agent_i, -1] = np.asarray(obs, dtype=float)
        history_act[agent_i] = np.roll(history_act[agent_i], shift=-1, axis=0)
        history_act[agent_i, -1] = np.asarray(applied_actions[agent_i], dtype=float)


def decode_step_info(info, num_agents: int, num_landmarks: int, fallback_actions: np.ndarray) -> dict:
    """复用统一动作链解码，避免策略独立测试与训练端语义漂移。

    四层动作的含义固定为 ``raw -> constrained -> rule -> smoothed``，其中
    ``applied_action`` 是最终平滑执行动作的兼容字段。策略独立测试虽然关闭了
    智能体的 COLREGs 接管，也仍可能经过基础边界约束和执行平滑，因此不能只读取
    ``applied_action`` 后丢失中间动作层。
    """
    return decode_action_chain_info(
        info,
        num_agents,
        num_landmarks,
        fallback_actions,
    )


def run_episode(
    env,
    model: MASAC,
    history_length: int,
    action_dim: int,
    max_steps: int,
    device: str,
    rnn_active: bool,
    episode_id: int = 1,
) -> dict:
    obs_n = env.reset()
    # Resetting a scene must never re-enable assistance for this evaluator.
    disable_agent_colregs_intervention(env)
    num_agents = len(obs_n)
    num_landmarks = int(getattr(env.world, "num_landmarks", 1))
    obs_dim = len(obs_n[0])
    history_obs = np.zeros((len(obs_n), history_length, obs_dim), dtype=float)
    history_act = np.zeros((len(obs_n), history_length, action_dim), dtype=float)
    trajectory = {}
    rewards = []
    filter_active = []
    correction_norms = []
    # 与训练端 TensorBoard 四个动作链标签同义。即使 policy-only 关闭规则
    # 接管，也记录约束层和平滑层，便于区分策略本身与执行器处理造成的差异。
    action_chain_norms = {
        "raw_to_constrained_norm": [],
        "constrained_to_rule_norm": [],
        "rule_to_smoothed_norm": [],
        "raw_to_smoothed_norm": [],
    }
    done_reason = "none"

    # 仅观察的规则审计：policy-only 的规则内化指标必须使用策略原始动作，
    # 不能把任何安全过滤器动作误记为策略已经学会。
    colregs_audit = COLREGsEventAudit(
        env.world,
        episode_id=episode_id,
        variant_name="policy_only",
    )

    append_history(trajectory, collect_snapshot(env))

    for step_i in range(max_steps):
        his_all, obs_all = make_policy_inputs(obs_n, history_obs, history_act, device)
        with torch.no_grad():
            action_tensors = model.act(his_all, obs_all, noise=0.0)
        action_array = torch.stack(action_tensors).detach().cpu().numpy()[:, 0, :]
        colregs_reports = colregs_audit.reports_for_current_state()
        next_obs, reward_n, done_n, info = env.step([action_array[i] for i in range(action_array.shape[0])])

        step_info = decode_step_info(info, num_agents, num_landmarks, action_array)
        colregs_audit.record_step(
            step=step_i + 1,
            # 以环境回传的 raw 动作作为审计输入，确保日志与训练时保存的
            # 原始策略动作完全一致，而不是仅依赖测试端的候选数组。
            policy_action=step_info["raw_actions"][0],
            constrained_action=step_info["constrained_actions"][0],
            rule_action=step_info["rule_actions"][0],
            smoothed_action=step_info["smoothed_actions"][0],
            applied_action=step_info["applied_actions"][0],
            filter_active=step_info["filter_active"][0],
            filter_mode=getattr(env.world.agents[0], "action_filter_mode", ""),
            done_reason=step_info["done_reason"][0],
            reports=colregs_reports,
        )
        applied_actions = step_info["applied_actions"]
        # Match main.py exactly: history is only advanced when the recurrent
        # network path is enabled. With RNN=False the actor was trained with a
        # zero history tensor at every environment step.
        if rnn_active:
            update_history(history_obs, history_act, obs_n, applied_actions)
        obs_n = next_obs

        append_history(trajectory, collect_snapshot(env))
        rewards.append(float(np.sum(reward_n)))
        filter_active.extend([bool(v) for v in step_info["filter_active"]])
        correction_norms.extend([float(v) for v in step_info["action_correction_norm"]])
        for key, values in step_info["action_chain_norms"].items():
            # 固定数组与旧字典均由统一解码器归一为每个 agent 一个 L2 范数。
            action_chain_norms[key].extend(
                [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]
            )
        reasons = [r for r in step_info["done_reason"] if r and r != "none"]
        if reasons:
            done_reason = reasons[0]
        if bool(np.any(done_n)):
            break

    colregs_event_rows = colregs_audit.finalize(
        final_step=len(rewards),
        done_reason=done_reason,
    )
    result = {
        "trajectory": trajectory,
        "steps": len(rewards),
        "reward_sum": float(np.sum(rewards)) if rewards else 0.0,
        "done_reason": done_reason,
        "filter_active_rate": float(np.mean(filter_active)) if filter_active else 0.0,
        # 兼容旧结果字段；其当前定义是 constrained -> rule 的纯规则修正量。
        "mean_action_correction": float(np.mean(correction_norms)) if correction_norms else 0.0,
        "action_chain_mean_norms": {
            key: float(np.mean(values)) if values else 0.0
            for key, values in action_chain_norms.items()
        },
        "colregs_step_rows": colregs_audit.step_rows,
        "colregs_event_rows": colregs_event_rows,
    }
    result["colregs_episode_summary"] = colregs_audit.episode_summary("policy_only")
    return result


def positions_array(track: dict) -> np.ndarray:
    if not track["positions"]:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(track["positions"], dtype=float)


def trajectory_motion_summary(result: dict, environment_dt: float) -> dict:
    """Summarize frame-to-frame motion and expose discontinuities directly."""
    dt = max(float(environment_dt), 1.0e-12)
    summary = {}
    for group in ("agents", "targets", "dynamic_obstacles"):
        group_rows = []
        for track in result["trajectory"].get(group, []):
            positions = positions_array(track)
            if positions.shape[0] <= 1:
                maximum_step_km = 0.0
                mean_step_km = 0.0
            else:
                step_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                maximum_step_km = float(np.max(step_distances))
                mean_step_km = float(np.mean(step_distances))
            group_rows.append(
                {
                    "name": track["name"],
                    "max_step_m": maximum_step_km * 1000.0,
                    "mean_step_m": mean_step_km * 1000.0,
                    "max_equivalent_speed_mps": maximum_step_km * 1000.0 / dt,
                }
            )
        summary[group] = group_rows
    return summary


def plot_units(env) -> tuple[float, str]:
    map_width_km = float(getattr(env.world, "map_half_size", 2.0)) * 2.0
    return (1000.0, "m") if map_width_km <= 2.0 + 1.0e-12 else (1.0, "km")


def plot_track(
    ax,
    track: dict,
    label: str,
    color,
    linewidth: float,
    alpha: float,
    coordinate_scale: float,
) -> None:
    positions = positions_array(track) * coordinate_scale
    if positions.shape[0] == 0:
        return
    if positions.shape[0] > 1:
        ax.plot(positions[:, 0], positions[:, 1], color=color, linewidth=linewidth, alpha=alpha)
    ax.scatter(
        positions[0, 0],
        positions[0, 1],
        s=24,
        color=color,
        marker="x",
        alpha=0.7,
        zorder=5,
    )
    circle = matplotlib.patches.Circle(
        positions[-1],
        radius=float(track["size"]) * coordinate_scale,
        facecolor=color,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
        zorder=6,
        label=label,
    )
    ax.add_patch(circle)
    ax.text(
        positions[-1, 0],
        positions[-1, 1] + float(track["size"]) * coordinate_scale * 1.25,
        label,
        ha="center",
        va="bottom",
        fontsize=8,
    )


def draw_trajectory(
    result: dict,
    env,
    output_path: Path,
    title: str,
    show_collision_boundary: bool = True,
    static_obstacle_visual_ratio=DEFAULT_STATIC_OBSTACLE_VISUAL_RATIO,
) -> None:
    import matplotlib.pyplot as plt

    world = env.world
    half_size = float(getattr(world, "map_half_size", 2.0))
    coordinate_scale, coordinate_unit = plot_units(env)
    display_half_size = half_size * coordinate_scale
    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-display_half_size, display_half_size)
    ax.set_ylim(-display_half_size, display_half_size)
    ax.set_xlabel("X position (%s)" % coordinate_unit)
    ax.set_ylabel("Y position (%s)" % coordinate_unit)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    boundary = matplotlib.patches.Rectangle(
        (-display_half_size, -display_half_size),
        2.0 * display_half_size,
        2.0 * display_half_size,
        fill=False,
        edgecolor="black",
        linewidth=1.0,
        alpha=0.8,
    )
    ax.add_patch(boundary)

    trajectory = result["trajectory"]
    for idx, track in enumerate(trajectory.get("agents", [])):
        plot_track(
            ax, track, "USV %d" % idx, "#1f77b4", 2.2, 0.95, coordinate_scale
        )
    for idx, track in enumerate(trajectory.get("targets", [])):
        plot_track(
            ax, track, "Target %d" % idx, "#d62728", 2.2, 0.9, coordinate_scale
        )
    for idx, track in enumerate(trajectory.get("dynamic_obstacles", [])):
        plot_track(
            ax, track, "Dyn %d" % idx, "#2ca02c", 1.3, 0.65, coordinate_scale
        )

    static_tracks = trajectory.get("static_obstacles", [])
    static_theme_map = static_obstacle_theme_map(
        static_tracks,
        ratio=static_obstacle_visual_ratio,
    )
    for idx, track in enumerate(static_tracks):
        positions = positions_array(track)
        if positions.shape[0] == 0:
            continue
        draw_static_obstacle(
            ax,
            positions[-1] * coordinate_scale,
            float(track["size"]) * coordinate_scale,
            idx,
            label="Static obstacle (circular collision model)" if idx == 0 else None,
            show_collision_boundary=show_collision_boundary,
            visual_theme=static_theme_map.get(idx),
        )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_frame(
    ax,
    result: dict,
    env,
    frame_index: int,
    title: str,
    show_collision_boundary: bool = True,
    static_obstacle_visual_ratio=DEFAULT_STATIC_OBSTACLE_VISUAL_RATIO,
) -> None:
    world = env.world
    half_size = float(getattr(world, "map_half_size", 2.0))
    coordinate_scale, coordinate_unit = plot_units(env)
    display_half_size = half_size * coordinate_scale
    ax.clear()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-display_half_size, display_half_size)
    ax.set_ylim(-display_half_size, display_half_size)
    ax.set_xlabel("X position (%s)" % coordinate_unit)
    ax.set_ylabel("Y position (%s)" % coordinate_unit)
    ax.set_title("%s | step %d" % (title, frame_index))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (-display_half_size, -display_half_size),
            2.0 * display_half_size,
            2.0 * display_half_size,
            fill=False,
            edgecolor="black",
            linewidth=1.0,
            alpha=0.8,
        )
    )

    trajectory = result["trajectory"]
    groups = [
        ("agents", "USV", "#1f77b4", 2.2, 0.95),
        ("targets", "Target", "#d62728", 2.2, 0.9),
        ("dynamic_obstacles", "Dyn", "#2ca02c", 1.3, 0.65),
    ]
    for group, label_prefix, color, linewidth, alpha in groups:
        for idx, full_track in enumerate(trajectory.get(group, [])):
            clipped = copy.deepcopy(full_track)
            clipped["positions"] = full_track["positions"][: frame_index + 1]
            plot_track(
                ax,
                clipped,
                "%s %d" % (label_prefix, idx),
                color,
                linewidth,
                alpha,
                coordinate_scale,
            )

    static_tracks = trajectory.get("static_obstacles", [])
    static_theme_map = static_obstacle_theme_map(
        static_tracks,
        ratio=static_obstacle_visual_ratio,
    )
    for idx, track in enumerate(static_tracks):
        positions = positions_array(track)
        if positions.shape[0] == 0:
            continue
        draw_static_obstacle(
            ax,
            positions[0] * coordinate_scale,
            float(track["size"]) * coordinate_scale,
            idx,
            show_collision_boundary=show_collision_boundary,
            visual_theme=static_theme_map.get(idx),
        )


def save_gif(
    result: dict,
    env,
    output_path: Path,
    title: str,
    fps: int,
    show_collision_boundary: bool = True,
    static_obstacle_visual_ratio=DEFAULT_STATIC_OBSTACLE_VISUAL_RATIO,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image

    frame_count = result["steps"] + 1
    fig, ax = plt.subplots(figsize=(8, 8), dpi=110)
    canvas = FigureCanvasAgg(fig)
    frames = []
    stride = max(1, frame_count // 180)
    for frame_index in range(0, frame_count, stride):
        draw_frame(
            ax,
            result,
            env,
            frame_index,
            title,
            show_collision_boundary=show_collision_boundary,
            static_obstacle_visual_ratio=static_obstacle_visual_ratio,
        )
        canvas.draw()
        width, height = canvas.get_width_height()
        buffer = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        frames.append(buffer.reshape(height, width, 3).copy())
    plt.close(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=max(int(1000 / max(fps, 1)), 1),
        loop=0,
    )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available to PyTorch.")
    return requested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained USV policy without COLREGs action-filter assistance."
    )
    parser.add_argument("config", nargs="?", default="CTIPSAC_T", help="Config file stem or path, e.g. CTIPSAC_T")
    parser.add_argument("--checkpoint", choices=("last", "best"), default="last", help="Which saved policy to load.")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Override model_dir. Defaults to D:/USV/logs/<TRAINING_RUN_NAME>/model_dir.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for PNG/GIF outputs. Defaults to D:/USV/logs/<TRAINING_RUN_NAME>/evaluation.",
    )
    parser.add_argument(
        "--episodes",
        "--max-episodes",
        dest="episodes",
        type=positive_int,
        default=1,
        help="Number of independent test episodes. Default: 1.",
    )
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=None,
        help="Maximum environment steps per episode; overrides config episode_length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base test seed. Episode i uses seed + i.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. Default: auto.",
    )
    parser.add_argument(
        "--save-gif",
        dest="save_gif",
        action="store_true",
        default=True,
        help="Save an animated GIF of the circle/trail view. This is enabled by default.",
    )
    parser.add_argument(
        "--no-save-gif",
        dest="save_gif",
        action="store_false",
        help="Skip GIF output and only save the summary PNG.",
    )
    parser.add_argument("--fps", type=positive_int, default=12, help="GIF frame rate when --save-gif is used.")
    boundary_group = parser.add_mutually_exclusive_group()
    boundary_group.add_argument(
        "--show-collision-boundary",
        dest="show_collision_boundary",
        action="store_true",
        help="Show the circular collision boundary in PNG/GIF output (default).",
    )
    boundary_group.add_argument(
        "--hide-collision-boundary",
        dest="show_collision_boundary",
        action="store_false",
        help="Hide the circular collision boundary; keep it only as an invisible clipping boundary.",
    )
    parser.set_defaults(show_collision_boundary=True)
    parser.add_argument(
        "--static-obstacle-ratio",
        type=static_obstacle_ratio_arg,
        default=None,
        metavar="L:M:S",
        help="Visual size-class ratio large:medium:small; default reads the config or 2:3:5.",
    )
    parser.add_argument("--show", action="store_true", help="Open matplotlib window after saving the PNG.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    project_root = Path(__file__).resolve().parent
    config, config_path, config_stem = read_config(project_root, args.config)
    static_obstacle_visual_ratio = (
        args.static_obstacle_ratio
        if args.static_obstacle_ratio is not None
        else parse_static_obstacle_visual_ratio(
            hp_get(
                config,
                "STATIC_OBSTACLE_VISUAL_RATIO",
                fallback=format_static_obstacle_visual_ratio(
                    DEFAULT_STATIC_OBSTACLE_VISUAL_RATIO
                ),
            )
        )
    )
    training_run_name = hp_get(config, "TRAINING_RUN_NAME", fallback=config_stem).strip()
    if (
        not training_run_name
        or training_run_name in {".", ".."}
        or os.path.basename(training_run_name) != training_run_name
    ):
        raise ValueError("TRAINING_RUN_NAME must be a single directory name, not a path.")
    base_seed = int(args.seed if args.seed is not None else hp_getint(config, "SEED", fallback=1))
    seed_everything(base_seed)

    control_model = hp_get(config, "control_model", fallback="usv_3dof")
    if control_model not in ACTION_DIMS:
        raise ValueError("Unsupported control_model: %s" % control_model)
    action_dim = ACTION_DIMS[control_model]
    history_length = hp_getint(config, "HISTORY_LENGTH")
    max_steps = int(args.max_steps if args.max_steps is not None else hp_getint(config, "episode_length"))
    device = resolve_device(args.device)

    env = make_eval_env(config, max_episode_steps=max_steps)
    disable_agent_colregs_intervention(env)
    obs_dim = int(env.observation_space[0].shape[0])
    target_mode = hp_get(config, "target_observation_mode", fallback="direct_noisy")
    model_dir = Path(args.model_dir) if args.model_dir else default_model_dir(training_run_name)
    validate_checkpoint_metadata(config, model_dir, obs_dim, action_dim, target_mode)
    validate_runtime_environment(env, config)

    model = make_model(config, action_dim, device)
    checkpoint_path = load_model_weights(
        model,
        model_dir,
        args.checkpoint,
        hp_getbool(config, "AUTOMATIC_ENTROPY"),
        hp_getbool(config, "CTIP_active", fallback=False),
        device,
    )

    rnn_active = hp_getbool(config, "RNN")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else evaluation_output_dir(training_run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("config_path       = %s" % config_path)
    print("scenario_profile  = %s" % hp_get(config, "SCENARIO_PROFILE", fallback="DEFAULT"))
    print("curriculum_stage  = %s" % hp_get(config, "CURRICULUM_STAGE", fallback="0"))
    print("checkpoint_path   = %s" % checkpoint_path)
    print("obs_dim           = %d" % obs_dim)
    print("action_dim        = %d" % action_dim)
    print("rnn_active        = %s" % rnn_active)
    print("agent_rule_filter = disabled (policy-only evaluation)")
    print("obstacle_colregs  = unchanged")
    print("device            = %s" % device)
    print("episodes          = %d" % args.episodes)
    print("max_steps         = %d" % max_steps)
    print("collision_boundary= %s" % args.show_collision_boundary)
    print("static_ratio      = %s" % format_static_obstacle_visual_ratio(static_obstacle_visual_ratio))

    episode_results = []
    png_paths = []
    for episode_i in range(args.episodes):
        episode_number = episode_i + 1
        episode_seed = base_seed + episode_i
        seed_everything(episode_seed)
        result = run_episode(
            env,
            model,
            history_length,
            action_dim,
            max_steps,
            device,
            rnn_active,
            episode_id=episode_number,
        )
        if result["filter_active_rate"] != 0.0:
            raise RuntimeError(
                "Policy-only evaluation detected an unexpected COLREGs action-filter intervention."
            )
        validate_runtime_environment(env, config)
        environment_summary = runtime_environment_summary(env)
        motion_summary = trajectory_motion_summary(
            result, environment_summary["environment_dt_s"]
        )

        episode_suffix = "" if args.episodes == 1 else "_ep%03d" % episode_number
        file_stem = "%s_%s_policy_only%s_seed%d_%s" % (
            config_stem,
            args.checkpoint,
            episode_suffix,
            episode_seed,
            timestamp,
        )
        png_path = output_dir / (file_stem + ".png")
        gif_path = output_dir / (file_stem + ".gif")
        title = "%s | policy-only | checkpoint=%s | episode=%d/%d | steps=%d | done=%s" % (
            config_stem,
            args.checkpoint,
            episode_number,
            args.episodes,
            result["steps"],
            result["done_reason"],
        )
        draw_trajectory(
            result,
            env,
            png_path,
            title,
            show_collision_boundary=args.show_collision_boundary,
            static_obstacle_visual_ratio=static_obstacle_visual_ratio,
        )
        if args.save_gif:
            save_gif(
                result,
                env,
                gif_path,
                title,
                args.fps,
                show_collision_boundary=args.show_collision_boundary,
                static_obstacle_visual_ratio=static_obstacle_visual_ratio,
            )

        episode_results.append(result)
        png_paths.append(png_path)
        print("episode           = %d/%d" % (episode_number, args.episodes))
        print("seed              = %d" % episode_seed)
        print("map_size_m        = %.1f" % environment_summary["map_size_m"])
        print("environment_dt_s  = %.3f" % environment_summary["environment_dt_s"])
        print("usv_substeps      = %d" % environment_summary["usv_substeps"])
        print("agent_radii_m     = %s" % environment_summary["agent_radii_m"])
        print("target_radii_m    = %s" % environment_summary["target_radii_m"])
        print("dynamic_radii_m   = %s" % environment_summary["dynamic_radii_m"])
        print("static_radii_m    = %s" % environment_summary["static_radii_m"])
        print("dynamic_speeds_mps= %s" % environment_summary["dynamic_speeds_mps"])
        print("trajectory_motion = %s" % motion_summary)
        print("steps             = %d" % result["steps"])
        print("reward_sum        = %.6f" % result["reward_sum"])
        print("done_reason       = %s" % result["done_reason"])
        print("filter_active_rate= %.6f" % result["filter_active_rate"])
        # 此字段保留旧名称兼容性，但数值现在严格表示纯 COLREGs 规则修正，
        # 而不是 EMA 平滑差异。policy-only 正常情况下应接近 0。
        print("mean_rule_correction = %.6f" % result["mean_action_correction"])
        print("action_chain_mean    = %s" % result["action_chain_mean_norms"])
        print("trajectory_png    = %s" % png_path)
        if args.save_gif:
            print("trajectory_gif    = %s" % gif_path)

    print_terminal_report(
        episode_results,
        event_rows=[
            event
            for result in episode_results
            for event in result.get("colregs_event_rows", [])
        ],
        audit_episode_rows=[
            summary
            for result in episode_results
            for summary in [result.get("colregs_episode_summary", {})]
        ],
        variant_name="policy_only",
        title="see_trained_policy_only 终端评估 / policy-only evaluation",
    )

    if args.show:
        for png_path in png_paths:
            image = plt.imread(png_path)
            plt.figure(figsize=(8, 8))
            plt.imshow(image)
            plt.axis("off")
        plt.show()
    env.close()


if __name__ == "__main__":
    main()
