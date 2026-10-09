"""Shared physical-scale configuration for training and evaluation environments."""

from __future__ import annotations

import numpy as np


DEFAULT_ENVIRONMENT_CONFIG = {
    "environment_dt": 30.0,
    "usv_3dof_dt": 0.1,
    "max_episode_steps": 100.0,
    "agent_radius": 0.04,
    "target_radius": 0.05,
    "dynamic_obstacle_min_size": 0.05,
    "dynamic_obstacle_max_size": 0.15,
    "agent_target_spawn_margin": 0.10,
    "static_obstacle_clearance": 0.22,
    "vessel_static_clearance": 0.28,
    "vessel_vessel_spawn_clearance": 0.38,
    "static_obstacle_spawn_margin": 0.04,
    "dynamic_obstacle_spawn_margin": 0.08,
    "dynamic_obstacle_max_turn_rate_deg_s": 0.343774677,
    "dynamic_obstacle_heading_noise_deg_s": 0.190985932,
    "dynamic_obstacle_max_accel_mps2": 1.0,
    # Target scripted-vessel motion parameters. These are SI values and are
    # converted to the environment's km/km/s representation in the scenario.
    "target_min_speed_mps": 0.8,
    "target_max_speed_mps": 2.0,
    "target_initial_speed_mps": 1.4,
    "target_max_accel_mps2": 0.15,
    "target_max_decel_mps2": 0.20,
    "target_max_turn_rate_deg_s": 4.0,
    "target_heading_noise_deg_s": 1.0,
    "target_speed_change_interval_min_s": 3.0,
    "target_speed_change_interval_max_s": 8.0,
    "target_speed_step_mps": 0.20,
    "target_emergency_stop_speed_mps": 0.0,
    "target_colregs_enabled": True,
    "target_predictive_avoidance_enabled": True,
    "observation_velocity_scale": 0.05,
    "observation_boundary_risk_distance": 0.30,
    "timeout_termination_enabled": False,
    "action_smoothing_alpha": 1.0,
    "scripted_action_is_velocity": False,
    "usv_max_surge_speed": 3.0,
    "usv_max_reverse_speed": 0.5,
    "usv_max_sway_speed": 1.0,
    "usv_max_thrust_rate": 120.0,
    "usv_max_rudder_angle_deg": 35.0,
    "usv_max_rudder_rate_deg_s": 45.0,
    "usv_max_yaw_rate_deg_s": 90.0,
    "colregs_risk_time_horizon": 240.0,
    "colregs_min_cpa_distance": 0.20,
    "colregs_safety_buffer": 0.08,
    "colregs_emergency_distance": 0.12,
    "geometric_avoidance_buffer": 0.22,
    "geometric_vessel_emergency_buffer": 0.25,
    "predictive_avoidance_time_horizon": 240.0,
    "predictive_vessel_buffer": 0.35,
    "boundary_turn_margin": 0.30,
    "motion_execution_safety_buffer": 0.0,
    # 是否在会遇事件期间锁存 COLREGs 规则义务。默认关闭以兼容旧训练
    # 配置和旧 checkpoint；只有配置显式设为 true 时才启用。
    "colregs_persistent_state_enabled": False,
    # 持久化规则状态在连续多少步确认无风险后才解除。
    "colregs_persistent_exit_steps": 3,
    # 持久化状态机的滞回阈值。进入阈值用于审计/配置记录；风险报告本身
    # 仍由 COLREGs 引擎判定。退出阈值应比进入阈值更宽松，避免状态抖动。
    "colregs_persistent_enter_dcpa": 0.20,
    "colregs_persistent_enter_tcpa": 240.0,
    "colregs_persistent_exit_dcpa": 0.24,
    "colregs_persistent_exit_tcpa": 270.0,
    # 持久化规则状态的最大保持步数；0 表示不启用异常保护。
    "colregs_persistent_max_hold_steps": 0.0,
    # 可选 DAgger 教师聚合，默认关闭以兼容旧实验。
    "dagger_enabled": False,
    "dagger_teacher_weight": 1.0,
    # 训练期是否启用 COLREGs 动作过滤（安全层）；默认 True 兼容旧实验
    "agent_colregs_action_filter_enabled": True,
    # ===== P0-1/P0-2 会遇生成与接受判据（默认 0 = 关闭，保持旧行为） =====
    "encounter_generation_mode": 0.0,      # 0=legacy 随机放置；1=CPA 倒推生成
    "encounter_tcpa_min_s": 60.0,
    "encounter_tcpa_max_s": 300.0,
    "encounter_dcpa_min_m": 10.0,
    "encounter_dcpa_max_m": 60.0,
    "encounter_accept_tcpa_max_s": 300.0,  # 接受判据（M2）
    "encounter_accept_dcpa_max_m": 120.0,
    "encounter_max_resample": 200.0,
    "encounter_head_on_weight": 1.0,
    "encounter_crossing_weight": 1.0,
    "encounter_overtaking_weight": 1.0,
    # 动态船多身份抽样（默认关闭；开启后按 configs/vessel_identities.json 的权重抽样）
    "vessel_identity_sampling_enabled": False,
    "vessel_identity_length_scale": 1.0,
}

BOOLEAN_ENVIRONMENT_KEYS = {
    "scripted_action_is_velocity",
    "timeout_termination_enabled",
    "target_colregs_enabled",
    "target_predictive_avoidance_enabled",
    "colregs_persistent_state_enabled",
    "dagger_enabled",
    "agent_colregs_action_filter_enabled",
    "vessel_identity_sampling_enabled",
}


def read_environment_config(config, section="hyperparam"):
    """Read optional physical-scale fields while preserving legacy defaults."""
    resolved = dict(DEFAULT_ENVIRONMENT_CONFIG)
    for key, fallback in DEFAULT_ENVIRONMENT_CONFIG.items():
        if key in BOOLEAN_ENVIRONMENT_KEYS:
            resolved[key] = config.getboolean(section, key, fallback=fallback)
        else:
            resolved[key] = config.getfloat(section, key, fallback=fallback)
    return validate_environment_config(resolved)


def validate_environment_config(values):
    resolved = dict(DEFAULT_ENVIRONMENT_CONFIG)
    if values:
        resolved.update(values)

    numeric_keys = set(DEFAULT_ENVIRONMENT_CONFIG) - BOOLEAN_ENVIRONMENT_KEYS
    for key in numeric_keys:
        value = float(resolved[key])
        if not np.isfinite(value):
            raise ValueError("Environment parameter '%s' must be finite." % key)
        resolved[key] = value
    for key in BOOLEAN_ENVIRONMENT_KEYS:
        resolved[key] = bool(resolved[key])

    positive_keys = {
        "environment_dt",
        "usv_3dof_dt",
        "agent_radius",
        "target_radius",
        "dynamic_obstacle_min_size",
        "dynamic_obstacle_max_size",
        "observation_velocity_scale",
        "observation_boundary_risk_distance",
        "usv_max_surge_speed",
        "usv_max_thrust_rate",
        "usv_max_rudder_angle_deg",
        "usv_max_rudder_rate_deg_s",
        "usv_max_yaw_rate_deg_s",
        "colregs_risk_time_horizon",
        "agent_target_spawn_margin",
        "static_obstacle_clearance",
        "vessel_static_clearance",
        "vessel_vessel_spawn_clearance",
        "dynamic_obstacle_max_turn_rate_deg_s",
        "dynamic_obstacle_heading_noise_deg_s",
        "dynamic_obstacle_max_accel_mps2",
        "target_max_accel_mps2",
        "target_max_decel_mps2",
        "target_speed_change_interval_min_s",
        "target_speed_change_interval_max_s",
        "target_max_turn_rate_deg_s",
        "colregs_persistent_exit_steps",
        "colregs_persistent_enter_dcpa",
        "colregs_persistent_enter_tcpa",
        "colregs_persistent_exit_dcpa",
        "colregs_persistent_exit_tcpa",
    }
    for key in positive_keys:
        if resolved[key] <= 0.0:
            raise ValueError("Environment parameter '%s' must be positive." % key)

    nonnegative_keys = {
        "usv_max_reverse_speed",
        "usv_max_sway_speed",
        "colregs_min_cpa_distance",
        "colregs_safety_buffer",
        "colregs_emergency_distance",
        "geometric_avoidance_buffer",
        "geometric_vessel_emergency_buffer",
        "predictive_avoidance_time_horizon",
        "predictive_vessel_buffer",
        "boundary_turn_margin",
        "motion_execution_safety_buffer",
        "static_obstacle_spawn_margin",
        "dynamic_obstacle_spawn_margin",
        "target_min_speed_mps",
        "target_max_speed_mps",
        "target_initial_speed_mps",
        "target_speed_step_mps",
        "target_emergency_stop_speed_mps",
        "target_heading_noise_deg_s",
        "colregs_persistent_max_hold_steps",
    }
    for key in nonnegative_keys:
        if resolved[key] < 0.0:
            raise ValueError("Environment parameter '%s' must be non-negative." % key)

    if int(round(resolved["colregs_persistent_exit_steps"])) < 1:
        raise ValueError("colregs_persistent_exit_steps must be at least 1.")
    resolved["colregs_persistent_exit_steps"] = int(round(resolved["colregs_persistent_exit_steps"]))
    if resolved["colregs_persistent_enter_dcpa"] < 0.0 or resolved["colregs_persistent_exit_dcpa"] < 0.0:
        raise ValueError("Persistent DCPA thresholds must be non-negative.")
    if resolved["colregs_persistent_enter_tcpa"] < 0.0 or resolved["colregs_persistent_exit_tcpa"] < 0.0:
        raise ValueError("Persistent TCPA thresholds must be non-negative.")
    if resolved["colregs_persistent_exit_dcpa"] < resolved["colregs_persistent_enter_dcpa"]:
        raise ValueError("Persistent exit DCPA must be >= enter DCPA.")
    if resolved["colregs_persistent_exit_tcpa"] < resolved["colregs_persistent_enter_tcpa"]:
        raise ValueError("Persistent exit TCPA must be >= enter TCPA.")
    if float(resolved["dagger_teacher_weight"]) < 0.0:
        raise ValueError("dagger_teacher_weight must be non-negative.")

    if resolved["dynamic_obstacle_max_size"] < resolved["dynamic_obstacle_min_size"]:
        raise ValueError("Dynamic obstacle size range must satisfy min_size <= max_size.")
    if resolved["target_max_speed_mps"] < resolved["target_min_speed_mps"]:
        raise ValueError("Target speed range must satisfy min_speed <= max_speed.")
    if not (
        resolved["target_min_speed_mps"]
        <= resolved["target_initial_speed_mps"]
        <= resolved["target_max_speed_mps"]
    ):
        raise ValueError("Target initial speed must lie within the target speed range.")
    if resolved["target_emergency_stop_speed_mps"] > resolved["target_max_speed_mps"]:
        raise ValueError("Target emergency stop speed cannot exceed target max speed.")
    if resolved["target_speed_change_interval_max_s"] < resolved["target_speed_change_interval_min_s"]:
        raise ValueError("Target speed-change interval must satisfy min <= max.")
    if not 0.0 < resolved["action_smoothing_alpha"] <= 1.0:
        raise ValueError("action_smoothing_alpha must be in (0, 1].")

    substeps = resolved["environment_dt"] / resolved["usv_3dof_dt"]
    if not np.isclose(substeps, round(substeps), rtol=0.0, atol=1.0e-9):
        raise ValueError("environment_dt must be an integer multiple of usv_3dof_dt.")
    resolved["usv_substeps"] = int(round(substeps))
    resolved["max_episode_steps"] = int(round(resolved["max_episode_steps"]))
    if resolved["max_episode_steps"] <= 0:
        raise ValueError("max_episode_steps must be positive.")
    return resolved
