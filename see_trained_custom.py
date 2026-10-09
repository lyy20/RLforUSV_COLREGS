# -*- coding: utf-8 -*-
"""读取独立测试配置，在自定义场景中评估已经训练好的 USV 策略。

终端示例（在项目根目录执行）：

    # 完全按照模板执行自定义测试。
    python see_trained_custom.py CUSTOM_EVALUATION_TEMPLATE.txt

    # 临时覆盖回合数、步数与规则模式；不会改写模板文件。
    python see_trained_custom.py CUSTOM_EVALUATION_TEMPLATE.txt --episodes 3 --max-steps 100 --agent-rule-mode policy_only

    # 运行 PN 公平基线。五个名称可任意组合，且同一个 seed 下共享初始场景。
    python see_trained_custom.py CUSTOM_EVALUATION_PN_BASELINE_0823.txt --variants sac,pn_only,sac_pn,sac_colregs,sac_pn_colregs

    # 临时指定模型目录和输出目录，适合 smoke 或迁移实验。
    python see_trained_custom.py CUSTOM_EVALUATION_TEMPLATE.txt --model-dir D:/USV/logs/SAC_3DOF_BODY_FRAME_STAGE_1/model_dir --output-dir D:/USV/custom_eval

    # 只显示静态障碍物内部环境图案，隐藏其圆形碰撞边界。
    python see_trained_custom.py CUSTOM_EVALUATION_TEMPLATE.txt --hide-collision-boundary

配置职责被严格拆成两部分：训练配置只负责重建网络并加载权重，自定义
配置只负责创建测试分布。动态/静态障碍物的实际数量可以改变，但网络的
动态槽、静态槽、实体编码器和隐藏层尺寸始终沿用训练配置。
"""

from __future__ import annotations

import argparse
import configparser
import copy
import csv
import json
import os
import random
import secrets
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch

from see_trained_once import (
    ACTION_DIMS,
    DEFAULT_STATIC_OBSTACLE_VISUAL_RATIO,
    append_history,
    collect_snapshot,
    draw_frame,
    draw_trajectory,
    format_static_obstacle_visual_ratio,
    headings_array,
    hp_get,
    hp_getbool,
    hp_getfloat,
    hp_getint,
    load_model_weights,
    make_model,
    make_policy_inputs,
    parse_static_obstacle_visual_ratio,
    static_obstacle_ratio_arg,
    read_config,
    read_reward_config,
    resolve_device,
    seed_everything,
    torch_load_checkpoint,
    trajectory_motion_summary,
    update_history,
)
from tracking.target_pf import Target
from guidance.proportional_navigation import (
    PNConfig,
    PNDiagnostics,
    PNVesselParameters,
    ProportionalNavigationGuidance,
)
from evaluation.colregs_event_audit import (
    AUDIT_EPISODE_FIELDS,
    AUDIT_EVENT_FIELDS,
    AUDIT_RULE_FIELDS,
    AUDIT_STEP_FIELDS,
    AUDIT_TEST_FIELDS,
    COLREGsEventAudit,
    DEFAULT_AUDIT_CONFIG,
    aggregate_audit_rows,
    build_rule_summary_rows,
    write_audit_csv,
    write_audit_json,
)
from evaluation.test_terminal_report import print_multi_variant_terminal_report
from utilities import envs
from utilities.info_schema import (
    REWARD_COMPONENT_KEYS,
    REWARD_METRIC_KEYS,
    decode_done_reason,
    field_slice,
    has_rule_metadata,
    is_fixed_info_array,
)
from utilities.paths import evaluation_output_dir, training_model_dir
from utilities.scenario_config import read_environment_config
from utilities.observation_schema import (
    BASE_OBS_DIM,
    DYNAMIC_OBS_FEATURE_DIM,
    OBSERVATION_LAYOUT_VERSION,
    STATIC_OBS_FEATURE_DIM,
)


VALID_AGENT_RULE_MODES = {"policy_only", "assisted", "both"}
VALID_PLACEMENT_MODES = {"random", "seeded_random", "fixed", "stress"}
VALID_TARGET_MOVEMENTS = {"static", "escape", "linear", "random", "levy"}
VALID_DYNAMIC_MOVEMENTS = {"linear", "random", "levy"}
VALID_CURRENT_MODES = {"off", "random", "fixed"}

# 新的五组公平对比变体。每组先形成候选动作，再统一进入 env.step()：
# 1) 环境内部先做动作范围约束；2) 如开启则由 COLREGs 动作集投影；
# 3) 做跨环境步动作平滑；4) 最后由 3DOF 的执行器限幅和子步积分执行。
# 因此 PN 绝不会绕开已有安全层或物理限制。
VALID_EVALUATION_VARIANTS = {
    "sac",
    "pn_only",
    "sac_pn",
    "sac_colregs",
    "sac_pn_colregs",
}
_VARIANT_RULE_MODE = {
    "sac": "policy_only",
    "pn_only": "policy_only",
    "sac_pn": "policy_only",
    "sac_colregs": "assisted",
    "sac_pn_colregs": "assisted",
}
_VARIANT_GUIDANCE_MODE = {
    "sac": "sac",
    "pn_only": "pn_only",
    "sac_pn": "sac_pn",
    "sac_colregs": "sac",
    "sac_pn_colregs": "sac_pn",
}


@dataclass(frozen=True)
class GuidanceSettings:
    """自定义测试阶段的 PN 运行参数，不会被训练代码读取。"""

    pn_config: PNConfig
    blend_weight: float
    blend_thrust: bool
    use_measured_target_position: bool


@dataclass(frozen=True)
class EvaluationVariant:
    """一次公平对照的动作来源与智能体规则层开关。

    ``name`` 用于文件名、CSV 和汇总表；``rule_mode`` 只控制智能体自身的
    COLREGs 动作过滤器；``guidance_mode`` 决定候选动作来自 SAC、PN 或二者
    的融合。动态障碍船是否遵规仍由 ``DYNAMIC_VESSEL_COLREGS`` 独立控制。
    """

    name: str
    rule_mode: str
    guidance_mode: str


def _parse_evaluation_variants(
    config: configparser.ConfigParser,
    variants_override: str | None,
    legacy_rule_mode_override: str | None,
) -> tuple[list[EvaluationVariant], bool, str | None]:
    """解析新旧两种测试配置，并保持旧 ``AGENT_RULE_MODE`` 的行为不变。

    返回值依次为：待运行的变体、是否使用新变体配置、旧规则模式（新配置时为
    ``None``）。旧配置继续输出 ``policy_only`` / ``assisted`` 文件名，避免
    已有测试脚本、统计表和人工习惯因 PN 功能被破坏。
    """
    configured_variants = ""
    if config.has_option("evaluation", "EVALUATION_VARIANTS"):
        configured_variants = config.get("evaluation", "EVALUATION_VARIANTS").strip()
    raw_variants = (variants_override or configured_variants).strip()
    if raw_variants:
        if legacy_rule_mode_override is not None:
            raise ValueError(
                "--variants 与 --agent-rule-mode 不能同时使用；前者已经逐项定义了规则层。"
            )
        names = [item.strip().lower() for item in raw_variants.split(",") if item.strip()]
        if not names:
            raise ValueError("EVALUATION_VARIANTS 至少需要一个有效变体名称。")
        unknown = [name for name in names if name not in VALID_EVALUATION_VARIANTS]
        if unknown:
            raise ValueError(
                "EVALUATION_VARIANTS 含无效值：%s；可选：%s。"
                % (", ".join(unknown), ", ".join(sorted(VALID_EVALUATION_VARIANTS)))
            )
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            raise ValueError("EVALUATION_VARIANTS 不能重复：%s。" % ", ".join(duplicate_names))
        return (
            [
                EvaluationVariant(
                    name=name,
                    rule_mode=_VARIANT_RULE_MODE[name],
                    guidance_mode=_VARIANT_GUIDANCE_MODE[name],
                )
                for name in names
            ],
            True,
            None,
        )

    legacy_rule_mode = _choice(
        legacy_rule_mode_override or _get(config, "evaluation", "AGENT_RULE_MODE"),
        VALID_AGENT_RULE_MODES,
        "AGENT_RULE_MODE",
    )
    legacy_modes = ["policy_only", "assisted"] if legacy_rule_mode == "both" else [legacy_rule_mode]
    return (
        [
            EvaluationVariant(name=mode, rule_mode=mode, guidance_mode="sac")
            for mode in legacy_modes
        ],
        False,
        legacy_rule_mode,
    )


def _parse_guidance_settings(
    config: configparser.ConfigParser,
    variants: list[EvaluationVariant],
    success_radius_m: float,
) -> GuidanceSettings | None:
    """仅在 PN 参与候选动作时读取 [guidance]，避免影响旧测试配置。

    这里的角度配置统一以 deg/deg/s 写入文本，脚本读取后立即转换为 rad/rad/s；
    这样配置文件更接近船舶工程常用单位，而 PN 数学计算始终保持 SI 单位。
    """
    pn_required = any(variant.guidance_mode in {"pn_only", "sac_pn"} for variant in variants)
    if not pn_required:
        return None
    if not config.has_section("guidance"):
        raise ValueError(
            "EVALUATION_VARIANTS 包含 PN 变体，但测试配置缺少 [guidance] 段。"
        )

    pn_config = PNConfig(
        navigation_constant=_getfloat(config, "guidance", "PN_NAVIGATION_CONSTANT"),
        max_range_m=_getfloat(config, "guidance", "PN_MAX_RANGE_M"),
        min_speed_mps=_getfloat(config, "guidance", "PN_MIN_SPEED_MPS"),
        min_closing_speed_mps=_getfloat(config, "guidance", "PN_MIN_CLOSING_SPEED_MPS"),
        max_lateral_accel_mps2=_getfloat(config, "guidance", "PN_MAX_LATERAL_ACCEL_MPS2"),
        max_yaw_rate_rad_s=np.deg2rad(
            _getfloat(config, "guidance", "PN_MAX_YAW_RATE_DEG_S")
        ),
        los_rate_filter_alpha=_getfloat(config, "guidance", "PN_LOS_RATE_FILTER_ALPHA"),
        los_rate_deadband_rad_s=np.deg2rad(
            _getfloat(config, "guidance", "PN_LOS_RATE_DEADBAND_DEG_S")
        ),
        yaw_rate_tracking_gain=_getfloat(config, "guidance", "PN_YAW_RATE_TRACKING_GAIN"),
        heading_assist_enabled=_getbool(config, "guidance", "PN_HEADING_ASSIST_ENABLED"),
        heading_assist_gain=_getfloat(config, "guidance", "PN_HEADING_ASSIST_GAIN"),
        cruise_speed_mps=_getfloat(config, "guidance", "PN_CRUISE_SPEED_MPS"),
        min_approach_speed_mps=_getfloat(config, "guidance", "PN_MIN_APPROACH_SPEED_MPS"),
        slowdown_range_m=_getfloat(config, "guidance", "PN_SLOWDOWN_RANGE_M"),
        goal_radius_m=_getfloat(
            config, "guidance", "PN_GOAL_RADIUS_M", success_radius_m
        ),
        speed_control_gain=_getfloat(config, "guidance", "PN_SPEED_CONTROL_GAIN"),
        target_velocity_source=_get(config, "guidance", "PN_TARGET_VELOCITY_SOURCE"),
        static_target_zero_velocity=_getbool(
            config, "guidance", "PN_STATIC_TARGET_ZERO_VELOCITY"
        ),
        target_velocity_filter_alpha=_getfloat(
            config, "guidance", "PN_TARGET_VELOCITY_FILTER_ALPHA"
        ),
        target_speed_max_mps=_getfloat(config, "guidance", "PN_TARGET_SPEED_MAX_MPS"),
    )
    blend_weight = _getfloat(config, "guidance", "PN_BLEND_WEIGHT")
    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError("PN_BLEND_WEIGHT 必须位于 [0, 1]。")
    settings = GuidanceSettings(
        pn_config=pn_config,
        blend_weight=float(blend_weight),
        blend_thrust=_getbool(config, "guidance", "PN_BLEND_THRUST"),
        use_measured_target_position=_getbool(
            config, "guidance", "PN_USE_MEASURED_TARGET_POSITION"
        ),
    )
    # 在真正创建环境和进入回合前先进行数学/物理范围验证。
    settings.pn_config.validate()
    return settings


def _get(config, section, option, fallback=None):
    if not config.has_section(section):
        if fallback is not None:
            return fallback
        raise ValueError("测试配置缺少 [%s] 段。" % section)
    if not config.has_option(section, option):
        if fallback is not None:
            return fallback
        raise ValueError("测试配置缺少必填项 [%s] %s。" % (section, option))
    return config.get(section, option)


def _getint(config, section, option, fallback=None):
    value = _get(config, section, option, fallback)
    return int(value)


def _getfloat(config, section, option, fallback=None):
    value = _get(config, section, option, fallback)
    return float(value)


def _getbool(config, section, option, fallback=None):
    if config.has_section(section) and config.has_option(section, option):
        return config.getboolean(section, option)
    if fallback is None:
        raise ValueError("测试配置缺少必填项 [%s] %s。" % (section, option))
    return bool(fallback)


def _optional_path(raw_value: str, project_root: Path) -> Path | None:
    raw_value = str(raw_value or "").strip()
    if not raw_value:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(raw_value)))
    return path if path.is_absolute() else project_root / path


def _safe_name(value: str, field_name: str) -> str:
    value = str(value).strip()
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError("%s 必须是单个目录名，不能是路径。" % field_name)
    return value


def _choice(value: str, valid_values: set[str], field_name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in valid_values:
        raise ValueError(
            "%s=%r 无效，可选值为：%s。"
            % (field_name, value, ", ".join(sorted(valid_values)))
        )
    return normalized


def _clone_config(source: configparser.ConfigParser) -> configparser.ConfigParser:
    cloned = configparser.ConfigParser(interpolation=None)
    for section in source.sections():
        cloned.add_section(section)
        for key, value in source.items(section):
            cloned.set(section, key, value)
    return cloned


def _set_hp(config: configparser.ConfigParser, key: str, value) -> None:
    config.set("hyperparam", key, str(value))


def read_custom_config(path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser(interpolation=None)
    config.optionxform = str
    with path.open("r", encoding="utf-8-sig") as file:
        config.read_file(file)
    for section in (
        "policy",
        "evaluation",
        "scene",
        "entities",
        "movement",
        "placement",
        "disturbance",
        "dynamics",
        "output",
    ):
        if not config.has_section(section):
            raise ValueError("测试配置缺少 [%s] 段。" % section)
    return config


def parse_colregs_audit_config(config: configparser.ConfigParser) -> dict:
    """读取测试专用 COLREGs 事件审计参数。

    该段是可选的，旧测试配置自动使用稳定默认值。所有阈值仅影响
    测试统计，不参与环境动作、奖励或策略推理。
    """
    values = dict(DEFAULT_AUDIT_CONFIG)
    if config.has_section("colregs_audit"):
        int_fields = {
            "ENTER_CONFIRM_STEPS": "enter_confirm_steps",
            "EXIT_CONFIRM_STEPS": "exit_confirm_steps",
            "EARLY_ACTION_WINDOW_STEPS": "early_action_window_steps",
            "REACTION_DEADLINE_STEPS": "reaction_deadline_steps",
        }
        float_fields = {
            "ACTION_COMPLIANCE_THRESHOLD": "action_compliance_threshold",
            "ANTI_RULE_ACTION_THRESHOLD": "anti_rule_action_threshold",
            "EVENT_EXIT_DISTANCE_M": "event_exit_distance_m",
            "SAFE_CLEARANCE_M": "safe_clearance_m",
            "SAFE_DCPA_M": "safe_dcpa_m",
            "RESOLUTION_TCPA_MAX_S": "resolution_tcpa_max_s",
            "STAND_ON_EMERGENCY_DISTANCE_M": "stand_on_emergency_distance_m",
        }
        for option, key in int_fields.items():
            if config.has_option("colregs_audit", option):
                values[key] = int(config.get("colregs_audit", option))
        for option, key in float_fields.items():
            if config.has_option("colregs_audit", option):
                values[key] = float(config.get("colregs_audit", option))

    if values["enter_confirm_steps"] < 1 or values["exit_confirm_steps"] < 1:
        raise ValueError("COLREGs 事件进入/退出确认步数必须大于等于 1。")
    for key in ("action_compliance_threshold", "anti_rule_action_threshold"):
        if not 0.0 <= values[key] <= 1.0:
            raise ValueError("%s 必须位于 [0, 1]。" % key)
    for key in (
        "event_exit_distance_m",
        "safe_clearance_m",
        "safe_dcpa_m",
        "resolution_tcpa_max_s",
        "stand_on_emergency_distance_m",
    ):
        if values[key] < 0.0:
            raise ValueError("%s 不能为负数。" % key)
    return values


def build_effective_environment_config(
    training_config: configparser.ConfigParser,
    custom_config: configparser.ConfigParser,
    max_steps_override: int | None,
) -> configparser.ConfigParser:
    """在训练配置副本上只覆盖测试环境字段，保留全部网络契约字段。"""
    effective = _clone_config(training_config)

    training_agents = hp_getint(training_config, "num_agents")
    training_targets = hp_getint(training_config, "num_landmarks")
    test_agents = _getint(custom_config, "scene", "NUM_AGENTS", training_agents)
    test_targets = _getint(custom_config, "scene", "NUM_TARGETS", training_targets)
    if test_agents != training_agents or test_targets != training_targets:
        raise ValueError(
            "NUM_AGENTS/NUM_TARGETS 属于策略网络契约，必须与训练配置一致："
            "训练=(%d, %d)，测试=(%d, %d)。"
            % (training_agents, training_targets, test_agents, test_targets)
        )

    map_size_m = _getfloat(custom_config, "scene", "MAP_SIZE_M")
    observation_range_m = _getfloat(custom_config, "scene", "OBSERVATION_RANGE_M")
    success_radius_m = _getfloat(custom_config, "scene", "TARGET_SUCCESS_RADIUS_M")
    dynamic_count = _getint(custom_config, "scene", "NUM_DYNAMIC_OBSTACLES")
    static_count = _getint(custom_config, "scene", "NUM_STATIC_OBSTACLES")
    if map_size_m <= 0.0 or observation_range_m <= 0.0 or success_radius_m <= 0.0:
        raise ValueError("地图尺寸、观测范围和任务完成半径必须为正数。")
    if dynamic_count < 0 or static_count < 0:
        raise ValueError("NUM_DYNAMIC_OBSTACLES 和 NUM_STATIC_OBSTACLES 不能为负数。")
    dynamic_min_size = _getfloat(custom_config, "entities", "DYNAMIC_RADIUS_MIN_M")
    dynamic_max_size = _getfloat(custom_config, "entities", "DYNAMIC_RADIUS_MAX_M")
    static_min_size = _getfloat(custom_config, "entities", "STATIC_RADIUS_MIN_M")
    static_max_size = _getfloat(custom_config, "entities", "STATIC_RADIUS_MAX_M")
    if dynamic_min_size <= 0.0 or static_min_size <= 0.0:
        raise ValueError("动态/静态障碍物半径下限必须大于 0。")
    if dynamic_max_size < dynamic_min_size or static_max_size < static_min_size:
        raise ValueError("动态/静态障碍物半径必须满足 min <= max。")
    dynamic_min_speed = _getfloat(custom_config, "movement", "DYNAMIC_MIN_SPEED_MPS")
    dynamic_max_speed = _getfloat(custom_config, "movement", "DYNAMIC_MAX_SPEED_MPS")
    if dynamic_min_speed < 0.0 or dynamic_max_speed < dynamic_min_speed:
        raise ValueError("动态障碍船速度必须满足 0 <= min <= max。")

    max_steps = int(
        max_steps_override
        if max_steps_override is not None
        else _getint(custom_config, "evaluation", "MAX_STEPS")
    )
    if max_steps <= 0:
        raise ValueError("MAX_STEPS 必须大于 0。")

    target_movement = _choice(
        _get(custom_config, "movement", "TARGET_MOVEMENT"),
        VALID_TARGET_MOVEMENTS,
        "TARGET_MOVEMENT",
    )
    dynamic_movement = _choice(
        _get(custom_config, "movement", "DYNAMIC_MOVEMENT"),
        VALID_DYNAMIC_MOVEMENTS,
        "DYNAMIC_MOVEMENT",
    )

    target_min_speed = _getfloat(
        custom_config,
        "movement",
        "TARGET_MIN_SPEED_MPS",
        0.0 if target_movement == "static" else 0.8,
    )
    target_max_speed = _getfloat(
        custom_config,
        "movement",
        "TARGET_MAX_SPEED_MPS",
        0.0 if target_movement == "static" else 2.0,
    )
    target_initial_speed = _getfloat(
        custom_config,
        "movement",
        "TARGET_INITIAL_SPEED_MPS",
        _getfloat(custom_config, "movement", "TARGET_SPEED_MPS", target_max_speed),
    )
    if target_min_speed < 0.0 or target_max_speed < target_min_speed:
        raise ValueError("目标船速度必须满足 0 <= min <= max。")
    if not target_min_speed <= target_initial_speed <= target_max_speed:
        raise ValueError("TARGET_INITIAL_SPEED_MPS 必须位于目标速度上下限之内。")

    scalar_hp = {
        "num_agents": test_agents,
        "num_landmarks": test_targets,
        "num_obstacles": dynamic_count,
        "num_static_obstacles": static_count,
        "map_half_size": map_size_m / 2000.0,
        "ob_range": observation_range_m / 1000.0,
        "rew_dis_th": success_radius_m / 1000.0,
        "episode_length": max_steps,
        "landmark_movable": _getbool(custom_config, "movement", "TARGET_MOVABLE"),
        "movement": target_movement,
        "landmark_vel": target_initial_speed / 1000.0,
        "max_vel": target_max_speed / 1000.0,
        "target_min_speed_mps": target_min_speed,
        "target_max_speed_mps": target_max_speed,
        "target_initial_speed_mps": target_initial_speed,
        "target_max_accel_mps2": _getfloat(custom_config, "movement", "TARGET_MAX_ACCEL_MPS2", 0.15),
        "target_max_decel_mps2": _getfloat(custom_config, "movement", "TARGET_MAX_DECEL_MPS2", 0.20),
        "target_max_turn_rate_deg_s": _getfloat(custom_config, "movement", "TARGET_MAX_TURN_RATE_DEG_S", 4.0),
        "target_heading_noise_deg_s": _getfloat(custom_config, "movement", "TARGET_HEADING_NOISE_DEG_S", 1.0),
        "target_speed_change_interval_min_s": _getfloat(custom_config, "movement", "TARGET_SPEED_CHANGE_INTERVAL_MIN_S", 3.0),
        "target_speed_change_interval_max_s": _getfloat(custom_config, "movement", "TARGET_SPEED_CHANGE_INTERVAL_MAX_S", 8.0),
        "target_speed_step_mps": _getfloat(custom_config, "movement", "TARGET_SPEED_STEP_MPS", 0.20),
        "target_emergency_stop_speed_mps": _getfloat(custom_config, "movement", "TARGET_EMERGENCY_STOP_SPEED_MPS", 0.0),
        "target_colregs_enabled": _getbool(custom_config, "movement", "TARGET_COLREGS_ENABLED", True),
        "target_predictive_avoidance_enabled": _getbool(custom_config, "movement", "TARGET_PREDICTIVE_AVOIDANCE_ENABLED", True),
        "obstacle_movable": _getbool(custom_config, "movement", "DYNAMIC_MOVABLE"),
        "obstacle_movement": dynamic_movement,
        "random_vel": _getbool(custom_config, "movement", "DYNAMIC_RANDOM_SPEED"),
        "dynamic_obstacle_min_speed": dynamic_min_speed / 1000.0,
        "dynamic_obstacle_max_speed": dynamic_max_speed / 1000.0,
        "agent_radius": _getfloat(custom_config, "entities", "AGENT_RADIUS_M") / 1000.0,
        "target_radius": _getfloat(custom_config, "entities", "TARGET_RADIUS_M") / 1000.0,
        "dynamic_obstacle_min_size": dynamic_min_size / 1000.0,
        "dynamic_obstacle_max_size": dynamic_max_size / 1000.0,
        "static_obstacle_min_size": static_min_size / 1000.0,
        "static_obstacle_max_size": static_max_size / 1000.0,
        "agent_target_spawn_margin": _getfloat(custom_config, "placement", "AGENT_TARGET_MARGIN_M") / 1000.0,
        "static_obstacle_clearance": _getfloat(custom_config, "placement", "STATIC_CLEARANCE_M") / 1000.0,
        "vessel_static_clearance": _getfloat(custom_config, "placement", "VESSEL_STATIC_CLEARANCE_M") / 1000.0,
        "vessel_vessel_spawn_clearance": _getfloat(custom_config, "placement", "VESSEL_VESSEL_CLEARANCE_M") / 1000.0,
        "static_obstacle_spawn_margin": _getfloat(custom_config, "placement", "STATIC_BOUNDARY_MARGIN_M") / 1000.0,
        "dynamic_obstacle_spawn_margin": _getfloat(custom_config, "placement", "DYNAMIC_BOUNDARY_MARGIN_M") / 1000.0,
        "target_observation_mode": _get(custom_config, "disturbance", "TARGET_OBSERVATION_MODE"),
        "target_position_noise_std": _getfloat(custom_config, "disturbance", "TARGET_POSITION_NOISE_STD_M") / 1000.0,
        "max_current_vel": _getfloat(custom_config, "disturbance", "MAX_CURRENT_SPEED_MPS") / 1000.0,
        "SCENARIO": _get(custom_config, "scene", "SCENARIO", hp_get(training_config, "SCENARIO")),
        "SCENARIO_PROFILE": "CUSTOM_EVALUATION_" + _safe_name(
            _get(custom_config, "evaluation", "TEST_NAME"), "TEST_NAME"
        ),
        # 测试阶段可独立打开持久化规则状态；旧测试配置缺省为关闭，
        # 因而继续使用原有的逐步规则过滤行为。
        "colregs_persistent_state_enabled": _getbool(
            custom_config,
            "evaluation",
            "PERSISTENT_RULE_STATE_ENABLED",
            False,
        ),
        "colregs_persistent_exit_steps": _getint(
            custom_config,
            "evaluation",
            "PERSISTENT_RULE_STATE_EXIT_STEPS",
            3,
        ),
        # 以下四项允许测试配置显式覆盖训练配置中的 DCPA/TCPA 滞回阈值；
        # 未填写时沿用断点对应的训练值，保证旧测试配置语义不变。
        "colregs_persistent_enter_dcpa": _getfloat(
            custom_config,
            "evaluation",
            "PERSISTENT_RULE_STATE_ENTER_DCPA_M",
            hp_getfloat(training_config, "colregs_persistent_enter_dcpa", fallback=0.20)
            * 1000.0,
        ) / 1000.0,
        "colregs_persistent_enter_tcpa": _getfloat(
            custom_config,
            "evaluation",
            "PERSISTENT_RULE_STATE_ENTER_TCPA_S",
            hp_getfloat(training_config, "colregs_persistent_enter_tcpa", fallback=240.0),
        ),
        "colregs_persistent_exit_dcpa": _getfloat(
            custom_config,
            "evaluation",
            "PERSISTENT_RULE_STATE_EXIT_DCPA_M",
            hp_getfloat(training_config, "colregs_persistent_exit_dcpa", fallback=0.24)
            * 1000.0,
        ) / 1000.0,
        "colregs_persistent_exit_tcpa": _getfloat(
            custom_config,
            "evaluation",
            "PERSISTENT_RULE_STATE_EXIT_TCPA_S",
            hp_getfloat(training_config, "colregs_persistent_exit_tcpa", fallback=270.0),
        ),
        # 可选最大保持步数；缺省为 0，保持旧测试配置语义。
        "colregs_persistent_max_hold_steps": _getint(
            custom_config,
            "evaluation",
            "PERSISTENT_RULE_STATE_MAX_HOLD_STEPS",
            0,
        ),
    }
    for key, value in scalar_hp.items():
        _set_hp(effective, key, value)

    preserve_dynamics = _getbool(
        custom_config, "dynamics", "PRESERVE_TRAINING_DYNAMICS", True
    )
    if not preserve_dynamics:
        dynamics_map = {
            "environment_dt": "ENVIRONMENT_DT_S",
            "usv_3dof_dt": "USV_3DOF_DT_S",
            "usv_max_surge_speed": "USV_MAX_SURGE_SPEED_MPS",
            "usv_max_reverse_speed": "USV_MAX_REVERSE_SPEED_MPS",
            "usv_max_sway_speed": "USV_MAX_SWAY_SPEED_MPS",
            "usv_max_thrust_rate": "USV_MAX_THRUST_RATE_NPS",
            "usv_max_rudder_angle_deg": "USV_MAX_RUDDER_ANGLE_DEG",
            "usv_max_rudder_rate_deg_s": "USV_MAX_RUDDER_RATE_DEG_S",
            "usv_max_yaw_rate_deg_s": "USV_MAX_YAW_RATE_DEG_S",
            "action_smoothing_alpha": "ACTION_SMOOTHING_ALPHA",
            "dynamic_obstacle_max_turn_rate_deg_s": "DYNAMIC_MAX_TURN_RATE_DEG_S",
            "dynamic_obstacle_heading_noise_deg_s": "DYNAMIC_HEADING_NOISE_DEG_S",
            "dynamic_obstacle_max_accel_mps2": "DYNAMIC_MAX_ACCEL_MPS2",
        }
        for hp_key, custom_key in dynamics_map.items():
            _set_hp(effective, hp_key, _getfloat(custom_config, "dynamics", custom_key))

    # 这些尺度随地图与实体大小改变，允许自定义测试显式覆盖。
    physical_maps = {
        "motion_execution_safety_buffer": ("dynamics", "MOTION_EXECUTION_SAFETY_BUFFER_M", 1000.0),
        "colregs_risk_time_horizon": ("dynamics", "COLREGS_RISK_TIME_HORIZON_S", 1.0),
        "colregs_min_cpa_distance": ("dynamics", "COLREGS_MIN_CPA_DISTANCE_M", 1000.0),
        "colregs_safety_buffer": ("dynamics", "COLREGS_SAFETY_BUFFER_M", 1000.0),
        "colregs_emergency_distance": ("dynamics", "COLREGS_EMERGENCY_DISTANCE_M", 1000.0),
        "geometric_avoidance_buffer": ("dynamics", "GEOMETRIC_AVOIDANCE_BUFFER_M", 1000.0),
        "geometric_vessel_emergency_buffer": ("dynamics", "GEOMETRIC_VESSEL_BUFFER_M", 1000.0),
        "predictive_avoidance_time_horizon": ("dynamics", "PREDICTIVE_HORIZON_S", 1.0),
        "predictive_vessel_buffer": ("dynamics", "PREDICTIVE_VESSEL_BUFFER_M", 1000.0),
        "boundary_turn_margin": ("dynamics", "BOUNDARY_TURN_MARGIN_M", 1000.0),
        # body-frame 观测中的边界风险提前量，单位同样从 m 换成 km。
        "observation_boundary_risk_distance": (
            "dynamics",
            "OBSERVATION_BOUNDARY_RISK_DISTANCE_M",
            1000.0,
        ),
    }
    for hp_key, (section, custom_key, divisor) in physical_maps.items():
        _set_hp(effective, hp_key, _getfloat(custom_config, section, custom_key) / divisor)

    # 归一化尺度应跟随自定义地图/速度，槽数量则绝不能改变。
    _set_hp(effective, "observation_base_position_scale", map_size_m / 2000.0)
    _set_hp(
        effective,
        "observation_velocity_scale",
        max(_getfloat(custom_config, "movement", "DYNAMIC_MAX_SPEED_MPS") / 1000.0, 1.0e-6),
    )
    _set_hp(
        effective,
        "observation_base_velocity_scale",
        max(_getfloat(custom_config, "dynamics", "USV_MAX_SURGE_SPEED_MPS") / 1000.0, 1.0e-6),
    )
    return effective


def make_custom_env(config: configparser.ConfigParser):
    environment_config = read_environment_config(config)
    environment_config["max_episode_steps"] = hp_getint(config, "episode_length")
    return envs.make_env(
        hp_get(config, "SCENARIO"),
        num_agents=hp_getint(config, "num_agents"),
        num_landmarks=hp_getint(config, "num_landmarks"),
        num_obstacles=hp_getint(config, "num_obstacles"),
        num_static_obstacles=hp_getint(config, "num_static_obstacles", fallback=0),
        map_half_size=hp_getfloat(config, "map_half_size"),
        static_obstacle_min_size=hp_getfloat(config, "static_obstacle_min_size"),
        static_obstacle_max_size=hp_getfloat(config, "static_obstacle_max_size"),
        ob_range=hp_getfloat(config, "ob_range"),
        # 网络槽数量始终从训练配置副本读取，实际障碍物数量不会修改这里。
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
        target_position_noise_std=hp_getfloat(config, "target_position_noise_std", fallback=0.0),
        scenario_profile=hp_get(config, "SCENARIO_PROFILE"),
        dynamic_obstacle_min_speed=hp_getfloat(config, "dynamic_obstacle_min_speed"),
        dynamic_obstacle_max_speed=hp_getfloat(config, "dynamic_obstacle_max_speed"),
        reward_config=read_reward_config(config),
        environment_config=environment_config,
        benchmark=True,
    )


def validate_checkpoint_network_contract(
    training_config: configparser.ConfigParser,
    model_dir: Path,
    obs_dim: int,
    action_dim: int,
) -> None:
    """只严格检查网络契约；场景差异是本测试工具的有意设计。"""
    state_path = model_dir / "training_state_last.pt"
    if not state_path.exists():
        print("checkpoint_note   = training_state_last.pt 不存在，将由权重形状完成最终校验")
        return
    state = torch_load_checkpoint(state_path, map_location="cpu")
    if not isinstance(state, dict):
        return

    if state.get("obs_dim") is not None and int(state["obs_dim"]) != int(obs_dim):
        raise ValueError(
            "观测维度与断点不一致：checkpoint=%d，custom_env=%d。请勿修改训练槽数量。"
            % (int(state["obs_dim"]), int(obs_dim))
        )
    if state.get("action_dim") is not None and int(state["action_dim"]) != int(action_dim):
        raise ValueError(
            "动作维度与断点不一致：checkpoint=%d，current=%d。"
            % (int(state["action_dim"]), int(action_dim))
        )

    saved = state.get("network_contract")
    if not isinstance(saved, dict):
        return
    current = {
        "policy_algorithm": hp_get(training_config, "DNN"),
        "control_model": hp_get(training_config, "control_model", fallback="usv_3dof"),
        "num_agents": hp_getint(training_config, "num_agents"),
        "num_landmarks": hp_getint(training_config, "num_landmarks"),
        "dynamic_observation_slots": hp_getint(training_config, "num_ob"),
        "static_observation_slots": hp_getint(training_config, "num_static_ob_slots", fallback=1),
        "observation_layout": hp_get(
            training_config,
            "OBSERVATION_LAYOUT",
            fallback=OBSERVATION_LAYOUT_VERSION,
        ),
        "base_observation_dim": int(BASE_OBS_DIM),
        "dynamic_observation_feature_dim": int(DYNAMIC_OBS_FEATURE_DIM),
        "static_observation_feature_dim": int(STATIC_OBS_FEATURE_DIM),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "hidden_dim_1": hp_getint(training_config, "DIM_1"),
        "hidden_dim_2": hp_getint(training_config, "DIM_2"),
        "rnn": hp_getbool(training_config, "RNN"),
        "history_length": hp_getint(training_config, "HISTORY_LENGTH"),
        "automatic_entropy_tuning": hp_getbool(training_config, "AUTOMATIC_ENTROPY"),
        "entity_encoder_enabled": hp_getbool(training_config, "ENTITY_ENCODER_ENABLED", fallback=False),
        "entity_encoder_latent_dim": hp_getint(training_config, "ENTITY_ENCODER_LATENT_DIM", fallback=32),
        "ctip_active": hp_getbool(training_config, "CTIP_active", fallback=False),
        "icm_active": hp_getbool(training_config, "ICM", fallback=False),
    }
    mismatches = [
        "%s: checkpoint=%r training_config=%r" % (key, saved[key], value)
        for key, value in current.items()
        if key in saved and saved[key] != value
    ]
    if mismatches:
        raise ValueError("训练配置与断点网络契约不一致：" + "; ".join(mismatches))


def _json_value(config, option, expected_type):
    raw = _get(config, "placement", option, "").strip()
    if not raw:
        return expected_type()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("%s 不是合法 JSON：%s" % (option, error)) from error
    if not isinstance(value, expected_type):
        raise ValueError("%s 的 JSON 顶层类型必须是 %s。" % (option, expected_type.__name__))
    return value


def _position_km(row: dict, field_name: str) -> np.ndarray:
    value = row.get("position_m")
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("%s 中 position_m 必须是 [x_m, y_m]。" % field_name)
    position = np.asarray(value, dtype=float) / 1000.0
    if not np.all(np.isfinite(position)):
        raise ValueError("%s 的 position_m 必须是有限数。" % field_name)
    return position


def _apply_fixed_layout(env, custom_config) -> None:
    world = env.world
    agent_rows = _json_value(custom_config, "FIXED_AGENT_JSON", list)
    target_rows = _json_value(custom_config, "FIXED_TARGET_JSON", list)
    dynamic_rows = _json_value(custom_config, "FIXED_DYNAMIC_JSON", list)
    static_rows = _json_value(custom_config, "FIXED_STATIC_JSON", list)
    expected = {
        "FIXED_AGENT_JSON": (agent_rows, world.num_agents),
        "FIXED_TARGET_JSON": (target_rows, world.num_landmarks),
        "FIXED_DYNAMIC_JSON": (dynamic_rows, world.num_obstacles),
        "FIXED_STATIC_JSON": (static_rows, world.num_static_obstacles),
    }
    for name, (rows, count) in expected.items():
        if len(rows) != count:
            raise ValueError("%s 实体数应为 %d，实际为 %d。" % (name, count, len(rows)))

    for index, (agent, row) in enumerate(zip(world.agents[:world.num_agents], agent_rows)):
        agent.state.p_pos = _position_km(row, "FIXED_AGENT_JSON[%d]" % index)
        heading = np.deg2rad(float(row.get("heading_deg", 0.0)))
        speed = float(row.get("speed_mps", 0.0)) / 1000.0
        agent.state.p_vel = np.array([np.cos(heading), np.sin(heading)]) * speed
        agent.state.p_vel_old = agent.state.p_vel.copy()
        agent.state.p_pos_origin = agent.state.p_pos.copy()
        agent.state.usv_3dof = None
        agent.action_smoothed_u = np.zeros(world.agent_action_dim, dtype=float)
        world.angle[index] = float(heading)

    for index, (target, row) in enumerate(zip(world.landmarks[:world.num_landmarks], target_rows)):
        target.state.p_pos = _position_km(row, "FIXED_TARGET_JSON[%d]" % index)
        target.ra = float(np.deg2rad(float(row.get("heading_deg", 0.0))))
        fixed_speed_mps = float(row.get("speed_mps", 0.0))
        target_config = getattr(world, "environment_config", {})
        target_min_mps = float(target_config.get("target_min_speed_mps", 0.0))
        target_max_mps = float(target_config.get("target_max_speed_mps", fixed_speed_mps))
        if target.movable and not target_min_mps <= fixed_speed_mps <= target_max_mps:
            raise ValueError(
                "FIXED_TARGET_JSON[%d] 的 speed_mps 必须位于目标速度范围 [%.3f, %.3f] 内。"
                % (index, target_min_mps, target_max_mps)
            )
        target.landmark_vel = (fixed_speed_mps / 1000.0) if target.movable else 0.0
        target.max_speed = (target_max_mps / 1000.0) if target.movable else 0.0
        target.current_landmark_vel = target.landmark_vel
        target.target_speed_intent = target.landmark_vel
        target.target_speed_hold_steps = 0
        target.target_nominal_heading = target.ra
        target.target_random_vector = np.array([np.cos(target.ra), np.sin(target.ra)])
        target.target_levy_hold_steps = 0
        target.state.p_vel = np.array([np.cos(target.ra), np.sin(target.ra)]) * target.landmark_vel
        estimate = world.landmarks[index + world.num_landmarks]
        estimate.state.p_pos = target.state.p_pos.copy()
        estimate.state.p_vel = target.state.p_vel.copy()

    for index, (obstacle, row) in enumerate(zip(world.obstacles[:world.num_obstacles], dynamic_rows)):
        obstacle.state.p_pos = _position_km(row, "FIXED_DYNAMIC_JSON[%d]" % index)
        obstacle.size = float(row.get("radius_m")) / 1000.0
        obstacle.ra = float(np.deg2rad(float(row.get("heading_deg", 0.0))))
        obstacle.obstacle_vel = float(row.get("speed_mps", 0.0)) / 1000.0
        obstacle.max_speed = obstacle.obstacle_vel
        obstacle.state.p_vel = np.array([np.cos(obstacle.ra), np.sin(obstacle.ra)]) * obstacle.obstacle_vel
        estimate = world.obstacles[index + world.num_obstacles]
        estimate.size = obstacle.size
        estimate.state.p_pos = obstacle.state.p_pos.copy()
        estimate.state.p_vel = obstacle.state.p_vel.copy()

    for index, (obstacle, row) in enumerate(zip(world.static_obstacles[:world.num_static_obstacles], static_rows)):
        obstacle.state.p_pos = _position_km(row, "FIXED_STATIC_JSON[%d]" % index)
        obstacle.size = float(row.get("radius_m")) / 1000.0
        obstacle.state.p_vel = np.zeros(2, dtype=float)
        obstacle.action.u = np.zeros(2, dtype=float)

    for agent in world.agents[:world.num_agents]:
        agent.reward_prev_target_distance = float(
            np.linalg.norm(world.landmarks[0].state.p_pos - agent.state.p_pos)
        )
        agent.reward_prev_action = np.zeros(world.agent_action_dim, dtype=float)
    world.landmarks_estimated = [
        Target(L_pos=target.state.p_pos, V=target.state.p_vel)
        for target in world.landmarks[:world.num_landmarks]
    ]
    world.obstacles_estimated = [
        Target(L_pos=obstacle.state.p_pos, V=obstacle.state.p_vel)
        for obstacle in world.obstacles[:world.num_obstacles]
    ]
    _validate_initial_layout(world)


def _apply_stress_layout(env) -> None:
    """保留随机位置和间距，只把动态船航向调整为穿越追踪主航路。"""
    world = env.world
    agent = world.agents[0]
    target = world.landmarks[0]
    route_midpoint = 0.5 * (agent.state.p_pos + target.state.p_pos)
    route = target.state.p_pos - agent.state.p_pos
    route_norm = max(float(np.linalg.norm(route)), 1.0e-12)
    route_normal = np.array([-route[1], route[0]]) / route_norm
    for index, obstacle in enumerate(world.obstacles[:world.num_obstacles]):
        desired = route_midpoint - obstacle.state.p_pos
        if np.linalg.norm(desired) <= 1.0e-9:
            desired = route_normal * (1.0 if index % 2 == 0 else -1.0)
        obstacle.ra = float(np.arctan2(desired[1], desired[0]))
        obstacle.state.p_vel = np.array([np.cos(obstacle.ra), np.sin(obstacle.ra)]) * obstacle.obstacle_vel


def _validate_initial_layout(world) -> None:
    entities = []
    entities.extend(world.agents[:world.num_agents])
    entities.extend(world.landmarks[:world.num_landmarks])
    entities.extend(world.obstacles[:world.num_obstacles])
    entities.extend(world.static_obstacles[:world.num_static_obstacles])
    half_size = float(world.map_half_size)
    for entity in entities:
        pos = np.asarray(entity.state.p_pos, dtype=float)
        radius = float(entity.size)
        if np.any(np.abs(pos) + radius > half_size + 1.0e-12):
            raise ValueError("固定实体 %s 超出地图边界。" % entity.name)
        if radius <= 0.0:
            raise ValueError("固定实体 %s 的 radius_m 必须大于 0。" % entity.name)
    for i, entity_a in enumerate(entities):
        for entity_b in entities[i + 1:]:
            distance = float(np.linalg.norm(entity_a.state.p_pos - entity_b.state.p_pos))
            if distance <= float(entity_a.size + entity_b.size):
                raise ValueError(
                    "固定布局初始重叠：%s 与 %s，中心距=%.3f m。"
                    % (entity_a.name, entity_b.name, distance * 1000.0)
                )


def configure_runtime_before_reset(env, custom_config, rule_mode: str) -> None:
    world = env.world
    world.agent_colregs_action_filter_enabled = rule_mode == "assisted"
    dynamic_colregs = _getbool(custom_config, "movement", "DYNAMIC_VESSEL_COLREGS")
    for obstacle in world.obstacles[:world.num_obstacles]:
        obstacle.colregs_compliant = dynamic_colregs
    scenario = getattr(env.reset_callback, "__self__", None)
    if scenario is not None:
        scenario.scene_generation_max_retries = _getint(
            custom_config, "placement", "MAX_PLACEMENT_RETRIES", 50
        )


def configure_runtime_after_reset(env, custom_config, placement_mode: str) -> bool:
    """返回是否需要重新计算观测（fixed/stress 修改了 reset 后的状态）。"""
    world = env.world
    current_mode = _choice(
        _get(custom_config, "disturbance", "CURRENT_MODE"),
        VALID_CURRENT_MODES,
        "CURRENT_MODE",
    )
    if current_mode == "off":
        world.ocean_current_enabled = False
        world.vel_ocean_current = 0.0
    elif current_mode == "fixed":
        world.ocean_current_enabled = True
        world.vel_ocean_current = _getfloat(
            custom_config, "disturbance", "FIXED_CURRENT_SPEED_MPS"
        ) / 1000.0
        world.angle_ocean_current = np.deg2rad(
            _getfloat(custom_config, "disturbance", "FIXED_CURRENT_DIRECTION_DEG")
        )
    else:
        world.ocean_current_enabled = True

    if placement_mode == "fixed":
        _apply_fixed_layout(env, custom_config)
        changed = True
    elif placement_mode == "stress":
        _apply_stress_layout(env)
        changed = True
    else:
        changed = False
    # PN 在执行第一步前就要读取地固速度。这里同步洋流至 3DOF 模型，确保
    # 导引器使用的速度与随后环境积分使用的速度处于同一个流场下。
    if hasattr(world, "_sync_usv_3dof_current"):
        world._sync_usv_3dof_current()
    if hasattr(world, "_invalidate_colregs_cache"):
        world._invalidate_colregs_cache()
    return changed


def _decode_info(info, num_agents: int, num_landmarks: int, fallback_actions) -> dict:
    fallback = np.asarray(fallback_actions, dtype=float).reshape(num_agents, -1)
    width = fallback.shape[1]
    decoded = {
        "raw_actions": fallback.copy(),
        "constrained_actions": fallback.copy(),
        # 当前动作契约：rule 是纯 COLREGs 输出，smoothed 是 EMA 后的
        # 动作；旧 info 没有这两个字段时分别退化为 constrained/applied。
        "rule_actions": fallback.copy(),
        "smoothed_actions": fallback.copy(),
        "applied_actions": fallback.copy(),
        "filter_active": np.zeros(num_agents, dtype=bool),
        "correction_norm": np.zeros(num_agents, dtype=float),
        "done_reason": ["none"] * num_agents,
        "reward_components": np.zeros((num_agents, len(REWARD_COMPONENT_KEYS)), dtype=float),
        "reward_metrics": np.zeros((num_agents, len(REWARD_METRIC_KEYS)), dtype=float),
        "action_chain_norms": {
            # 初始化阶段四层动作都等于 fallback，理论差异为零。不能在
            # ``decoded`` 字典尚未构造完成时引用它，否则自定义测试会触发
            # UnboundLocalError；后续由 derive_chain_norms() 统一重算。
            "raw_to_constrained_norm": np.zeros(num_agents, dtype=float),
            "constrained_to_rule_norm": np.zeros(num_agents, dtype=float),
            "rule_to_smoothed_norm": np.zeros(num_agents, dtype=float),
            "raw_to_smoothed_norm": np.zeros(num_agents, dtype=float),
        },
    }

    def coerce_matrix(value, default):
        """将字典 info 中的动作安全整理成当前测试动作形状。"""
        if value is None:
            return default.copy()
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return default.copy()
        if array.size != default.size:
            return default.copy()
        return array.reshape(default.shape).copy()

    def derive_chain_norms():
        """用解码后的动作层重新计算诊断范数，避免旧字段误标。"""
        return {
            "raw_to_constrained_norm": np.linalg.norm(
                decoded["constrained_actions"] - decoded["raw_actions"], axis=1
            ),
            "constrained_to_rule_norm": np.linalg.norm(
                decoded["rule_actions"] - decoded["constrained_actions"], axis=1
            ),
            "rule_to_smoothed_norm": np.linalg.norm(
                decoded["smoothed_actions"] - decoded["rule_actions"], axis=1
            ),
            "raw_to_smoothed_norm": np.linalg.norm(
                decoded["smoothed_actions"] - decoded["raw_actions"], axis=1
            ),
        }

    if isinstance(info, dict):
        decoded["raw_actions"] = coerce_matrix(
            info.get("raw_actions"), decoded["raw_actions"]
        )
        decoded["constrained_actions"] = coerce_matrix(
            info.get("constrained_actions"), decoded["constrained_actions"]
        )
        decoded["applied_actions"] = coerce_matrix(
            info.get("applied_actions"), decoded["applied_actions"]
        )
        decoded["rule_actions"] = coerce_matrix(
            info.get("rule_actions"), decoded["constrained_actions"]
        )
        decoded["smoothed_actions"] = coerce_matrix(
            info.get("smoothed_actions"), decoded["applied_actions"]
        )
        decoded["filter_active"] = [
            bool(v) for v in info.get("filter_active", decoded["filter_active"])
        ]
        decoded["correction_norm"] = [
            float(v)
            for v in info.get("action_correction_norm", decoded["correction_norm"])
        ]
        decoded["done_reason"] = [
            str(v) for v in info.get("done_reason", decoded["done_reason"])
        ]
        decoded["action_chain_norms"] = derive_chain_norms()
        return decoded

    if is_fixed_info_array(info, num_landmarks=num_landmarks):
        for source, target in (
            ("raw_action", "raw_actions"),
            ("constrained_action", "constrained_actions"),
            ("applied_action", "applied_actions"),
        ):
            decoded[target] = np.asarray(info[:, field_slice(source, num_landmarks)], dtype=float)[:, :width]
        if has_rule_metadata(info):
            decoded["rule_actions"] = np.asarray(
                info[:, field_slice("rule_action", num_landmarks)], dtype=float
            )[:, :width]
            decoded["smoothed_actions"] = np.asarray(
                info[:, field_slice("smoothed_action", num_landmarks)], dtype=float
            )[:, :width]
            decoded["action_chain_norms"] = {
                key: np.asarray(
                    info[:, field_slice(key, num_landmarks)], dtype=float
                ).reshape(-1)
                for key in decoded["action_chain_norms"]
            }
        else:
            decoded["rule_actions"] = decoded["constrained_actions"].copy()
            decoded["smoothed_actions"] = decoded["applied_actions"].copy()
            decoded["action_chain_norms"] = derive_chain_norms()
        decoded["filter_active"] = np.asarray(
            info[:, field_slice("filter_active", num_landmarks)], dtype=float
        ).reshape(-1).astype(bool)
        decoded["correction_norm"] = np.asarray(
            info[:, field_slice("action_correction_norm", num_landmarks)], dtype=float
        ).reshape(-1)
        decoded["done_reason"] = [
            decode_done_reason(value)
            for value in np.asarray(info[:, field_slice("done_reason", num_landmarks)]).reshape(-1)
        ]
        decoded["reward_components"] = np.asarray(
            info[:, field_slice("reward_components", num_landmarks)], dtype=float
        )
        decoded["reward_metrics"] = np.asarray(
            info[:, field_slice("reward_metrics", num_landmarks)], dtype=float
        )
    else:
        decoded["action_chain_norms"] = derive_chain_norms()
    return decoded


def _agent_physical_state(agent, world) -> dict:
    state = getattr(agent.state, "usv_3dof", None)
    if state is None:
        heading = float(world.angle[0]) if world.angle else 0.0
        return {
            "heading_rad": heading,
            "u_mps": float(np.linalg.norm(agent.state.p_vel)) * 1000.0,
            "v_mps": 0.0,
            "yaw_rate_rad_s": 0.0,
            "thrust_n": 0.0,
            "rudder_rad": 0.0,
        }
    return {
        "heading_rad": float(state.psi),
        "u_mps": float(state.u),
        "v_mps": float(state.v),
        "yaw_rate_rad_s": float(state.r),
        "thrust_n": float(state.thrust),
        "rudder_rad": float(state.rudder),
    }


def _finite_vector2_or_none(values) -> np.ndarray | None:
    """返回有限二维向量；估计实体尚未初始化时显式返回 None。"""
    if values is None:
        return None
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != 2 or not np.all(np.isfinite(vector)):
        return None
    return vector.copy()


def _guidance_target_state(
    world,
    agent_index: int,
    settings: GuidanceSettings,
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """获取 PN 的目标测量位置、速度来源和静止标记。

    ``direct_noisy`` 观测模式会把带测量误差的目标坐标写入
    ``world.landmarks[i + world.num_landmarks]``。默认让 PN 使用该坐标，
    而不是偷看真实目标位置；把 ``PN_USE_MEASURED_TARGET_POSITION=false``
    时才显式使用真实位置，适合作为理想传感器上界而非公平默认值。
    """
    target_index = min(int(agent_index), int(world.num_landmarks) - 1)
    target = world.landmarks[target_index]
    estimate_index = target_index + int(world.num_landmarks)
    estimate = (
        world.landmarks[estimate_index]
        if estimate_index < len(world.landmarks)
        else None
    )
    true_position = _finite_vector2_or_none(getattr(target.state, "p_pos", None))
    measured_position = _finite_vector2_or_none(
        getattr(getattr(estimate, "state", None), "p_pos", None)
    )
    if settings.use_measured_target_position and measured_position is not None:
        position = measured_position
    elif true_position is not None:
        position = true_position
    else:
        raise ValueError("PN 无法读取目标位置，目标或其测量实体尚未初始化。")

    source = str(settings.pn_config.target_velocity_source).strip().lower()
    if source == "world_oracle":
        velocity = _finite_vector2_or_none(getattr(target.state, "p_vel", None))
    elif source == "measured_state":
        velocity = _finite_vector2_or_none(
            getattr(getattr(estimate, "state", None), "p_vel", None)
        )
    else:
        # finite_difference/zero 的速度在 PN 内部自己构建，这里不传真值。
        velocity = None

    target_is_static = not bool(getattr(target, "movable", False))
    velocity_mps = None if velocity is None else velocity * 1000.0
    return position * 1000.0, velocity_mps, target_is_static


def _inactive_pn_diagnostics(reason: str = "not_selected") -> PNDiagnostics:
    """为 SAC-only 行填充统一 PN CSV 列，避免不同变体的列结构漂移。"""
    return PNDiagnostics(
        active=False,
        reason=reason,
        target_velocity_source="not_used",
        range_m=float("nan"),
        closing_speed_mps=float("nan"),
        raw_los_rate_rad_s=float("nan"),
        filtered_los_rate_rad_s=float("nan"),
        lateral_accel_command_mps2=float("nan"),
        yaw_rate_command_rad_s=float("nan"),
        heading_error_rad=float("nan"),
        heading_capture_yaw_rate_rad_s=float("nan"),
        desired_yaw_rate_rad_s=float("nan"),
        target_speed_estimate_mps=float("nan"),
        desired_surge_speed_mps=float("nan"),
        thrust_command_n=float("nan"),
        pn_thrust_norm=float("nan"),
        pn_rudder_norm=float("nan"),
    )


def _make_episode_guidance(
    env,
    settings: GuidanceSettings | None,
    num_agents: int,
) -> list[ProportionalNavigationGuidance | None]:
    """为每个智能体创建独立 PN 记忆，绝不跨回合共享 LOS/速度滤波状态。"""
    if settings is None:
        return [None] * num_agents
    vessel = PNVesselParameters.from_usv_config(env.world.usv_3dof_model.config)
    return [
        ProportionalNavigationGuidance(settings.pn_config, vessel)
        for _ in range(num_agents)
    ]


def _pn_action_for_agent(
    env,
    agent_index: int,
    guidance: ProportionalNavigationGuidance,
    settings: GuidanceSettings,
) -> tuple[np.ndarray, PNDiagnostics]:
    """按当前真实 3DOF 状态和目标测量构造一条 PN 候选动作。"""
    world = env.world
    agent = world.agents[agent_index]
    own_state = getattr(agent.state, "usv_3dof", None)
    own_position_m = np.asarray(agent.state.p_pos, dtype=float) * 1000.0
    if own_state is None:
        heading = float(world.angle[agent_index]) if len(world.angle) > agent_index else 0.0
        own_velocity_world_mps = np.asarray(agent.state.p_vel, dtype=float) * 1000.0
        own_surge_speed_mps = float(np.linalg.norm(own_velocity_world_mps))
        own_yaw_rate_rad_s = 0.0
    else:
        if hasattr(world, "_sync_usv_3dof_current"):
            world._sync_usv_3dof_current()
        x_dot, y_dot, _ = world.usv_3dof_model.earth_fixed_velocity(
            own_state.psi, own_state.u, own_state.v, own_state.r
        )
        heading = float(own_state.psi)
        own_velocity_world_mps = np.asarray([x_dot, y_dot], dtype=float)
        own_surge_speed_mps = float(own_state.u)
        own_yaw_rate_rad_s = float(own_state.r)

    target_position_m, target_velocity_mps, target_is_static = _guidance_target_state(
        world, agent_index, settings
    )
    environment_dt_s = float(world.environment_config["environment_dt"])
    return guidance.compute_action(
        own_position_m=own_position_m,
        own_velocity_world_mps=own_velocity_world_mps,
        own_heading_rad=heading,
        own_yaw_rate_rad_s=own_yaw_rate_rad_s,
        own_surge_speed_mps=own_surge_speed_mps,
        target_position_m=target_position_m,
        target_state_velocity_mps=target_velocity_mps,
        target_is_static=target_is_static,
        dt_s=environment_dt_s,
    )


def run_custom_episode(
    env,
    model,
    custom_config,
    evaluation_variant: EvaluationVariant,
    episode_id: int,
    placement_mode: str,
    history_length: int,
    action_dim: int,
    max_steps: int,
    device: str,
    rnn_active: bool,
    guidance_settings: GuidanceSettings | None,
    colregs_audit_config: dict | None = None,
) -> dict:
    """执行一个测试回合，并把候选动作与实际执行动作逐层记录。

    五组对照在这一处汇合：SAC/PN 只负责形成 ``candidate_actions``；真正的
    执行链仍保持环境原有次序，即范围约束 -> 可选 COLREGs 投影 -> 动作平滑
    -> 3DOF 执行器速率限制 -> 0.1 s 子步积分。这样不会因为测试基线而绕开
    物理模型或安全层，也不会修改任何训练流程。
    """
    configure_runtime_before_reset(env, custom_config, evaluation_variant.rule_mode)
    obs_n = env.reset()
    changed = configure_runtime_after_reset(env, custom_config, placement_mode)
    if changed:
        obs_n = [env._get_obs(agent) for agent in env.world.policy_agents]

    num_agents = len(obs_n)
    num_landmarks = int(env.world.num_landmarks)
    obs_dim = int(len(obs_n[0]))
    history_obs = np.zeros((num_agents, history_length, obs_dim), dtype=float)
    history_act = np.zeros((num_agents, history_length, action_dim), dtype=float)
    if evaluation_variant.guidance_mode in {"pn_only", "sac_pn"} and action_dim != 2:
        raise ValueError("PN 仅适配当前 [推力, 舵角] 两维 usv_3dof 动作空间。")
    # 一个多变体任务共用同一份 [guidance] 配置时，SAC / SAC+COLREGs 基线
    # 也不应创建 PN 对象。这样可以明确保证它们只执行原策略动作。
    guidance_modules = (
        _make_episode_guidance(env, guidance_settings, num_agents)
        if evaluation_variant.guidance_mode in {"pn_only", "sac_pn"}
        else [None] * num_agents
    )
    trajectory = {}
    append_history(trajectory, collect_snapshot(env))
    initial_agent_position = np.asarray(env.world.agents[0].state.p_pos, dtype=float).copy()
    initial_target_position = np.asarray(env.world.landmarks[0].state.p_pos, dtype=float).copy()

    rewards = []
    step_rows = []
    colregs_audit = COLREGsEventAudit(
        env.world,
        episode_id=episode_id,
        config=colregs_audit_config,
        variant_name=evaluation_variant.name,
    )
    done_reason = "none"
    for step_index in range(1, max_steps + 1):
        # Shadow reports are evaluated before the candidate action is sent to
        # env.step(). They are observations only and cannot alter the rollout.
        colregs_reports = colregs_audit.reports_for_current_state()
        needs_sac_action = evaluation_variant.guidance_mode in {"sac", "sac_pn"}
        needs_pn_action = evaluation_variant.guidance_mode in {"pn_only", "sac_pn"}
        policy_actions = np.full((num_agents, action_dim), np.nan, dtype=float)
        if needs_sac_action:
            histories, observations = make_policy_inputs(
                obs_n, history_obs, history_act, device
            )
            with torch.no_grad():
                action_tensors = model.act(histories, observations, noise=0.0)
            policy_actions = torch.stack(action_tensors).detach().cpu().numpy()[:, 0, :]

        pn_actions = np.full((num_agents, action_dim), np.nan, dtype=float)
        pn_diagnostics = [_inactive_pn_diagnostics() for _ in range(num_agents)]
        if needs_pn_action:
            if guidance_settings is None:
                raise AssertionError("PN 变体缺少已验证的 GuidanceSettings。")
            for agent_index, guidance in enumerate(guidance_modules):
                if guidance is None:
                    raise AssertionError("PN 变体没有创建回合内导引器。")
                pn_action, pn_diag = _pn_action_for_agent(
                    env, agent_index, guidance, guidance_settings
                )
                pn_actions[agent_index] = pn_action
                pn_diagnostics[agent_index] = pn_diag

        if evaluation_variant.guidance_mode == "sac":
            # SAC 与 SAC+COLREGs 都把同一条网络动作作为候选动作；二者的唯一
            # 差异由 environment 内部的 COLREGs 过滤器开关决定。
            candidate_actions = policy_actions.copy()
        elif evaluation_variant.guidance_mode == "pn_only":
            # PN-only 不调用 model.act()，避免 SAC 网络输出被隐式混入基线。
            candidate_actions = pn_actions.copy()
        elif evaluation_variant.guidance_mode == "sac_pn":
            candidate_actions = policy_actions.copy()
            beta = guidance_settings.blend_weight
            # SAC+PN 的默认融合只修正舵角：
            # delta_mix = (1-beta) * delta_SAC + beta * delta_PN。
            # 这让 SAC 保留速度/推力决策，PN 仅给出面向目标 LOS 的转向先验。
            # 若 PN_BLEND_THRUST=true，才对推力也按同一凸组合融合。
            candidate_actions[:, 1] = (
                (1.0 - beta) * policy_actions[:, 1] + beta * pn_actions[:, 1]
            )
            if guidance_settings.blend_thrust:
                candidate_actions[:, 0] = (
                    (1.0 - beta) * policy_actions[:, 0] + beta * pn_actions[:, 0]
                )
        else:
            raise AssertionError(
                "未处理的 guidance_mode=%s。" % evaluation_variant.guidance_mode
            )

        if not np.all(np.isfinite(candidate_actions)):
            raise ValueError("候选动作出现 NaN/Inf，已拒绝送入环境。")
        # PN 与凸组合理论上已处于 [-1,1]；此处仅作为与 COLREGs 输入范围一致的
        # 最后一层数值保护，不改变 SAC 正常 tanh 输出的语义。
        candidate_actions = np.clip(candidate_actions, -1.0, 1.0)
        next_obs, reward_n, done_n, info = env.step(
            [candidate_actions[index] for index in range(num_agents)]
        )
        decoded = _decode_info(info, num_agents, num_landmarks, candidate_actions)
        colregs_audit.record_step(
            step=step_index,
            policy_action=policy_actions[0],
            constrained_action=decoded["constrained_actions"][0],
            applied_action=decoded["applied_actions"][0],
            filter_active=decoded["filter_active"][0],
            filter_mode=getattr(env.world.agents[0], "action_filter_mode", ""),
            done_reason=decoded["done_reason"][0],
            reports=colregs_reports,
            rule_action=decoded["rule_actions"][0],
            smoothed_action=decoded["smoothed_actions"][0],
        )
        if rnn_active:
            update_history(history_obs, history_act, obs_n, decoded["applied_actions"])
        obs_n = next_obs
        append_history(trajectory, collect_snapshot(env))

        step_reward = float(np.sum(reward_n))
        rewards.append(step_reward)
        reasons = [value for value in decoded["done_reason"] if value != "none"]
        if reasons:
            done_reason = reasons[0]

        for agent_index, agent in enumerate(env.world.agents[:num_agents]):
            physical = _agent_physical_state(agent, env.world)
            metrics = decoded["reward_metrics"][agent_index]
            components = decoded["reward_components"][agent_index]
            row = {
                "step": step_index,
                "agent_index": agent_index,
                "evaluation_variant": evaluation_variant.name,
                "guidance_mode": evaluation_variant.guidance_mode,
                "agent_rule_mode": evaluation_variant.rule_mode,
                "reward": step_reward,
                "done": bool(done_n[agent_index]),
                "done_reason": decoded["done_reason"][agent_index],
                "filter_active": bool(decoded["filter_active"][agent_index]),
                "correction_norm": float(decoded["correction_norm"][agent_index]),
                # 四段动作链与训练端 TensorBoard 同义，可把 PN/SAC 候选动作
                # 的变化与 COLREGs 规则修正、EMA 平滑修正分开统计。
                "raw_to_constrained_norm": float(
                    decoded["action_chain_norms"]["raw_to_constrained_norm"][agent_index]
                ),
                "constrained_to_rule_norm": float(
                    decoded["action_chain_norms"]["constrained_to_rule_norm"][agent_index]
                ),
                "rule_to_smoothed_norm": float(
                    decoded["action_chain_norms"]["rule_to_smoothed_norm"][agent_index]
                ),
                "raw_to_smoothed_norm": float(
                    decoded["action_chain_norms"]["raw_to_smoothed_norm"][agent_index]
                ),
                "x_m": float(agent.state.p_pos[0]) * 1000.0,
                "y_m": float(agent.state.p_pos[1]) * 1000.0,
                "speed_mps": float(np.linalg.norm(agent.state.p_vel)) * 1000.0,
            }
            row["policy_action_available"] = int(needs_sac_action)
            row["pn_action_available"] = int(needs_pn_action)
            for action_name, values in (
                ("policy", policy_actions[agent_index]),
                ("pn_candidate", pn_actions[agent_index]),
                ("candidate", candidate_actions[agent_index]),
            ):
                row[action_name + "_thrust_norm"] = float(values[0])
                row[action_name + "_rudder_norm"] = (
                    float(values[1]) if values.size > 1 else 0.0
                )
            for action_name, values in (
                ("raw", decoded["raw_actions"][agent_index]),
                ("constrained", decoded["constrained_actions"][agent_index]),
                ("rule", decoded["rule_actions"][agent_index]),
                ("smoothed", decoded["smoothed_actions"][agent_index]),
                ("applied", decoded["applied_actions"][agent_index]),
            ):
                row[action_name + "_thrust_norm"] = float(values[0])
                row[action_name + "_rudder_norm"] = float(values[1]) if values.size > 1 else 0.0
            # PN 中间量以 SI 单位输出，便于逐步复算 Vc、LOS 角速度、a_n 与舵角
            # 融合效果。SAC-only 行同样保留这些列，但以 not_selected/NaN 标示。
            row.update(pn_diagnostics[agent_index].as_dict())
            row.update(physical)
            for key, value in zip(REWARD_COMPONENT_KEYS, components):
                row["reward_" + key] = float(value)
            for key, value in zip(REWARD_METRIC_KEYS, metrics):
                row[key + "_m"] = float(value) * 1000.0
            step_rows.append(row)
        if bool(np.any(done_n)):
            break

    colregs_event_rows = colregs_audit.finalize(
        final_step=len(rewards),
        done_reason=done_reason,
    )
    colregs_episode_summary = colregs_audit.episode_summary(evaluation_variant.name)

    result = {
        "trajectory": trajectory,
        "step_rows": step_rows,
        "steps": len(rewards),
        "reward_sum": float(np.sum(rewards)) if rewards else 0.0,
        "done_reason": done_reason,
        "initial_agent_position": initial_agent_position,
        "initial_target_position": initial_target_position,
        "colregs_step_rows": colregs_audit.step_rows,
        "colregs_event_rows": colregs_event_rows,
        "colregs_episode_summary": colregs_episode_summary,
    }
    result["summary"] = summarize_episode(
        result, env, custom_config, evaluation_variant, guidance_settings
    )
    return result


def summarize_episode(
    result,
    env,
    custom_config,
    evaluation_variant: EvaluationVariant,
    guidance_settings: GuidanceSettings | None,
) -> dict:
    agent_positions = np.asarray(
        result["trajectory"]["agents"][0]["positions"], dtype=float
    )
    target_positions = np.asarray(
        result["trajectory"]["targets"][0]["positions"], dtype=float
    )
    if len(agent_positions) > 1:
        path_length_m = float(np.sum(np.linalg.norm(np.diff(agent_positions, axis=0), axis=1))) * 1000.0
    else:
        path_length_m = 0.0
    start_distance_m = float(
        np.linalg.norm(result["initial_target_position"] - result["initial_agent_position"])
    ) * 1000.0
    final_distance_m = float(np.linalg.norm(target_positions[-1] - agent_positions[-1])) * 1000.0
    target_distance_reduction_m = max(start_distance_m - final_distance_m, 0.0)
    # 未完成回合也可计算该指标：实际接近目标的净距离 / 实际航迹长度。
    # 对静止目标它受三角不等式约束，通常位于 [0, 1]；动态目标则应结合
    # target_distance_reduction_m 一并解读，而不能单独作为成功率替代品。
    path_efficiency = (
        target_distance_reduction_m / path_length_m
        if path_length_m > 1.0e-12
        else 0.0
    )

    rows = result["step_rows"]
    filter_values = [float(row["filter_active"]) for row in rows]
    corrections = [row["correction_norm"] for row in rows]
    clearances = [row["min_clearance_m"] for row in rows if np.isfinite(row["min_clearance_m"])]
    yaw_rates = np.asarray([row["yaw_rate_rad_s"] for row in rows], dtype=float)
    rudders = np.asarray([row["rudder_rad"] for row in rows], dtype=float)
    applied_actions = np.asarray(
        [[row["applied_thrust_norm"], row["applied_rudder_norm"]] for row in rows],
        dtype=float,
    )
    pn_active = np.asarray([row["pn_active"] for row in rows], dtype=float)
    pn_los_rates = np.asarray([row["pn_los_rate_rad_s"] for row in rows], dtype=float)
    pn_lateral_accels = np.asarray(
        [row["pn_lateral_accel_command_mps2"] for row in rows], dtype=float
    )
    pn_yaw_commands = np.asarray(
        [row["pn_yaw_rate_command_rad_s"] for row in rows], dtype=float
    )
    pn_heading_errors = np.asarray(
        [row["pn_heading_error_rad"] for row in rows], dtype=float
    )
    pn_heading_capture_yaw_commands = np.asarray(
        [row["pn_heading_capture_yaw_rate_rad_s"] for row in rows], dtype=float
    )
    pn_desired_yaw_commands = np.asarray(
        [row["pn_desired_yaw_rate_rad_s"] for row in rows], dtype=float
    )
    pn_target_speeds = np.asarray(
        [row["pn_target_speed_estimate_mps"] for row in rows], dtype=float
    )
    policy_actions = np.asarray(
        [[row["policy_thrust_norm"], row["policy_rudder_norm"]] for row in rows],
        dtype=float,
    )
    candidate_actions = np.asarray(
        [[row["candidate_thrust_norm"], row["candidate_rudder_norm"]] for row in rows],
        dtype=float,
    )
    valid_policy_rows = np.all(np.isfinite(policy_actions), axis=1)
    candidate_policy_delta = (
        np.linalg.norm(candidate_actions[valid_policy_rows] - policy_actions[valid_policy_rows], axis=1)
        if np.any(valid_policy_rows)
        else np.asarray([], dtype=float)
    )

    def finite_mean(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else float("nan")

    return {
        "evaluation_variant": evaluation_variant.name,
        "guidance_mode": evaluation_variant.guidance_mode,
        "rule_mode": evaluation_variant.rule_mode,
        "pn_blend_weight": (
            guidance_settings.blend_weight
            if evaluation_variant.guidance_mode == "sac_pn" and guidance_settings is not None
            else 0.0
        ),
        "pn_blend_thrust": int(
            guidance_settings is not None
            and evaluation_variant.guidance_mode == "sac_pn"
            and guidance_settings.blend_thrust
        ),
        "steps": result["steps"],
        "reward_sum": result["reward_sum"],
        "done_reason": result["done_reason"],
        "success": int(result["done_reason"] == "success"),
        "collision": int(result["done_reason"] in {"collision", "target_collision"}),
        "out_of_bounds": int(result["done_reason"] == "out_of_bounds"),
        "timeout": int(result["done_reason"] == "timeout"),
        "start_target_distance_m": start_distance_m,
        "final_target_distance_m": final_distance_m,
        "target_distance_reduction_m": target_distance_reduction_m,
        "path_length_m": path_length_m,
        "path_efficiency": path_efficiency,
        "minimum_clearance_m": min(clearances) if clearances else float("nan"),
        "filter_active_rate": float(np.mean(filter_values)) if filter_values else 0.0,
        "mean_action_correction": float(np.mean(corrections)) if corrections else 0.0,
        "mean_abs_yaw_rate_rad_s": float(np.mean(np.abs(yaw_rates))) if yaw_rates.size else 0.0,
        "mean_abs_yaw_rate_change_rad_s": float(np.mean(np.abs(np.diff(yaw_rates)))) if yaw_rates.size > 1 else 0.0,
        "mean_abs_rudder_change_rad": float(np.mean(np.abs(np.diff(rudders)))) if rudders.size > 1 else 0.0,
        "mean_action_change": float(np.mean(np.linalg.norm(np.diff(applied_actions, axis=0), axis=1))) if len(applied_actions) > 1 else 0.0,
        "pn_active_rate": float(np.mean(pn_active)) if pn_active.size else 0.0,
        "mean_abs_pn_los_rate_rad_s": finite_mean(np.abs(pn_los_rates)),
        "mean_abs_pn_lateral_accel_mps2": finite_mean(np.abs(pn_lateral_accels)),
        "mean_abs_pn_yaw_rate_command_rad_s": finite_mean(np.abs(pn_yaw_commands)),
        "mean_abs_pn_heading_error_rad": finite_mean(np.abs(pn_heading_errors)),
        "mean_abs_pn_heading_capture_yaw_rate_rad_s": finite_mean(
            np.abs(pn_heading_capture_yaw_commands)
        ),
        "mean_abs_pn_desired_yaw_rate_rad_s": finite_mean(
            np.abs(pn_desired_yaw_commands)
        ),
        "mean_pn_target_speed_estimate_mps": finite_mean(pn_target_speeds),
        "mean_candidate_policy_delta": (
            float(np.mean(candidate_policy_delta)) if candidate_policy_delta.size else float("nan")
        ),
    }


def save_trajectory_csv(result: dict, path: Path) -> None:
    rows = []
    for group, tracks in result["trajectory"].items():
        for entity_index, track in enumerate(tracks):
            headings = headings_array(track)
            for step, position in enumerate(track["positions"]):
                is_vessel = group != "static_obstacles"
                heading_rad = float(headings[step]) if is_vessel and step < len(headings) else None
                rows.append({
                    "step": step,
                    "entity_group": group,
                    "entity_index": entity_index,
                    "entity_name": track["name"],
                    "x_m": float(position[0]) * 1000.0,
                    "y_m": float(position[1]) * 1000.0,
                    "radius_m": float(track["size"]) * 1000.0,
                    "heading_rad": heading_rad if heading_rad is not None else "",
                    "heading_deg": float(np.rad2deg(heading_rad)) if heading_rad is not None else "",
                })
    write_csv(path, rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_gif_with_stride(
    result,
    env,
    output_path: Path,
    title: str,
    fps: int,
    stride: int,
    show_collision_boundary: bool = True,
    static_obstacle_visual_ratio=DEFAULT_STATIC_OBSTACLE_VISUAL_RATIO,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image

    frame_count = result["steps"] + 1
    frame_indices = list(range(0, frame_count, max(1, stride)))
    if frame_indices[-1] != frame_count - 1:
        frame_indices.append(frame_count - 1)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=110)
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.09, top=0.88)
    canvas = FigureCanvasAgg(fig)
    frames = []
    for frame_index in frame_indices:
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
        rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(rgba[:, :, :3].copy())
    plt.close(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=max(int(1000 / max(fps, 1)), 1),
        loop=0,
    )


def write_effective_config(path: Path, config: configparser.ConfigParser) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("# 该文件是自定义测试配置与训练配置合并后的最终生效值。\n")
        file.write("# 网络结构字段来自训练配置；场景字段来自自定义测试配置。\n\n")
        config.write(file)


def _finite_summary_mean(rows: list[dict], field_name: str) -> float:
    """计算汇总指标的有限均值，避免单个无定义指标污染整组对比。"""
    values = np.asarray([row.get(field_name, float("nan")) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def build_variant_summaries(
    all_summaries: list[dict],
    evaluation_variants: list[EvaluationVariant],
) -> tuple[list[dict], dict]:
    """按五组动作来源汇总论文常用的任务、安全、效率与 PN 诊断指标。

    ``evaluation_variant`` 是唯一分组键，不能再按 ``rule_mode`` 分组：
    SAC、PN-only 和 SAC+PN 都可能关闭 COLREGs，但它们的候选动作来源完全不同。
    """
    mean_fields = (
        ("mean_reward", "reward_sum"),
        ("mean_steps", "steps"),
        ("mean_final_target_distance_m", "final_target_distance_m"),
        ("mean_target_distance_reduction_m", "target_distance_reduction_m"),
        ("mean_path_length_m", "path_length_m"),
        ("mean_path_efficiency", "path_efficiency"),
        ("mean_minimum_clearance_m", "minimum_clearance_m"),
        ("mean_filter_active_rate", "filter_active_rate"),
        ("mean_action_correction", "mean_action_correction"),
        ("mean_candidate_policy_delta", "mean_candidate_policy_delta"),
        ("mean_pn_active_rate", "pn_active_rate"),
        ("mean_abs_pn_los_rate_rad_s", "mean_abs_pn_los_rate_rad_s"),
        ("mean_abs_pn_lateral_accel_mps2", "mean_abs_pn_lateral_accel_mps2"),
        ("mean_abs_pn_yaw_rate_command_rad_s", "mean_abs_pn_yaw_rate_command_rad_s"),
        (
            "mean_abs_pn_heading_capture_yaw_rate_rad_s",
            "mean_abs_pn_heading_capture_yaw_rate_rad_s",
        ),
        ("mean_abs_pn_desired_yaw_rate_rad_s", "mean_abs_pn_desired_yaw_rate_rad_s"),
        ("mean_abs_pn_heading_error_rad", "mean_abs_pn_heading_error_rad"),
        ("mean_pn_target_speed_estimate_mps", "mean_pn_target_speed_estimate_mps"),
    )
    rows_out = []
    aggregate = {}
    for variant in evaluation_variants:
        rows = [
            row for row in all_summaries
            if row["evaluation_variant"] == variant.name
        ]
        if not rows:
            raise AssertionError("变体 %s 没有产生任何回合结果。" % variant.name)
        summary = {
            "evaluation_variant": variant.name,
            "guidance_mode": variant.guidance_mode,
            "agent_rule_mode": variant.rule_mode,
            "episodes": len(rows),
            "success_rate": _finite_summary_mean(rows, "success"),
            "collision_rate": _finite_summary_mean(rows, "collision"),
            "out_of_bounds_rate": _finite_summary_mean(rows, "out_of_bounds"),
            "timeout_rate": _finite_summary_mean(rows, "timeout"),
        }
        for output_name, source_name in mean_fields:
            summary[output_name] = _finite_summary_mean(rows, source_name)
        rows_out.append(summary)
        aggregate[variant.name] = dict(summary)
    return rows_out, aggregate


PAIR_METRICS = (
    "success",
    "collision",
    "out_of_bounds",
    "timeout",
    "reward_sum",
    "minimum_clearance_m",
    "encounter_events",
    "policy_rule_follow_rate",
    "safe_resolution_rate",
    "internalization_success_rate",
)


def build_filter_on_off_pair_summary(all_summaries: list[dict]) -> tuple[list[dict], dict]:
    """Pair policy-only and assisted episodes by identical seed.

    The trajectories can diverge after the first filtered action, so this is
    deliberately an episode-level paired comparison rather than an event-ID
    join.  It remains valid alongside the PN variants, which are ignored here.
    """
    by_seed: dict[int, dict[str, dict]] = {}
    for row in all_summaries:
        variant = str(row.get("evaluation_variant", ""))
        if variant not in {"policy_only", "assisted"}:
            continue
        seed = int(row["seed"])
        variants = by_seed.setdefault(seed, {})
        if variant in variants:
            raise ValueError("seed=%d has duplicate %s rows." % (seed, variant))
        variants[variant] = row

    rows = []
    for seed in sorted(by_seed):
        variants = by_seed[seed]
        if set(variants) != {"policy_only", "assisted"}:
            continue
        policy = variants["policy_only"]
        assisted = variants["assisted"]
        row = {
            "seed": seed,
            "episode": int(policy.get("episode", assisted.get("episode", 0))),
            "policy_done_reason": policy.get("done_reason", ""),
            "assisted_done_reason": assisted.get("done_reason", ""),
        }
        for metric in PAIR_METRICS:
            policy_value = _finite_summary_mean([policy], metric)
            assisted_value = _finite_summary_mean([assisted], metric)
            row["policy_" + metric] = policy_value
            row["assisted_" + metric] = assisted_value
            row[metric + "_assisted_minus_policy"] = (
                assisted_value - policy_value
                if np.isfinite(policy_value) and np.isfinite(assisted_value)
                else float("nan")
            )
        rows.append(row)

    aggregate = {
        "paired_episodes": len(rows),
        "unpaired_seeds": sum(
            1 for variants in by_seed.values()
            if set(variants) != {"policy_only", "assisted"}
        ),
    }
    for metric in PAIR_METRICS:
        aggregate["mean_" + metric + "_assisted_minus_policy"] = _finite_summary_mean(
            rows, metric + "_assisted_minus_policy"
        )
    return rows, aggregate


def _json_safe(value):
    """将 NaN/Inf 变为 JSON ``null``，保持 manifest 可被标准解析器读取。

    CSV 中仍保留 ``nan``，因为数值统计软件通常把它识别为缺失值；JSON 则不应
    依赖 Python 专有的 ``NaN`` 字面量。SAC-only 的 PN 指标正是这一类“不适用”。
    """
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取 TXT/INI 配置，在自定义场景中评估训练好的 USV 策略。"
    )
    parser.add_argument("test_config", help="自定义测试配置文件路径，例如 CUSTOM_EVALUATION_TEMPLATE.txt。")
    parser.add_argument("--training-config", default=None, help="临时覆盖 TRAINING_CONFIG，主要用于 smoke/迁移检查。")
    parser.add_argument("--training-run-name", default=None, help="临时覆盖 TRAINING_RUN_NAME，必须与被覆盖的训练配置一致。")
    parser.add_argument("--episodes", type=int, default=None, help="临时覆盖 EPISODES。")
    parser.add_argument("--max-steps", type=int, default=None, help="临时覆盖 MAX_STEPS。")
    parser.add_argument("--seed", type=int, default=None, help="临时覆盖 BASE_SEED。")
    parser.add_argument(
        "--variants",
        default=None,
        help=(
            "临时覆盖 EVALUATION_VARIANTS，例如 sac,pn_only,sac_pn,sac_colregs,sac_pn_colregs；"
            "不能与 --agent-rule-mode 同时使用。"
        ),
    )
    parser.add_argument("--agent-rule-mode", choices=sorted(VALID_AGENT_RULE_MODES), default=None, help="临时覆盖 AGENT_RULE_MODE。")
    parser.add_argument("--placement-mode", choices=sorted(VALID_PLACEMENT_MODES), default=None, help="临时覆盖 PLACEMENT_MODE。")
    parser.add_argument("--checkpoint", choices=("last", "best"), default=None, help="临时覆盖 CHECKPOINT。")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None, help="临时覆盖 DEVICE。")
    parser.add_argument("--model-dir", default=None, help="临时覆盖 MODEL_DIR。")
    parser.add_argument("--output-dir", default=None, help="临时覆盖本次测试的最终输出目录。")
    parser.add_argument("--no-gif", action="store_true", help="本次运行不生成 GIF，不修改配置文件。")
    boundary_group = parser.add_mutually_exclusive_group()
    boundary_group.add_argument(
        "--show-collision-boundary",
        dest="show_collision_boundary",
        action="store_true",
        help="显示静态障碍物圆形碰撞边界；未指定时读取配置，默认显示。",
    )
    boundary_group.add_argument(
        "--hide-collision-boundary",
        dest="show_collision_boundary",
        action="store_false",
        help="隐藏静态障碍物圆形碰撞边界，仅显示内部环境图案。",
    )
    parser.set_defaults(show_collision_boundary=None)
    parser.add_argument(
        "--static-obstacle-ratio",
        type=static_obstacle_ratio_arg,
        default=None,
        metavar="L:M:S",
        help="临时覆盖静态障碍物视觉分档比例（大岛:沙岛:礁石），默认读取配置或 2:3:5。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    custom_path = Path(args.test_config)
    if not custom_path.is_absolute():
        custom_path = project_root / custom_path
    custom_path = custom_path.resolve()
    custom = read_custom_config(custom_path)
    static_obstacle_visual_ratio = (
        args.static_obstacle_ratio
        if args.static_obstacle_ratio is not None
        else parse_static_obstacle_visual_ratio(
            _get(
                custom,
                "output",
                "STATIC_OBSTACLE_VISUAL_RATIO",
                format_static_obstacle_visual_ratio(
                    DEFAULT_STATIC_OBSTACLE_VISUAL_RATIO
                ),
            )
        )
    )

    training_config_name = (
        args.training_config or _get(custom, "policy", "TRAINING_CONFIG")
    ).strip()
    training_config, training_config_path, training_config_stem = read_config(
        project_root, training_config_name
    )
    configured_run_name = _safe_name(
        args.training_run_name or _get(custom, "policy", "TRAINING_RUN_NAME"),
        "TRAINING_RUN_NAME",
    )
    actual_run_name = _safe_name(
        hp_get(training_config, "TRAINING_RUN_NAME", fallback=training_config_stem),
        "训练配置中的 TRAINING_RUN_NAME",
    )
    if configured_run_name != actual_run_name:
        raise ValueError(
            "测试配置的 TRAINING_RUN_NAME=%s 与训练配置中的 %s 不一致。"
            % (configured_run_name, actual_run_name)
        )

    evaluation_variants, uses_variant_configuration, legacy_rule_mode = _parse_evaluation_variants(
        custom,
        args.variants,
        args.agent_rule_mode,
    )
    placement_mode = _choice(
        args.placement_mode or _get(custom, "placement", "PLACEMENT_MODE"),
        VALID_PLACEMENT_MODES,
        "PLACEMENT_MODE",
    )
    current_mode = _choice(
        _get(custom, "disturbance", "CURRENT_MODE"),
        VALID_CURRENT_MODES,
        "CURRENT_MODE",
    )
    del current_mode

    episodes = int(args.episodes if args.episodes is not None else _getint(custom, "evaluation", "EPISODES"))
    max_steps = int(args.max_steps if args.max_steps is not None else _getint(custom, "evaluation", "MAX_STEPS"))
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("EPISODES 和 MAX_STEPS 必须大于 0。")
    configured_seed = int(args.seed if args.seed is not None else _getint(custom, "evaluation", "BASE_SEED"))
    if placement_mode == "random":
        base_seed = secrets.randbelow(2**31 - episodes - 1)
        print("random_base_seed  = %d（请保存此值以复现实验）" % base_seed)
    else:
        base_seed = configured_seed

    effective = build_effective_environment_config(
        training_config, custom, max_steps_override=max_steps
    )
    control_model = hp_get(training_config, "control_model", fallback="usv_3dof")
    if control_model not in ACTION_DIMS:
        raise ValueError("不支持的 control_model：%s。" % control_model)
    action_dim = ACTION_DIMS[control_model]
    device = resolve_device(args.device or _get(custom, "policy", "DEVICE", "auto").strip().lower())
    checkpoint_kind = args.checkpoint or _get(custom, "policy", "CHECKPOINT", "last").strip().lower()
    if checkpoint_kind not in {"last", "best"}:
        raise ValueError("CHECKPOINT 只能是 last 或 best。")

    model_dir_override = args.model_dir or _get(custom, "policy", "MODEL_DIR", "")
    model_dir = _optional_path(model_dir_override, project_root) or training_model_dir(actual_run_name)

    # 先创建一个最终测试环境，用它检查观测维度；网络仍严格由训练配置构造。
    seed_everything(base_seed)
    probe_env = make_custom_env(effective)
    obs_dim = int(probe_env.observation_space[0].shape[0])
    validate_checkpoint_network_contract(training_config, model_dir, obs_dim, action_dim)
    guidance_settings = _parse_guidance_settings(
        custom,
        evaluation_variants,
        _getfloat(custom, "scene", "TARGET_SUCCESS_RADIUS_M"),
    )
    if guidance_settings is not None:
        # 在加载网络和运行完整回合前就检查 PN 请求的艏摇能力是否落在当前
        # 3DOF 硬上限内。该构造不会推进环境状态，也不会触碰训练断点。
        ProportionalNavigationGuidance(
            guidance_settings.pn_config,
            PNVesselParameters.from_usv_config(probe_env.world.usv_3dof_model.config),
        )
    probe_env.close()

    model = make_model(training_config, action_dim, device)
    checkpoint_path = load_model_weights(
        model,
        model_dir,
        checkpoint_kind,
        hp_getbool(training_config, "AUTOMATIC_ENTROPY"),
        hp_getbool(training_config, "CTIP_active", fallback=False),
        device,
    )

    test_name = _safe_name(_get(custom, "evaluation", "TEST_NAME"), "TEST_NAME")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    explicit_output = args.output_dir
    configured_output_root = _get(custom, "output", "OUTPUT_ROOT", "")
    if explicit_output:
        output_dir = _optional_path(explicit_output, project_root)
    elif configured_output_root.strip():
        output_dir = _optional_path(configured_output_root, project_root) / test_name / timestamp
    else:
        output_dir = evaluation_output_dir(actual_run_name) / test_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    save_png = _getbool(custom, "output", "SAVE_PNG", True)
    save_gif = _getbool(custom, "output", "SAVE_GIF", True) and not args.no_gif
    show_collision_boundary = (
        bool(args.show_collision_boundary)
        if args.show_collision_boundary is not None
        else _getbool(custom, "output", "SHOW_COLLISION_BOUNDARY", True)
    )
    save_trajectory = _getbool(custom, "output", "SAVE_TRAJECTORY_CSV", True)
    save_action = _getbool(custom, "output", "SAVE_ACTION_STATE_CSV", True)
    save_summary = _getbool(custom, "output", "SAVE_EPISODE_SUMMARY_CSV", True)
    save_effective = _getbool(custom, "output", "SAVE_EFFECTIVE_CONFIG", True)
    save_manifest = _getbool(custom, "output", "SAVE_JSON_MANIFEST", True)
    save_colregs_audit = _getbool(
        custom, "output", "SAVE_COLREGS_AUDIT", True
    )
    gif_every = _getint(custom, "output", "SAVE_GIF_EVERY", 1)
    gif_stride = _getint(custom, "output", "GIF_FRAME_STRIDE", 1)
    gif_fps = _getint(custom, "output", "GIF_FPS", 12)
    if gif_every <= 0 or gif_stride <= 0 or gif_fps <= 0:
        raise ValueError("SAVE_GIF_EVERY、GIF_FRAME_STRIDE 和 GIF_FPS 必须大于 0。")

    if save_effective:
        write_effective_config(output_dir / "effective_environment_config.txt", effective)
        shutil.copy2(custom_path, output_dir / "input_custom_config.txt")
    history_length = hp_getint(training_config, "HISTORY_LENGTH")
    rnn_active = hp_getbool(training_config, "RNN")
    colregs_audit_config = parse_colregs_audit_config(custom)

    print("training_config   = %s" % training_config_path)
    print("custom_config     = %s" % custom_path)
    print("training_run_name = %s" % actual_run_name)
    print("checkpoint_path   = %s" % checkpoint_path)
    print("output_dir        = %s" % output_dir)
    print("evaluation_variants = %s" % ",".join(variant.name for variant in evaluation_variants))
    if not uses_variant_configuration:
        print("legacy_rule_mode = %s" % legacy_rule_mode)
    if guidance_settings is not None:
        print(
            "pn_settings      = N=%.3f blend=%.3f yaw_limit=%.3f deg/s"
            % (
                guidance_settings.pn_config.navigation_constant,
                guidance_settings.blend_weight,
                np.rad2deg(guidance_settings.pn_config.max_yaw_rate_rad_s),
            )
        )
    print("placement_mode    = %s" % placement_mode)
    print("episodes          = %d" % episodes)
    print("max_steps         = %d" % max_steps)
    print("collision_boundary= %s" % show_collision_boundary)
    print("static_ratio      = %s" % format_static_obstacle_visual_ratio(static_obstacle_visual_ratio))
    print("obs_dim/action_dim= %d/%d" % (obs_dim, action_dim))
    print("observation_slots = dynamic:%d static:%d" % (
        hp_getint(training_config, "num_ob"),
        hp_getint(training_config, "num_static_ob_slots", fallback=1),
    ))
    dynamic_count = hp_getint(effective, "num_obstacles")
    static_count = hp_getint(effective, "num_static_obstacles")
    if dynamic_count > hp_getint(training_config, "num_ob"):
        print("visibility_note   = 动态船多于动态槽，策略每步只看到最近的槽数量实体")
    if static_count > hp_getint(training_config, "num_static_ob_slots", fallback=1):
        print("visibility_note   = 静态障碍多于静态槽，策略每步只看到最近的槽数量实体")

    all_summaries = []
    manifest_episodes = []
    all_colregs_step_rows = []
    all_colregs_event_rows = []
    all_colregs_episode_rows = []
    for episode_index in range(episodes):
        episode_seed = base_seed + episode_index
        for variant in evaluation_variants:
            seed_everything(episode_seed)
            env = make_custom_env(effective)
            result = run_custom_episode(
                env=env,
                model=model,
                custom_config=custom,
                evaluation_variant=variant,
                episode_id=episode_index + 1,
                placement_mode=placement_mode,
                history_length=history_length,
                action_dim=action_dim,
                max_steps=max_steps,
                device=device,
                rnn_active=rnn_active,
                guidance_settings=guidance_settings,
                colregs_audit_config=colregs_audit_config,
            )
            summary = dict(result["summary"])
            audit_summary = dict(result["colregs_episode_summary"])
            summary.update({
                "episode": episode_index + 1,
                "seed": episode_seed,
                "placement_mode": placement_mode,
            })
            audit_summary.update(
                {
                    "episode": episode_index + 1,
                    "seed": episode_seed,
                    "placement_mode": placement_mode,
                    "done_reason": result["done_reason"],
                }
            )
            summary.update(audit_summary)
            all_summaries.append(summary)
            all_colregs_step_rows.extend(result["colregs_step_rows"])
            all_colregs_event_rows.extend(result["colregs_event_rows"])
            all_colregs_episode_rows.append(audit_summary)
            stem = "episode_%03d_seed_%d_%s" % (
                episode_index + 1,
                episode_seed,
                variant.name,
            )
            title = "%s | %s | episode=%d | steps=%d | done=%s" % (
                test_name,
                variant.name,
                episode_index + 1,
                result["steps"],
                result["done_reason"],
            )
            if save_png:
                draw_trajectory(
                    result,
                    env,
                    output_dir / (stem + ".png"),
                    title,
                    show_collision_boundary=show_collision_boundary,
                    static_obstacle_visual_ratio=static_obstacle_visual_ratio,
                )
            if save_gif and episode_index % gif_every == 0:
                save_gif_with_stride(
                    result,
                    env,
                    output_dir / (stem + ".gif"),
                    title,
                    gif_fps,
                    gif_stride,
                    show_collision_boundary=show_collision_boundary,
                    static_obstacle_visual_ratio=static_obstacle_visual_ratio,
                )
            if save_trajectory:
                save_trajectory_csv(result, output_dir / (stem + "_trajectory.csv"))
            if save_action:
                write_csv(output_dir / (stem + "_action_state.csv"), result["step_rows"])

            motion = trajectory_motion_summary(
                result, read_environment_config(effective)["environment_dt"]
            )
            print(
                "episode=%d seed=%d variant=%s steps=%d done=%s reward=%.4f "
                "min_clearance_m=%.3f filter_rate=%.4f"
                % (
                    episode_index + 1,
                    episode_seed,
                    variant.name,
                    result["steps"],
                    result["done_reason"],
                    result["reward_sum"],
                    summary["minimum_clearance_m"],
                    summary["filter_active_rate"],
                )
            )
            manifest_episodes.append({"summary": summary, "motion": motion})
            if variant.rule_mode == "policy_only" and summary["filter_active_rate"] != 0.0:
                raise AssertionError("policy_only 模式仍检测到规则动作过滤器介入。")
            env.close()

    variant_summary_rows, aggregate = build_variant_summaries(
        all_summaries, evaluation_variants
    )
    filter_pair_rows, filter_dependency = build_filter_on_off_pair_summary(
        all_summaries
    )
    if save_summary:
        write_csv(output_dir / "episode_summary.csv", all_summaries)
        write_csv(output_dir / "variant_summary.csv", variant_summary_rows)
    # The paired filter-on/off result is a primary evaluation artifact and is
    # written whenever both variants were actually run, independent of the
    # optional per-episode CSV switch.
    if filter_pair_rows:
        write_csv(output_dir / "filter_on_off_pair_summary.csv", filter_pair_rows)
    if save_colregs_audit:
        write_audit_csv(
            output_dir / "colregs_step_log.csv",
            all_colregs_step_rows,
            AUDIT_STEP_FIELDS,
        )
        write_audit_csv(
            output_dir / "colregs_event_log.csv",
            all_colregs_event_rows,
            AUDIT_EVENT_FIELDS,
        )
        write_audit_csv(
            output_dir / "colregs_episode_summary.csv",
            all_colregs_episode_rows,
            AUDIT_EPISODE_FIELDS,
        )
        colregs_test_rows = [
            aggregate_audit_rows(
                all_colregs_event_rows,
                all_colregs_episode_rows,
                variant.name,
            )
            for variant in evaluation_variants
        ]
        colregs_rule_rows = []
        for variant in evaluation_variants:
            colregs_rule_rows.extend(
                build_rule_summary_rows(
                    all_colregs_event_rows,
                    variant.name,
                )
            )
        write_audit_csv(
            output_dir / "colregs_test_summary.csv",
            colregs_test_rows,
            AUDIT_TEST_FIELDS,
        )
        write_audit_csv(
            output_dir / "colregs_rule_summary.csv",
            colregs_rule_rows,
            AUDIT_RULE_FIELDS,
        )
        print("colregs_test_summary = %s" % colregs_test_rows)
        write_audit_json(
            output_dir / "colregs_config.json",
            {
                "scope": "dynamic_obstacle_vessels_only",
                "target_vessel_included": False,
                "counting_unit": "one_event_per_dynamic_vessel_per_episode",
                "audit_is_shadow_only": True,
                "config": colregs_audit_config,
                "test_summary": colregs_test_rows,
                "rule_summary": colregs_rule_rows,
                "filter_dependency": filter_dependency,
            },
        )
    if save_manifest:
        guidance_manifest = None
        if guidance_settings is not None:
            guidance_manifest = {
                "pn_config": asdict(guidance_settings.pn_config),
                "pn_max_yaw_rate_deg_s": float(
                    np.rad2deg(guidance_settings.pn_config.max_yaw_rate_rad_s)
                ),
                "pn_los_rate_deadband_deg_s": float(
                    np.rad2deg(guidance_settings.pn_config.los_rate_deadband_rad_s)
                ),
                "pn_blend_weight": guidance_settings.blend_weight,
                "pn_blend_thrust": guidance_settings.blend_thrust,
                "pn_use_measured_target_position": guidance_settings.use_measured_target_position,
            }
        manifest = {
            "created_at": timestamp,
            "training_config": str(training_config_path),
            "custom_config": str(custom_path),
            "training_run_name": actual_run_name,
            "checkpoint": str(checkpoint_path),
            "device": device,
            "base_seed": base_seed,
            "placement_mode": placement_mode,
            "show_collision_boundary": show_collision_boundary,
            "static_obstacle_visual_ratio": format_static_obstacle_visual_ratio(
                static_obstacle_visual_ratio
            ),
            "evaluation_variants": [asdict(variant) for variant in evaluation_variants],
            "uses_evaluation_variants": uses_variant_configuration,
            "legacy_agent_rule_mode": legacy_rule_mode,
            "guidance": guidance_manifest,
            "colregs_audit": {
                "scope": "dynamic_obstacle_vessels_only",
                "target_vessel_included": False,
                "counting_unit": "one_event_per_dynamic_vessel_per_episode",
                "config": colregs_audit_config,
                "saved": save_colregs_audit,
            },
            "network_contract": {
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "dynamic_observation_slots": hp_getint(training_config, "num_ob"),
                "static_observation_slots": hp_getint(training_config, "num_static_ob_slots", fallback=1),
            },
            "persistent_rule_state": {
                "enabled": bool(read_environment_config(effective).get("colregs_persistent_state_enabled", False)),
                "exit_steps": int(read_environment_config(effective).get("colregs_persistent_exit_steps", 3)),
            },
            "aggregate": aggregate,
            "filter_dependency": filter_dependency,
            "episodes": manifest_episodes,
        }
        with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as file:
            json.dump(_json_safe(manifest), file, ensure_ascii=False, indent=2)

    episode_results_by_variant = {
        variant.name: [
            row for row in all_summaries
            if row.get("evaluation_variant") == variant.name
        ]
        for variant in evaluation_variants
    }
    print_multi_variant_terminal_report(
        episode_results_by_variant,
        event_rows=all_colregs_event_rows,
        audit_episode_rows=all_colregs_episode_rows,
        title="see_trained_custom 终端评估 / custom evaluation",
    )

    print("aggregate         = %s" % aggregate)
    if filter_pair_rows:
        print("filter_dependency = %s" % filter_dependency)
    print("evaluation_done   = %s" % output_dir)


if __name__ == "__main__":
        main()
