"""比例导引 PN 的独立数学与边界自检。

在项目根目录运行：

    python scripts/check_proportional_navigation.py

这些检查不加载训练网络，也不修改训练数据。它们验证 PN 的坐标符号、
目标速度估计、航向捕获量纲、3DOF 物理限幅和动作有限性，适合作为后续
修改导引参数后的回归检查。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamics.usv_3dof import USV3DOFConfig
from guidance.proportional_navigation import (
    PNConfig,
    PNVesselParameters,
    ProportionalNavigationGuidance,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def make_guidance(**config_overrides) -> ProportionalNavigationGuidance:
    values = {
        "target_velocity_source": "measured_state",
        "los_rate_filter_alpha": 0.0,
        "heading_assist_enabled": True,
        "heading_assist_gain": 1.0,
        "goal_radius_m": 20.0,
        "slowdown_range_m": 50.0,
        "max_range_m": 1000.0,
    }
    values.update(config_overrides)
    return ProportionalNavigationGuidance(
        PNConfig(**values),
        PNVesselParameters.from_usv_config(USV3DOFConfig()),
    )


def test_positive_los_rate_turns_left() -> None:
    """目标 LOS 逆时针转动时，PN 应给出正舵角，即向左转。"""
    guidance = make_guidance()
    action, diag = guidance.compute_action(
        own_position_m=[0.0, 0.0],
        own_velocity_world_mps=[2.0, 0.0],
        own_heading_rad=0.0,
        own_yaw_rate_rad_s=0.0,
        own_surge_speed_mps=2.0,
        target_position_m=[100.0, 0.0],
        target_state_velocity_mps=[0.0, 1.0],
        dt_s=1.0,
    )
    require(diag.raw_los_rate_rad_s > 0.0, "正 LOS 角速度场景构造失败。")
    require(diag.lateral_accel_command_mps2 > 0.0, "PN 未生成正法向加速度。")
    require(action[1] > 0.0, "正 LOS 角速度应对应正舵角（向左）。")
    print("PASS positive_los_rate: rudder_norm=%.6f" % action[1])


def test_negative_los_rate_turns_right() -> None:
    """目标 LOS 顺时针转动时，PN 应给出负舵角，即向右转。"""
    guidance = make_guidance()
    action, diag = guidance.compute_action(
        own_position_m=[0.0, 0.0],
        own_velocity_world_mps=[2.0, 0.0],
        own_heading_rad=0.0,
        own_yaw_rate_rad_s=0.0,
        own_surge_speed_mps=2.0,
        target_position_m=[100.0, 0.0],
        target_state_velocity_mps=[0.0, -1.0],
        dt_s=1.0,
    )
    require(diag.raw_los_rate_rad_s < 0.0, "负 LOS 角速度场景构造失败。")
    require(diag.lateral_accel_command_mps2 < 0.0, "PN 未生成负法向加速度。")
    require(action[1] < 0.0, "负 LOS 角速度应对应负舵角（向右）。")
    print("PASS negative_los_rate: rudder_norm=%.6f" % action[1])


def test_static_target_heading_capture() -> None:
    """静止目标、初始不闭合时，航向捕获项仍应把船首转向目标。"""
    guidance = make_guidance(target_velocity_source="finite_difference")
    action, diag = guidance.compute_action(
        own_position_m=[0.0, 0.0],
        own_velocity_world_mps=[0.0, 0.0],
        own_heading_rad=0.0,
        own_yaw_rate_rad_s=0.0,
        own_surge_speed_mps=0.0,
        target_position_m=[100.0, 100.0],
        dt_s=1.0,
        target_is_static=True,
    )
    require(abs(diag.lateral_accel_command_mps2) <= 1.0e-12, "静止且零闭合速度不应产生 PN 法向加速度。")
    require(diag.heading_capture_yaw_rate_rad_s > 0.0, "航向捕获应先生成正艏摇角速度。")
    require(
        np.isclose(
            diag.desired_yaw_rate_rad_s,
            diag.heading_capture_yaw_rate_rad_s,
            atol=1.0e-12,
        ),
        "静止且零闭合速度时，最终艏摇参考应只来自航向捕获项。",
    )
    require(action[1] > 0.0, "目标位于左前方时航向捕获应给出左转舵角。")
    require(action[0] > -1.0, "PN-only 速度保持器应给出非零前进推力。")
    print("PASS static_heading_capture: rudder_norm=%.6f thrust_norm=%.6f" % (action[1], action[0]))


def test_heading_capture_uses_yaw_rate_units() -> None:
    """验证 r_acq=k_psi*e_psi，而不是把角度直接叠加到舵角控制器。"""
    heading_gain = 0.50  # 1/s
    yaw_limit = np.deg2rad(10.0)
    guidance = make_guidance(
        heading_assist_gain=heading_gain,
        max_yaw_rate_rad_s=yaw_limit,
    )
    _, diag = guidance.compute_action(
        own_position_m=[0.0, 0.0],
        own_velocity_world_mps=[0.0, 0.0],
        own_heading_rad=0.0,
        own_yaw_rate_rad_s=0.0,
        own_surge_speed_mps=0.0,
        target_position_m=[0.0, 100.0],
        dt_s=1.0,
        target_is_static=True,
    )
    expected_capture_rate = min(heading_gain * (np.pi / 2.0), yaw_limit)
    require(
        np.isclose(diag.heading_capture_yaw_rate_rad_s, expected_capture_rate, atol=1.0e-12),
        "航向捕获项未按 k_psi[1/s] * e_psi[rad] 生成角速度。",
    )
    require(
        abs(diag.desired_yaw_rate_rad_s) <= yaw_limit + 1.0e-12,
        "航向捕获与 PN 叠加后的最终艏摇参考没有限幅。",
    )
    print("PASS heading_capture_units: desired_yaw_rate=%.6f" % diag.desired_yaw_rate_rad_s)


def test_rejects_yaw_rate_above_3dof_limit() -> None:
    """PN 不能请求超过 3DOF 状态硬限幅的艏摇角速度。"""
    vessel = PNVesselParameters.from_usv_config(
        USV3DOFConfig(max_yaw_rate=np.deg2rad(10.0))
    )
    try:
        ProportionalNavigationGuidance(
            PNConfig(max_yaw_rate_rad_s=np.deg2rad(12.0)),
            vessel,
        )
    except ValueError as error:
        require("超过当前 3DOF 模型的硬上限" in str(error), "物理上限报错信息不明确。")
    else:
        raise AssertionError("PN 接受了超过 3DOF 最大艏摇角速度的配置。")
    print("PASS yaw_rate_physical_limit")


def test_static_target_rejects_difference_noise() -> None:
    """静止目标强制零速度，位置测量变化不能生成虚假目标速度。"""
    guidance = make_guidance(target_velocity_source="finite_difference")
    guidance.compute_action(
        own_position_m=[0.0, 0.0],
        own_velocity_world_mps=[1.0, 0.0],
        own_heading_rad=0.0,
        own_yaw_rate_rad_s=0.0,
        own_surge_speed_mps=1.0,
        target_position_m=[100.0, 0.0],
        dt_s=1.0,
        target_is_static=True,
    )
    _, diag = guidance.compute_action(
        own_position_m=[1.0, 0.0],
        own_velocity_world_mps=[1.0, 0.0],
        own_heading_rad=0.0,
        own_yaw_rate_rad_s=0.0,
        own_surge_speed_mps=1.0,
        target_position_m=[104.0, -3.0],
        dt_s=1.0,
        target_is_static=True,
    )
    require(diag.target_speed_estimate_mps <= 1.0e-12, "静止目标不应从测量差分得到假速度。")
    print("PASS static_velocity_rejection")


def test_limits_and_degenerate_geometry() -> None:
    """极端相对运动和零距离都不能导致动作越界或 NaN。"""
    guidance = make_guidance(
        max_lateral_accel_mps2=0.15,
        max_yaw_rate_rad_s=np.deg2rad(5.0),
        target_speed_max_mps=2.0,
    )
    action, diag = guidance.compute_action(
        own_position_m=[0.0, 0.0],
        own_velocity_world_mps=[3.0, 0.0],
        own_heading_rad=0.0,
        own_yaw_rate_rad_s=0.0,
        own_surge_speed_mps=3.0,
        target_position_m=[10.0, 1.0],
        target_state_velocity_mps=[-20.0, 20.0],
        dt_s=1.0,
    )
    require(np.all(np.isfinite(action)), "极端相对运动产生了非有限动作。")
    require(np.max(np.abs(action)) <= 1.0 + 1.0e-12, "PN 动作未被归一化限幅。")
    require(abs(diag.lateral_accel_command_mps2) <= 0.15 + 1.0e-12, "法向加速度限幅失效。")
    require(abs(diag.yaw_rate_command_rad_s) <= np.deg2rad(5.0) + 1.0e-12, "艏摇角速度限幅失效。")

    degenerate_action, degenerate_diag = guidance.compute_action(
        own_position_m=[0.0, 0.0],
        own_velocity_world_mps=[0.0, 0.0],
        own_heading_rad=0.0,
        own_yaw_rate_rad_s=0.0,
        own_surge_speed_mps=0.0,
        target_position_m=[0.0, 0.0],
        dt_s=1.0,
    )
    require(np.all(np.isfinite(degenerate_action)), "零距离几何产生了非有限动作。")
    require(not degenerate_diag.active, "零距离几何应返回非激活 PN 诊断。")
    print("PASS limits_and_degenerate_geometry")


def main() -> None:
    test_positive_los_rate_turns_left()
    test_negative_los_rate_turns_right()
    test_static_target_heading_capture()
    test_heading_capture_uses_yaw_rate_units()
    test_rejects_yaw_rate_above_3dof_limit()
    test_static_target_rejects_difference_noise()
    test_limits_and_degenerate_geometry()
    print("ALL PN CHECKS PASSED")


if __name__ == "__main__":
    main()
