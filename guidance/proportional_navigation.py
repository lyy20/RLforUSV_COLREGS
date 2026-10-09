"""测试基线用的比例导引（Proportional Navigation, PN）实现。

本模块只产生归一化候选动作 ``[thrust_norm, rudder_norm]``，其范围为
``[-1, 1]``。它不直接修改世界状态、不绕过 COLREGs，也不绕过 3DOF
模型中的推力/舵角变化率、最大舵角、最大艏摇角速度等物理限制。

坐标与单位约定
------------
* 位置和世界速度均使用地固坐标系，单位分别为 m、m/s；
* 艏向、LOS 角速度和艏摇角速度使用 rad、rad/s；
* 机体系正 x 指向船首，正 y 指向左舷；因此正舵角/正艏摇为逆时针左转；
* ``target_position_m - own_position_m`` 定义为从本船指向目标的 LOS 向量。

经典 PN 的核心量为

.. math::

    \boldsymbol r &= \boldsymbol p_T-\boldsymbol p_O, \\
    \boldsymbol v_{rel} &= \boldsymbol v_T-\boldsymbol v_O, \\
    V_c &= \max\left(-\frac{\boldsymbol r^\mathsf{T}\boldsymbol v_{rel}}
    {\|\boldsymbol r\|},0\right), \\
    \dot\lambda &= \frac{r_xv_{rel,y}-r_yv_{rel,x}}
    {\|\boldsymbol r\|^2}, \\
    a_n &= N V_c\dot\lambda.

其中 ``N`` 是比例导引系数，``a_n`` 为期望法向加速度。项目中的 USV
不是可直接命令横向加速度的质点，而是由推力和舵角驱动的欠驱动 3DOF
模型。因此本实现采用“PN 外环 + 舵角跟踪内环”：将 ``a_n`` 转成期望艏摇
角速度，再由舵角命令使实际艏摇角速度向其靠近。静止目标且初始艏向偏离
很大时，纯 PN 可能因初始闭合速度为零而暂时不转向，所以额外加入可关闭的
LOS 航向捕获项。这一项只帮助进入 PN 有效追踪几何，不替代 PN 主公式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from dynamics.usv_3dof import wrap_angle


_EPSILON = 1.0e-9
_VALID_TARGET_VELOCITY_SOURCES = {
    "finite_difference",
    "measured_state",
    "world_oracle",
    "zero",
}


def _as_vector2(values: Sequence[float], name: str) -> np.ndarray:
    """将输入安全地转换为二维有限向量，避免 PN 将 NaN 传入环境。"""
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != 2 or not np.all(np.isfinite(vector)):
        raise ValueError("%s 必须是两个有限浮点数。" % name)
    return vector.copy()


def _clip_unit(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


@dataclass(frozen=True)
class PNVesselParameters:
    """PN 推力保持器所需的 3DOF 物理参数。

    ``damping_surge`` 与 ``damping_surge_quad`` 与当前
    :mod:`dynamics.usv_3dof` 的纵荡阻尼定义一致：

    .. math:: D_u(u)=d_u u+d_{u2}|u|u.

    因而在稳态前进速度 ``u_ref`` 附近，前馈推力可取 ``D_u(u_ref)``。
    这只是给 PN-only 基线一个合理的速度保持器，并不改写 3DOF 方程。
    """

    min_thrust: float
    max_thrust: float
    damping_surge: float
    damping_surge_quad: float
    # 这是 3DOF 状态积分中实际使用的硬限幅，而不是 PN 自行假设的能力上界。
    # PN 的目标艏摇角速度不得超过它，否则导引器会持续请求模型不可能实现的转向。
    max_yaw_rate_rad_s: float

    @classmethod
    def from_usv_config(cls, config) -> "PNVesselParameters":
        return cls(
            min_thrust=float(config.min_thrust),
            max_thrust=float(config.max_thrust),
            damping_surge=float(config.damping_surge),
            damping_surge_quad=float(config.damping_surge_quad),
            max_yaw_rate_rad_s=float(config.max_yaw_rate),
        )

    def validate(self) -> None:
        if not np.isfinite(self.min_thrust) or not np.isfinite(self.max_thrust):
            raise ValueError("PN 推力上下限必须是有限数。")
        if self.max_thrust <= self.min_thrust:
            raise ValueError("PN 推力上限必须大于推力下限。")
        if self.damping_surge < 0.0 or self.damping_surge_quad < 0.0:
            raise ValueError("PN 纵荡阻尼不能为负数。")
        if not np.isfinite(self.max_yaw_rate_rad_s) or self.max_yaw_rate_rad_s <= 0.0:
            raise ValueError("3DOF 最大艏摇角速度必须是有限正数。")


@dataclass(frozen=True)
class PNConfig:
    """比例导引与其 3DOF 舵角/速度适配器的配置。

    单位约定：

    * ``max_yaw_rate_rad_s`` 为 PN 允许请求的最大艏摇角速度，必须不大于
      3DOF 模型的 ``max_yaw_rate``；
    * ``yaw_rate_tracking_gain`` 的单位为 s，使
      ``K_r (r_des-r)`` 成为无量纲归一化舵角命令；
    * ``heading_assist_gain`` 的单位为 1/s，使 ``k_psi e_psi`` 成为
      艏摇角速度。该项仅用于静止目标或初始不闭合时的航向捕获。
    """

    navigation_constant: float = 3.0
    max_range_m: float = 1000.0
    min_speed_mps: float = 0.30
    min_closing_speed_mps: float = 0.05
    max_lateral_accel_mps2: float = 1.0
    max_yaw_rate_rad_s: float = np.deg2rad(12.0)
    los_rate_filter_alpha: float = 0.80
    los_rate_deadband_rad_s: float = np.deg2rad(0.10)
    yaw_rate_tracking_gain: float = 2.0
    heading_assist_enabled: bool = True
    heading_assist_gain: float = 1.0
    cruise_speed_mps: float = 2.0
    min_approach_speed_mps: float = 0.70
    slowdown_range_m: float = 150.0
    goal_radius_m: float = 100.0
    speed_control_gain: float = 25.0
    target_velocity_source: str = "finite_difference"
    static_target_zero_velocity: bool = True
    target_velocity_filter_alpha: float = 0.70
    target_speed_max_mps: float = 3.0

    def validate(self) -> None:
        """在每次测试开始前拒绝会导致非物理行为的配置。"""
        positive_values = {
            "PN_NAVIGATION_CONSTANT": self.navigation_constant,
            "PN_MAX_RANGE_M": self.max_range_m,
            "PN_MIN_SPEED_MPS": self.min_speed_mps,
            "PN_MAX_LATERAL_ACCEL_MPS2": self.max_lateral_accel_mps2,
            "PN_MAX_YAW_RATE_DEG_S": self.max_yaw_rate_rad_s,
            "PN_CRUISE_SPEED_MPS": self.cruise_speed_mps,
            "PN_SLOWDOWN_RANGE_M": self.slowdown_range_m,
            "PN_GOAL_RADIUS_M": self.goal_radius_m,
            "PN_SPEED_CONTROL_GAIN": self.speed_control_gain,
            "PN_TARGET_SPEED_MAX_MPS": self.target_speed_max_mps,
        }
        for name, value in positive_values.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("%s 必须为有限正数。" % name)
        nonnegative_values = {
            "PN_MIN_CLOSING_SPEED_MPS": self.min_closing_speed_mps,
            "PN_LOS_RATE_DEADBAND_DEG_S": self.los_rate_deadband_rad_s,
            "PN_YAW_RATE_TRACKING_GAIN": self.yaw_rate_tracking_gain,
            "PN_HEADING_ASSIST_GAIN": self.heading_assist_gain,
            "PN_MIN_APPROACH_SPEED_MPS": self.min_approach_speed_mps,
        }
        for name, value in nonnegative_values.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("%s 必须为有限非负数。" % name)
        for name, value in {
            "PN_LOS_RATE_FILTER_ALPHA": self.los_rate_filter_alpha,
            "PN_TARGET_VELOCITY_FILTER_ALPHA": self.target_velocity_filter_alpha,
        }.items():
            if not 0.0 <= value < 1.0:
                raise ValueError("%s 必须位于 [0, 1) 内。" % name)
        if self.slowdown_range_m <= self.goal_radius_m:
            raise ValueError("PN_SLOWDOWN_RANGE_M 必须大于 PN_GOAL_RADIUS_M。")
        if self.cruise_speed_mps < self.min_approach_speed_mps:
            raise ValueError("PN_CRUISE_SPEED_MPS 必须不小于 PN_MIN_APPROACH_SPEED_MPS。")
        source = str(self.target_velocity_source).strip().lower()
        if source not in _VALID_TARGET_VELOCITY_SOURCES:
            raise ValueError(
                "PN_TARGET_VELOCITY_SOURCE=%r 无效，可选：%s。"
                % (self.target_velocity_source, ", ".join(sorted(_VALID_TARGET_VELOCITY_SOURCES)))
            )


@dataclass
class _PNState:
    """单个测试回合内部保存的测量记忆，reset 后绝不跨回合泄漏。"""

    previous_target_position_m: Optional[np.ndarray] = None
    filtered_target_velocity_mps: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=float)
    )
    filtered_los_rate_rad_s: float = 0.0
    los_rate_initialized: bool = False


@dataclass(frozen=True)
class PNDiagnostics:
    """用于 CSV 与后续论文统计的 PN 中间量。"""

    active: bool
    reason: str
    target_velocity_source: str
    range_m: float
    closing_speed_mps: float
    raw_los_rate_rad_s: float
    filtered_los_rate_rad_s: float
    lateral_accel_command_mps2: float
    # 仅由经典 PN a_n / V_g 得到的外环艏摇角速度命令。
    yaw_rate_command_rad_s: float
    heading_error_rad: float
    # 航向捕获项和两项叠加、限幅后的最终期望艏摇角速度。
    heading_capture_yaw_rate_rad_s: float
    desired_yaw_rate_rad_s: float
    target_speed_estimate_mps: float
    desired_surge_speed_mps: float
    thrust_command_n: float
    pn_thrust_norm: float
    pn_rudder_norm: float

    def as_dict(self) -> dict:
        """返回仅包含标量的字典，便于直接写入逐步动作 CSV。"""
        return {
            "pn_active": int(self.active),
            "pn_reason": self.reason,
            "pn_target_velocity_source": self.target_velocity_source,
            "pn_range_m": self.range_m,
            "pn_closing_speed_mps": self.closing_speed_mps,
            "pn_los_rate_rad_s": self.raw_los_rate_rad_s,
            "pn_los_rate_filtered_rad_s": self.filtered_los_rate_rad_s,
            "pn_lateral_accel_command_mps2": self.lateral_accel_command_mps2,
            "pn_yaw_rate_command_rad_s": self.yaw_rate_command_rad_s,
            "pn_heading_error_rad": self.heading_error_rad,
            "pn_heading_capture_yaw_rate_rad_s": self.heading_capture_yaw_rate_rad_s,
            "pn_desired_yaw_rate_rad_s": self.desired_yaw_rate_rad_s,
            "pn_target_speed_estimate_mps": self.target_speed_estimate_mps,
            "pn_desired_surge_speed_mps": self.desired_surge_speed_mps,
            "pn_thrust_command_n": self.thrust_command_n,
            "pn_thrust_norm": self.pn_thrust_norm,
            "pn_rudder_norm": self.pn_rudder_norm,
        }


class ProportionalNavigationGuidance:
    """适配当前欠驱动 3DOF USV 的测试期比例导引器。"""

    def __init__(self, config: PNConfig, vessel: PNVesselParameters):
        config.validate()
        vessel.validate()
        if config.max_yaw_rate_rad_s > vessel.max_yaw_rate_rad_s + _EPSILON:
            raise ValueError(
                "PN_MAX_YAW_RATE_DEG_S=%.3f 超过当前 3DOF 模型的硬上限 %.3f deg/s；"
                "请降低 PN 配置或使用与训练一致的动力学配置。"
                % (
                    np.rad2deg(config.max_yaw_rate_rad_s),
                    np.rad2deg(vessel.max_yaw_rate_rad_s),
                )
            )
        self.config = config
        self.vessel = vessel
        self.state = _PNState()

    def reset(self) -> None:
        """清除上一回合的 LOS 与目标速度滤波状态。"""
        self.state = _PNState()

    def _estimate_target_velocity(
        self,
        target_position_m: np.ndarray,
        dt_s: float,
        target_is_static: bool,
        state_velocity_mps: Optional[Sequence[float]],
    ) -> np.ndarray:
        """从允许的测量来源估计目标地固速度。

        ``finite_difference`` 使用相邻测量位置：

        .. math:: \hat{\boldsymbol v}_T(k)=
            \frac{\hat{\boldsymbol p}_T(k)-\hat{\boldsymbol p}_T(k-1)}{\Delta t}.

        随后进行一阶低通并按物理速度上限裁剪。对当前静止目标场景，
        ``static_target_zero_velocity`` 会强制输出零，避免位置测量噪声经差分
        被放大成虚假的目标机动。
        """
        cfg = self.config
        source = str(cfg.target_velocity_source).strip().lower()
        previous = self.state.previous_target_position_m
        self.state.previous_target_position_m = target_position_m.copy()

        if target_is_static and cfg.static_target_zero_velocity:
            self.state.filtered_target_velocity_mps = np.zeros(2, dtype=float)
            return self.state.filtered_target_velocity_mps.copy()
        if source == "zero":
            raw_velocity = np.zeros(2, dtype=float)
        elif source == "finite_difference":
            if previous is None or dt_s <= _EPSILON:
                raw_velocity = np.zeros(2, dtype=float)
            else:
                raw_velocity = (target_position_m - previous) / dt_s
        else:
            # ``measured_state`` 与 ``world_oracle`` 的差别由调用方决定：
            # 前者传传感器/估计实体速度，后者只在理想上界诊断时传真实速度。
            raw_velocity = (
                _as_vector2(state_velocity_mps, "target_state_velocity_mps")
                if state_velocity_mps is not None
                else np.zeros(2, dtype=float)
            )

        raw_speed = float(np.linalg.norm(raw_velocity))
        if raw_speed > cfg.target_speed_max_mps:
            raw_velocity *= cfg.target_speed_max_mps / max(raw_speed, _EPSILON)

        alpha = cfg.target_velocity_filter_alpha
        self.state.filtered_target_velocity_mps = (
            alpha * self.state.filtered_target_velocity_mps
            + (1.0 - alpha) * raw_velocity
        )
        return self.state.filtered_target_velocity_mps.copy()

    def _desired_surge_speed(self, range_m: float) -> float:
        """在成功半径前平滑减速，避免 PN-only 在目标周边持续全速冲刺。"""
        cfg = self.config
        if range_m <= cfg.goal_radius_m:
            return 0.0
        ratio = np.clip(
            (range_m - cfg.goal_radius_m)
            / max(cfg.slowdown_range_m - cfg.goal_radius_m, _EPSILON),
            0.0,
            1.0,
        )
        return float(
            cfg.min_approach_speed_mps
            + ratio * (cfg.cruise_speed_mps - cfg.min_approach_speed_mps)
        )

    def _normalized_thrust(self, thrust_n: float) -> float:
        """反解项目中的线性归一化推力映射，确保 PN 与 SAC 同一动作语义。"""
        clipped = float(np.clip(thrust_n, self.vessel.min_thrust, self.vessel.max_thrust))
        normalized = (
            2.0
            * (clipped - self.vessel.min_thrust)
            / (self.vessel.max_thrust - self.vessel.min_thrust)
            - 1.0
        )
        return _clip_unit(normalized)

    def _inactive_diagnostics(self, reason: str, source: str, range_m: float) -> PNDiagnostics:
        zero_thrust_norm = self._normalized_thrust(self.vessel.min_thrust)
        return PNDiagnostics(
            active=False,
            reason=reason,
            target_velocity_source=source,
            range_m=range_m,
            closing_speed_mps=0.0,
            raw_los_rate_rad_s=0.0,
            filtered_los_rate_rad_s=0.0,
            lateral_accel_command_mps2=0.0,
            yaw_rate_command_rad_s=0.0,
            heading_error_rad=0.0,
            heading_capture_yaw_rate_rad_s=0.0,
            desired_yaw_rate_rad_s=0.0,
            target_speed_estimate_mps=0.0,
            desired_surge_speed_mps=0.0,
            thrust_command_n=self.vessel.min_thrust,
            pn_thrust_norm=zero_thrust_norm,
            pn_rudder_norm=0.0,
        )

    def compute_action(
        self,
        own_position_m: Sequence[float],
        own_velocity_world_mps: Sequence[float],
        own_heading_rad: float,
        own_yaw_rate_rad_s: float,
        own_surge_speed_mps: float,
        target_position_m: Sequence[float],
        dt_s: float,
        target_is_static: bool = False,
        target_state_velocity_mps: Optional[Sequence[float]] = None,
    ) -> tuple[np.ndarray, PNDiagnostics]:
        """计算一条 PN 候选动作，返回 ``[推力归一化值, 舵角归一化值]``。

        方向检查：在当前 ``x`` 向右、``y`` 向上、逆时针为正的约定下，若
        目标 LOS 逆时针旋转（``dot(lambda)>0``）且正在闭合，则
        ``a_n=N*V_c*dot(lambda)>0``，最终舵角命令也偏正，即向左转；这与
        本项目 3DOF 正舵角产生正艏摇的符号一致。
        """
        cfg = self.config
        source = str(cfg.target_velocity_source).strip().lower()
        own_position = _as_vector2(own_position_m, "own_position_m")
        own_velocity = _as_vector2(own_velocity_world_mps, "own_velocity_world_mps")
        target_position = _as_vector2(target_position_m, "target_position_m")
        if not np.isfinite(own_heading_rad) or not np.isfinite(own_yaw_rate_rad_s):
            raise ValueError("own_heading_rad 与 own_yaw_rate_rad_s 必须为有限数。")
        if not np.isfinite(own_surge_speed_mps) or not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("own_surge_speed_mps 与 dt_s 必须为有效有限数，且 dt_s > 0。")

        target_velocity = self._estimate_target_velocity(
            target_position,
            float(dt_s),
            bool(target_is_static),
            target_state_velocity_mps,
        )
        line_of_sight = target_position - own_position
        range_m = float(np.linalg.norm(line_of_sight))
        if range_m <= _EPSILON:
            diagnostics = self._inactive_diagnostics("coincident_position", source, range_m)
            return np.array([diagnostics.pn_thrust_norm, diagnostics.pn_rudder_norm]), diagnostics
        if range_m > cfg.max_range_m:
            diagnostics = self._inactive_diagnostics("outside_guidance_range", source, range_m)
            return np.array([diagnostics.pn_thrust_norm, diagnostics.pn_rudder_norm]), diagnostics

        relative_velocity = target_velocity - own_velocity
        closing_speed = max(
            -float(np.dot(line_of_sight, relative_velocity)) / max(range_m, _EPSILON),
            0.0,
        )
        # 二维叉积的 z 分量。该式就是 LOS 方位角 lambda 的时间导数。
        raw_los_rate = float(
            (line_of_sight[0] * relative_velocity[1]
             - line_of_sight[1] * relative_velocity[0])
            / max(range_m * range_m, _EPSILON)
        )
        if not self.state.los_rate_initialized:
            filtered_los_rate = raw_los_rate
            self.state.los_rate_initialized = True
        else:
            filtered_los_rate = (
                cfg.los_rate_filter_alpha * self.state.filtered_los_rate_rad_s
                + (1.0 - cfg.los_rate_filter_alpha) * raw_los_rate
            )
        self.state.filtered_los_rate_rad_s = float(filtered_los_rate)
        if abs(filtered_los_rate) < cfg.los_rate_deadband_rad_s:
            filtered_los_rate = 0.0

        # PN 主公式。闭合速度很小时将其置零，避免微小的几何噪声产生持续摆舵。
        effective_closing_speed = (
            closing_speed if closing_speed >= cfg.min_closing_speed_mps else 0.0
        )
        lateral_accel = float(np.clip(
            cfg.navigation_constant * effective_closing_speed * filtered_los_rate,
            -cfg.max_lateral_accel_mps2,
            cfg.max_lateral_accel_mps2,
        ))
        ground_speed = max(float(np.linalg.norm(own_velocity)), cfg.min_speed_mps)
        # 理想平面转弯近似 a_n = V_g r。因而 a_n [m/s^2] / V_g [m/s]
        # 直接得到 PN 外环的期望艏摇角速度 r_PN [rad/s]。地速过小时使用
        # min_speed_mps 防止除零；最终命令还会受 PN 和 3DOF 两层上限约束。
        pn_yaw_rate_command = float(np.clip(
            lateral_accel / ground_speed,
            -cfg.max_yaw_rate_rad_s,
            cfg.max_yaw_rate_rad_s,
        ))

        # 只在 PN 法向加速度之外补充航向捕获。它解决静止目标、初始艏向偏差时
        # Vc=0 而纯 PN 暂不转向的问题；关闭该项即可得到严格的纯 PN 外环。
        los_bearing = float(np.arctan2(line_of_sight[1], line_of_sight[0]))
        heading_error = wrap_angle(los_bearing - float(own_heading_rad))
        # 航向捕获同样先转换成角速度：r_acq = sat(k_psi * e_psi)。
        # k_psi 的单位为 1/s，e_psi 的单位为 rad，所以二者乘积为 rad/s；
        # 这避免旧实现把“角度误差”直接加到“角速度误差”上的量纲错误。
        heading_capture_yaw_rate = 0.0
        if cfg.heading_assist_enabled:
            heading_capture_yaw_rate = float(np.clip(
                cfg.heading_assist_gain * heading_error,
                -cfg.max_yaw_rate_rad_s,
                cfg.max_yaw_rate_rad_s,
            ))
        # 最终参考艏摇角速度同时包含 PN 截获趋势和初始 LOS 对准趋势。
        desired_yaw_rate = float(np.clip(
            pn_yaw_rate_command + heading_capture_yaw_rate,
            -cfg.max_yaw_rate_rad_s,
            cfg.max_yaw_rate_rad_s,
        ))
        # K_r 的单位为 s。r_des-r 为 rad/s，乘积为无量纲数，再裁剪成与
        # SAC 完全相同的 [-1, 1] 归一化舵角动作语义。
        rudder_norm = _clip_unit(
            cfg.yaw_rate_tracking_gain
            * (desired_yaw_rate - float(own_yaw_rate_rad_s))
        )

        desired_surge_speed = self._desired_surge_speed(range_m)
        # 推力前馈抵消纵荡阻尼，P 项补偿当前 u 与参考 u_ref 的差异：
        # T_cmd = sat[d_u*u_ref+d_u2*|u_ref|*u_ref+K_u*(u_ref-u)].
        drag_feedforward = (
            self.vessel.damping_surge * desired_surge_speed
            + self.vessel.damping_surge_quad
            * abs(desired_surge_speed)
            * desired_surge_speed
        )
        thrust_command = float(np.clip(
            drag_feedforward
            + cfg.speed_control_gain * (desired_surge_speed - float(own_surge_speed_mps)),
            self.vessel.min_thrust,
            self.vessel.max_thrust,
        ))
        thrust_norm = self._normalized_thrust(thrust_command)
        diagnostics = PNDiagnostics(
            active=True,
            reason="active",
            target_velocity_source=source,
            range_m=range_m,
            closing_speed_mps=closing_speed,
            raw_los_rate_rad_s=raw_los_rate,
            filtered_los_rate_rad_s=float(filtered_los_rate),
            lateral_accel_command_mps2=lateral_accel,
            yaw_rate_command_rad_s=pn_yaw_rate_command,
            heading_error_rad=heading_error,
            heading_capture_yaw_rate_rad_s=heading_capture_yaw_rate,
            desired_yaw_rate_rad_s=desired_yaw_rate,
            target_speed_estimate_mps=float(np.linalg.norm(target_velocity)),
            desired_surge_speed_mps=desired_surge_speed,
            thrust_command_n=thrust_command,
            pn_thrust_norm=thrust_norm,
            pn_rudder_norm=rudder_norm,
        )
        return np.array([thrust_norm, rudder_norm], dtype=float), diagnostics
