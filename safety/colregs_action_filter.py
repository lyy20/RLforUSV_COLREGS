from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .colregs import EncounterType


@dataclass(frozen=True)
class ActionFilterDecision:
    raw_action: Tuple[float, ...]
    constrained_action: Tuple[float, ...]
    # 纯 COLREGs 规则层输出；这里不包含后续 EMA 平滑器的影响。
    rule_action: Tuple[float, ...]
    applied_action: Tuple[float, ...]
    mode: str
    reason: str
    active: bool = False
    encounter_type: Optional[str] = None
    target_id: Optional[str] = None
    give_way_vessel: Optional[str] = None
    stand_on_vessel: Optional[str] = None
    # 规则教师审计所需的几何量；不进入策略网络输入。
    dcpa: Optional[float] = None
    tcpa: Optional[float] = None

    def as_dict(self):
        return {
            "raw_action": self.raw_action,
            "constrained_action": self.constrained_action,
            "rule_action": self.rule_action,
            "applied_action": self.applied_action,
            "mode": self.mode,
            "reason": self.reason,
            "active": self.active,
            "encounter_type": self.encounter_type,
            "target_id": self.target_id,
            "give_way_vessel": self.give_way_vessel,
            "stand_on_vessel": self.stand_on_vessel,
            "dcpa": self.dcpa,
            "tcpa": self.tcpa,
        }


@dataclass(frozen=True)
class COLREGsActionSetConfig:
    action_dim: int = 2
    min_thrust: float = -1.0
    max_thrust: float = 1.0
    min_rudder: float = -1.0
    max_rudder: float = 1.0
    emergency_distance: float = 0.12

    head_on_actions: Tuple[Tuple[float, float], ...] = (
        (0.15, -0.85),
        (0.35, -0.70),
        (-0.10, -1.00),
    )
    crossing_give_way_actions: Tuple[Tuple[float, float], ...] = (
        (-0.20, -0.90),
        (0.00, -0.75),
        (0.20, -1.00),
    )
    overtaking_actions: Tuple[Tuple[float, float], ...] = (
        (0.10, -0.55),
        (0.25, -0.45),
        (-0.15, -0.70),
    )
    undefined_risk_actions: Tuple[Tuple[float, float], ...] = (
        (-0.25, -0.80),
        (0.00, -0.65),
    )
    emergency_stand_on_actions: Tuple[Tuple[float, float], ...] = (
        (-0.20, -0.55),
        (0.00, -0.45),
    )


class COLREGsActionSetFilter:
    """Project normalized USV actions onto a COLREGs-compliant action set."""

    def __init__(self, config: COLREGsActionSetConfig | None = None):
        self.config = config or COLREGsActionSetConfig()

    def filter_action(self, world, agent, raw_action) -> ActionFilterDecision:
        raw = self._coerce_action(raw_action)
        constrained = self._constrain_action(raw)

        if not getattr(world, "colregs_persistent_state_enabled", False):
            # 关闭开关时清空可能由前一次运行留下的状态，确保行为与旧版
            # 逐步过滤器完全一致，不会因为运行时切换而残留规则义务。
            agent.colregs_rule_state = None
            agent.colregs_rule_state_steps = 0
            agent.colregs_rule_state_exit_steps = 0

        if not getattr(world, "agent_colregs_action_filter_enabled", False):
            # 规则过滤器关闭（例如 policy_only 测试）时不保留上一变体
            # 的会遇义务，避免 assisted -> policy_only 切换产生隐性状态。
            agent.colregs_rule_state = None
            agent.colregs_rule_state_steps = 0
            agent.colregs_rule_state_exit_steps = 0
            return self._decision(raw, constrained, constrained, "disabled", "filter_disabled")
        if getattr(world, "control_model", None) != "usv_3dof":
            agent.colregs_rule_state = None
            agent.colregs_rule_state_steps = 0
            agent.colregs_rule_state_exit_steps = 0
            return self._decision(raw, constrained, constrained, "pass_through", "not_usv_3dof")
        if not getattr(world, "colregs_enabled", False) or not getattr(agent, "is_vessel", False):
            agent.colregs_rule_state = None
            agent.colregs_rule_state_steps = 0
            agent.colregs_rule_state_exit_steps = 0
            return self._decision(raw, constrained, constrained, "pass_through", "colregs_unavailable")

        report = self._critical_report(world, agent)
        agent.colregs_last_report = report

        # 可选的会遇义务锁存：风险报告在阈值附近可能逐步进出，若每步
        # 重新分类会造成规则动作抖动。持久化状态只保存目标/会遇类型/角色，
        # 具体动作仍基于当前报告（或最近报告）计算；连续确认无风险后释放。
        if getattr(world, "colregs_persistent_state_enabled", False):
            report = self._apply_persistent_rule_state(world, agent, report)
        if report is None:
            return self._decision(raw, constrained, constrained, "network", "no_collision_risk")

        rule_state = getattr(agent, "colregs_rule_state", None)
        if rule_state:
            encounter_type = rule_state["encounter_type"]
            give_way_vessel = rule_state["give_way_vessel"]
            stand_on_vessel = rule_state["stand_on_vessel"]
        else:
            encounter_type = report.encounter_type
            give_way_vessel = report.give_way_vessel
            stand_on_vessel = report.stand_on_vessel
        agent_is_give_way = give_way_vessel in (agent.name, "both")
        agent_is_stand_on = stand_on_vessel in (agent.name, "both")
        if agent_is_give_way:
            action_set, mode = self._give_way_action_set(
                self._state_encounter_type(encounter_type)
            )
            applied = self._nearest_action(constrained, action_set)
            return self._decision(raw, constrained, applied, mode, "colregs_give_way", True, report)

        if agent_is_stand_on:
            if report.distance < self.config.emergency_distance:
                applied = self._nearest_action(constrained, self.config.emergency_stand_on_actions)
                return self._decision(raw, constrained, applied, "colregs_stand_on_emergency", "stand_on_emergency", True, report)
            applied = self._keep_course_action(constrained)
            return self._decision(raw, constrained, applied, "colregs_stand_on", "keep_course_and_speed", True, report)

        applied = self._nearest_action(constrained, self.config.undefined_risk_actions)
        return self._decision(raw, constrained, applied, "colregs_undefined_risk", "undefined_risk", True, report)

    @staticmethod
    def _state_encounter_type(value):
        if isinstance(value, EncounterType):
            return value
        try:
            return EncounterType(str(value))
        except (TypeError, ValueError):
            return EncounterType.COLLISION_RISK_UNDEFINED

    def _apply_persistent_rule_state(self, world, agent, report):
        state = getattr(agent, "colregs_rule_state", None)
        exit_steps = max(int(getattr(world, "colregs_persistent_exit_steps", 3)), 1)
        max_hold_steps = max(int(getattr(world, "colregs_persistent_max_hold_steps", 0)), 0)
        enter_dcpa = float(getattr(world, "colregs_persistent_enter_dcpa", np.inf))
        enter_tcpa = float(getattr(world, "colregs_persistent_enter_tcpa", np.inf))
        exit_dcpa = float(getattr(world, "colregs_persistent_exit_dcpa", np.inf))
        exit_tcpa = float(getattr(world, "colregs_persistent_exit_tcpa", np.inf))

        if report is not None:
            # 进入阈值只控制是否锁存规则义务；硬风险报告仍由 COLREGs
            # 引擎产生，避免改变关闭状态机时的旧安全语义。
            enter_match = (
                np.isfinite(report.dcpa)
                and report.dcpa <= enter_dcpa
                and np.isfinite(report.tcpa)
                and 0.0 <= report.tcpa <= enter_tcpa
            ) or report.distance <= float(self.config.emergency_distance)
            # 新目标只有在满足进入门槛时才替换已锁存会遇；否则优先寻找
            # 仍然存在的旧目标报告，避免出现“旧会遇义务 + 新目标几何量”
            # 的不一致组合。若旧目标已消失，则清空旧状态并按普通风险报告
            # 处理当前目标（当前目标未达进入门槛时不会重新锁存）。
            if state is not None and state.get("target_id") != report.target_id and not enter_match:
                locked_target_id = state.get("target_id")
                locked_reports = tuple(world.evaluate_colregs_for_entity(agent))
                locked_report = next(
                    (
                        item
                        for item in locked_reports
                        if item.target_id == locked_target_id and item.risk_of_collision
                    ),
                    None,
                )
                if locked_report is not None:
                    state["last_report"] = locked_report
                    agent.colregs_rule_state_steps = int(
                        getattr(agent, "colregs_rule_state_steps", 0)
                    ) + 1
                    agent.colregs_rule_state_exit_steps = 0
                    return locked_report
                agent.colregs_rule_state = None
                agent.colregs_rule_state_steps = 0
                agent.colregs_rule_state_exit_steps = 0
                state = None
                return report
            if state is None or state.get("target_id") != report.target_id:
                if report.give_way_vessel in (agent.name, "both"):
                    phase = "GIVE_WAY"
                elif report.stand_on_vessel in (agent.name, "both"):
                    phase = "STAND_ON"
                else:
                    phase = "RISK_DETECTED"
                agent.colregs_rule_state = {
                    # 显式状态机状态：风险检测后锁存让路/直航义务。
                    "phase": phase,
                    "target_id": report.target_id,
                    "encounter_type": report.encounter_type,
                    "give_way_vessel": report.give_way_vessel,
                    "stand_on_vessel": report.stand_on_vessel,
                    "last_report": report,
                }
                agent.colregs_rule_state_steps = 1
            else:
                # 目标仍处于风险会遇中，保持初始义务，更新几何量供本步动作使用。
                state["last_report"] = report
                agent.colregs_rule_state_steps = int(
                    getattr(agent, "colregs_rule_state_steps", 0)
                ) + 1
            agent.colregs_rule_state_exit_steps = 0
            if max_hold_steps > 0 and agent.colregs_rule_state_steps >= max_hold_steps:
                # 异常保护：防止目标/传感器异常导致规则义务无限锁存。
                agent.colregs_rule_state = None
                agent.colregs_rule_state_steps = 0
            return report

        if state is None:
            agent.colregs_rule_state_exit_steps = 0
            return None

        # 风险报告消失后，重新读取当前目标的实时几何量。不能使用旧报告，
        # 否则旧 DCPA/TCPA 会一直满足退出条件而导致状态永不释放。
        target_id = state.get("target_id")
        current_reports = tuple(world.evaluate_colregs_for_entity(agent))
        current_report = next(
            (item for item in current_reports if item.target_id == target_id), None
        )
        if current_report is not None:
            current_dcpa = float(getattr(current_report, "dcpa", np.inf))
            current_tcpa = float(getattr(current_report, "tcpa", np.inf))
            # 只有 DCPA 与 TCPA 同时仍处于退出滞回窗口内，才继续保持
            # 会遇义务。碰撞风险本身是两个条件的联合约束；使用 ``or``
            # 会在 DCPA 已经安全但 TCPA 仍为正（或反之）时无限拖长规则
            # 动作，明显压制追踪策略恢复。
            still_within_hysteresis = (
                np.isfinite(current_dcpa)
                and current_dcpa <= exit_dcpa
                and np.isfinite(current_tcpa)
                and 0.0 <= current_tcpa <= exit_tcpa
            )
            if still_within_hysteresis:
                agent.colregs_rule_state_exit_steps = 0
                state["last_report"] = current_report
                return current_report

        agent.colregs_rule_state_exit_steps = int(
            getattr(agent, "colregs_rule_state_exit_steps", 0)
        ) + 1
        if agent.colregs_rule_state_exit_steps >= exit_steps:
            agent.colregs_rule_state = None
            agent.colregs_rule_state_steps = 0
            agent.colregs_rule_state_exit_steps = 0
            return None
        # 退出确认期间继续履行最近一次规则义务，防止刚越过阈值就恢复追踪。
        return state.get("last_report")

    def _coerce_action(self, action):
        data = np.asarray(action, dtype=float).reshape(-1)
        if data.size < self.config.action_dim:
            padded = np.zeros(self.config.action_dim, dtype=float)
            padded[:data.size] = data
            data = padded
        return data[:self.config.action_dim]

    def _constrain_action(self, action):
        data = np.nan_to_num(np.asarray(action, dtype=float), nan=0.0, posinf=1.0, neginf=-1.0)
        data = data.copy()
        data[0] = np.clip(data[0], self.config.min_thrust, self.config.max_thrust)
        data[1] = np.clip(data[1], self.config.min_rudder, self.config.max_rudder)
        return data

    @staticmethod
    def _critical_report(world, agent):
        reports = world.evaluate_colregs_for_entity(agent)
        agent.colregs_last_reports = reports
        risk_reports = [report for report in reports if report.risk_of_collision]
        if not risk_reports:
            return None

        def report_key(report):
            tcpa = report.tcpa if np.isfinite(report.tcpa) else 1.0e9
            return tcpa, report.dcpa, report.distance

        return min(risk_reports, key=report_key)

    def _give_way_action_set(self, encounter_type):
        if encounter_type == EncounterType.HEAD_ON:
            return self.config.head_on_actions, "colregs_head_on"
        if encounter_type == EncounterType.CROSSING_GIVE_WAY:
            return self.config.crossing_give_way_actions, "colregs_crossing_give_way"
        if encounter_type == EncounterType.OVERTAKING_GIVE_WAY:
            return self.config.overtaking_actions, "colregs_overtaking_give_way"
        return self.config.undefined_risk_actions, "colregs_give_way"

    @staticmethod
    def _nearest_action(action, action_set: Sequence[Tuple[float, float]]):
        candidates = np.asarray(action_set, dtype=float)
        distances = np.linalg.norm(candidates - action.reshape(1, -1), axis=1)
        return candidates[int(np.argmin(distances))]

    @staticmethod
    def _keep_course_action(action):
        keep_course = np.asarray(action, dtype=float).copy()
        keep_course[1] = 0.0
        return keep_course

    @staticmethod
    def _as_tuple(action):
        return tuple(float(x) for x in np.asarray(action, dtype=float).reshape(-1))

    def _decision(
        self,
        raw,
        constrained,
        applied,
        mode,
        reason,
        active=False,
        report=None,
        rule_action=None,
    ):
        # 过滤器本身不负责 EMA 平滑；因此当前第三个动作参数就是纯规则
        # 动作。保留 ``applied_action`` 字段是为了兼容旧的 info/测试代码，
        # 但环境核心会显式读取 ``rule_action`` 作为平滑器输入。
        if rule_action is None:
            rule_action = applied
        return ActionFilterDecision(
            raw_action=self._as_tuple(raw),
            constrained_action=self._as_tuple(constrained),
            rule_action=self._as_tuple(rule_action),
            applied_action=self._as_tuple(applied),
            mode=mode,
            reason=reason,
            active=active,
            encounter_type=None if report is None else report.encounter_type.value,
            target_id=None if report is None else report.target_id,
            give_way_vessel=None if report is None else report.give_way_vessel,
            stand_on_vessel=None if report is None else report.stand_on_vessel,
            dcpa=None if report is None else float(report.dcpa),
            tcpa=None if report is None else float(report.tcpa),
        )
