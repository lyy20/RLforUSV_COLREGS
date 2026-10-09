"""Episode-level COLREGs internalization audit for evaluation runs.

The audit is deliberately side-effect free: it reads the world state and the
policy action, but never changes an action, reward, or environment transition.
One event is tracked per dynamic obstacle vessel, so repeated risk reports over
many environment steps are counted once rather than once per step.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from safety.colregs import COLREGsReport, EncounterType


DEFAULT_AUDIT_CONFIG = {
    "enter_confirm_steps": 2,
    "exit_confirm_steps": 3,
    "early_action_window_steps": 5,
    "reaction_deadline_steps": 5,
    "action_compliance_threshold": 0.70,
    "anti_rule_action_threshold": 0.20,
    "event_exit_distance_m": 0.0,
    # Zero only proves non-overlap and is too weak for an internalization claim.
    "safe_clearance_m": 10.0,
    "safe_dcpa_m": 30.0,
    "resolution_tcpa_max_s": 0.0,
    "stand_on_emergency_distance_m": 0.0,
}

AUDIT_STEP_FIELDS = (
    "episode_id", "step", "ownship_id", "target_id", "event_id", "risk_of_collision",
    "encounter_type", "applicable_rules", "required_actions",
    "recommended_actions", "give_way_vessel", "stand_on_vessel", "distance_m",
    "tcpa_s", "dcpa_m", "relative_bearing_deg", "heading_difference_deg",
    "policy_action_available", "raw_thrust", "raw_rudder", "constrained_thrust",
    "constrained_rudder", "rule_thrust", "rule_rudder", "smoothed_thrust",
    "smoothed_rudder", "applied_thrust", "applied_rudder", "filter_active",
    "filter_mode", "policy_rudder_direction", "heading_rad", "yaw_rate_rad_s",
    "actual_rudder_rad", "clearance_m", "action_compliant", "anti_rule_action",
    "action_present", "physical_action_compliant", "physical_anti_rule_action",
    "physical_action_present", "rule_reference_thrust", "rule_reference_rudder",
    "policy_rule_action_distance", "stand_on_emergency_action",
    # 四段动作链的范数均在归一化动作空间中计算，便于识别规则层与
    # 平滑器各自的影响；action_correction_norm 保留为纯规则修正别名。
    "action_correction_norm", "raw_to_constrained_norm",
    "constrained_to_rule_norm", "rule_to_smoothed_norm",
    "raw_to_smoothed_norm", "done_reason",
)

AUDIT_EVENT_FIELDS = (
    "event_id", "evaluation_variant", "episode_id", "ownship_id", "target_id",
    "encounter_type", "encounter_type_history", "type_transition_count", "role",
    "applicable_rules", "give_way_vessel", "stand_on_vessel", "start_step",
    "end_step", "duration_steps", "start_distance_m", "minimum_distance_m",
    "minimum_clearance_m", "minimum_dcpa_m", "minimum_tcpa_s",
    "resolution_clearance_m", "resolution_dcpa_m", "resolution_tcpa_s",
    "first_compliant_step", "reaction_delay_steps", "reaction_delay_seconds",
    "action_compliance_ratio", "anti_rule_action_ratio", "early_action_ok",
    "policy_rule_followed", "safe_resolution", "collision", "out_of_bounds",
    "risk_action_steps", "rule_violation_steps", "anti_rule_steps",
    "stand_on_emergency_steps", "rule_violation_event",
    "filter_active_steps", "filter_active_ratio", "filter_activated_in_event",
    "mean_action_correction", "mean_rule_action_correction",
    "mean_smoothing_correction", "mean_total_action_correction",
    "done_reason", "failure_reason",
)

AUDIT_EPISODE_FIELDS = (
    "evaluation_variant", "episode_id", "episode", "seed", "placement_mode",
    "done_reason", "encounter_events",
    "rule_followed_events", "successful_resolution_events", "compliant_and_safe_events",
    "policy_rule_follow_rate", "safe_resolution_rate", "internalization_success_rate",
    "risk_action_steps", "rule_violation_steps", "anti_rule_steps",
    "rule_violation_step_rate", "rule_violation_events", "stand_on_emergency_steps",
    "multi_vessel_risk_steps", "multi_vessel_risk_events", "mean_reaction_delay_steps",
    "filter_activated_events", "filter_active_event_rate",
)

AUDIT_TEST_FIELDS = (
    "evaluation_variant", "episodes", "episodes_with_encounter", "total_encounters",
    "rule_followed_events", "successful_resolution_events", "compliant_and_safe_events",
    "policy_rule_follow_rate", "safe_resolution_rate", "conditional_resolution_rate",
    "internalization_success_rate", "collision_events", "out_of_bounds_events",
    "risk_action_steps", "rule_violation_steps", "anti_rule_steps",
    "rule_violation_step_rate", "rule_violation_events", "stand_on_emergency_steps",
    "policy_rule_follow_rate_ci95_low", "policy_rule_follow_rate_ci95_high",
    "safe_resolution_rate_ci95_low", "safe_resolution_rate_ci95_high",
    "internalization_success_rate_ci95_low", "internalization_success_rate_ci95_high",
    "multi_vessel_risk_events", "filter_activated_events", "filter_active_event_rate",
)

AUDIT_RULE_FIELDS = (
    "evaluation_variant", "encounter_type", "events", "rule_followed_events",
    "successful_resolution_events", "compliant_and_safe_events",
    "policy_rule_follow_rate", "safe_resolution_rate",
    "internalization_success_rate", "mean_minimum_clearance_m",
    "mean_resolution_clearance_m", "mean_resolution_dcpa_m",
    "mean_reaction_delay_seconds", "risk_action_steps",
    "rule_violation_steps", "rule_violation_step_rate", "rule_violation_events",
)


def _finite(value, default=float("nan")):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float("nan")


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval used for event-level rates in the test summary."""
    n = float(total)
    if n <= 0.0:
        return float("nan"), float("nan")
    p = float(successes) / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    margin = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _value(value):
    """Convert numpy scalars and non-finite values to CSV/JSON-safe values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else ""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _encounter_value(report: Optional[COLREGsReport]) -> str:
    if report is None:
        return EncounterType.SAFE_PASSAGE.value
    encounter = getattr(report, "encounter_type", report)
    return getattr(encounter, "value", str(encounter))


def _is_risk_report(report: Optional[COLREGsReport]) -> bool:
    if report is None or not bool(getattr(report, "risk_of_collision", False)):
        return False
    # Undefined-risk is still a collision-risk judgement. It is counted as an
    # event so the denominator reflects every rule-layer warning; its event
    # type remains explicit in the CSV for separate analysis.
    return _encounter_value(report) != EncounterType.SAFE_PASSAGE.value


def _report_text(values: Iterable) -> str:
    return "|".join(str(getattr(value, "value", value)) for value in (values or ()))


def _dynamic_obstacles(world):
    """Return real dynamic vessels only; target and static objects are excluded."""
    return [
        obstacle
        for obstacle in world.obstacles[: world.num_obstacles]
        if getattr(obstacle, "is_vessel", False)
        and getattr(obstacle, "vessel_role", "") == "dynamic_obstacle"
    ]


def _report_clearance_m(report: COLREGsReport, agent, other) -> float:
    """Compute the pre-action hull clearance represented by a report."""
    center_distance_m = _finite(getattr(report, "distance", np.nan)) * 1000.0
    return center_distance_m - (float(agent.size) + float(other.size)) * 1000.0


def _heading_and_yaw_rate(world, agent):
    state = getattr(agent.state, "usv_3dof", None)
    if state is not None:
        return _finite(state.psi, 0.0), _finite(state.r, 0.0)
    heading = world._entity_heading(agent, np.asarray(agent.state.p_vel, dtype=float))
    return _finite(heading, 0.0), 0.0


def _action_direction(report: COLREGsReport, rudder: float, yaw_rate: float) -> tuple[bool, bool, bool]:
    """Return (compliant, anti_rule, action_present) for the ownship action.

    The evaluator intentionally uses coarse directional semantics. COLREGs
    permits a set of safe trajectories, so exact equality to a discrete rule
    action would incorrectly label valid actions as violations.
    """
    encounter = _encounter_value(report)
    rudder_sign = np.sign(_finite(rudder, 0.0))
    yaw_sign = np.sign(_finite(yaw_rate, 0.0))
    effective_sign = rudder_sign if rudder_sign != 0.0 else -yaw_sign
    # In this project negative rudder/yaw means starboard turn.
    starboard = effective_sign < 0.0
    port = effective_sign > 0.0
    action_present = abs(_finite(rudder, 0.0)) > 1.0e-4 or abs(_finite(yaw_rate, 0.0)) > 1.0e-5

    if encounter == EncounterType.HEAD_ON.value:
        return starboard, port, action_present
    if encounter == EncounterType.CROSSING_GIVE_WAY.value:
        return starboard, port, action_present
    if encounter == EncounterType.OVERTAKING_GIVE_WAY.value:
        # Overtaking has no single mandatory rudder sign. A deliberate, safe
        # course alteration is considered compliant at the action layer.
        return action_present, False, action_present
    if encounter in {
        EncounterType.CROSSING_STAND_ON.value,
        EncounterType.OVERTAKEN_STAND_ON.value,
    }:
        # Stand-on means initially maintain course and speed. The event logic
        # allows emergency action later when the risk does not resolve.
        return not action_present, False, action_present
    return action_present, False, action_present


def _rule_reference_action(
    report: COLREGsReport,
    constrained_action,
    world,
    stand_on_emergency: bool = False,
):
    """Build a non-executed normalized reference action for diagnostics only."""
    if not _is_risk_report(report):
        return None
    action = np.asarray(constrained_action, dtype=float).reshape(-1)
    if action.size < 2:
        return None
    reference = action[:2].copy()
    ownship_id = str(getattr(world.agents[0], "name", "agent 0"))
    filter_object = getattr(world, "agent_colregs_action_filter", None)
    if filter_object is None:
        return None
    give_way = getattr(report, "give_way_vessel", None) in {ownship_id, "both"}
    stand_on = getattr(report, "stand_on_vessel", None) in {ownship_id, "both"}
    if give_way:
        action_set, _ = filter_object._give_way_action_set(report.encounter_type)
        return np.asarray(filter_object._nearest_action(reference, action_set), dtype=float)
    if stand_on:
        if stand_on_emergency:
            return np.asarray(
                filter_object._nearest_action(
                    reference,
                    filter_object.config.emergency_stand_on_actions,
                ),
                dtype=float,
            )
        return np.asarray(filter_object._keep_course_action(reference), dtype=float)
    action_set = filter_object.config.undefined_risk_actions
    return np.asarray(filter_object._nearest_action(reference, action_set), dtype=float)


@dataclass
class _Event:
    event_id: str
    episode_id: int
    ownship_id: str
    target_id: str
    encounter_type_at_entry: str
    role: str
    applicable_rules: str
    give_way_vessel: str
    stand_on_vessel: str
    start_step: int
    start_distance_m: float
    reports: List[str] = field(default_factory=list)
    rows: List[dict] = field(default_factory=list)
    clearances_m: List[float] = field(default_factory=list)
    dcpa_m: List[float] = field(default_factory=list)
    tcpa_s: List[float] = field(default_factory=list)
    compliant_actions: List[bool] = field(default_factory=list)
    anti_rule_actions: List[bool] = field(default_factory=list)
    action_present: List[bool] = field(default_factory=list)
    filter_active_steps: List[bool] = field(default_factory=list)
    action_corrections: List[float] = field(default_factory=list)
    first_compliant_step: Optional[int] = None
    end_step: Optional[int] = None
    last_risk_step: Optional[int] = None
    exit_counter: int = 0
    type_history: List[str] = field(default_factory=list)
    collision: bool = False
    out_of_bounds: bool = False
    done_reason: str = "none"
    resolved: bool = False

    def add_step(
        self,
        row: dict,
        report: COLREGsReport,
        compliant: bool,
        anti_rule: bool,
        action_present: bool,
        count_for_compliance: bool = True,
        filter_active: bool = False,
        action_correction: float = 0.0,
    ):
        self.rows.append(row)
        self.reports.append(_encounter_value(report))
        encounter = _encounter_value(report)
        if not self.type_history or self.type_history[-1] != encounter:
            self.type_history.append(encounter)
        self.clearances_m.append(_finite(row.get("clearance_m")))
        self.dcpa_m.append(_finite(getattr(report, "dcpa", np.nan)) * 1000.0)
        self.tcpa_s.append(_finite(getattr(report, "tcpa", np.nan)))
        self.filter_active_steps.append(bool(filter_active))
        self.action_corrections.append(_finite(action_correction, 0.0))
        if count_for_compliance:
            self.compliant_actions.append(bool(compliant))
            self.anti_rule_actions.append(bool(anti_rule))
            self.action_present.append(bool(action_present))
            if compliant and self.first_compliant_step is None:
                self.first_compliant_step = int(row["step"])
        self.last_risk_step = int(row["step"])


class COLREGsEventAudit:
    """Track dynamic-vessel encounters for one evaluation variant."""

    def __init__(
        self,
        world,
        episode_id: int,
        config: Optional[dict] = None,
        variant_name: str = "policy_only",
    ):
        self.world = world
        self.episode_id = int(episode_id)
        self.variant_name = str(variant_name)
        self.config = dict(DEFAULT_AUDIT_CONFIG)
        if config:
            self.config.update(config)
        self._states: Dict[str, dict] = {}
        self._events: List[_Event] = []
        self._event_sequence = 0
        self.step_rows: List[dict] = []
        self.multi_vessel_risk_steps = 0
        self.multi_vessel_risk_events = 0
        self._multi_vessel_active = False

    @property
    def events(self) -> list[_Event]:
        return self._events

    def _event_key(self, target) -> str:
        return str(getattr(target, "name", id(target)))

    def _new_event(self, step: int, target, report: COLREGsReport, row: dict) -> _Event:
        self._event_sequence += 1
        ownship_id = str(getattr(self.world.agents[0], "name", "agent_0"))
        variant = self.variant_name.replace(" ", "_")
        event = _Event(
            event_id="ep%04d_%s_event%04d_%s" % (
                self.episode_id,
                variant,
                self._event_sequence,
                self._event_key(target).replace(" ", "_"),
            ),
            episode_id=self.episode_id,
            ownship_id=ownship_id,
            target_id=str(getattr(target, "name", "dynamic_obstacle")),
            encounter_type_at_entry=_encounter_value(report),
            role="give_way" if getattr(report, "give_way_vessel", None) in {"both", getattr(self.world.agents[0], "name", "") } else (
                "stand_on" if getattr(report, "stand_on_vessel", None) in {"both", getattr(self.world.agents[0], "name", "")} else "undefined"
            ),
            applicable_rules=_report_text(getattr(report, "applicable_rules", ())),
            give_way_vessel=str(getattr(report, "give_way_vessel", "") or ""),
            stand_on_vessel=str(getattr(report, "stand_on_vessel", "") or ""),
            start_step=int(step),
            start_distance_m=_finite(row.get("distance_m")),
        )
        self._events.append(event)
        return event

    def _reports(self, agent, targets):
        reports = self.world.evaluate_colregs_for_entity(agent, targets)
        return {str(target.name): report for target, report in zip(targets, reports)}

    def reports_for_current_state(self):
        """Return shadow reports before an action is applied."""
        agent = self.world.agents[0]
        targets = _dynamic_obstacles(self.world)
        return self._reports(agent, targets)

    def record_step(
        self,
        step: int,
        policy_action,
        constrained_action,
        applied_action,
        filter_active: bool,
        filter_mode: str,
        done_reason: str = "none",
        reports: Optional[dict] = None,
        rule_action=None,
        smoothed_action=None,
    ):
        agent = self.world.agents[0]
        targets = _dynamic_obstacles(self.world)
        reports = reports if reports is not None else self._reports(agent, targets)
        risk_targets = [target for target in targets if _is_risk_report(reports.get(str(target.name)))]
        if len(risk_targets) >= 2:
            self.multi_vessel_risk_steps += 1
        if len(risk_targets) >= 2 and not self._multi_vessel_active:
            self.multi_vessel_risk_events += 1
            self._multi_vessel_active = True
        elif len(risk_targets) < 2:
            self._multi_vessel_active = False

        policy = np.asarray(policy_action, dtype=float).reshape(-1)
        constrained = np.asarray(constrained_action, dtype=float).reshape(-1)
        applied = np.asarray(applied_action, dtype=float).reshape(-1)
        # 测试审计也遵循训练动作契约：规则动作是纯 COLREGs 输出，
        # smoothed_action 是 EMA 后的动作，applied_action 是最终执行兼容字段。
        # 旧调用方没有新参数时，使用不混淆语义的退化回退。
        rule = (
            np.asarray(rule_action, dtype=float).reshape(-1)
            if rule_action is not None
            else constrained.copy()
        )
        smoothed = (
            np.asarray(smoothed_action, dtype=float).reshape(-1)
            if smoothed_action is not None
            else applied.copy()
        )
        state = getattr(agent.state, "usv_3dof", None)
        heading, yaw_rate = _heading_and_yaw_rate(self.world, agent)
        own_rudder = (
            _finite(getattr(state, "rudder", np.nan))
            if state is not None
            else _finite(applied[1] if applied.size > 1 else np.nan)
        )
        policy_rudder = _finite(policy[1] if policy.size > 1 else 0.0, 0.0)
        policy_action_available = bool(
            policy.size >= 2 and np.all(np.isfinite(policy[:2]))
        )
        action_correction = float(np.linalg.norm(rule[:2] - constrained[:2]))
        raw_to_constrained = float(np.linalg.norm(policy[:2] - constrained[:2]))
        constrained_to_rule = float(np.linalg.norm(rule[:2] - constrained[:2]))
        rule_to_smoothed = float(np.linalg.norm(smoothed[:2] - rule[:2]))
        raw_to_smoothed = float(np.linalg.norm(smoothed[:2] - policy[:2]))
        stand_on_emergency_distance_m = float(
            self.config.get("stand_on_emergency_distance_m", 0.0)
        )
        if stand_on_emergency_distance_m <= 0.0:
            filter_config = getattr(
                getattr(self.world, "agent_colregs_action_filter", None),
                "config",
                None,
            )
            stand_on_emergency_distance_m = float(
                getattr(filter_config, "emergency_distance", 0.0)
            ) * 1000.0
        for target in targets:
            target_id = str(target.name)
            report = reports.get(target_id)
            if report is None:
                continue
            clearance = _report_clearance_m(report, agent, target)
            # The primary internalization label is based on the SAC output,
            # never on the rule-filtered/applied command. Physical response is
            # still logged separately for interpreting 3DOF actuator delay.
            if policy_action_available:
                compliant, anti_rule, action_present = _action_direction(
                    report, policy_rudder, 0.0
                )
            else:
                compliant, anti_rule, action_present = False, False, False
            physical_compliant, physical_anti_rule, physical_action_present = _action_direction(
                report, own_rudder, yaw_rate
            )
            distance_m = _finite(getattr(report, "distance", np.nan)) * 1000.0
            tcpa_s = _finite(getattr(report, "tcpa", np.nan))
            is_stand_on = _encounter_value(report) in {
                EncounterType.CROSSING_STAND_ON.value,
                EncounterType.OVERTAKEN_STAND_ON.value,
            }
            stand_on_emergency = bool(
                is_stand_on
                and action_present
                and (
                    distance_m <= stand_on_emergency_distance_m
                    or (
                        np.isfinite(tcpa_s)
                        and tcpa_s
                        <= float(self.config["reaction_deadline_steps"])
                        * _finite(
                            getattr(self.world, "environment_config", {}).get(
                                "environment_dt", 1.0
                            ),
                            1.0,
                        )
                    )
                )
            )
            if stand_on_emergency:
                compliant = True
                anti_rule = False
            physical_stand_on_emergency = bool(
                is_stand_on
                and physical_action_present
                and distance_m <= stand_on_emergency_distance_m
            )
            if physical_stand_on_emergency:
                physical_compliant = True
                physical_anti_rule = False
            reference_action = _rule_reference_action(
                report,
                constrained,
                self.world,
                stand_on_emergency=stand_on_emergency,
            )
            policy_reference_distance = (
                float(np.linalg.norm(policy[:2] - reference_action))
                if reference_action is not None and policy_action_available
                else np.nan
            )
            state_info = self._states.get(target_id)
            event = state_info.get("event") if state_info is not None else None
            risk = _is_risk_report(report)
            exit_distance_m = float(self.config.get("event_exit_distance_m", 0.0))
            if (
                not risk
                and event is not None
                and exit_distance_m > 0.0
                and _finite(getattr(report, "distance", np.nan)) * 1000.0 <= exit_distance_m
            ):
                # Hysteresis prevents a noisy report from closing an event
                # while the vessel pair is still inside the configured exit
                # radius.
                risk = True
            row = {
                "episode_id": self.episode_id,
                "step": int(step),
                "ownship_id": str(getattr(agent, "name", "agent_0")),
                "target_id": target_id,
                "event_id": self._states.get(target_id, {}).get("event_id", ""),
                "risk_of_collision": int(bool(getattr(report, "risk_of_collision", False))),
                "encounter_type": _encounter_value(report),
                "applicable_rules": _report_text(getattr(report, "applicable_rules", ())),
                "required_actions": _report_text(getattr(report, "required_actions", ())),
                "recommended_actions": _report_text(getattr(report, "recommended_actions", ())),
                "give_way_vessel": str(getattr(report, "give_way_vessel", "") or ""),
                "stand_on_vessel": str(getattr(report, "stand_on_vessel", "") or ""),
                "distance_m": distance_m,
                "tcpa_s": tcpa_s,
                "dcpa_m": _finite(getattr(report, "dcpa", np.nan)) * 1000.0,
                "relative_bearing_deg": np.rad2deg(_finite(getattr(report, "relative_bearing_rad", np.nan))),
                "heading_difference_deg": np.rad2deg(_finite(getattr(report, "heading_difference_rad", np.nan))),
                "raw_thrust": _finite(policy[0] if policy.size > 0 else np.nan),
                "raw_rudder": _finite(policy[1] if policy.size > 1 else np.nan),
                "policy_action_available": int(policy_action_available),
                 "constrained_thrust": _finite(constrained[0] if constrained.size > 0 else np.nan),
                 "constrained_rudder": _finite(constrained[1] if constrained.size > 1 else np.nan),
                 "rule_thrust": _finite(rule[0] if rule.size > 0 else np.nan),
                 "rule_rudder": _finite(rule[1] if rule.size > 1 else np.nan),
                 "smoothed_thrust": _finite(smoothed[0] if smoothed.size > 0 else np.nan),
                 "smoothed_rudder": _finite(smoothed[1] if smoothed.size > 1 else np.nan),
                 "applied_thrust": _finite(applied[0] if applied.size > 0 else np.nan),
                 "applied_rudder": _finite(applied[1] if applied.size > 1 else np.nan),
                "filter_active": int(bool(filter_active)),
                "filter_mode": str(filter_mode or ""),
                "policy_rudder_direction": "starboard" if np.sign(_finite(policy[1] if policy.size > 1 else 0.0)) < 0 else ("port" if np.sign(_finite(policy[1] if policy.size > 1 else 0.0)) > 0 else "neutral"),
                "heading_rad": heading,
                "yaw_rate_rad_s": yaw_rate,
                "actual_rudder_rad": own_rudder,
                "clearance_m": clearance,
                "action_compliant": int(bool(compliant)),
                "anti_rule_action": int(bool(anti_rule)),
                "action_present": int(bool(action_present)),
                "physical_action_compliant": int(bool(physical_compliant)),
                "physical_anti_rule_action": int(bool(physical_anti_rule)),
                "physical_action_present": int(bool(physical_action_present)),
                "rule_reference_thrust": (
                    _finite(reference_action[0])
                    if reference_action is not None
                    else np.nan
                ),
                "rule_reference_rudder": (
                    _finite(reference_action[1])
                    if reference_action is not None
                    else np.nan
                ),
                "policy_rule_action_distance": policy_reference_distance,
                 "stand_on_emergency_action": int(stand_on_emergency),
                 "action_correction_norm": action_correction,
                 "raw_to_constrained_norm": raw_to_constrained,
                 "constrained_to_rule_norm": constrained_to_rule,
                 "rule_to_smoothed_norm": rule_to_smoothed,
                 "raw_to_smoothed_norm": raw_to_smoothed,
                 "done_reason": str(done_reason or "none"),
             }
            state_info = self._states.setdefault(target_id, {
                "candidate_steps": 0,
                "exit_steps": 0,
                "event_id": "",
                "event": None,
                "pending": [],
            })
            event = state_info.get("event")
            if risk:
                state_info["candidate_steps"] += 1
                state_info["exit_steps"] = 0
                if event is None:
                    state_info["pending"].append(
                        (
                            row,
                            report,
                            compliant,
                            anti_rule,
                            action_present,
                            bool(filter_active),
                            action_correction,
                        )
                    )
                    if state_info["candidate_steps"] < int(self.config["enter_confirm_steps"]):
                        self.step_rows.append(row)
                        continue
                    pending = state_info["pending"]
                    event = self._new_event(
                        int(pending[0][0]["step"]),
                        target,
                        pending[0][1],
                        pending[0][0],
                    )
                    state_info["event"] = event
                    state_info["event_id"] = event.event_id
                    for (
                        pending_row,
                        pending_report,
                        pending_compliant,
                        pending_anti,
                        pending_present,
                        pending_filter_active,
                        pending_correction,
                    ) in pending:
                        pending_row["event_id"] = event.event_id
                        event.add_step(
                            pending_row,
                            pending_report,
                            pending_compliant,
                            pending_anti,
                            pending_present,
                            filter_active=pending_filter_active,
                            action_correction=pending_correction,
                        )
                    state_info["pending"] = []
                else:
                    row["event_id"] = event.event_id
                    event.add_step(
                        row,
                        report,
                        compliant,
                        anti_rule,
                        action_present,
                        filter_active=filter_active,
                        action_correction=action_correction,
                    )
            elif event is not None:
                state_info["candidate_steps"] = 0
                state_info["exit_steps"] += 1
                row["event_id"] = event.event_id
                # Keep a short resolving tail, then close the event once the
                # report is stable. These rows aid debugging but are not counted
                # as risk/compliance steps by the event summary.
                if state_info["exit_steps"] < int(self.config["exit_confirm_steps"]):
                    event.add_step(
                        row,
                        report,
                        compliant,
                        anti_rule,
                        action_present,
                        count_for_compliance=False,
                        filter_active=filter_active,
                        action_correction=action_correction,
                    )
                else:
                    self._close_event(
                        event,
                        step - state_info["exit_steps"],
                        done_reason,
                        resolved=True,
                    )
                    state_info["event"] = None
                    state_info["event_id"] = ""
                    state_info["exit_steps"] = 0
            else:
                # A non-risk step between two risk reports must not count as
                # confirmation for the next event.
                state_info["candidate_steps"] = 0
                state_info["pending"] = []
            self.step_rows.append(row)

    def _close_event(
        self,
        event: _Event,
        end_step: int,
        done_reason: str = "none",
        resolved: bool = False,
    ):
        if event.end_step is not None:
            return
        event.end_step = max(int(end_step), event.start_step)
        event.done_reason = str(done_reason or "none")
        event.collision = event.done_reason in {"collision", "target_collision"}
        event.out_of_bounds = event.done_reason == "out_of_bounds"
        event.resolved = bool(resolved)

    def finalize(self, final_step: int, done_reason: str = "none"):
        for state_info in self._states.values():
            event = state_info.get("event")
            if event is not None:
                self._close_event(
                    event,
                    final_step,
                    done_reason,
                    # A task success does not prove that an active encounter
                    # was resolved; only a confirmed risk exit does.
                    resolved=False,
                )
                state_info["event"] = None
        return [self.event_row(event) for event in self._events]

    def event_row(self, event: _Event) -> dict:
        clearances = np.asarray(event.clearances_m, dtype=float)
        clearances = clearances[np.isfinite(clearances)]
        dcpas = np.asarray(event.dcpa_m, dtype=float)
        dcpas = dcpas[np.isfinite(dcpas)]
        tcpas = np.asarray(event.tcpa_s, dtype=float)
        tcpas = tcpas[np.isfinite(tcpas)]
        compliance = float(np.mean(event.compliant_actions)) if event.compliant_actions else 0.0
        anti_ratio = float(np.mean(event.anti_rule_actions)) if event.anti_rule_actions else 0.0
        first_step = event.first_compliant_step
        reaction_delay = (first_step - event.start_step) if first_step is not None else np.nan
        early_limit = event.start_step + int(self.config["early_action_window_steps"])
        early_action_ok = first_step is not None and first_step <= early_limit
        policy_followed = bool(
            early_action_ok
            and (
                first_step is not None
                and first_step - event.start_step
                <= int(self.config["reaction_deadline_steps"])
            )
            and compliance >= float(self.config["action_compliance_threshold"])
            and anti_ratio <= float(self.config["anti_rule_action_threshold"])
            and not event.collision
        )
        safe_clearance = float(self.config.get("safe_clearance_m", 0.0))
        safe_dcpa = float(self.config.get("safe_dcpa_m", 0.0))
        resolution_tcpa_max = float(self.config.get("resolution_tcpa_max_s", 0.0))
        min_clearance = float(np.min(clearances)) if clearances.size else np.nan
        # The event minimum describes closest approach. Resolution safety must
        # be checked at the final confirmed exit tail instead.
        resolution_row = event.rows[-1] if event.rows else {}
        resolution_clearance = _finite(resolution_row.get("clearance_m"))
        resolution_dcpa = _finite(resolution_row.get("dcpa_m"))
        resolution_tcpa = _finite(resolution_row.get("tcpa_s"))
        safe_resolution = bool(
            event.resolved
            and not event.collision
            and not event.out_of_bounds
            and (
                np.isfinite(resolution_clearance)
                and resolution_clearance >= max(safe_clearance, 0.0)
                and np.isfinite(resolution_dcpa)
                and resolution_dcpa >= max(safe_dcpa, 0.0)
                and (
                    not np.isfinite(resolution_tcpa)
                    or resolution_tcpa <= resolution_tcpa_max
                )
            )
            and str(event.done_reason) not in {
                "collision",
                "target_collision",
                "out_of_bounds",
                "timeout",
            }
        )
        if event.collision:
            failure_reason = "collision"
        elif event.out_of_bounds:
            failure_reason = "out_of_bounds"
        elif event.done_reason == "timeout":
            failure_reason = "timeout"
        elif not event.resolved:
            failure_reason = "episode_ended_before_resolution"
        elif not policy_followed:
            failure_reason = "late_or_noncompliant"
        else:
            failure_reason = ""
        filter_active_steps = int(sum(event.filter_active_steps))
        filter_active_ratio = (
            float(filter_active_steps / len(event.filter_active_steps))
            if event.filter_active_steps
            else 0.0
        )
        mean_action_correction = (
            float(np.mean(event.action_corrections))
            if event.action_corrections
            else 0.0
        )
        rule_corrections = np.asarray(
            [_finite(row.get("constrained_to_rule_norm")) for row in event.rows],
            dtype=float,
        )
        smoothing_corrections = np.asarray(
            [_finite(row.get("rule_to_smoothed_norm")) for row in event.rows],
            dtype=float,
        )
        total_corrections = np.asarray(
            [_finite(row.get("raw_to_smoothed_norm")) for row in event.rows],
            dtype=float,
        )
        rule_corrections = rule_corrections[np.isfinite(rule_corrections)]
        smoothing_corrections = smoothing_corrections[np.isfinite(smoothing_corrections)]
        total_corrections = total_corrections[np.isfinite(total_corrections)]
        risk_action_steps = int(len(event.compliant_actions))
        rule_violation_steps = int(sum(not value for value in event.compliant_actions))
        anti_rule_steps = int(sum(bool(value) for value in event.anti_rule_actions))
        stand_on_emergency_steps = int(
            sum(int(row.get("stand_on_emergency_action", 0)) for row in event.rows)
        )
        return {
            "event_id": event.event_id,
            "evaluation_variant": self.variant_name,
            "episode_id": event.episode_id,
            "ownship_id": event.ownship_id,
            "target_id": event.target_id,
            "encounter_type": event.encounter_type_at_entry,
            "encounter_type_history": "|".join(event.type_history),
            "type_transition_count": max(len(event.type_history) - 1, 0),
            "role": event.role,
            "applicable_rules": event.applicable_rules,
            "give_way_vessel": event.give_way_vessel,
            "stand_on_vessel": event.stand_on_vessel,
            "start_step": event.start_step,
            "end_step": event.end_step if event.end_step is not None else event.start_step,
            "duration_steps": max((event.end_step or event.start_step) - event.start_step + 1, 1),
            "start_distance_m": event.start_distance_m,
            "minimum_distance_m": (
                float(np.min([_finite(row.get("distance_m")) for row in event.rows]))
                if event.rows
                else np.nan
            ),
            "minimum_clearance_m": min_clearance,
            "minimum_dcpa_m": float(np.min(dcpas)) if dcpas.size else np.nan,
            "minimum_tcpa_s": float(np.min(tcpas)) if tcpas.size else np.nan,
            "resolution_clearance_m": resolution_clearance,
            "resolution_dcpa_m": resolution_dcpa,
            "resolution_tcpa_s": resolution_tcpa,
            "first_compliant_step": first_step if first_step is not None else "",
            "reaction_delay_steps": reaction_delay,
            "reaction_delay_seconds": reaction_delay * _finite(
                getattr(self.world, "environment_config", {}).get("environment_dt", 1.0),
                1.0,
            ) if np.isfinite(reaction_delay) else np.nan,
            "action_compliance_ratio": compliance,
            "anti_rule_action_ratio": anti_ratio,
            "early_action_ok": int(early_action_ok),
            "policy_rule_followed": int(policy_followed),
            "safe_resolution": int(safe_resolution),
            "collision": int(event.collision),
            "out_of_bounds": int(event.out_of_bounds),
            "risk_action_steps": risk_action_steps,
            "rule_violation_steps": rule_violation_steps,
            "anti_rule_steps": anti_rule_steps,
            "stand_on_emergency_steps": stand_on_emergency_steps,
            "rule_violation_event": int(not policy_followed),
            "filter_active_steps": filter_active_steps,
            "filter_active_ratio": filter_active_ratio,
            "filter_activated_in_event": int(filter_active_steps > 0),
            "mean_action_correction": mean_action_correction,
            "mean_rule_action_correction": (
                float(np.mean(rule_corrections)) if rule_corrections.size else 0.0
            ),
            "mean_smoothing_correction": (
                float(np.mean(smoothing_corrections))
                if smoothing_corrections.size else 0.0
            ),
            "mean_total_action_correction": (
                float(np.mean(total_corrections)) if total_corrections.size else 0.0
            ),
            "done_reason": event.done_reason,
            "failure_reason": failure_reason,
        }

    def episode_summary(self, variant_name: str) -> dict:
        rows = [self.event_row(event) for event in self._events]
        total = len(rows)
        followed = sum(int(row["policy_rule_followed"]) for row in rows)
        safe = sum(int(row["safe_resolution"]) for row in rows)
        compliant_safe = sum(int(row["policy_rule_followed"] and row["safe_resolution"]) for row in rows)
        risk_action_steps = sum(int(row.get("risk_action_steps", 0)) for row in rows)
        rule_violation_steps = sum(int(row.get("rule_violation_steps", 0)) for row in rows)
        anti_rule_steps = sum(int(row.get("anti_rule_steps", 0)) for row in rows)
        rule_violation_events = sum(int(row.get("rule_violation_event", 0)) for row in rows)
        stand_on_emergency_steps = sum(
            int(row.get("stand_on_emergency_steps", 0)) for row in rows
        )
        return {
            "evaluation_variant": variant_name,
            "episode_id": self.episode_id,
            "encounter_events": total,
            "rule_followed_events": followed,
            "successful_resolution_events": safe,
            "compliant_and_safe_events": compliant_safe,
            "policy_rule_follow_rate": _safe_div(followed, total),
            "safe_resolution_rate": _safe_div(safe, total),
            "internalization_success_rate": _safe_div(compliant_safe, total),
            "risk_action_steps": risk_action_steps,
            "rule_violation_steps": rule_violation_steps,
            "anti_rule_steps": anti_rule_steps,
            "rule_violation_step_rate": _safe_div(rule_violation_steps, risk_action_steps),
            "rule_violation_events": rule_violation_events,
            "stand_on_emergency_steps": stand_on_emergency_steps,
            "multi_vessel_risk_steps": self.multi_vessel_risk_steps,
            "multi_vessel_risk_events": self.multi_vessel_risk_events,
            "filter_activated_events": sum(
                int(row.get("filter_activated_in_event", 0)) for row in rows
            ),
            "filter_active_event_rate": _safe_div(
                sum(int(row.get("filter_activated_in_event", 0)) for row in rows),
                total,
            ),
            "mean_reaction_delay_steps": _safe_div(
                sum(float(row["reaction_delay_steps"]) for row in rows if str(row["reaction_delay_steps"]) not in {"", "nan"} and np.isfinite(_finite(row["reaction_delay_steps"]))),
                sum(1 for row in rows if np.isfinite(_finite(row["reaction_delay_steps"]))),
            ),
        }


def aggregate_audit_rows(event_rows: list[dict], episode_rows: list[dict], variant_name: str) -> dict:
    """Aggregate N-test results strictly at event level."""
    rows = [row for row in event_rows if row.get("evaluation_variant", variant_name) == variant_name]
    total = len(rows)
    followed = sum(int(row.get("policy_rule_followed", 0)) for row in rows)
    safe = sum(int(row.get("safe_resolution", 0)) for row in rows)
    compliant_safe = sum(int(row.get("policy_rule_followed", 0) and row.get("safe_resolution", 0)) for row in rows)
    risk_action_steps = sum(int(row.get("risk_action_steps", 0)) for row in rows)
    rule_violation_steps = sum(int(row.get("rule_violation_steps", 0)) for row in rows)
    anti_rule_steps = sum(int(row.get("anti_rule_steps", 0)) for row in rows)
    rule_violation_events = sum(int(row.get("rule_violation_event", 0)) for row in rows)
    stand_on_emergency_steps = sum(
        int(row.get("stand_on_emergency_steps", 0)) for row in rows
    )
    follow_ci_low, follow_ci_high = _wilson_interval(followed, total)
    safe_ci_low, safe_ci_high = _wilson_interval(safe, total)
    internal_ci_low, internal_ci_high = _wilson_interval(compliant_safe, total)
    encounters_per_episode = [row.get("encounter_events", 0) for row in episode_rows if row.get("evaluation_variant") == variant_name]
    return {
        "evaluation_variant": variant_name,
        "episodes": len([row for row in episode_rows if row.get("evaluation_variant") == variant_name]),
        "episodes_with_encounter": sum(int(value > 0) for value in encounters_per_episode),
        "total_encounters": total,
        "rule_followed_events": followed,
        "successful_resolution_events": safe,
        "compliant_and_safe_events": compliant_safe,
        "policy_rule_follow_rate": _safe_div(followed, total),
        "safe_resolution_rate": _safe_div(safe, total),
        "conditional_resolution_rate": _safe_div(compliant_safe, followed),
        "internalization_success_rate": _safe_div(compliant_safe, total),
        "risk_action_steps": risk_action_steps,
        "rule_violation_steps": rule_violation_steps,
        "anti_rule_steps": anti_rule_steps,
        "rule_violation_step_rate": _safe_div(rule_violation_steps, risk_action_steps),
        "rule_violation_events": rule_violation_events,
        "stand_on_emergency_steps": stand_on_emergency_steps,
        "policy_rule_follow_rate_ci95_low": follow_ci_low,
        "policy_rule_follow_rate_ci95_high": follow_ci_high,
        "safe_resolution_rate_ci95_low": safe_ci_low,
        "safe_resolution_rate_ci95_high": safe_ci_high,
        "internalization_success_rate_ci95_low": internal_ci_low,
        "internalization_success_rate_ci95_high": internal_ci_high,
        "collision_events": sum(int(row.get("collision", 0)) for row in rows),
        "out_of_bounds_events": sum(int(row.get("out_of_bounds", 0)) for row in rows),
        "multi_vessel_risk_events": sum(int(row.get("multi_vessel_risk_events", 0)) for row in episode_rows if row.get("evaluation_variant") == variant_name),
        "filter_activated_events": sum(
            int(row.get("filter_activated_in_event", 0)) for row in rows
        ),
        "filter_active_event_rate": _safe_div(
            sum(int(row.get("filter_activated_in_event", 0)) for row in rows),
            total,
        ),
    }


def build_rule_summary_rows(event_rows: list[dict], variant_name: str) -> list[dict]:
    """Aggregate encounters by classified COLREGs type for paper tables."""
    variant_rows = [
        row for row in event_rows
        if row.get("evaluation_variant", variant_name) == variant_name
    ]
    encounter_types = sorted({str(row.get("encounter_type", "")) for row in variant_rows})
    output = []
    for encounter_type in encounter_types:
        rows = [
            row for row in variant_rows
            if str(row.get("encounter_type", "")) == encounter_type
        ]
        count = len(rows)
        followed = sum(int(row.get("policy_rule_followed", 0)) for row in rows)
        safe = sum(int(row.get("safe_resolution", 0)) for row in rows)
        compliant_safe = sum(
            int(row.get("policy_rule_followed", 0) and row.get("safe_resolution", 0))
            for row in rows
        )
        risk_action_steps = sum(int(row.get("risk_action_steps", 0)) for row in rows)
        rule_violation_steps = sum(int(row.get("rule_violation_steps", 0)) for row in rows)
        rule_violation_events = sum(int(row.get("rule_violation_event", 0)) for row in rows)
        clearances = np.asarray(
            [_finite(row.get("minimum_clearance_m")) for row in rows],
            dtype=float,
        )
        delays = np.asarray(
            [_finite(row.get("reaction_delay_seconds")) for row in rows],
            dtype=float,
        )
        clearances = clearances[np.isfinite(clearances)]
        resolution_clearances = np.asarray(
            [_finite(row.get("resolution_clearance_m")) for row in rows],
            dtype=float,
        )
        resolution_dcpas = np.asarray(
            [_finite(row.get("resolution_dcpa_m")) for row in rows],
            dtype=float,
        )
        resolution_clearances = resolution_clearances[np.isfinite(resolution_clearances)]
        resolution_dcpas = resolution_dcpas[np.isfinite(resolution_dcpas)]
        delays = delays[np.isfinite(delays)]
        output.append(
            {
                "evaluation_variant": variant_name,
                "encounter_type": encounter_type,
                "events": count,
                "rule_followed_events": followed,
                "successful_resolution_events": safe,
                "compliant_and_safe_events": compliant_safe,
                "policy_rule_follow_rate": _safe_div(followed, count),
                "safe_resolution_rate": _safe_div(safe, count),
                "internalization_success_rate": _safe_div(compliant_safe, count),
                "mean_minimum_clearance_m": float(np.mean(clearances)) if clearances.size else np.nan,
                "mean_resolution_clearance_m": (
                    float(np.mean(resolution_clearances))
                    if resolution_clearances.size else np.nan
                ),
                "mean_resolution_dcpa_m": (
                    float(np.mean(resolution_dcpas))
                    if resolution_dcpas.size else np.nan
                ),
                "mean_reaction_delay_seconds": float(np.mean(delays)) if delays.size else np.nan,
                "risk_action_steps": risk_action_steps,
                "rule_violation_steps": rule_violation_steps,
                "rule_violation_step_rate": _safe_div(
                    rule_violation_steps, risk_action_steps
                ),
                "rule_violation_events": rule_violation_events,
            }
        )
    return output


def write_audit_csv(path: Path, rows: list[dict], fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Keep an explicit header for empty tests, which makes downstream data
        # pipelines distinguish “no encounters” from a missing file.
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            if fieldnames:
                csv.writer(file).writerow(list(fieldnames))
        return
    fieldnames = list(fieldnames or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _value(row.get(key, "")) for key in fieldnames})


def write_audit_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(payload), file, ensure_ascii=False, indent=2)
