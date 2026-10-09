from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


EPS = 1.0e-9


class EncounterType(str, Enum):
    SAFE_PASSAGE = "safe_passage"
    HEAD_ON = "head_on"
    CROSSING_GIVE_WAY = "crossing_give_way"
    CROSSING_STAND_ON = "crossing_stand_on"
    OVERTAKING_GIVE_WAY = "overtaking_give_way"
    OVERTAKEN_STAND_ON = "overtaken_stand_on"
    COLLISION_RISK_UNDEFINED = "collision_risk_undefined"


class COLREGsAction(str, Enum):
    KEEP_LOOKOUT = "keep_lookout"
    USE_SAFE_SPEED = "use_safe_speed"
    ASSESS_COLLISION_RISK = "assess_collision_risk"
    ALTER_COURSE_STARBOARD = "alter_course_starboard"
    AVOID_CROSSING_AHEAD = "avoid_crossing_ahead"
    PASS_ASTERN = "pass_astern"
    KEEP_CLEAR = "keep_clear"
    KEEP_COURSE_AND_SPEED = "keep_course_and_speed"
    MONITOR_GIVE_WAY_VESSEL = "monitor_give_way_vessel"
    REDUCE_SPEED_IF_NEEDED = "reduce_speed_if_needed"
    TAKE_EARLY_SUBSTANTIAL_ACTION = "take_early_substantial_action"


RULE_LOOKOUT = "Rule 5: maintain proper lookout"
RULE_SAFE_SPEED = "Rule 6: proceed at safe speed"
RULE_RISK = "Rule 7: use available means to determine risk of collision"
RULE_ACTION = "Rule 8: take early and substantial action to avoid collision"
RULE_OVERTAKING = "Rule 13: overtaking vessel shall keep out of the way"
RULE_HEAD_ON = "Rule 14: power-driven vessels in head-on shall alter to starboard"
RULE_CROSSING = "Rule 15: vessel with the other on starboard side shall keep out of the way"
RULE_GIVE_WAY = "Rule 16: give-way vessel shall take early and substantial action"
RULE_STAND_ON = "Rule 17: stand-on vessel shall keep course and speed"


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def signed_angle_between(reference_heading: float, target_angle: float) -> float:
    """Positive is port side, negative is starboard side."""
    return wrap_to_pi(target_angle - reference_heading)


def angle_abs(angle: float) -> float:
    return abs(wrap_to_pi(angle))


def vector_heading(vector: Iterable[float], fallback: float = 0.0) -> float:
    data = np.asarray(vector, dtype=float).reshape(2)
    if np.linalg.norm(data) <= EPS:
        return wrap_to_pi(fallback)
    return wrap_to_pi(float(np.arctan2(data[1], data[0])))


def velocity_from_heading(heading: float, speed: float) -> np.ndarray:
    return np.array([np.cos(heading) * speed, np.sin(heading) * speed], dtype=float)


@dataclass(frozen=True)
class VesselState:
    vessel_id: str
    position: np.ndarray
    velocity: np.ndarray
    heading: float
    speed: float
    radius: float = 0.0
    role: str = "vessel"
    colregs_compliant: bool = False

    @classmethod
    def from_kinematics(
        cls,
        vessel_id: str,
        position: Iterable[float],
        velocity: Optional[Iterable[float]] = None,
        heading: Optional[float] = None,
        speed: Optional[float] = None,
        radius: float = 0.0,
        role: str = "vessel",
        colregs_compliant: bool = False,
    ) -> "VesselState":
        position_array = np.asarray(position, dtype=float).reshape(2)
        if velocity is None:
            resolved_heading = wrap_to_pi(0.0 if heading is None else heading)
            resolved_speed = 0.0 if speed is None else float(speed)
            velocity_array = velocity_from_heading(resolved_heading, resolved_speed)
        else:
            velocity_array = np.asarray(velocity, dtype=float).reshape(2)
            resolved_speed = float(np.linalg.norm(velocity_array)) if speed is None else float(speed)
            resolved_heading = (
                vector_heading(velocity_array, fallback=0.0)
                if heading is None
                else wrap_to_pi(float(heading))
            )
        return cls(
            vessel_id=str(vessel_id),
            position=position_array,
            velocity=velocity_array,
            heading=resolved_heading,
            speed=resolved_speed,
            radius=float(radius),
            role=role,
            colregs_compliant=bool(colregs_compliant),
        )


@dataclass(frozen=True)
class COLREGsConfig:
    risk_time_horizon: float = 180.0
    min_cpa_distance: float = 0.12
    safety_buffer: float = 0.02
    min_speed: float = 1.0e-8
    head_on_bearing_deg: float = 15.0
    head_on_course_deg: float = 25.0
    overtaking_sector_deg: float = 112.5


@dataclass(frozen=True)
class COLREGsReport:
    ownship_id: str
    target_id: str
    encounter_type: EncounterType
    risk_of_collision: bool
    give_way_vessel: Optional[str]
    stand_on_vessel: Optional[str]
    applicable_rules: Tuple[str, ...]
    required_actions: Tuple[COLREGsAction, ...]
    recommended_actions: Tuple[COLREGsAction, ...]
    distance: float
    dcpa: float
    tcpa: float
    relative_bearing_rad: float
    target_relative_bearing_rad: float
    heading_difference_rad: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "ownship_id": self.ownship_id,
            "target_id": self.target_id,
            "encounter_type": self.encounter_type.value,
            "risk_of_collision": self.risk_of_collision,
            "give_way_vessel": self.give_way_vessel,
            "stand_on_vessel": self.stand_on_vessel,
            "applicable_rules": self.applicable_rules,
            "required_actions": tuple(action.value for action in self.required_actions),
            "recommended_actions": tuple(action.value for action in self.recommended_actions),
            "distance": self.distance,
            "dcpa": self.dcpa,
            "tcpa": self.tcpa,
            "relative_bearing_rad": self.relative_bearing_rad,
            "target_relative_bearing_rad": self.target_relative_bearing_rad,
            "heading_difference_rad": self.heading_difference_rad,
        }


class COLREGsEngine:
    """Rule-level COLREGs evaluator for 2D vessel encounters.

    This module only describes maritime-rule obligations. It does not modify
    policy actions, rewards, environment transitions, or obstacle movement.
    """

    def __init__(self, config: Optional[COLREGsConfig] = None):
        self.config = config or COLREGsConfig()

    def evaluate(self, ownship: VesselState, target: VesselState) -> COLREGsReport:
        metrics = self._relative_metrics(ownship, target)
        risk = self.has_collision_risk(ownship, target, metrics)
        encounter = self.classify_encounter(ownship, target, metrics, risk)
        rules, required, recommended, give_way, stand_on = self._obligations(
            encounter, ownship.vessel_id, target.vessel_id
        )
        return COLREGsReport(
            ownship_id=ownship.vessel_id,
            target_id=target.vessel_id,
            encounter_type=encounter,
            risk_of_collision=risk,
            give_way_vessel=give_way,
            stand_on_vessel=stand_on,
            applicable_rules=rules,
            required_actions=required,
            recommended_actions=recommended,
            distance=metrics["distance"],
            dcpa=metrics["dcpa"],
            tcpa=metrics["tcpa"],
            relative_bearing_rad=metrics["relative_bearing"],
            target_relative_bearing_rad=metrics["target_relative_bearing"],
            heading_difference_rad=metrics["heading_difference"],
        )

    def evaluate_many(
        self,
        ownship: VesselState,
        targets: Iterable[VesselState],
    ) -> Tuple[COLREGsReport, ...]:
        return tuple(self.evaluate(ownship, target) for target in targets if target.vessel_id != ownship.vessel_id)

    def classify_encounter(
        self,
        ownship: VesselState,
        target: VesselState,
        metrics: Optional[Dict[str, float]] = None,
        risk: Optional[bool] = None,
    ) -> EncounterType:
        metrics = metrics or self._relative_metrics(ownship, target)
        risk = self.has_collision_risk(ownship, target, metrics) if risk is None else risk
        if not risk:
            return EncounterType.SAFE_PASSAGE

        if self._is_head_on(metrics):
            return EncounterType.HEAD_ON
        if self._ownship_is_overtaking_target(metrics):
            return EncounterType.OVERTAKING_GIVE_WAY
        if self._target_is_overtaking_ownship(metrics):
            return EncounterType.OVERTAKEN_STAND_ON
        if metrics["relative_bearing"] < 0.0:
            return EncounterType.CROSSING_GIVE_WAY
        if metrics["relative_bearing"] > 0.0:
            return EncounterType.CROSSING_STAND_ON
        return EncounterType.COLLISION_RISK_UNDEFINED

    def has_collision_risk(
        self,
        ownship: VesselState,
        target: VesselState,
        metrics: Optional[Dict[str, float]] = None,
    ) -> bool:
        metrics = metrics or self._relative_metrics(ownship, target)
        clearance = ownship.radius + target.radius + self.config.safety_buffer
        cpa_limit = max(self.config.min_cpa_distance, clearance)
        if metrics["distance"] <= cpa_limit:
            return True
        return 0.0 <= metrics["tcpa"] <= self.config.risk_time_horizon and metrics["dcpa"] <= cpa_limit

    def _relative_metrics(self, ownship: VesselState, target: VesselState) -> Dict[str, float]:
        relative_position = target.position - ownship.position
        relative_velocity = target.velocity - ownship.velocity
        distance = float(np.linalg.norm(relative_position))
        rel_speed_sq = float(np.dot(relative_velocity, relative_velocity))
        if rel_speed_sq <= EPS:
            tcpa = np.inf
            dcpa = distance
        else:
            tcpa = float(-np.dot(relative_position, relative_velocity) / rel_speed_sq)
            closest_position = relative_position + relative_velocity * max(tcpa, 0.0)
            dcpa = float(np.linalg.norm(closest_position))

        bearing_to_target = vector_heading(relative_position, fallback=ownship.heading)
        bearing_from_target = vector_heading(-relative_position, fallback=target.heading)
        relative_bearing = signed_angle_between(ownship.heading, bearing_to_target)
        target_relative_bearing = signed_angle_between(target.heading, bearing_from_target)
        heading_difference = angle_abs(target.heading - ownship.heading)
        return {
            "distance": distance,
            "tcpa": tcpa,
            "dcpa": dcpa,
            "relative_bearing": relative_bearing,
            "target_relative_bearing": target_relative_bearing,
            "heading_difference": heading_difference,
        }

    def _is_head_on(self, metrics: Dict[str, float]) -> bool:
        bearing_limit = np.deg2rad(self.config.head_on_bearing_deg)
        course_limit = np.deg2rad(self.config.head_on_course_deg)
        return (
            angle_abs(metrics["relative_bearing"]) <= bearing_limit
            and angle_abs(metrics["target_relative_bearing"]) <= bearing_limit
            and abs(np.pi - metrics["heading_difference"]) <= course_limit
        )

    def _ownship_is_overtaking_target(self, metrics: Dict[str, float]) -> bool:
        overtaking_limit = np.deg2rad(self.config.overtaking_sector_deg)
        return angle_abs(metrics["target_relative_bearing"]) >= overtaking_limit

    def _target_is_overtaking_ownship(self, metrics: Dict[str, float]) -> bool:
        overtaking_limit = np.deg2rad(self.config.overtaking_sector_deg)
        return angle_abs(metrics["relative_bearing"]) >= overtaking_limit

    def _obligations(
        self,
        encounter: EncounterType,
        ownship_id: str,
        target_id: str,
    ) -> Tuple[
        Tuple[str, ...],
        Tuple[COLREGsAction, ...],
        Tuple[COLREGsAction, ...],
        Optional[str],
        Optional[str],
    ]:
        base_rules = (RULE_LOOKOUT, RULE_SAFE_SPEED, RULE_RISK)
        base_actions = (
            COLREGsAction.KEEP_LOOKOUT,
            COLREGsAction.USE_SAFE_SPEED,
            COLREGsAction.ASSESS_COLLISION_RISK,
        )

        if encounter == EncounterType.SAFE_PASSAGE:
            return base_rules, base_actions, base_actions, None, None

        if encounter == EncounterType.HEAD_ON:
            actions = base_actions + (
                COLREGsAction.ALTER_COURSE_STARBOARD,
                COLREGsAction.TAKE_EARLY_SUBSTANTIAL_ACTION,
            )
            return (
                base_rules + (RULE_ACTION, RULE_HEAD_ON),
                actions,
                actions,
                "both",
                "both",
            )

        if encounter == EncounterType.CROSSING_GIVE_WAY:
            required = base_actions + (
                COLREGsAction.KEEP_CLEAR,
                COLREGsAction.AVOID_CROSSING_AHEAD,
                COLREGsAction.PASS_ASTERN,
                COLREGsAction.ALTER_COURSE_STARBOARD,
                COLREGsAction.TAKE_EARLY_SUBSTANTIAL_ACTION,
            )
            return (
                base_rules + (RULE_ACTION, RULE_CROSSING, RULE_GIVE_WAY),
                required,
                required + (COLREGsAction.REDUCE_SPEED_IF_NEEDED,),
                ownship_id,
                target_id,
            )

        if encounter == EncounterType.CROSSING_STAND_ON:
            required = base_actions + (
                COLREGsAction.KEEP_COURSE_AND_SPEED,
                COLREGsAction.MONITOR_GIVE_WAY_VESSEL,
            )
            return (
                base_rules + (RULE_CROSSING, RULE_STAND_ON),
                required,
                required,
                target_id,
                ownship_id,
            )

        if encounter == EncounterType.OVERTAKING_GIVE_WAY:
            required = base_actions + (
                COLREGsAction.KEEP_CLEAR,
                COLREGsAction.TAKE_EARLY_SUBSTANTIAL_ACTION,
            )
            return (
                base_rules + (RULE_ACTION, RULE_OVERTAKING, RULE_GIVE_WAY),
                required,
                required + (COLREGsAction.REDUCE_SPEED_IF_NEEDED,),
                ownship_id,
                target_id,
            )

        if encounter == EncounterType.OVERTAKEN_STAND_ON:
            required = base_actions + (
                COLREGsAction.KEEP_COURSE_AND_SPEED,
                COLREGsAction.MONITOR_GIVE_WAY_VESSEL,
            )
            return (
                base_rules + (RULE_OVERTAKING, RULE_STAND_ON),
                required,
                required,
                target_id,
                ownship_id,
            )

        required = base_actions + (
            COLREGsAction.REDUCE_SPEED_IF_NEEDED,
            COLREGsAction.TAKE_EARLY_SUBSTANTIAL_ACTION,
        )
        return base_rules + (RULE_ACTION,), required, required, None, None
