from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utilities.paths import colregs_visualization_dir


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def vessel(vessel_id, position, velocity, heading=None, radius=0.01):
    from safety.colregs import VesselState

    return VesselState.from_kinematics(
        vessel_id=vessel_id,
        position=position,
        velocity=velocity,
        heading=heading,
        radius=radius,
    )


def check_rule_classification() -> None:
    from safety.colregs import COLREGsAction, COLREGsEngine, EncounterType

    engine = COLREGsEngine()

    cases = [
        (
            "head_on",
            vessel("own", [0.0, 0.0], [1.0, 0.0], heading=0.0),
            vessel("target", [1.0, 0.0], [-1.0, 0.0], heading=np.pi),
            EncounterType.HEAD_ON,
            COLREGsAction.ALTER_COURSE_STARBOARD,
            "both",
        ),
        (
            "crossing_give_way",
            vessel("own", [0.0, 0.0], [1.0, 0.0], heading=0.0),
            vessel("target", [1.0, -1.0], [0.0, 1.0], heading=np.pi / 2.0),
            EncounterType.CROSSING_GIVE_WAY,
            COLREGsAction.PASS_ASTERN,
            "own",
        ),
        (
            "crossing_stand_on",
            vessel("own", [0.0, 0.0], [1.0, 0.0], heading=0.0),
            vessel("target", [1.0, 1.0], [0.0, -1.0], heading=-np.pi / 2.0),
            EncounterType.CROSSING_STAND_ON,
            COLREGsAction.KEEP_COURSE_AND_SPEED,
            "target",
        ),
        (
            "overtaking_give_way",
            vessel("own", [0.0, 0.0], [1.0, 0.0], heading=0.0),
            vessel("target", [1.0, 0.0], [0.5, 0.0], heading=0.0),
            EncounterType.OVERTAKING_GIVE_WAY,
            COLREGsAction.KEEP_CLEAR,
            "own",
        ),
        (
            "overtaken_stand_on",
            vessel("own", [0.0, 0.0], [1.0, 0.0], heading=0.0),
            vessel("target", [-1.0, 0.0], [1.5, 0.0], heading=0.0),
            EncounterType.OVERTAKEN_STAND_ON,
            COLREGsAction.KEEP_COURSE_AND_SPEED,
            "target",
        ),
        (
            "safe_passage",
            vessel("own", [0.0, 0.0], [1.0, 0.0], heading=0.0),
            vessel("target", [1.0, 1.0], [1.0, 0.0], heading=0.0),
            EncounterType.SAFE_PASSAGE,
            COLREGsAction.KEEP_LOOKOUT,
            None,
        ),
    ]

    for name, ownship, target, expected, expected_action, expected_give_way in cases:
        report = engine.evaluate(ownship, target)
        require(report.encounter_type == expected, "%s classified as %s" % (name, report.encounter_type))
        require(expected_action in report.required_actions, "%s missing required action %s" % (name, expected_action))
        require(report.give_way_vessel == expected_give_way, "%s give-way mismatch" % name)
        print(
            "PASS %-22s encounter=%s risk=%s tcpa=%.3f dcpa=%.3f"
            % (name, report.encounter_type.value, report.risk_of_collision, report.tcpa, report.dcpa)
        )


def check_environment_colregs() -> None:
    from utilities.envs import make_env

    env = make_env(
        "usv_tracking_base",
        num_agents=1,
        num_landmarks=1,
        num_obstacles=5,
        num_ob=3,
        control_model="usv_3dof",
    )
    obs = env.reset()
    agent = env.world.agents[0]
    obstacle_vessels = env.world.dynamic_obstacle_vessels()
    reports = env.world.evaluate_colregs_for_entity(agent, obstacle_vessels)

    require(getattr(agent, "is_vessel", False), "Agent is not marked as a vessel.")
    require(getattr(env.world.landmarks[0], "is_vessel", False), "Target is not marked as a vessel.")
    require(
        not getattr(env.world.landmarks[0], "colregs_compliant", True),
        "Target should be a non-compliant vessel controlled by escape logic.",
    )
    require(len(obstacle_vessels) == env.world.num_obstacles, "Dynamic obstacles are not all vessel-marked.")
    require(all(ob.colregs_compliant for ob in obstacle_vessels), "Dynamic obstacle vessel is not COLREGs-compliant.")
    require(len(reports) == env.world.num_obstacles, "Agent did not receive one report per dynamic obstacle.")

    env.step([np.array([0.0, 0.0], dtype=float)])
    require(len(env.world.colregs_last_reports) > 0, "World did not cache COLREGs reports after step.")
    print(
        "PASS environment_colregs vessels=%d agent_reports=%d cached_reports=%d obs_shape=%s"
        % (len(obstacle_vessels), len(reports), len(env.world.colregs_last_reports), np.asarray(obs[0]).shape)
    )


def check_colregs_obstacle_control() -> None:
    import multiagent.scenarios as scenarios

    scenario = scenarios.load("usv_tracking_base.py").Scenario()
    world = scenario.make_world(
        num_agents=1,
        num_landmarks=1,
        num_obstacles=1,
        num_ob=1,
        control_model="usv_3dof",
        obstacle_movement="steady",
    )
    agent = world.agents[0]
    obstacle = world.obstacles[0]

    agent.state.p_pos = np.array([0.0, 0.0], dtype=float)
    agent.state.p_vel = np.array([0.0003, 0.0], dtype=float)
    agent.size = 0.04

    obstacle.state.p_pos = np.array([0.08, 0.0], dtype=float)
    obstacle.state.p_vel = np.array([-0.0003, 0.0], dtype=float)
    obstacle.size = 0.04
    obstacle.obstacle_vel = 0.0003
    obstacle.max_speed = 0.0003
    obstacle.ra = np.pi

    previous_heading = obstacle.ra
    scenario._move_obstacle(obstacle, world)
    heading_delta = ((obstacle.ra - previous_heading + np.pi) % (2.0 * np.pi)) - np.pi

    require(obstacle.colregs_last_report is not None, "Obstacle did not produce a critical COLREGs report.")
    require(
        obstacle.colregs_last_report.encounter_type.value == "head_on",
        "Obstacle did not classify the encounter as head-on.",
    )
    require(
        obstacle.colregs_control_action == "head_on_starboard",
        "Obstacle did not select the head-on starboard maneuver.",
    )
    require(heading_delta < 0.0, "Starboard maneuver should reduce heading in this coordinate convention.")
    print(
        "PASS obstacle_control action=%s heading_delta=%.6f speed=%.8f"
        % (obstacle.colregs_control_action, heading_delta, float(np.linalg.norm(obstacle.action.u)))
    )


def check_dynamic_obstacle_colregs_pair() -> None:
    import multiagent.scenarios as scenarios

    scenario = scenarios.load("usv_tracking_base.py").Scenario()
    world = scenario.make_world(
        num_agents=1,
        num_landmarks=1,
        num_obstacles=2,
        num_ob=1,
        control_model="usv_3dof",
        obstacle_movement="steady",
    )
    agent = world.agents[0]
    target = world.landmarks[0]
    moving = world.obstacles[0]
    other = world.obstacles[1]

    agent.state.p_pos = np.array([-0.9, 0.9], dtype=float)
    agent.state.p_vel = np.zeros(2, dtype=float)
    target.state.p_pos = np.array([-0.9, -0.9], dtype=float)
    target.state.p_vel = np.zeros(2, dtype=float)

    moving.state.p_pos = np.array([0.0, 0.0], dtype=float)
    moving.state.p_vel = np.array([0.0003, 0.0], dtype=float)
    moving.size = 0.04
    moving.obstacle_vel = 0.0003
    moving.max_speed = 0.0003
    moving.ra = 0.0

    other.state.p_pos = np.array([0.08, 0.0], dtype=float)
    other.state.p_vel = np.array([-0.0003, 0.0], dtype=float)
    other.size = 0.04
    other.obstacle_vel = 0.0003
    other.max_speed = 0.0003
    other.ra = np.pi

    previous_heading = moving.ra
    scenario._move_obstacle(moving, world)
    heading_delta = ((moving.ra - previous_heading + np.pi) % (2.0 * np.pi)) - np.pi

    require(moving.colregs_last_report is not None, "Moving obstacle did not evaluate the other obstacle.")
    require(
        moving.colregs_last_report.target_id == other.name,
        "Moving obstacle did not select the closest obstacle vessel as COLREGs target.",
    )
    require(
        moving.colregs_last_report.encounter_type.value == "head_on",
        "Dynamic obstacle pair was not classified as head-on.",
    )
    require(moving.colregs_control_action == "head_on_starboard", "Dynamic obstacle did not apply starboard action.")
    require(heading_delta < 0.0, "Dynamic obstacle pair should turn to starboard.")
    print(
        "PASS obstacle_pair_colregs target=%s action=%s geometry=%s"
        % (moving.colregs_last_report.target_id, moving.colregs_control_action, moving.geometric_avoidance_active)
    )


def check_static_geometric_avoidance() -> None:
    import multiagent.scenarios as scenarios

    scenario = scenarios.load("usv_tracking_base.py").Scenario()
    world = scenario.make_world(
        num_agents=1,
        num_landmarks=1,
        num_obstacles=2,
        num_ob=1,
        control_model="usv_3dof",
        obstacle_movement="steady",
    )
    agent = world.agents[0]
    target = world.landmarks[0]
    moving = world.obstacles[0]
    static_obstacle = world.obstacles[1]

    agent.state.p_pos = np.array([-0.9, 0.9], dtype=float)
    agent.state.p_vel = np.zeros(2, dtype=float)
    target.state.p_pos = np.array([-0.9, -0.9], dtype=float)
    target.state.p_vel = np.zeros(2, dtype=float)

    moving.state.p_pos = np.array([0.0, 0.0], dtype=float)
    moving.state.p_vel = np.array([0.0003, 0.0], dtype=float)
    moving.size = 0.04
    moving.obstacle_vel = 0.0003
    moving.max_speed = 0.0003
    moving.ra = 0.0

    static_obstacle.name = "static_obstacle"
    static_obstacle.state.p_pos = np.array([0.12, 0.03], dtype=float)
    static_obstacle.state.p_vel = np.zeros(2, dtype=float)
    static_obstacle.size = 0.06
    static_obstacle.movable = False
    static_obstacle.is_vessel = False
    static_obstacle.colregs_compliant = False
    static_obstacle.vessel_role = "static_obstacle"

    previous_heading = moving.ra
    scenario._move_obstacle(moving, world)
    heading_delta = ((moving.ra - previous_heading + np.pi) % (2.0 * np.pi)) - np.pi

    require(moving.geometric_avoidance_active, "Moving obstacle did not activate geometric avoidance.")
    require(
        moving.geometric_avoidance_target == static_obstacle.name,
        "Moving obstacle did not select the nearby static obstacle.",
    )
    require(abs(heading_delta) <= scenario.geometric_avoidance_max_turn + 1.0e-9, "Geometric turn limit failed.")
    require(abs(heading_delta) > 1.0e-9, "Geometric avoidance did not change heading.")
    print(
        "PASS static_geometric_avoidance target=%s heading_delta=%.6f distance=%.6f"
        % (moving.geometric_avoidance_target, heading_delta, moving.geometric_avoidance_distance)
    )


def check_smooth_target_escape() -> None:
    import multiagent.scenarios as scenarios

    scenario = scenarios.load("usv_tracking_base.py").Scenario()
    world = scenario.make_world(
        num_agents=1,
        num_landmarks=1,
        num_obstacles=1,
        num_ob=1,
        control_model="usv_3dof",
        movement="escape",
    )
    scenario.target_escape_noise = 0.0
    agent = world.agents[0]
    target = world.landmarks[0]

    agent.state.p_pos = np.array([0.0, 0.0], dtype=float)
    agent.state.p_vel = np.zeros(2, dtype=float)
    target.state.p_pos = np.array([1.0, 0.0], dtype=float)
    target.state.p_vel = np.zeros(2, dtype=float)
    target.landmark_vel = 0.0003
    target.max_speed = 0.0003
    target.current_landmark_vel = 0.0003
    target.ra = np.pi

    previous_heading = target.ra
    previous_speed = target.current_landmark_vel
    scenario._move_target(target, world)
    heading_delta = ((target.ra - previous_heading + np.pi) % (2.0 * np.pi)) - np.pi

    require(target.target_escape_control_action == "smooth_escape", "Target did not use smooth escape.")
    require(abs(heading_delta) <= scenario.target_max_turn + 1.0e-9, "Target heading change was not limited.")
    require(target.current_landmark_vel <= previous_speed, "Target speed should change smoothly toward cruise speed.")
    require(
        abs(target.current_landmark_vel - previous_speed) < previous_speed,
        "Target speed changed too abruptly.",
    )
    print(
        "PASS smooth_target_escape heading_delta=%.6f speed %.8f->%.8f"
        % (heading_delta, previous_speed, target.current_landmark_vel)
    )


def _set_pose(entity, position, velocity=None, heading=None):
    entity.state.p_pos = np.array(position, dtype=float)
    entity.state.p_vel = np.zeros(2, dtype=float) if velocity is None else np.array(velocity, dtype=float)
    if heading is not None and hasattr(entity, "ra"):
        entity.ra = float(heading)


def _integrate_script_entity(entity, world) -> None:
    if not entity.movable:
        return
    if entity.action.u is None:
        action = np.zeros(world.dim_p, dtype=float)
    else:
        action = np.asarray(entity.action.u, dtype=float)
    entity.state.p_vel = np.asarray(entity.state.p_vel, dtype=float) * (1.0 - world.damping)
    entity.state.p_vel += (action / entity.mass) * world.dt
    if entity.max_speed is not None:
        speed = float(np.linalg.norm(entity.state.p_vel))
        if speed > entity.max_speed and speed > 1.0e-12:
            entity.state.p_vel = entity.state.p_vel / speed * entity.max_speed
    entity.state.p_pos = np.asarray(entity.state.p_pos, dtype=float) + entity.state.p_vel * world.dt


def _entity_visual_kind(entity):
    role = getattr(entity, "vessel_role", "")
    if role == "ownship" or "agent" in entity.name:
        return "agent"
    if role == "target" or "landmark" in entity.name:
        return "target"
    if role == "static_obstacle" or not getattr(entity, "movable", False):
        return "static"
    return "obstacle"


def _new_visual_case(title, entities):
    palette = ["#16a34a", "#f97316", "#7c3aed", "#0891b2", "#db2777"]
    colors_by_kind = {
        "agent": "#2563eb",
        "target": "#dc2626",
        "static": "#525252",
    }
    return {
        "title": title,
        "entities": {
            entity.name: {
                "kind": _entity_visual_kind(entity),
                "size": float(entity.size),
                "color": colors_by_kind.get(_entity_visual_kind(entity), palette[index % len(palette)]),
            }
            for index, entity in enumerate(entities)
        },
        "trajectories": {entity.name: [] for entity in entities},
        "headings": {entity.name: [] for entity in entities},
        "notes": [],
        "proof_records": [],
        "proof_summary": "",
    }


def _record_visual_case(case, entities, note):
    for entity in entities:
        case["trajectories"][entity.name].append(np.asarray(entity.state.p_pos, dtype=float).copy())
        if hasattr(entity, "ra"):
            heading = float(entity.ra)
        elif np.linalg.norm(entity.state.p_vel) > 1.0e-12:
            heading = float(np.arctan2(entity.state.p_vel[1], entity.state.p_vel[0]))
        else:
            heading = None
        case["headings"][entity.name].append(heading)
    case["notes"].append(note)


def _make_tracking_world(num_obstacles=2, movement="escape", obstacle_movement="steady"):
    import multiagent.scenarios as scenarios

    scenario = scenarios.load("usv_tracking_base.py").Scenario()
    world = scenario.make_world(
        num_agents=1,
        num_landmarks=1,
        num_obstacles=num_obstacles,
        num_ob=1,
        landmark_vel=0.0003,
        max_vel=0.0003,
        control_model="usv_3dof",
        movement=movement,
        obstacle_movement=obstacle_movement,
    )
    world.damping = 0.0
    return scenario, world


def _prepare_far_agent_and_target(world):
    agent = world.agents[0]
    target = world.landmarks[0]
    _set_pose(agent, [-0.9, 0.9], [0.0, 0.0])
    _set_pose(target, [-0.9, -0.9], [0.0, 0.0], heading=0.0)
    target.landmark_vel = 0.0
    target.current_landmark_vel = 0.0
    target.max_speed = 0.0
    target.movable = False


def _simulate_dynamic_obstacle_colregs(steps):
    scenario, world = _make_tracking_world(num_obstacles=2, movement="linear", obstacle_movement="steady")
    _prepare_far_agent_and_target(world)
    left = world.obstacles[0]
    right = world.obstacles[1]

    _set_pose(left, [-0.34, 0.025], [0.0012, 0.0], heading=0.0)
    left.size = 0.04
    left.obstacle_vel = 0.0012
    left.max_speed = 0.0012

    _set_pose(right, [0.34, -0.025], [-0.0012, 0.0], heading=np.pi)
    right.size = 0.04
    right.obstacle_vel = 0.0012
    right.max_speed = 0.0012

    entities = [left, right]
    case = _new_visual_case("Dynamic vessels: COLREGs head-on", entities)
    _record_visual_case(case, entities, "initial")
    for _ in range(steps):
        for obstacle in entities:
            scenario._move_obstacle(obstacle, world)
        note = "%s | %s" % (left.colregs_control_action, right.colregs_control_action)
        if left.geometric_avoidance_active or right.geometric_avoidance_active:
            note += " | geom"
        for obstacle in entities:
            _integrate_script_entity(obstacle, world)
        world.evaluate_all_colregs()
        _record_visual_case(case, entities, note)
    return case


def _simulate_static_geometric_avoidance(steps):
    scenario, world = _make_tracking_world(num_obstacles=2, movement="linear", obstacle_movement="steady")
    _prepare_far_agent_and_target(world)
    moving = world.obstacles[0]
    static_obstacle = world.obstacles[1]

    _set_pose(moving, [-0.48, 0.0], [0.0012, 0.0], heading=0.0)
    moving.size = 0.04
    moving.obstacle_vel = 0.0012
    moving.max_speed = 0.0012

    static_obstacle.name = "static obstacle"
    _set_pose(static_obstacle, [0.0, 0.035], [0.0, 0.0], heading=0.0)
    static_obstacle.size = 0.075
    static_obstacle.movable = False
    static_obstacle.is_vessel = False
    static_obstacle.colregs_compliant = False
    static_obstacle.vessel_role = "static_obstacle"

    entities = [moving, static_obstacle]
    case = _new_visual_case("Static obstacle: geometric avoidance", entities)
    _record_visual_case(case, entities, "initial")
    for _ in range(steps):
        scenario._move_obstacle(moving, world)
        note = "geom=%s target=%s" % (
            moving.geometric_avoidance_active,
            moving.geometric_avoidance_target,
        )
        _integrate_script_entity(moving, world)
        _record_visual_case(case, entities, note)
    return case


def _simulate_smooth_target_escape(steps):
    scenario, world = _make_tracking_world(num_obstacles=1, movement="escape", obstacle_movement="steady")
    scenario.target_escape_noise = 0.04
    scenario.target_min_speed_factor = 0.85
    agent = world.agents[0]
    target = world.landmarks[0]
    obstacle = world.obstacles[0]

    _set_pose(agent, [0.0, 0.0], [0.0, 0.0])
    _set_pose(target, [0.54, 0.0], [0.0, 0.0], heading=np.pi / 2.0)
    target.landmark_vel = 0.0003
    target.current_landmark_vel = 0.0003
    target.max_speed = 0.0003
    target.movable = True

    _set_pose(obstacle, [0.85, 0.85], [0.0, 0.0], heading=0.0)
    obstacle.movable = False
    obstacle.is_vessel = False

    entities = [agent, target]
    case = _new_visual_case("Target: smooth escape from USV", entities)
    _record_visual_case(case, entities, "initial")
    agent_speed = 0.00036
    for _ in range(steps):
        to_target = np.asarray(target.state.p_pos) - np.asarray(agent.state.p_pos)
        distance = float(np.linalg.norm(to_target))
        if distance > 1.0e-12:
            agent.state.p_vel = to_target / distance * agent_speed
        else:
            agent.state.p_vel = np.zeros(2, dtype=float)
        agent.state.p_pos = np.asarray(agent.state.p_pos, dtype=float) + agent.state.p_vel * world.dt

        scenario._move_target(target, world)
        note = "escape dist=%.3f speed=%.5f" % (
            target.target_escape_last_distance,
            target.current_landmark_vel,
        )
        _integrate_script_entity(target, world)
        _record_visual_case(case, entities, note)
    return case


def _heading_velocity(heading, speed):
    return np.array([np.cos(heading) * speed, np.sin(heading) * speed], dtype=float)


def _configure_visual_ship(entity, name, position, heading, speed, size=0.04):
    entity.name = name
    entity.size = size
    entity.movable = True
    entity.is_vessel = True
    entity.colregs_compliant = True
    entity.vessel_role = "dynamic_obstacle"
    entity.obstacle_vel = speed
    entity.max_speed = speed
    entity.ra = float(heading)
    entity.action.u = _heading_velocity(heading, speed)
    entity.state.p_pos = np.array(position, dtype=float)
    entity.state.p_vel = _heading_velocity(heading, speed)
    entity.colregs_last_report = None
    entity.colregs_last_reports = ()
    entity.colregs_control_action = "none"
    entity.geometric_avoidance_active = False
    entity.geometric_avoidance_target = None
    entity.geometric_avoidance_distance = None


def _configure_visual_static(entity, name, position, size=0.08):
    entity.name = name
    entity.size = size
    entity.movable = False
    entity.is_vessel = False
    entity.colregs_compliant = False
    entity.vessel_role = "static_obstacle"
    entity.obstacle_vel = 0.0
    entity.max_speed = 0.0
    entity.ra = 0.0
    entity.action.u = np.zeros(2, dtype=float)
    entity.state.p_pos = np.array(position, dtype=float)
    entity.state.p_vel = np.zeros(2, dtype=float)


def _make_complex_visual_world():
    scenario, world = _make_tracking_world(num_obstacles=6, movement="linear", obstacle_movement="steady")
    world.damping = 0.0

    agent = world.agents[0]
    target = world.landmarks[0]
    _set_pose(agent, [3.0, 3.0], [0.0, 0.0])
    _set_pose(target, [-3.0, -3.0], [0.0, 0.0], heading=0.0)
    agent.is_vessel = False
    target.is_vessel = False
    target.movable = False
    target.landmark_vel = 0.0
    target.current_landmark_vel = 0.0
    target.max_speed = 0.0
    return scenario, world


def _complex_entities(world):
    return list(world.obstacles[:6])


def _dynamic_visual_ships(world):
    return list(world.obstacles[:4])


def _static_visual_obstacles(world):
    return list(world.obstacles[4:6])


def _require_complex_case_shape(entities, title):
    dynamic_count = sum(
        1 for entity in entities
        if getattr(entity, "movable", False) and getattr(entity, "is_vessel", False)
    )
    static_count = sum(
        1 for entity in entities
        if not getattr(entity, "movable", False) and not getattr(entity, "is_vessel", False)
    )
    require(dynamic_count >= 4, "%s has fewer than four dynamic ships." % title)
    require(static_count >= 2, "%s has fewer than two static obstacles." % title)


def _rule_labels(rules):
    labels = []
    for rule in rules:
        prefix = str(rule).split(":", 1)[0].strip()
        labels.append(prefix.replace("Rule ", "R"))
    return tuple(labels)


def _proof_record(world, ownship, target, label, expected_encounter):
    report = world.evaluate_colregs_between(ownship, target)
    require(
        report.encounter_type.value == expected_encounter,
        "%s expected %s but got %s." % (label, expected_encounter, report.encounter_type.value),
    )
    return {
        "label": label,
        "ownship": ownship.name,
        "target": target.name,
        "encounter": report.encounter_type.value,
        "risk": bool(report.risk_of_collision),
        "rules": tuple(report.applicable_rules),
        "rule_labels": _rule_labels(report.applicable_rules),
        "give_way": report.give_way_vessel,
        "stand_on": report.stand_on_vessel,
        "required_actions": tuple(action.value for action in report.required_actions),
        "recommended_actions": tuple(action.value for action in report.recommended_actions),
    }


def _attach_visual_proofs(case, records):
    case["proof_records"] = list(records)
    rules = []
    encounters = []
    for record in records:
        rules.extend(record["rule_labels"])
        encounters.append(record["encounter"])

    def rule_key(label):
        try:
            return int(label[1:])
        except (ValueError, IndexError):
            return 999

    compact_rules = ",".join(sorted(set(rules), key=rule_key))
    compact_encounters = ",".join(sorted(set(encounters)))
    case["proof_summary"] = "Rules %s | %s" % (compact_rules, compact_encounters)


def _active_ship_note(ships):
    active_actions = []
    active_geometry = []
    for ship in ships:
        action = getattr(ship, "colregs_control_action", "none")
        if action not in ("none", "nominal"):
            active_actions.append("%s:%s" % (ship.name, action))
        if getattr(ship, "geometric_avoidance_active", False):
            active_geometry.append("%s->%s" % (ship.name, ship.geometric_avoidance_target))
    if not active_actions and not active_geometry:
        return "nominal"
    parts = active_actions[:3]
    if len(active_actions) > 3:
        parts.append("+%d actions" % (len(active_actions) - 3))
    if active_geometry:
        parts.append("geom " + ",".join(active_geometry[:2]))
    return " | ".join(parts)


def _simulate_complex_case(title, setup_fn, proof_fn, steps):
    scenario, world = _make_complex_visual_world()
    setup_fn(world)
    entities = _complex_entities(world)
    ships = _dynamic_visual_ships(world)
    _require_complex_case_shape(entities, title)
    case = _new_visual_case(title, entities)
    _attach_visual_proofs(case, proof_fn(world))
    _record_visual_case(case, entities, "initial")
    for _ in range(steps):
        for ship in ships:
            scenario._move_obstacle(ship, world)
        note = _active_ship_note(ships)
        for ship in ships:
            _integrate_script_entity(ship, world)
        world.evaluate_all_colregs()
        _record_visual_case(case, entities, note)
    return case


def _setup_head_on_case(world):
    ships = _dynamic_visual_ships(world)
    statics = _static_visual_obstacles(world)
    _configure_visual_ship(ships[0], "ship H0", [-0.24, 0.02], 0.0, 0.0015)
    _configure_visual_ship(ships[1], "ship H1", [0.24, -0.02], np.pi, 0.0015)
    _configure_visual_ship(ships[2], "ship H2", [-0.75, 0.45], 0.0, 0.0008)
    _configure_visual_ship(ships[3], "ship H3", [0.75, -0.45], np.pi, 0.0008)
    _configure_visual_static(statics[0], "static H0", [-0.02, 0.25], 0.075)
    _configure_visual_static(statics[1], "static H1", [0.18, -0.30], 0.075)


def _proof_head_on_case(world):
    ships = _dynamic_visual_ships(world)
    return [
        _proof_record(world, ships[0], ships[1], "Rule14 both alter starboard", "head_on"),
        _proof_record(world, ships[1], ships[0], "Rule14 reciprocal", "head_on"),
    ]


def _setup_crossing_case(world):
    ships = _dynamic_visual_ships(world)
    statics = _static_visual_obstacles(world)
    _configure_visual_ship(ships[0], "ship C0 give-way", [-0.28, 0.02], 0.0, 0.0018)
    _configure_visual_ship(ships[1], "ship C1 stand-on", [0.02, -0.28], np.pi / 2.0, 0.0018)
    _configure_visual_ship(ships[2], "ship C2", [-0.72, -0.38], 0.20, 0.00075)
    _configure_visual_ship(ships[3], "ship C3", [0.72, 0.38], -2.95, 0.00075)
    _configure_visual_static(statics[0], "static C0", [-0.18, 0.30], 0.075)
    _configure_visual_static(statics[1], "static C1", [0.33, -0.18], 0.075)


def _proof_crossing_case(world):
    ships = _dynamic_visual_ships(world)
    return [
        _proof_record(world, ships[0], ships[1], "Rule15/16 crossing give-way", "crossing_give_way"),
        _proof_record(world, ships[1], ships[0], "Rule15/17 crossing stand-on", "crossing_stand_on"),
    ]


def _setup_overtaking_case(world):
    ships = _dynamic_visual_ships(world)
    statics = _static_visual_obstacles(world)
    _configure_visual_ship(ships[0], "ship O0 overtaking", [-0.44, -0.06], 0.0, 0.0019)
    _configure_visual_ship(ships[1], "ship O1 stand-on", [-0.25, -0.06], 0.0, 0.0006)
    _configure_visual_ship(ships[2], "ship O2", [-0.65, 0.42], -0.10, 0.00075)
    _configure_visual_ship(ships[3], "ship O3", [0.70, -0.42], 3.05, 0.00075)
    _configure_visual_static(statics[0], "static O0", [-0.03, 0.22], 0.075)
    _configure_visual_static(statics[1], "static O1", [0.35, -0.24], 0.075)


def _proof_overtaking_case(world):
    ships = _dynamic_visual_ships(world)
    return [
        _proof_record(world, ships[0], ships[1], "Rule13/16 overtaking keep-clear", "overtaking_give_way"),
        _proof_record(world, ships[1], ships[0], "Rule13/17 overtaken stand-on", "overtaken_stand_on"),
    ]


def _setup_dense_mixed_case(world):
    ships = _dynamic_visual_ships(world)
    statics = _static_visual_obstacles(world)
    _configure_visual_ship(ships[0], "ship M0", [-0.24, 0.08], 0.0, 0.0018)
    _configure_visual_ship(ships[1], "ship M1", [0.24, 0.06], np.pi, 0.0018)
    _configure_visual_ship(ships[2], "ship M2", [0.00, -0.27], np.pi / 2.0, 0.0015)
    _configure_visual_ship(ships[3], "ship M3", [-0.42, 0.08], 0.0, 0.0030)
    _configure_visual_static(statics[0], "static M0", [-0.10, 0.36], 0.080)
    _configure_visual_static(statics[1], "static M1", [0.26, -0.16], 0.085)


def _proof_dense_mixed_case(world):
    ships = _dynamic_visual_ships(world)
    return [
        _proof_record(world, ships[0], ships[1], "Rule14 in dense traffic", "head_on"),
        _proof_record(world, ships[0], ships[2], "Rule15/16 dense crossing", "crossing_give_way"),
        _proof_record(world, ships[3], ships[0], "Rule13 dense overtaking", "overtaking_give_way"),
    ]


def _setup_base_rules_case(world):
    ships = _dynamic_visual_ships(world)
    statics = _static_visual_obstacles(world)
    _configure_visual_ship(ships[0], "ship B0", [-0.78, -0.55], 0.15, 0.00075)
    _configure_visual_ship(ships[1], "ship B1", [-0.72, 0.55], -0.15, 0.00075)
    _configure_visual_ship(ships[2], "ship B2", [0.78, -0.55], np.pi - 0.15, 0.00075)
    _configure_visual_ship(ships[3], "ship B3", [0.72, 0.55], -np.pi + 0.15, 0.00075)
    _configure_visual_static(statics[0], "static B0", [-0.05, 0.0], 0.075)
    _configure_visual_static(statics[1], "static B1", [0.28, 0.28], 0.075)


def _proof_base_rules_case(world):
    ships = _dynamic_visual_ships(world)
    return [
        _proof_record(world, ships[0], ships[1], "Rule5/6/7 safe-passage baseline", "safe_passage"),
        _proof_record(world, ships[2], ships[3], "Rule5/6/7 reciprocal baseline", "safe_passage"),
    ]


def _simulate_complex_rule_visual_cases(steps):
    return [
        _simulate_complex_case("R5/R6/R7 Baseline", _setup_base_rules_case, _proof_base_rules_case, steps),
        _simulate_complex_case("R14 Head-on", _setup_head_on_case, _proof_head_on_case, steps),
        _simulate_complex_case("R15/R16/R17 Crossing", _setup_crossing_case, _proof_crossing_case, steps),
        _simulate_complex_case("R13/R16/R17 Overtaking", _setup_overtaking_case, _proof_overtaking_case, steps),
        _simulate_complex_case("Mixed Dense Traffic", _setup_dense_mixed_case, _proof_dense_mixed_case, steps),
    ]


def _assert_rule_coverage(cases):
    required_rules = {
        "Rule 5",
        "Rule 6",
        "Rule 7",
        "Rule 8",
        "Rule 13",
        "Rule 14",
        "Rule 15",
        "Rule 16",
        "Rule 17",
    }
    covered_rules = set()
    for case in cases:
        for record in case["proof_records"]:
            for rule in record["rules"]:
                covered_rules.add(str(rule).split(":", 1)[0])
    missing = sorted(required_rules - covered_rules)
    require(not missing, "Visualization proof is missing rules: %s" % ", ".join(missing))
    return tuple(sorted(covered_rules))


def _case_bounds(case):
    positions = []
    for trajectory in case["trajectories"].values():
        positions.extend(trajectory)
    data = np.asarray(positions, dtype=float)
    min_xy = data.min(axis=0)
    max_xy = data.max(axis=0)
    span = np.maximum(max_xy - min_xy, 0.35)
    margin = np.maximum(span * 0.15, 0.08)
    return min_xy - margin, max_xy + margin


def _draw_visual_case(ax, case, frame_index):
    from matplotlib.patches import Circle

    styles = {
        "agent": {"color": "#2563eb", "marker": "o"},
        "target": {"color": "#dc2626", "marker": "s"},
        "obstacle": {"color": "#16a34a", "marker": "^"},
        "static": {"color": "#525252", "marker": "X"},
    }
    ax.clear()
    frame_index = min(frame_index, len(case["notes"]) - 1)
    for name, trajectory in case["trajectories"].items():
        points = np.asarray(trajectory[: frame_index + 1], dtype=float)
        meta = case["entities"][name]
        style = styles[meta["kind"]]
        color = meta.get("color", style["color"])
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.8, alpha=0.75)
        ax.scatter(points[-1, 0], points[-1, 1], color=color, marker=style["marker"], s=42, zorder=3)
        ax.text(points[-1, 0], points[-1, 1], " " + name, color=color, fontsize=7)
        ax.add_patch(
            Circle(
                points[-1],
                meta["size"],
                fill=False,
                color=color,
                linestyle="--" if meta["kind"] == "static" else "-",
                linewidth=1.0,
                alpha=0.75,
            )
        )
        heading = case["headings"][name][frame_index]
        if heading is not None and meta["kind"] != "static":
            arrow_len = max(meta["size"] * 2.4, 0.04)
            ax.arrow(
                points[-1, 0],
                points[-1, 1],
                np.cos(heading) * arrow_len,
                np.sin(heading) * arrow_len,
                color=color,
                head_width=0.018,
                length_includes_head=True,
                alpha=0.8,
            )

    lower, upper = case["bounds"]
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    title_lines = [case["title"]]
    if case.get("proof_summary"):
        title_lines.append(case["proof_summary"])
    title_lines.append("step=%03d %s" % (frame_index, case["notes"][frame_index]))
    ax.set_title("\n".join(title_lines), fontsize=7.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _make_case_grid(plt, case_count):
    cols = 2 if case_count > 3 else case_count
    rows = int(np.ceil(case_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.8 * rows), constrained_layout=True)
    axes = np.asarray(axes, dtype=object).reshape(-1)
    for ax in axes[case_count:]:
        ax.axis("off")
    return fig, axes[:case_count]


def _numbered_fallback_path(path, index):
    return path.with_name("%s_%02d%s" % (path.stem, index, path.suffix))


def _open_writable_csv(path):
    for index in range(100):
        candidate = path if index == 0 else _numbered_fallback_path(path, index)
        try:
            return candidate, candidate.open("w", newline="", encoding="utf-8")
        except PermissionError:
            continue
    raise PermissionError("No writable CSV output path was available near %s" % path)


def _write_visualization_csv(cases, csv_path):
    actual_path, handle = _open_writable_csv(csv_path)
    with handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "step", "entity", "x", "y", "heading", "note"])
        for case in cases:
            max_steps = len(case["notes"])
            for step in range(max_steps):
                for name, trajectory in case["trajectories"].items():
                    pos = trajectory[step]
                    heading = case["headings"][name][step]
                    writer.writerow([case["title"], step, name, pos[0], pos[1], heading, case["notes"][step]])
    return actual_path


def _write_rule_proof_csv(cases, csv_path):
    actual_path, handle = _open_writable_csv(csv_path)
    with handle:
        writer = csv.writer(handle)
        writer.writerow([
            "case",
            "label",
            "ownship",
            "target",
            "encounter",
            "risk",
            "give_way",
            "stand_on",
            "rules",
            "required_actions",
            "recommended_actions",
        ])
        for case in cases:
            for record in case["proof_records"]:
                writer.writerow([
                    case["title"],
                    record["label"],
                    record["ownship"],
                    record["target"],
                    record["encounter"],
                    record["risk"],
                    record["give_way"],
                    record["stand_on"],
                    " | ".join(record["rules"]),
                    " | ".join(record["required_actions"]),
                    " | ".join(record["recommended_actions"]),
                ])
    return actual_path


def run_visualization(output_dir=None, steps=80, save_gif=True):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError as exc:
        raise RuntimeError("matplotlib and pillow are required for --visualize.") from exc

    np.random.seed(7)
    output_path = Path(output_dir) if output_dir is not None else colregs_visualization_dir()
    output_path.mkdir(parents=True, exist_ok=True)

    cases = _simulate_complex_rule_visual_cases(steps)
    covered_rules = _assert_rule_coverage(cases)
    for case in cases:
        case["bounds"] = _case_bounds(case)

    summary_png = output_path / "colregs_avoidance_summary.png"
    fig, axes = _make_case_grid(plt, len(cases))
    for ax, case in zip(axes, cases):
        _draw_visual_case(ax, case, len(case["notes"]) - 1)
    fig.savefig(summary_png, dpi=180)
    plt.close(fig)

    csv_path = output_path / "colregs_avoidance_trace.csv"
    csv_path = _write_visualization_csv(cases, csv_path)
    proof_csv_path = output_path / "colregs_rule_proof.csv"
    proof_csv_path = _write_rule_proof_csv(cases, proof_csv_path)

    gif_path = output_path / "colregs_avoidance_process.gif"
    if save_gif:
        fig, axes = _make_case_grid(plt, len(cases))

        def update(frame_index):
            for ax, case in zip(axes, cases):
                _draw_visual_case(ax, case, frame_index)
            return []

        animation = FuncAnimation(fig, update, frames=steps + 1, interval=140, blit=False)
        animation.save(gif_path, writer=PillowWriter(fps=8))
        plt.close(fig)
    else:
        gif_path = None

    print("VISUAL summary_png=%s" % summary_png)
    if gif_path is not None:
        print("VISUAL process_gif=%s" % gif_path)
    print("VISUAL trace_csv=%s" % csv_path)
    print("VISUAL proof_csv=%s" % proof_csv_path)
    print("VISUAL covered_rules=%s" % ", ".join(covered_rules))
    return summary_png, gif_path, csv_path, proof_csv_path


def run_checks() -> None:
    check_rule_classification()
    check_environment_colregs()
    check_colregs_obstacle_control()
    check_dynamic_obstacle_colregs_pair()
    check_static_geometric_avoidance()
    check_smooth_target_escape()
    print("ALL COLREGS CHECKS PASSED")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Check and visualize COLREGs / avoidance behavior.")
    parser.add_argument("--visualize", action="store_true", help="Generate trajectory PNG/GIF/CSV outputs.")
    parser.add_argument("--visualize-only", action="store_true", help="Skip assertions and only generate visualization.")
    parser.add_argument("--visual-steps", type=int, default=80, help="Number of simulated visualization steps.")
    parser.add_argument("--visual-output-dir", type=Path, default=None, help="Directory for visualization outputs.")
    parser.add_argument("--no-gif", action="store_true", help="Only save summary PNG and CSV.")
    args = parser.parse_args(argv)

    if not args.visualize_only:
        run_checks()
    if args.visualize or args.visualize_only:
        run_visualization(
            output_dir=args.visual_output_dir,
            steps=args.visual_steps,
            save_gif=not args.no_gif,
        )


if __name__ == "__main__":
    main()
