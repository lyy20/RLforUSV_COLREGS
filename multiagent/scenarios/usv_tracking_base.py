import numpy as np

from A_Star import calculate_direction_angle
from dynamics import world_to_body_vector
from multiagent.core import World, Agent, Landmark, Obstacle
from multiagent.scenario import BaseScenario
from safety import (
    COLREGsActionSetConfig,
    COLREGsActionSetFilter,
    COLREGsConfig,
    COLREGsEngine,
    EncounterType,
)
from tracking.target_pf import Target
from utilities.utilities import random_levy
from utilities.observation_schema import (
    BASE_INDEX,
    BASE_OBS_DIM,
    DYNAMIC_OBS_FEATURE_DIM,
    STATIC_OBS_FEATURE_DIM,
    observation_dim,
)
from utilities.scenario_config import validate_environment_config

done_state = False

# 奖励项的兼容默认值。配置文件可以覆盖这些值，但缺少某个键时仍使用这里的默认值。
DEFAULT_REWARD_CONFIG = {
    'success': 100.0,
    'collision': 120.0,
    'out_of_bounds': 100.0,
    'timeout': 25.0,
    'progress_weight': 12.0,
    'range_weight': 0.5,
    'safety_weight': 20.0,
    'boundary_weight': 8.0,
    'action_smooth_weight': 0.02,
    'yaw_rate_weight': 0.05,
    'rudder_smooth_weight': 0.10,
    'intervention_active_penalty': 0.05,
    'intervention_magnitude_weight': 0.25,
    'time_penalty': 0.05,
    'safe_clearance': 0.15,
    'boundary_safe_distance': 0.25,
    # 可选 COLREGs 奖励引导，默认关闭以保持旧实验 reward 语义不变。
    'rule_reward_enabled': False,
    'rule_action_alignment_weight': 0.0,
    'rule_cpa_weight': 0.0,
    'rule_obligation_penalty': 0.0,
}

class Scenario(BaseScenario):
    @staticmethod
    def _resolve_reward_config(reward_config):
        """合并配置文件奖励参数，并校验为有限浮点数。"""
        resolved = dict(DEFAULT_REWARD_CONFIG)
        if reward_config is None:
            return resolved
        for key in resolved:
            if key not in reward_config:
                continue
            value = float(reward_config[key])
            if not np.isfinite(value):
                raise ValueError("Reward parameter '{}' must be finite.".format(key))
            resolved[key] = value
        resolved['rule_reward_enabled'] = bool(reward_config.get('rule_reward_enabled', False))
        return resolved

    @staticmethod
    def _distance(pos_a, pos_b):
        return np.linalg.norm(np.asarray(pos_a) - np.asarray(pos_b))

    @staticmethod
    def _set_entity_colors(world):
        for agent in world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])
        for i, landmark in enumerate(world.landmarks):
            if i < world.num_landmarks:
                landmark.color = np.array([0.25, 0.25, 0.25])
            else:
                landmark.color = np.array([0.55, 0.0, 0.0])
        for i, obstacle in enumerate(world.obstacles):
            if i < world.num_obstacles:
                obstacle.color = np.array([0.0, 1.0, 0.0])
            else:
                obstacle.color = np.array([0.0, 1.0, 1.0])
        for obstacle in getattr(world, 'static_obstacles', []):
            obstacle.color = np.array([0.85, 0.45, 0.10])

    @staticmethod
    def _reset_agent_state(agent, world):
        # 动作链在 reset 时必须有确定且有限的初值。观测中会读取上一个
        # 平滑动作；如果这里保留 core.py 中的 None，np.asarray(None)
        # 会生成 NaN，并在第一步更新前污染 SAC 的网络输入。
        action_dim = int(
            getattr(world, 'agent_action_dim', getattr(world, 'dim_p', 2))
        )
        zero_action = np.zeros(max(action_dim, 1), dtype=np.float64)
        spawn_limit = max(getattr(world, 'map_half_size', 2.0) * 0.85, 0.5)
        agent.state.p_pos = np.random.uniform(-spawn_limit, spawn_limit, world.dim_p)
        agent.state.p_vel = np.zeros(world.dim_p)
        agent.state.p_vel_old = np.zeros(world.dim_p)
        agent.state.c = np.zeros(world.dim_c)
        agent.state.a_vel = 0.
        agent.state.p_pos_origin = agent.state.p_pos.copy()
        agent.state.usv_3dof = None
        # 统一初始化 raw -> constrained -> rule -> smoothed -> applied 的
        # 五个动作层级；这些字段同时被 info、历史观测和 replay 审计使用。
        agent.action_raw_u = zero_action.copy()
        agent.action_constrained_u = zero_action.copy()
        agent.action_rule_u = zero_action.copy()
        agent.action_smoothed_u = zero_action.copy()
        agent.action_executed_u = zero_action.copy()
        agent.action_clip_correction_norm = 0.0
        agent.action_rule_correction_norm = 0.0
        agent.action_smooth_correction_norm = 0.0
        agent.action_total_correction_norm = 0.0
        agent.action_filter_decision = None
        agent.action_filter_active = False
        agent.action_filter_mode = 'reset'
        # 每个回合重新开始时清空持久化规则义务，避免把上一回合的
        # 会遇类型、目标身份或退出计数带入新场景。
        agent.colregs_rule_state = None
        agent.colregs_rule_state_steps = 0
        agent.colregs_rule_state_exit_steps = 0
        # Action.u 是环境动作接口，也初始化为同一有限向量，避免某些
        # reset/观测回调在首次 step 前访问到 None。
        if hasattr(agent, 'action'):
            agent.action.u = zero_action.copy()
        agent.has_captured = False
        agent.consecutive_hold = 0
        agent.reward_prev_target_distance = None
        agent.reward_prev_action = None
        agent.reward_prev_rudder = 0.0
        agent.reward_last_components = {}
        agent.reward_last_metrics = {}
        agent.reward_done_reason = 'none'

    @staticmethod
    def _inside_map(pos, world, radius=0.0, margin=0.0):
        half_size = getattr(world, 'map_half_size', 2.0)
        limit = half_size - radius - margin
        pos = np.asarray(pos, dtype=float)
        return bool(np.all(pos >= -limit) and np.all(pos <= limit))

    def _random_point_in_map(self, world, radius=0.0, margin=0.0):
        half_size = getattr(world, 'map_half_size', 2.0)
        limit = max(half_size - radius - margin, 1.0e-6)
        return np.random.uniform(-limit, limit, world.dim_p)

    def _is_clear_of_entities(self, pos, radius, entities, clearance):
        valid_entities = [
            entity for entity in entities
            if entity is not None
            and getattr(entity.state, 'p_pos', None) is not None
        ]
        if not valid_entities:
            return True
        positions = np.asarray(
            [np.asarray(entity.state.p_pos, dtype=float) for entity in valid_entities],
            dtype=float,
        )
        sizes = np.asarray(
            [float(getattr(entity, 'size', 0.0)) for entity in valid_entities],
            dtype=float,
        )
        candidate = np.asarray(pos, dtype=float).reshape(1, -1)
        if (
            candidate.shape[1] != positions.shape[1]
            or not np.all(np.isfinite(candidate))
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(sizes))
        ):
            return False
        distances = np.linalg.norm(positions - candidate, axis=1)
        return bool(np.all(distances > float(radius) + sizes + float(clearance)))

    def _sample_clear_position(self, world, radius, entities, clearance=0.0, attempts=1000, margin=0.03):
        for _ in range(attempts):
            pos = self._random_point_in_map(world, radius=radius, margin=margin)
            if self._is_clear_of_entities(pos, radius, entities, clearance):
                return pos
        raise RuntimeError("Failed to sample a valid scenario position. Reduce obstacle density or clearances.")

    def _sample_agent_target_pair(self, world):
        map_width = getattr(world, 'map_half_size', 2.0) * 2.0
        min_dist = 0.45 * map_width
        max_dist = 0.75 * map_width
        agent = world.agents[0]
        target = world.landmarks[0]
        for _ in range(2000):
            agent_pos = self._random_point_in_map(
                world,
                radius=agent.size,
                margin=self.agent_target_spawn_margin,
            )
            distance = np.random.uniform(min_dist, max_dist)
            heading = np.random.uniform(0.0, 2.0 * np.pi)
            target_pos = agent_pos + self._heading_command(heading, distance)
            if self._inside_map(
                target_pos,
                world,
                radius=target.size,
                margin=self.agent_target_spawn_margin,
            ):
                return agent_pos, target_pos
        return self._sample_fallback_agent_target_pair(world, min_dist)

    def _sample_fallback_agent_target_pair(self, world, min_dist):
        agent = world.agents[0]
        target = world.landmarks[0]
        for _ in range(2000):
            agent_pos = self._random_point_in_map(
                world,
                radius=agent.size,
                margin=self.agent_target_spawn_margin,
            )
            target_pos = self._random_point_in_map(
                world,
                radius=target.size,
                margin=self.agent_target_spawn_margin,
            )
            if self._distance(agent_pos, target_pos) >= min_dist:
                return agent_pos, target_pos
        agent_pos = np.array([-0.65, -0.65]) * getattr(world, 'map_half_size', 2.0)
        target_pos = -agent_pos
        return agent_pos, target_pos

    @staticmethod
    def _reflect_levy_at_bounds(entity, world=None):
        """Reflect a Levy command against the actual map boundary.

        The old implementation used the fixed coordinates +/-0.8, which was
        inconsistent with the 600 m scene and with custom evaluation maps.
        This helper remains only for the legacy dynamic-obstacle Levy mode;
        target Levy motion now uses the shared constrained heading layer.
        """
        if entity.action.u is None:
            return
        half_size = float(getattr(world, 'map_half_size', 0.8))
        margin = float(getattr(entity, 'size', 0.0)) + float(
            getattr(world, 'boundary_turn_margin', 0.0)
        )
        limit = max(half_size - margin, 0.0)
        if entity.state.p_pos[0] > limit:
            entity.action.u[0] = -abs(entity.action.u[0])
        if entity.state.p_pos[0] < -limit:
            entity.action.u[0] = abs(entity.action.u[0])
        if entity.state.p_pos[1] > limit:
            entity.action.u[1] = -abs(entity.action.u[1])
        if entity.state.p_pos[1] < -limit:
            entity.action.u[1] = abs(entity.action.u[1])

    def _target_speed_intent_update(self, entity, world):
        """Update a low-frequency target speed intention.

        Speed changes are deliberately separated from the actual command. The
        intention changes every few seconds, while
        ``_limit_target_speed_change`` applies the physical acceleration and
        deceleration bounds at every environment step.
        """
        if self.movement == 'static':
            return 0.0
        hold_steps = int(getattr(entity, 'target_speed_hold_steps', 0))
        if hold_steps > 0:
            entity.target_speed_hold_steps = hold_steps - 1
            return float(getattr(entity, 'target_speed_intent', self.target_initial_speed))

        current = float(getattr(entity, 'target_speed_intent', self.target_initial_speed))
        step = self.target_speed_step
        if step > 1.0e-12:
            direction = int(np.random.choice([-1, 0, 1], p=[0.35, 0.30, 0.35]))
            current += direction * step
        else:
            current = float(np.random.uniform(self.target_min_speed, self.target_max_speed))
        current = float(np.clip(current, self.target_min_speed, self.target_max_speed))
        interval = np.random.uniform(
            self.target_speed_change_interval_min,
            self.target_speed_change_interval_max,
        )
        entity.target_speed_hold_steps = max(1, int(round(interval / max(world.dt, 1.0e-9))))
        entity.target_speed_intent = current
        return current

    def _target_nominal_motion(self, entity, world, previous_heading):
        """Return a desired heading and speed before safety-rule correction."""
        speed_intent = self._target_speed_intent_update(entity, world)

        if self.movement == 'linear':
            noise = np.random.uniform(
                -self.target_heading_noise,
                self.target_heading_noise,
            )
            entity.target_nominal_heading = self._wrap_to_pi(
                float(getattr(entity, 'target_nominal_heading', previous_heading)) + noise
            )
            return entity.target_nominal_heading, speed_intent

        if self.movement == 'random':
            previous_vector = np.asarray(
                getattr(
                    entity,
                    'target_random_vector',
                    self._heading_command(previous_heading, 1.0),
                ),
                dtype=float,
            )
            noise_vector = np.random.normal(0.0, 1.0, size=world.dim_p)
            random_vector = (
                self.target_random_correlation * previous_vector
                + np.sqrt(max(1.0 - self.target_random_correlation ** 2, 0.0)) * noise_vector
            )
            if np.linalg.norm(random_vector) <= 1.0e-9:
                random_vector = self._heading_command(previous_heading, 1.0)
            random_vector = random_vector / np.linalg.norm(random_vector)
            entity.target_random_vector = random_vector
            entity.target_nominal_heading = float(np.arctan2(random_vector[1], random_vector[0]))
            return entity.target_nominal_heading, speed_intent

        if self.movement == 'levy':
            hold_steps = int(getattr(entity, 'target_levy_hold_steps', 0))
            if hold_steps <= 0:
                levy_vector = np.asarray(random_levy(self.target_levy_beta), dtype=float).reshape(-1)
                if levy_vector.size < 2 or np.linalg.norm(levy_vector[:2]) <= 1.0e-9:
                    levy_vector = self._heading_command(previous_heading, 1.0)
                levy_vector = levy_vector[:2]
                entity.target_nominal_heading = float(np.arctan2(levy_vector[1], levy_vector[0]))
                entity.target_levy_hold_steps = max(
                    1,
                    int(np.clip(3.0 + np.linalg.norm(levy_vector) * 1.5, 3.0, 8.0)),
                )
            else:
                entity.target_levy_hold_steps = hold_steps - 1
            return float(getattr(entity, 'target_nominal_heading', previous_heading)), speed_intent

        if self.movement == 'escape':
            pursuers = [agent for agent in world.agents if agent.state.p_pos is not None]
            if not pursuers:
                return previous_heading, speed_intent

            pursuer = min(
                pursuers,
                key=lambda agent: self._distance(entity.state.p_pos, agent.state.p_pos),
            )
            dist_min = self._distance(entity.state.p_pos, pursuer.state.p_pos)
            escape_heading = calculate_direction_angle(pursuer.state.p_pos, entity.state.p_pos)
            old_bias = float(getattr(entity, 'target_escape_heading_bias', 0.0))
            noise = np.random.uniform(-self.target_escape_noise, self.target_escape_noise)
            bias = (
                self.target_escape_bias_smoothing * old_bias
                + (1.0 - self.target_escape_bias_smoothing) * noise
            )
            bias = float(np.clip(bias, -self.target_escape_max_bias, self.target_escape_max_bias))
            desired_heading = self._wrap_to_pi(escape_heading + bias)
            pressure_range = max(self.target_escape_pressure_range, 1.0e-9)
            pressure = np.clip((pressure_range - dist_min) / pressure_range, 0.0, 1.0)
            desired_speed = speed_intent + pressure * (self.target_max_speed - speed_intent)
            entity.target_escape_heading_bias = bias
            entity.target_escape_last_distance = float(dist_min)
            entity.target_escape_desired_heading = desired_heading
            entity.target_escape_control_action = 'smooth_escape'
            return desired_heading, float(np.clip(
                desired_speed,
                self.target_min_speed,
                self.target_max_speed,
            ))

        return previous_heading, speed_intent

    def _limit_target_speed_change(self, target, world):
        """Apply target acceleration/deceleration and normal speed bounds."""
        if target.action.u is None:
            return
        desired = np.asarray(target.action.u, dtype=float).reshape(-1)[:world.dim_p]
        desired_speed = float(np.linalg.norm(desired))
        current_speed = float(getattr(target, 'current_landmark_vel', 0.0))
        if current_speed <= 1.0e-12 and target.state.p_vel is not None:
            current_speed = float(np.linalg.norm(np.asarray(target.state.p_vel, dtype=float)))

        max_accel_delta = self.target_max_accel * float(world.dt)
        max_decel_delta = self.target_max_decel * float(world.dt)
        limited_speed = float(np.clip(
            desired_speed,
            current_speed - max_decel_delta,
            current_speed + max_accel_delta,
        ))

        emergency_action = (
            str(getattr(target, 'colregs_control_action', '')).endswith('emergency_action')
            or bool(getattr(target, 'geometric_avoidance_active', False))
        )
        lower_bound = self.target_emergency_stop_speed if emergency_action else self.target_min_speed
        limited_speed = float(np.clip(limited_speed, lower_bound, self.target_max_speed))
        if desired_speed > 1.0e-12:
            heading = float(np.arctan2(desired[1], desired[0]))
            target.action.u = self._heading_command(heading, limited_speed)
        else:
            target.action.u = np.zeros(world.dim_p, dtype=float)
            limited_speed = 0.0
        target.current_landmark_vel = limited_speed

    def _move_target(self, entity, world):
        if not entity.movable:
            return
        previous_heading = self._wrap_to_pi(entity.ra)
        if self.movement == 'static':
            entity.action.u = np.zeros(world.dim_p, dtype=float)
            entity.state.p_vel = np.zeros(world.dim_p, dtype=float)
            entity.current_landmark_vel = 0.0
            entity.target_escape_control_action = 'static'
            return
        nominal_heading, nominal_speed = self._target_nominal_motion(entity, world, previous_heading)
        entity.target_nominal_heading = self._wrap_to_pi(nominal_heading)
        # The target now follows the exact same safety ordering as a dynamic
        # obstacle, while its nominal command still comes from its selected
        # movement mode. This keeps the proven obstacle behavior unchanged.
        self._apply_colregs_to_obstacle(
            entity,
            world,
            entity.target_nominal_heading,
            nominal_speed,
            previous_heading,
            max_turn=self.target_max_turn,
        )
        base_speed = float(np.linalg.norm(entity.action.u)) if entity.action.u is not None else nominal_speed
        self._apply_geometric_avoidance_to_obstacle(
            entity,
            world,
            entity.ra,
            base_speed,
            previous_heading,
        )
        base_speed = float(np.linalg.norm(entity.action.u)) if entity.action.u is not None else nominal_speed
        self._apply_boundary_avoidance(
            entity,
            world,
            entity.ra,
            base_speed,
            previous_heading,
            self.target_max_turn,
            'target_boundary_turn',
        )
        self._limit_target_speed_change(entity, world)

    def _apply_target_motion_safety(self, entity, world, previous_heading):
        if entity.action.u is None:
            return
        base_speed = float(np.linalg.norm(entity.action.u))
        if base_speed <= 1.0e-12:
            return
        self._apply_geometric_avoidance_to_target(entity, world, entity.ra, base_speed, previous_heading)
        base_speed = float(np.linalg.norm(entity.action.u))
        self._apply_boundary_avoidance(
            entity,
            world,
            entity.ra,
            base_speed,
            previous_heading,
            self.target_max_turn,
            'escape_boundary_turn',
        )

    @staticmethod
    def _wrap_to_pi(angle):
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _heading_command(heading, speed):
        return np.array([np.cos(heading) * speed, np.sin(heading) * speed])

    def _planned_entity_velocity(self, entity):
        if getattr(entity, 'action', None) is not None and entity.action.u is not None:
            action_velocity = np.asarray(entity.action.u, dtype=float).reshape(-1)
            if action_velocity.size >= 2:
                return action_velocity[:2]
        if entity.state.p_vel is not None and np.linalg.norm(entity.state.p_vel) > 1.0e-12:
            return np.asarray(entity.state.p_vel, dtype=float)
        if hasattr(entity, 'obstacle_vel') and hasattr(entity, 'ra'):
            return self._heading_command(entity.ra, entity.obstacle_vel)
        if hasattr(entity, 'current_landmark_vel') and hasattr(entity, 'ra'):
            return self._heading_command(entity.ra, entity.current_landmark_vel)
        if hasattr(entity, 'landmark_vel') and hasattr(entity, 'ra'):
            return self._heading_command(entity.ra, entity.landmark_vel)
        return np.zeros(2, dtype=float)

    def _limit_heading_change(self, current_heading, desired_heading, max_turn=None):
        delta = self._wrap_to_pi(desired_heading - current_heading)
        if max_turn is None:
            max_turn = self.colregs_obstacle_max_turn
        delta = np.clip(delta, -max_turn, max_turn)
        return self._wrap_to_pi(current_heading + delta)

    def _boundary_desired_heading(self, entity, world, base_heading, base_speed):
        half_size = getattr(world, 'map_half_size', 2.0)
        margin = getattr(self, 'boundary_turn_margin', 0.28) + entity.size
        pos = np.asarray(entity.state.p_pos, dtype=float)
        step_distance = max(base_speed, 1.0e-9) * getattr(world, 'dt', 30.0)
        predicted = pos + self._heading_command(base_heading, step_distance)
        near_edge = np.any(np.abs(pos) > (half_size - margin))
        predicted_outside = not self._inside_map(predicted, world, radius=entity.size, margin=0.02)
        if not near_edge and not predicted_outside:
            return None

        inward = -pos
        if np.linalg.norm(inward) <= 1.0e-9:
            inward = -self._heading_command(base_heading, 1.0)
        return float(np.arctan2(inward[1], inward[0]))

    def _apply_boundary_avoidance(self, entity, world, base_heading, base_speed, previous_heading, max_turn, label):
        desired_heading = self._boundary_desired_heading(entity, world, base_heading, base_speed)
        if desired_heading is None:
            return False
        entity.ra = self._limit_heading_change(previous_heading, desired_heading, max_turn)
        speed_factor = getattr(self, 'boundary_speed_factor', 0.80)
        entity.action.u = self._heading_command(entity.ra, base_speed * speed_factor)
        entity.geometric_avoidance_active = True
        entity.geometric_avoidance_target = 'map_boundary'
        entity.geometric_avoidance_distance = self._distance(entity.state.p_pos, np.zeros(world.dim_p))
        if hasattr(entity, 'target_escape_control_action'):
            entity.target_escape_control_action = label
        if hasattr(entity, 'colregs_control_action') and entity.colregs_control_action in ('none', 'nominal'):
            entity.colregs_control_action = label
        return True

    def _target_avoidance_entities(self, target, world):
        entities = []
        entities.extend(world.agents)
        entities.extend(world.obstacles[:world.num_obstacles])
        entities.extend(getattr(world, 'static_obstacles', []))
        return [
            entity for entity in entities
            if entity is not target and entity.state.p_pos is not None
        ]

    def _apply_geometric_avoidance_to_target(self, target, world, base_heading, base_speed, previous_heading):
        self._reset_geometric_avoidance(target)
        if base_speed <= 1.0e-12:
            return

        pos = np.asarray(target.state.p_pos, dtype=float)
        forward = self._heading_command(base_heading, 1.0)
        repulsion = np.zeros(world.dim_p, dtype=float)
        closest_entity = None
        closest_distance = np.inf
        closest_trigger_distance = None

        for entity in self._target_avoidance_entities(target, world):
            delta = pos - np.asarray(entity.state.p_pos, dtype=float)
            distance = float(np.linalg.norm(delta))
            if not np.isfinite(distance):
                continue
            buffer = self.geometric_vessel_emergency_buffer if getattr(entity, 'is_vessel', False) else self.geometric_avoidance_buffer
            trigger_distance = target.size + entity.size + buffer
            if distance >= trigger_distance:
                continue
            away_direction = delta / distance if distance > 1.0e-9 else np.array([np.sin(base_heading), -np.cos(base_heading)])
            weight = ((trigger_distance - distance) / max(buffer, 1.0e-9)) ** 2
            repulsion += away_direction * weight
            if distance < closest_distance:
                closest_entity = entity
                closest_distance = distance
                closest_trigger_distance = trigger_distance

        if closest_entity is None:
            return

        desired_vector = forward + self.geometric_avoidance_gain * repulsion
        if np.linalg.norm(desired_vector) <= 1.0e-9:
            desired_vector = forward + np.array([np.sin(base_heading), -np.cos(base_heading)])
        desired_heading = float(np.arctan2(desired_vector[1], desired_vector[0]))
        target.ra = self._limit_heading_change(previous_heading, desired_heading, self.target_max_turn)
        proximity = np.clip(
            (closest_trigger_distance - closest_distance) / max(closest_trigger_distance, 1.0e-9),
            0.0,
            1.0,
        )
        speed_factor = max(
            self.geometric_avoidance_min_speed_factor,
            1.0 - self.geometric_avoidance_slowdown * proximity,
        )
        target.action.u = self._heading_command(target.ra, base_speed * speed_factor)
        target.geometric_avoidance_active = True
        target.geometric_avoidance_target = closest_entity.name
        target.geometric_avoidance_distance = float(closest_distance)
        target.target_escape_control_action = 'escape_avoid_obstacle'

    def _critical_colregs_report(self, obstacle, world):
        targets = self._colregs_targets_for_obstacle(obstacle, world)
        reports = world.evaluate_colregs_for_entity(obstacle, targets)
        obstacle.colregs_last_reports = reports
        risk_reports = [report for report in reports if report.risk_of_collision]
        if not risk_reports:
            obstacle.colregs_last_report = None
            obstacle.colregs_control_action = 'nominal'
            return None

        def report_key(report):
            tcpa = report.tcpa if np.isfinite(report.tcpa) else 1.0e9
            return tcpa, report.dcpa, report.distance

        report = min(risk_reports, key=report_key)
        obstacle.colregs_last_report = report
        return report

    def _colregs_targets_for_obstacle(self, obstacle, world):
        targets = []
        targets.extend(world.agents)
        targets.extend(world.landmarks[:world.num_landmarks])
        targets.extend(world.dynamic_obstacle_vessels())

        unique_targets = []
        seen_ids = set()
        for target in targets:
            if target is obstacle or target.state.p_pos is None:
                continue
            if not getattr(target, 'is_vessel', False):
                continue
            target_id = id(target)
            if target_id in seen_ids:
                continue
            seen_ids.add(target_id)
            unique_targets.append(target)
        return unique_targets

    def _apply_colregs_to_obstacle(
        self,
        obstacle,
        world,
        nominal_heading,
        nominal_speed,
        previous_heading,
        max_turn=None,
    ):
        heading_limit = self.colregs_obstacle_max_turn if max_turn is None else float(max_turn)
        if not getattr(world, 'colregs_enabled', False) or not getattr(obstacle, 'colregs_compliant', False):
            obstacle.ra = self._limit_heading_change(previous_heading, nominal_heading, heading_limit)
            obstacle.action.u = self._heading_command(obstacle.ra, nominal_speed)
            obstacle.colregs_control_action = 'disabled'
            return

        report = self._critical_colregs_report(obstacle, world)
        if report is None:
            obstacle.ra = self._limit_heading_change(previous_heading, nominal_heading, heading_limit)
            obstacle.action.u = self._heading_command(obstacle.ra, nominal_speed)
            return

        desired_heading = previous_heading
        speed_factor = 1.0
        encounter = report.encounter_type
        obstacle_is_give_way = report.give_way_vessel in (obstacle.name, 'both')
        obstacle_is_stand_on = report.stand_on_vessel in (obstacle.name, 'both')

        if encounter == EncounterType.HEAD_ON:
            desired_heading = previous_heading - self.colregs_starboard_turn
            speed_factor = self.colregs_give_way_speed_factor
            obstacle.colregs_control_action = 'head_on_starboard'
        elif obstacle_is_give_way:
            if encounter == EncounterType.CROSSING_GIVE_WAY:
                desired_heading = previous_heading - self.colregs_starboard_turn
                speed_factor = self.colregs_give_way_speed_factor
                obstacle.colregs_control_action = 'crossing_give_way_starboard'
            elif encounter == EncounterType.OVERTAKING_GIVE_WAY:
                desired_heading = previous_heading - 0.5 * self.colregs_starboard_turn
                speed_factor = self.colregs_overtaking_speed_factor
                obstacle.colregs_control_action = 'overtaking_keep_clear'
            else:
                desired_heading = previous_heading - 0.5 * self.colregs_starboard_turn
                speed_factor = self.colregs_give_way_speed_factor
                obstacle.colregs_control_action = 'give_way_keep_clear'
        elif obstacle_is_stand_on:
            desired_heading = previous_heading
            speed_factor = 1.0
            obstacle.colregs_control_action = 'stand_on_keep_course'
            if report.distance < self.colregs_emergency_distance:
                desired_heading = previous_heading - 0.5 * self.colregs_starboard_turn
                speed_factor = self.colregs_emergency_speed_factor
                obstacle.colregs_control_action = 'stand_on_emergency_action'
        else:
            desired_heading = previous_heading - 0.5 * self.colregs_starboard_turn
            speed_factor = self.colregs_give_way_speed_factor
            obstacle.colregs_control_action = 'undefined_risk_action'

        obstacle.ra = self._limit_heading_change(previous_heading, desired_heading, heading_limit)
        command_speed = max(nominal_speed * speed_factor, nominal_speed * self.colregs_min_speed_factor)
        obstacle.action.u = self._heading_command(obstacle.ra, command_speed)

    @staticmethod
    def _reset_geometric_avoidance(entity):
        entity.geometric_avoidance_active = False
        entity.geometric_avoidance_target = None
        entity.geometric_avoidance_distance = None

    def _avoidance_entities_for_obstacle(self, obstacle, world):
        candidates = []
        candidates.extend(world.agents)
        candidates.extend(world.landmarks[:world.num_landmarks])
        candidates.extend(world.obstacles[:world.num_obstacles])
        candidates.extend(getattr(world, 'static_obstacles', []))

        entities = []
        seen_ids = set()
        ignored_roles = {'target_estimation', 'obstacle_estimation'}
        for entity in candidates:
            if entity is obstacle or entity.state.p_pos is None:
                continue
            if getattr(entity, 'vessel_role', '') in ignored_roles:
                continue
            entity_id = id(entity)
            if entity_id in seen_ids:
                continue
            seen_ids.add(entity_id)
            entities.append(entity)
        return entities

    def _apply_geometric_avoidance_to_obstacle(self, obstacle, world, base_heading, base_speed, previous_heading):
        self._reset_geometric_avoidance(obstacle)
        if base_speed <= 1.0e-12:
            return

        pos = np.asarray(obstacle.state.p_pos, dtype=float)
        forward = self._heading_command(base_heading, 1.0)
        own_velocity = self._heading_command(base_heading, base_speed)
        repulsion = np.zeros(world.dim_p, dtype=float)
        closest_entity = None
        closest_distance = np.inf
        closest_trigger_distance = None
        active = False

        for entity in self._avoidance_entities_for_obstacle(obstacle, world):
            other_pos = np.asarray(entity.state.p_pos, dtype=float)
            delta = pos - other_pos
            distance = float(np.linalg.norm(delta))
            if not np.isfinite(distance):
                continue

            combined_radius = float(obstacle.size + entity.size)
            if getattr(entity, 'is_vessel', False) and (
                getattr(entity, 'vessel_role', '') != 'target'
                or getattr(self, 'target_predictive_avoidance_enabled', True)
            ):
                buffer = self.geometric_vessel_emergency_buffer
            else:
                buffer = self.geometric_avoidance_buffer
            trigger_distance = combined_radius + buffer
            current_active = distance < trigger_distance

            predictive_active = False
            predictive_distance = None
            predictive_trigger_distance = None
            predictive_away_direction = None
            if getattr(entity, 'is_vessel', False) and (
                getattr(entity, 'vessel_role', '') != 'target'
                or getattr(self, 'target_predictive_avoidance_enabled', True)
            ):
                other_velocity = self._planned_entity_velocity(entity)
                relative_position = other_pos - pos
                relative_velocity = other_velocity - own_velocity
                rel_speed_sq = float(np.dot(relative_velocity, relative_velocity))
                if rel_speed_sq > 1.0e-12:
                    tcpa = float(-np.dot(relative_position, relative_velocity) / rel_speed_sq)
                    horizon = getattr(self, 'predictive_avoidance_time_horizon', 180.0)
                    if 0.0 <= tcpa <= horizon:
                        closest_relative = relative_position + relative_velocity * tcpa
                        dcpa = float(np.linalg.norm(closest_relative))
                        predictive_trigger_distance = combined_radius + self.predictive_vessel_buffer
                        if dcpa < predictive_trigger_distance:
                            predictive_active = True
                            predictive_distance = dcpa
                            if dcpa > 1.0e-9:
                                predictive_away_direction = -closest_relative / dcpa
                            else:
                                predictive_away_direction = np.array([np.sin(base_heading), -np.cos(base_heading)])

            if not current_active and not predictive_active:
                continue

            active = True
            if current_active:
                if distance <= 1.0e-9:
                    away_direction = np.array([np.sin(base_heading), -np.cos(base_heading)])
                else:
                    away_direction = delta / distance
                weight = ((trigger_distance - distance) / max(buffer, 1.0e-9)) ** 2
                repulsion += away_direction * weight

            if predictive_active:
                predictive_weight = (
                    (predictive_trigger_distance - predictive_distance)
                    / max(self.predictive_vessel_buffer, 1.0e-9)
                ) ** 2
                repulsion += predictive_away_direction * predictive_weight * self.predictive_avoidance_gain

            candidate_distance = predictive_distance if predictive_active else distance
            candidate_trigger_distance = predictive_trigger_distance if predictive_active else trigger_distance
            if candidate_distance < closest_distance:
                closest_entity = entity
                closest_distance = candidate_distance
                closest_trigger_distance = candidate_trigger_distance

        if not active:
            return

        desired_vector = forward + self.geometric_avoidance_gain * repulsion
        if np.linalg.norm(desired_vector) <= 1.0e-9:
            desired_vector = forward + np.array([np.sin(base_heading), -np.cos(base_heading)])
        desired_heading = float(np.arctan2(desired_vector[1], desired_vector[0]))

        avoidance_turn_limit = (
            self.target_max_turn
            if getattr(obstacle, 'vessel_role', '') == 'target'
            else self.geometric_avoidance_max_turn
        )
        obstacle.ra = self._limit_heading_change(
            previous_heading,
            desired_heading,
            avoidance_turn_limit,
        )
        proximity = np.clip(
            (closest_trigger_distance - closest_distance) / max(closest_trigger_distance, 1.0e-9),
            0.0,
            1.0,
        )
        speed_factor = max(
            self.geometric_avoidance_min_speed_factor,
            1.0 - self.geometric_avoidance_slowdown * proximity,
        )
        obstacle.action.u = self._heading_command(obstacle.ra, base_speed * speed_factor)
        obstacle.geometric_avoidance_active = True
        obstacle.geometric_avoidance_target = closest_entity.name if closest_entity is not None else None
        obstacle.geometric_avoidance_distance = float(closest_distance)

    def _move_obstacle(self, obstacle, world):
        if not obstacle.movable:
            return
        previous_heading = self._wrap_to_pi(obstacle.ra)
        if self.obstacle_movement == 'linear':
            nominal_speed = obstacle.obstacle_vel
            nominal_heading = self._wrap_to_pi(
                previous_heading
                + np.random.uniform(
                    -self.dynamic_obstacle_heading_noise,
                    self.dynamic_obstacle_heading_noise,
                )
            )
        elif self.obstacle_movement == 'random':
            nominal_action = np.random.randn(2) / 2.
            nominal_speed = min(np.linalg.norm(nominal_action), max(obstacle.obstacle_vel, 1.0e-9))
            nominal_heading = np.arctan2(nominal_action[1], nominal_action[0])
        elif self.obstacle_movement == 'levy':
            beta = 1.9
            nominal_action = random_levy(beta)
            nominal_speed = min(np.linalg.norm(nominal_action), max(obstacle.obstacle_vel, 1.0e-9))
            nominal_heading = np.arctan2(nominal_action[1], nominal_action[0])
        else:
            nominal_speed = obstacle.obstacle_vel
            nominal_heading = previous_heading

        self._apply_colregs_to_obstacle(obstacle, world, nominal_heading, nominal_speed, previous_heading)
        base_speed = float(np.linalg.norm(obstacle.action.u)) if obstacle.action.u is not None else nominal_speed
        self._apply_geometric_avoidance_to_obstacle(obstacle, world, obstacle.ra, base_speed, previous_heading)
        base_speed = float(np.linalg.norm(obstacle.action.u)) if obstacle.action.u is not None else nominal_speed
        self._apply_boundary_avoidance(
            obstacle,
            world,
            obstacle.ra,
            base_speed,
            previous_heading,
            self.colregs_obstacle_max_turn,
            'boundary_turn',
        )
        self._limit_dynamic_obstacle_speed_change(obstacle, world)
        if self.obstacle_movement == 'levy':
            self._reflect_levy_at_bounds(obstacle, world)

    def _limit_dynamic_obstacle_speed_change(self, obstacle, world):
        if obstacle.action.u is None:
            return
        desired = np.asarray(obstacle.action.u, dtype=float)
        desired_speed = float(np.linalg.norm(desired))
        if desired_speed <= 1.0e-12:
            return
        current_speed = float(np.linalg.norm(np.asarray(obstacle.state.p_vel, dtype=float)))
        max_delta = max(float(self.dynamic_obstacle_max_accel) * float(world.dt), 0.0)
        limited_speed = float(np.clip(desired_speed, current_speed - max_delta, current_speed + max_delta))
        if self.dynamic_obstacle_speed_range is not None:
            min_speed, max_speed = self.dynamic_obstacle_speed_range
            limited_speed = float(np.clip(limited_speed, min_speed, max_speed))
        obstacle.action.u = desired * (limited_speed / desired_speed)

    def _add_nearby_entity(self, entity_inside, entity, agent, ob_range):
        dist = self._distance(entity.state.p_pos, agent.state.p_pos)
        if dist >= ob_range:
            return
        for i_in, entity_in in enumerate(entity_inside):
            if dist < self._distance(entity_in.state.p_pos, agent.state.p_pos):
                entity_inside.insert(i_in, entity)
                return
        entity_inside.append(entity)

    def _dynamic_entity_risk_key(self, entity, agent):
        """W1: 风险优先排序键 (closing, TCPA, DCPA, range)。

        距离近但正在远离的船，优先级低于距离远但正在交会的船。
        返回元组按字典序比较：closing=0 在前，其后 TCPA 小者优先。
        """
        p_rel = np.asarray(entity.state.p_pos, dtype=float) - np.asarray(agent.state.p_pos, dtype=float)
        rng = float(np.linalg.norm(p_rel))
        try:
            v_self = np.asarray(self._entity_world_velocity(agent), dtype=float)
        except Exception:
            v_self = np.zeros(2, dtype=float)
        try:
            v_other = np.asarray(self._entity_world_velocity(entity), dtype=float)
        except Exception:
            v_other = np.zeros(2, dtype=float)
        v_rel = v_other - v_self
        vv = float(np.dot(v_rel, v_rel))
        if vv <= 1.0e-12:
            return (1, 0.0, rng, rng)
        tcpa = -float(np.dot(p_rel, v_rel)) / vv
        if tcpa <= 0.0:
            return (1, 0.0, rng, rng)
        dcpa = float(np.linalg.norm(p_rel + v_rel * tcpa))
        return (0, tcpa, dcpa, rng)

    def _add_nearby_dynamic_entity(self, entity_inside, entity, agent, ob_range):
        """W1: 按风险（TCPA/DCPA）而非距离插入动态船槽位候选列表。"""
        if self._distance(entity.state.p_pos, agent.state.p_pos) >= ob_range:
            return
        key = self._dynamic_entity_risk_key(entity, agent)
        for i_in, entity_in in enumerate(entity_inside):
            if key < self._dynamic_entity_risk_key(entity_in, agent):
                entity_inside.insert(i_in, entity)
                return
        entity_inside.append(entity)

    def _sample_encounter_state(self, agent, target_landmark, obstacle, occupied, world,
                                 agent_nominal_speed=0.001):
        """P0-1：CPA 倒推生成会遇（几何保证必然交会）。

        数学：期望 TCPA = T、DCPA = d。取相对速度 v_rel = v_o - v_s、单位法向 n ⟂ v_rel，
        令 t=0 的相对位置 r = d·n - v_rel·T，则
            TCPA = -(r·v_rel)/|v_rel|^2 = T,   DCPA = |r + v_rel·T| = d。
        故 p_o(0) = p_s + d·n - v_rel·T。
        """
        env_cfg = getattr(world, 'environment_config', {}) or {}
        import math as _math
        p_s = np.asarray(agent.state.p_pos, dtype=float)
        p_t = np.asarray(target_landmark.state.p_pos, dtype=float)
        route = p_t - p_s
        route_len = float(np.linalg.norm(route))
        if route_len < 1.0e-9:
            return None
        psi_s = float(np.arctan2(route[1], route[0]))
        u_s = float(agent_nominal_speed)
        v_s = np.array([np.cos(psi_s), np.sin(psi_s)]) * u_s
        w_ho = float(env_cfg.get('encounter_head_on_weight', 1.0))
        w_cr = float(env_cfg.get('encounter_crossing_weight', 1.0))
        w_ot = float(env_cfg.get('encounter_overtaking_weight', 1.0))
        weights = np.array([max(w_ho, 0.0), max(w_cr, 0.0), max(w_ot, 0.0)], dtype=float)
        if float(weights.sum()) <= 0.0:
            weights = np.ones(3, dtype=float)
        kind = int(np.random.choice(3, p=weights / float(weights.sum())))
        u_min = float(env_cfg.get('dynamic_obstacle_min_speed', 0.001))
        u_max = float(env_cfg.get('dynamic_obstacle_max_speed', 0.002))
        if kind == 0:      # 对遇
            psi_o = psi_s + np.pi + np.deg2rad(np.random.uniform(-15.0, 15.0))
            u_o = np.random.uniform(u_min, u_max)
        elif kind == 1:    # 交叉
            side = 1.0 if np.random.rand() < 0.5 else -1.0
            psi_o = psi_s + side * np.deg2rad(np.random.uniform(45.0, 135.0))
            u_o = np.random.uniform(u_min, u_max)
        else:              # 追越（他船慢于我）
            psi_o = psi_s + np.deg2rad(np.random.uniform(-20.0, 20.0))
            u_o = min(np.random.uniform(u_min, u_max), max(u_s * 0.8, u_min))
        v_o = np.array([np.cos(psi_o), np.sin(psi_o)]) * u_o
        v_rel = v_o - v_s
        vv = float(np.dot(v_rel, v_rel))
        if vv <= 1.0e-16:
            return None
        T = np.random.uniform(float(env_cfg.get('encounter_tcpa_min_s', 60.0)),
                              float(env_cfg.get('encounter_tcpa_max_s', 300.0)))
        d = np.random.uniform(float(env_cfg.get('encounter_dcpa_min_m', 10.0)),
                              float(env_cfg.get('encounter_dcpa_max_m', 60.0))) / 1000.0
        n_hat = np.array([-v_rel[1], v_rel[0]]) / float(np.sqrt(vv))
        if np.random.rand() < 0.5:
            n_hat = -n_hat
        position = p_s + d * n_hat - v_rel * T
        if not self._inside_map(position, world, radius=obstacle.size,
                                margin=float(env_cfg.get('dynamic_obstacle_spawn_margin', 0.08))):
            return None
        if not self._is_clear_of_entities(position, obstacle.size, occupied,
                                          self.vessel_static_clearance):
            return None
        # M2 接受判据自检（构造保证成立，此处做防御性校验）
        r = position - p_s
        tcpa = -float(np.dot(r, v_rel)) / vv
        dcpa = float(np.linalg.norm(r + v_rel * tcpa))
        if tcpa > float(env_cfg.get('encounter_accept_tcpa_max_s', 300.0)) + 1.0e-6:
            return None
        if dcpa * 1000.0 > float(env_cfg.get('encounter_accept_dcpa_max_m', 120.0)) + 1.0e-6:
            return None
        return position, psi_o, u_o

    def _normalized_relative_position(self, entity, agent, world):
        scale = max(float(getattr(world, 'ob_range', 0.8)), 1.0e-9)
        rel_world = np.asarray(entity.state.p_pos, dtype=float) - np.asarray(agent.state.p_pos, dtype=float)
        rel_body = world_to_body_vector(self._agent_3dof_state(agent, world).psi, rel_world)
        return np.clip(rel_body / scale, -1.0, 1.0)

    def _normalized_distance(self, entity, agent, world):
        scale = max(float(getattr(world, 'ob_range', 0.8)), 1.0e-9)
        dist = self._distance(entity.state.p_pos, agent.state.p_pos)
        return float(np.clip(dist / scale, 0.0, 1.0))

    def _normalized_size(self, entity):
        scale = max(getattr(self, 'observation_size_scale', 0.15), 1.0e-9)
        return float(np.clip(float(entity.size) / scale, 0.0, 1.0))

    def _entity_world_velocity(self, entity):
        if entity.state.p_vel is not None and np.linalg.norm(entity.state.p_vel) > 1.0e-12:
            return np.asarray(entity.state.p_vel, dtype=float)
        return self._planned_entity_velocity(entity)

    def _agent_3dof_state(self, agent, world):
        """Return the physical USV state used by both dynamics and observation."""
        state = getattr(agent.state, 'usv_3dof', None)
        if state is not None:
            return state
        try:
            agent_index = world.agents.index(agent)
        except ValueError as exc:
            raise ValueError('Observed agent is not registered in world.agents.') from exc
        return world._ensure_usv_3dof_state(agent, agent_index)

    def _normalized_relative_velocity(self, entity, agent, world):
        scale = max(getattr(self, 'observation_velocity_scale', 0.05), 1.0e-9)
        rel_world = self._entity_world_velocity(entity) - self._entity_world_velocity(agent)
        rel_body = world_to_body_vector(self._agent_3dof_state(agent, world).psi, rel_world)
        return np.clip(rel_body / scale, -1.0, 1.0)

    def _boundary_observation(self, agent, world, psi):
        """Return normalized clearance and a body-frame outward risk vector.

        Four wall risks are activated continuously inside the configured
        warning distance. Their outward normals are summed before being
        rotated with the same ``R(psi).T`` convention as all entity vectors.
        """
        half_size = float(getattr(world, 'map_half_size', 2.0))
        usable_half_size = max(half_size - float(agent.size), 1.0e-9)
        x, y = np.asarray(agent.state.p_pos, dtype=float)
        wall_clearances = np.asarray(
            [
                usable_half_size - x,
                usable_half_size + x,
                usable_half_size - y,
                usable_half_size + y,
            ],
            dtype=float,
        )
        clearance = float(np.min(wall_clearances))
        clearance_normalized = float(np.clip(clearance / usable_half_size, -1.0, 1.0))

        warning_distance = max(float(self.observation_boundary_risk_distance), 1.0e-9)
        wall_risks = np.clip((warning_distance - wall_clearances) / warning_distance, 0.0, 1.0)
        outward_normals = np.asarray(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            dtype=float,
        )
        risk_world = np.sum(wall_risks[:, None] * outward_normals, axis=0)
        risk_norm = float(np.linalg.norm(risk_world))
        if risk_norm > 1.0:
            risk_world /= risk_norm
        risk_body = world_to_body_vector(psi, risk_world)
        return np.array(
            [clearance_normalized, risk_body[0], risk_body[1]],
            dtype=float,
        )

    def _dynamic_observation_slot(self, entity, agent, world):
        rel_pos = self._normalized_relative_position(entity, agent, world)
        rel_vel = self._normalized_relative_velocity(entity, agent, world)
        return np.array([
            rel_pos[0],
            rel_pos[1],
            self._normalized_distance(entity, agent, world),
            self._normalized_size(entity),
            rel_vel[0],
            rel_vel[1],
            1.0,  # type_mask: +1 means an occupied dynamic-vessel slot.
        ], dtype=float)

    def _static_observation_slot(self, entity, agent, world):
        rel_pos = self._normalized_relative_position(entity, agent, world)
        return np.array([
            rel_pos[0],
            rel_pos[1],
            self._normalized_distance(entity, agent, world),
            self._normalized_size(entity),
            -1.0,  # type_mask: -1 means an occupied static-obstacle slot.
        ], dtype=float)
    
    def make_world(
        self,
        num_agents=1,
        num_landmarks=1,
        num_obstacles=5,
        num_static_obstacles=8,
        map_half_size=2.0,
        static_obstacle_min_size=0.04,
        static_obstacle_max_size=0.10,
        ob_range=0.8,
        num_ob=3,
        num_static_ob_slots=1,
        landmark_depth=0.,
        landmark_movable=True,
        obstacle_movable=True,
        landmark_vel=0.0003,
        max_vel=0.0003,
        random_vel=False,
        movement='escape',
        obstacle_movement='linear',
        pre_method='LS',
        target_observation_mode='direct_noisy',
        target_position_noise_std=0.005,
        scenario_profile='DEFAULT',
        dynamic_obstacle_min_speed=None,
        dynamic_obstacle_max_speed=None,
        rew_err_th=0.03,
        rew_dis_th=0.3,
        max_range=2.0,
        max_current_vel=0.0003,
        range_dropping=0.01,
        control_model='usv_3dof',
        reward_config=None,
        environment_config=None,
    ):
        environment_config = validate_environment_config(environment_config)
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.num_agents = num_agents
        world.num_landmarks = num_landmarks
        world.num_obstacles = num_obstacles
        world.num_static_obstacles = num_static_obstacles
        world.map_half_size = map_half_size
        world.ob_range = ob_range
        world.num_ob = num_ob
        world.num_static_ob_slots = num_static_ob_slots
        world.scenario_profile = str(scenario_profile)
        world.environment_config = dict(environment_config)
        world.dt = environment_config['environment_dt']
        world.max_episode_steps = environment_config['max_episode_steps']
        world.timeout_termination_enabled = environment_config['timeout_termination_enabled']
        world.target_observation_mode = target_observation_mode
        world.target_position_noise_std = target_position_noise_std
        world.collaborative = True
        world.control_model = control_model
        world.agent_action_dim = world.usv_3dof_action_dim if control_model == 'usv_3dof' else 1
        world.usv_3dof_normalized_action = control_model == 'usv_3dof'
        world.agent_action_smoothing_alpha = environment_config['action_smoothing_alpha']
        world.scripted_action_is_velocity = environment_config['scripted_action_is_velocity']
        world.motion_execution_safety_buffer = environment_config['motion_execution_safety_buffer']
        world.colregs_persistent_state_enabled = bool(
            environment_config['colregs_persistent_state_enabled']
        )
        world.colregs_persistent_exit_steps = int(
            environment_config['colregs_persistent_exit_steps']
        )
        # 显式规则状态机的滞回阈值，默认值只在持久化开关打开时生效。
        world.colregs_persistent_enter_dcpa = float(
            environment_config['colregs_persistent_enter_dcpa']
        )
        world.colregs_persistent_enter_tcpa = float(
            environment_config['colregs_persistent_enter_tcpa']
        )
        world.colregs_persistent_exit_dcpa = float(
            environment_config['colregs_persistent_exit_dcpa']
        )
        world.colregs_persistent_exit_tcpa = float(
            environment_config['colregs_persistent_exit_tcpa']
        )
        world.colregs_persistent_max_hold_steps = int(
            round(environment_config.get('colregs_persistent_max_hold_steps', 0.0))
        )
        usv_config = world.usv_3dof_model.config
        usv_config.dt = environment_config['usv_3dof_dt']
        usv_config.max_surge_speed = environment_config['usv_max_surge_speed']
        usv_config.max_reverse_speed = environment_config['usv_max_reverse_speed']
        usv_config.max_sway_speed = environment_config['usv_max_sway_speed']
        usv_config.max_thrust_rate = environment_config['usv_max_thrust_rate']
        usv_config.max_rudder_angle = np.deg2rad(environment_config['usv_max_rudder_angle_deg'])
        usv_config.max_rudder_rate = np.deg2rad(environment_config['usv_max_rudder_rate_deg_s'])
        usv_config.max_yaw_rate = np.deg2rad(environment_config['usv_max_yaw_rate_deg_s'])
        world.colregs = COLREGsEngine(COLREGsConfig(
            risk_time_horizon=environment_config['colregs_risk_time_horizon'],
            min_cpa_distance=environment_config['colregs_min_cpa_distance'],
            safety_buffer=environment_config['colregs_safety_buffer'],
        ))
        world.colregs_risk_time_horizon = float(
            environment_config['colregs_risk_time_horizon']
        )
        world.agent_colregs_action_filter = COLREGsActionSetFilter(COLREGsActionSetConfig(
            emergency_distance=environment_config['colregs_emergency_distance'],
        ))
        world.agent_nominal_speed = 0.001
        world.heading_rate_gain = 0.3
        world.astar_heading_rate_gain = 1.0
        world.ocean_current_enabled = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = environment_config['agent_radius']
            agent.max_a_speed = 3.1415
            agent.is_vessel = True
            agent.colregs_compliant = False
            agent.vessel_role = 'ownship'
            # agent has_captured target
            agent.has_captured = False
            # Time ( agent has_captured target )
            agent.consecutive_hold = 0
        # add landmarks 预测地标需要大于本体
        world.landmarks = [Landmark() for i in range(num_landmarks*2)]
        for i, landmark in enumerate(world.landmarks):
            if i < num_landmarks:
                landmark.name = 'landmark %d' % i
                landmark.collide = False
                # Static mode is authoritative even for older configs that leave
                # landmark_movable enabled.
                landmark.movable = bool(landmark_movable and movement != 'static')
                landmark.is_vessel = True
                landmark.colregs_compliant = bool(
                    environment_config['target_colregs_enabled']
                )
                landmark.vessel_role = 'target'
                landmark.size = environment_config['target_radius']
            else:
                landmark.name = 'landmark_estimation %d' % (i-num_landmarks)
                landmark.collide = False
                landmark.movable = False
                landmark.size = 0.002
                landmark.is_vessel = False
                landmark.colregs_compliant = False
                landmark.vessel_role = 'target_estimation'
        # 创建障碍物
        # 翻倍障碍物，为了估计观测,且障碍物预测体需要大于本体
        world.obstacles = [Obstacle() for i in range(num_obstacles*2)]
        for i, obstacle in enumerate(world.obstacles):
            if i < num_obstacles:
                obstacle.name = 'obstacle %d' % i
                obstacle.collide = False
                obstacle.movable = obstacle_movable
                obstacle.is_vessel = True
                obstacle.colregs_compliant = True
                obstacle.vessel_role = 'dynamic_obstacle'
            else:
                obstacle.name = 'obstacle_estimation %d' % (i-num_obstacles)
                obstacle.collide = False
                obstacle.movable = False
                obstacle.is_vessel = False
                obstacle.colregs_compliant = False
                obstacle.vessel_role = 'obstacle_estimation'

        world.static_obstacles = [Obstacle() for i in range(num_static_obstacles)]
        for i, obstacle in enumerate(world.static_obstacles):
            obstacle.name = 'static_obstacle %d' % i
            obstacle.collide = False
            obstacle.movable = False
            obstacle.is_vessel = False
            obstacle.colregs_compliant = False
            obstacle.vessel_role = 'static_obstacle'

        # make initial conditions
        world.cov = np.ones(num_landmarks)/30.
        world.error = np.ones(num_landmarks)
        
        #make initial world current 
        self.max_vel_ocean_current = max_current_vel
        world.vel_ocean_current = 0 #initial random strength
        world.angle_ocean_current = 0 #initial landmark direction
        # world.vel_ocean_current = 0.05
        # world.angle_ocean_current = np.pi/2.*3.
        
        self.landmark_vel = landmark_vel
        # print('test',landmark_vel)
        
        #benchmark variables  基准参数   增加与障碍物的碰撞
        self.pre_error = np.ones(num_landmarks)
        self.agent_outofworld = 0
        self.landmark_collision = 0
        self.agent_collision = 0
        self.obstacle_collision = 0
        self.pre_error = np.ones(num_obstacles)
        #Scenario initial conditions
        self.max_landmark_depth = landmark_depth
        #set random target depth
        self.landmark_depth = 0
        # if self.landmark_depth<15.:
        #     self.landmark_depth = 15.
        self.ra = 0 #initial landmark direction
        self.movement = movement
        self.obstacle_movement = obstacle_movement
        self.dynamic_obstacle_min_size = environment_config['dynamic_obstacle_min_size']
        self.dynamic_obstacle_max_size = environment_config['dynamic_obstacle_max_size']
        self.dynamic_obstacle_heading_noise = np.deg2rad(
            environment_config['dynamic_obstacle_heading_noise_deg_s']
        ) * world.dt
        self.dynamic_obstacle_max_accel = (
            environment_config['dynamic_obstacle_max_accel_mps2'] / world.usv_3dof_m_per_km
        )
        self.colregs_obstacle_max_turn = np.deg2rad(
            environment_config['dynamic_obstacle_max_turn_rate_deg_s']
        ) * world.dt
        self.colregs_starboard_turn = 0.45
        self.colregs_give_way_speed_factor = 0.75
        self.colregs_overtaking_speed_factor = 0.85
        self.colregs_emergency_speed_factor = 0.5
        self.colregs_min_speed_factor = 0.35
        self.colregs_emergency_distance = environment_config['colregs_emergency_distance']
        self.geometric_avoidance_buffer = environment_config['geometric_avoidance_buffer']
        self.geometric_vessel_emergency_buffer = environment_config['geometric_vessel_emergency_buffer']
        self.geometric_avoidance_gain = 2.2
        self.geometric_avoidance_max_turn = self.colregs_obstacle_max_turn
        self.geometric_avoidance_slowdown = 0.6
        self.geometric_avoidance_min_speed_factor = 0.4
        self.predictive_avoidance_time_horizon = environment_config['predictive_avoidance_time_horizon']
        self.predictive_vessel_buffer = environment_config['predictive_vessel_buffer']
        self.predictive_avoidance_gain = 1.4
        self.boundary_turn_margin = environment_config['boundary_turn_margin']
        self.boundary_speed_factor = 0.80
        self.static_obstacle_min_size = static_obstacle_min_size
        self.static_obstacle_max_size = static_obstacle_max_size
        # A failed entity placement restarts the complete layout below. This
        # prevents a rare invalid partial layout from terminating a worker.
        self.scene_generation_max_retries = 50
        # 自定义测试允许在小地图中增加障碍物数量。CUSTOM_EVALUATION 仅改变
        # 场景生成器的选择，不改变训练配置或网络输入槽数量。
        is_custom_evaluation = str(scenario_profile).upper().startswith('CUSTOM_EVALUATION')
        self.is_small_scene_layout = bool(
            is_custom_evaluation
            or (
                float(map_half_size) <= 1.0 + 1.0e-9
                and int(num_obstacles) <= 2
                and int(num_static_obstacles) <= 3
            )
        )
        if self.is_small_scene_layout:
            self.static_obstacle_clearance = environment_config['static_obstacle_clearance']
            self.vessel_static_clearance = environment_config['vessel_static_clearance']
            self.vessel_vessel_spawn_clearance = environment_config['vessel_vessel_spawn_clearance']
            self.static_obstacle_spawn_margin = environment_config['static_obstacle_spawn_margin']
            self.dynamic_obstacle_spawn_margin = environment_config['dynamic_obstacle_spawn_margin']
        else:
            # Preserve the original larger-scene spacing requirements.
            self.static_obstacle_clearance = 0.22
            self.vessel_static_clearance = 0.28
            self.vessel_vessel_spawn_clearance = 0.38
            self.static_obstacle_spawn_margin = 0.04
            self.dynamic_obstacle_spawn_margin = 0.08
        self.target_min_speed = max(
            float(environment_config['target_min_speed_mps']) / world.usv_3dof_m_per_km,
            0.0,
        )
        self.target_max_speed = max(
            float(environment_config['target_max_speed_mps']) / world.usv_3dof_m_per_km,
            self.target_min_speed,
        )
        self.target_initial_speed = float(np.clip(
            float(environment_config['target_initial_speed_mps']) / world.usv_3dof_m_per_km,
            self.target_min_speed,
            self.target_max_speed,
        ))
        self.target_max_accel = max(
            float(environment_config['target_max_accel_mps2']) / world.usv_3dof_m_per_km,
            0.0,
        )
        self.target_max_decel = max(
            float(environment_config['target_max_decel_mps2']) / world.usv_3dof_m_per_km,
            0.0,
        )
        self.target_max_turn = np.deg2rad(
            float(environment_config['target_max_turn_rate_deg_s'])
        ) * world.dt
        self.target_heading_noise = np.deg2rad(
            float(environment_config['target_heading_noise_deg_s'])
        ) * world.dt
        self.target_speed_step = max(
            float(environment_config['target_speed_step_mps']) / world.usv_3dof_m_per_km,
            0.0,
        )
        self.target_speed_change_interval_min = float(
            environment_config['target_speed_change_interval_min_s']
        )
        self.target_speed_change_interval_max = float(
            environment_config['target_speed_change_interval_max_s']
        )
        self.target_emergency_stop_speed = max(
            float(environment_config['target_emergency_stop_speed_mps'])
            / world.usv_3dof_m_per_km,
            0.0,
        )
        self.target_random_correlation = 0.90
        self.target_levy_beta = 1.9
        self.target_predictive_avoidance_enabled = bool(
            environment_config['target_predictive_avoidance_enabled']
        )
        self.target_speed_smoothing = 0.15
        self.target_min_speed_factor = 0.55
        self.target_escape_pressure_range = 0.7
        self.target_escape_noise = 0.12
        self.target_escape_bias_smoothing = 0.9
        self.target_escape_max_bias = 0.25
        self.pre_method = pre_method
        self.target_observation_mode = str(target_observation_mode).lower()
        self.target_position_noise_std = max(float(target_position_noise_std), 0.0)
        configured_speed_bounds = (
            dynamic_obstacle_min_speed is not None
            or dynamic_obstacle_max_speed is not None
        )
        if configured_speed_bounds:
            min_speed = 0.0 if dynamic_obstacle_min_speed is None else float(dynamic_obstacle_min_speed)
            max_speed = min_speed if dynamic_obstacle_max_speed is None else float(dynamic_obstacle_max_speed)
            if min_speed < 0.0 or max_speed < min_speed:
                raise ValueError(
                    'Dynamic obstacle speed range must satisfy 0 <= min_speed <= max_speed.'
                )
            self.dynamic_obstacle_speed_range = (min_speed, max_speed)
        else:
            self.dynamic_obstacle_speed_range = None
        self.rew_err_th = rew_err_th
        self.rew_dis_th = rew_dis_th
        self.set_max_range = max_range
        # 奖励权重统一从配置文件传入；下面的属性名保持不变，兼容现有 reward() 逻辑。
        self.reward_config = self._resolve_reward_config(reward_config)
        self.reward_success = self.reward_config['success']
        self.reward_collision = self.reward_config['collision']
        self.reward_out_of_bounds = self.reward_config['out_of_bounds']
        self.reward_timeout = self.reward_config['timeout']
        self.reward_progress_weight = self.reward_config['progress_weight']
        self.reward_range_weight = self.reward_config['range_weight']
        self.reward_safety_weight = self.reward_config['safety_weight']
        self.reward_boundary_weight = self.reward_config['boundary_weight']
        self.reward_action_smooth_weight = self.reward_config['action_smooth_weight']
        self.reward_yaw_rate_weight = self.reward_config['yaw_rate_weight']
        self.reward_rudder_smooth_weight = self.reward_config['rudder_smooth_weight']
        self.reward_intervention_active_penalty = self.reward_config['intervention_active_penalty']
        self.reward_intervention_magnitude_weight = self.reward_config['intervention_magnitude_weight']
        self.reward_time_penalty = self.reward_config['time_penalty']
        self.reward_safe_clearance = self.reward_config['safe_clearance']
        self.reward_boundary_safe_distance = self.reward_config['boundary_safe_distance']
        self.rule_reward_enabled = bool(self.reward_config['rule_reward_enabled'])
        self.rule_action_alignment_weight = float(self.reward_config['rule_action_alignment_weight'])
        self.rule_cpa_weight = float(self.reward_config['rule_cpa_weight'])
        self.rule_obligation_penalty = float(self.reward_config['rule_obligation_penalty'])
        self.world_colregs_risk_time_horizon = float(
            getattr(world, 'colregs_risk_time_horizon', 240.0)
        )
        # Observation normalization scales for obstacle slots.
        self.observation_size_scale = max(
            environment_config['dynamic_obstacle_max_size'],
            static_obstacle_max_size,
            environment_config['agent_radius'],
            environment_config['target_radius'],
        )
        self.observation_velocity_scale = environment_config['observation_velocity_scale']
        self.observation_boundary_risk_distance = environment_config[
            'observation_boundary_risk_distance'
        ]
        self.agent_target_spawn_margin = environment_config['agent_target_spawn_margin']
        # Layout selection follows the physical scene contract instead of a
        # profile label. Curriculum/ablation configs such as ``1_STAGE_1`` and
        # ``2_STAGE_1`` use the same small-scene geometry but intentionally
        # carry different profile names.
        self.use_scaled_small_layout = self.is_small_scene_layout
        #random variables for target
        self.max_vel = max_vel
        self.random_vel = random_vel
        self.reset_world(world)
        #
        world.damping = landmark_vel/5.
        
        self.range_dropping = range_dropping
        
        return world

    def reset_world(self, world):
        """Regenerate a complete valid scene, retrying only failed layouts."""
        max_retries = max(int(getattr(self, 'scene_generation_max_retries', 50)), 1)
        last_error = None
        for layout_attempt in range(1, max_retries + 1):
            try:
                self._reset_world_once(world)
            except RuntimeError as error:
                last_error = error
                continue

            world.scene_generation_attempts = layout_attempt
            world.scene_generation_retries = layout_attempt - 1
            return

        raise RuntimeError(
            'Failed to generate a valid complete scenario after {} attempts. '
            'Last layout error: {}'.format(max_retries, last_error)
        )

    def _reset_world_once(self, world):
        if getattr(self, 'use_scaled_small_layout', False):
            self._reset_scaled_small_layout(world)
            return
        world.vel_ocean_current = np.random.rand(1).item(0)*self.max_vel_ocean_current #initial random strength
        world.angle_ocean_current = (np.random.rand(1)*np.pi*2.).item(0) #initial landmark direction
        self._set_entity_colors(world)

        # set random initial states
        for agent in world.agents:
            self._reset_agent_state(agent, world)
        spawn_agent = world.agents[-1]
        for i, landmark in enumerate(world.landmarks):
            if i < world.num_landmarks:
                dis = np.random.uniform(1.0 , 1.2)
                # dis = np.random.uniform(0.1, 1.) #改成这个避免碰撞次数太高
                rad = np.random.uniform(0, np.pi*2)
                landmark.state.p_pos = spawn_agent.state.p_pos + np.array([np.cos(rad),np.sin(rad)])*dis
                landmark.state.p_vel = np.zeros(world.dim_p)
            # if i < world.num_landmarks:
            #     dis = np.random.uniform(0.9, 1.8)
            #     rad = np.random.uniform(0, np.pi*2)
            #     landmark.state.p_pos = world.agents[0].state.p_pos + np.array([np.cos(rad),np.sin(rad)])*dis
            #     while landmark.state.p_pos[0] < -0.95 or landmark.state.p_pos[0] > 0.95 or landmark.state.p_pos[1] < -0.95 or landmark.state.p_pos[1] > 0.95:
            #         dis = np.random.uniform(0.4, 1.8)
            #         rad = np.random.uniform(0, np.pi * 2)
            #         landmark.state.p_pos = world.agents[0].state.p_pos + np.array([np.cos(rad), np.sin(rad)]) * dis
            #     landmark.state.p_vel = np.zeros(world.dim_p)
            else:
                landmark.state.p_pos = world.landmarks[i-world.num_landmarks].state.p_pos
                landmark.state.p_vel = np.zeros(world.dim_p)
        #障碍物随机定位，假设只有一个目标点，让障碍物不接触目标点
        target_landmark = world.landmarks[0]
        agent = world.agents[0]
        # 智能体与目标点之间的向量
        agent_to_target = target_landmark.state.p_pos - agent.state.p_pos
        agent_to_target_dist = self._distance(target_landmark.state.p_pos, agent.state.p_pos)
        # 智能体的大小，假设为一个常量，可根据实际情况调整
        agent_size = agent.size
        min_gap = agent_size * 4  # 两相近障碍物之间的最小缝隙
        for i, obstacle in enumerate(world.obstacles):
            if i < world.num_obstacles:
                obstacle.size = np.random.uniform(0.05, 0.15)
                world.obstacles[i + world.num_obstacles].size = obstacle.size
                valid_position = False
                while not valid_position:
                    # 以智能体与目标点连线为直径创建圆形区域
                    # 先随机生成一个角度
                    angle = np.random.uniform(0, 2 * np.pi)
                    # 随机生成一个半径
                    radius = np.random.uniform(0, (3 * agent_to_target_dist) / 2)
                    # 计算圆形区域内的随机点相对于智能体的偏移向量
                    offset = np.array([radius * np.cos(angle), radius * np.sin(angle)])
                    # 计算障碍物的位置
                    O_pos = agent.state.p_pos + agent_to_target / 2 + offset
                    # 检查是否覆盖目标点、智能体以及与其他障碍物重叠，并且距离目标 0.6 以上
                    valid = True
                    # 增加距离目标 0.6 以上的判断条件
                    if self._distance(O_pos, target_landmark.state.p_pos) <= 0.6 or self._distance(O_pos, agent.state.p_pos) <= (obstacle.size + agent_size + 0.3):
                        valid = False
                    for other_obstacle in world.obstacles[:i]:
                        dist = self._distance(O_pos, other_obstacle.state.p_pos)
                        if dist <= (obstacle.size + other_obstacle.size + min_gap):
                            valid = False
                            break
                    if valid:
                        valid_position = True

                obstacle.state.p_pos = O_pos
                obstacle.state.p_vel = np.zeros(world.dim_p)
            else:
                obstacle.state.p_pos = world.obstacles[i - world.num_obstacles].state.p_pos
                obstacle.state.p_vel = np.zeros(world.dim_p)

        # Scenario-generation v2: overwrite legacy placement with a bounded map,
        # distant pursuit endpoints, static point obstacles, and clear dynamic vessels.
        agent = world.agents[0]
        target_landmark = world.landmarks[0]
        agent_pos, target_pos = self._sample_agent_target_pair(world)
        agent.state.p_pos = agent_pos
        agent.state.p_pos_origin = agent_pos.copy()
        agent.state.p_vel = np.zeros(world.dim_p)
        agent.state.p_vel_old = np.zeros(world.dim_p)
        agent.state.usv_3dof = None
        target_landmark.state.p_pos = target_pos
        target_landmark.state.p_vel = np.zeros(world.dim_p)

        for i, landmark in enumerate(world.landmarks):
            if i < world.num_landmarks:
                if i > 0:
                    landmark.state.p_pos = self._sample_clear_position(
                        world,
                        landmark.size,
                        world.agents + world.landmarks[:i],
                        clearance=0.40,
                    )
                landmark.state.p_vel = np.zeros(world.dim_p)
            else:
                landmark.state.p_pos = world.landmarks[i - world.num_landmarks].state.p_pos.copy()
                landmark.state.p_vel = np.zeros(world.dim_p)

        fixed_entities = world.agents + world.landmarks[:world.num_landmarks]
        placed_static = []
        for obstacle in world.static_obstacles:
            obstacle.size = np.random.uniform(self.static_obstacle_min_size, self.static_obstacle_max_size)
            obstacle.state.p_pos = self._sample_clear_position(
                world,
                obstacle.size,
                fixed_entities + placed_static,
                clearance=self.static_obstacle_clearance,
                margin=self.static_obstacle_spawn_margin,
            )
            obstacle.state.p_vel = np.zeros(world.dim_p)
            obstacle.action.u = np.zeros(world.dim_p)
            placed_static.append(obstacle)

        agent_to_target = target_landmark.state.p_pos - agent.state.p_pos
        route_length = max(self._distance(target_landmark.state.p_pos, agent.state.p_pos), 1.0e-9)
        route_unit = agent_to_target / route_length
        route_normal = np.array([-route_unit[1], route_unit[0]])

        _enc_cfg = getattr(world, 'environment_config', {}) or {}
        _use_encounter_generation = float(
            _enc_cfg.get('encounter_generation_mode', 0.0)
        ) >= 0.5
        placed_dynamic = []
        for i, obstacle in enumerate(world.obstacles):
            if i < world.num_obstacles:
                obstacle.size = np.random.uniform(0.05, 0.15)
                world.obstacles[i + world.num_obstacles].size = obstacle.size
                occupied = fixed_entities + world.static_obstacles + placed_dynamic
                if _use_encounter_generation:
                    _max_try = int(round(float(_enc_cfg.get('encounter_max_resample', 200.0))))
                    _sampled = None
                    for _ in range(max(_max_try, 1)):
                        _sampled = self._sample_encounter_state(
                            agent, target_landmark, obstacle, occupied, world,
                            agent_nominal_speed=float(getattr(world, 'agent_nominal_speed', 0.001)),
                        )
                        if _sampled is not None:
                            break
                    if _sampled is not None:
                        obstacle.state.p_pos = _sampled[0]
                        obstacle.ra = float(_sampled[1])
                        obstacle.obstacle_vel = float(_sampled[2])
                        obstacle.max_speed = float(_sampled[2])
                        obstacle.state.p_vel = np.array(
                            [np.cos(obstacle.ra), np.sin(obstacle.ra)], dtype=float
                        ) * float(_sampled[2])
                        obstacle.encounter_generated = True
                        placed_dynamic.append(obstacle)
                        continue
                position = None
                for _ in range(800):
                    along = np.random.uniform(0.15, 0.90)
                    lateral = np.random.uniform(-0.75, 0.75)
                    candidate = agent.state.p_pos + agent_to_target * along + route_normal * lateral
                    if not self._inside_map(
                        candidate,
                        world,
                        radius=obstacle.size,
                        margin=self.dynamic_obstacle_spawn_margin,
                    ):
                        continue
                    if self._is_clear_of_entities(candidate, obstacle.size, occupied, self.vessel_static_clearance):
                        position = candidate
                        break
                if position is None:
                    position = self._sample_clear_position(
                        world,
                        obstacle.size,
                        occupied,
                        clearance=self.vessel_vessel_spawn_clearance,
                        margin=self.dynamic_obstacle_spawn_margin,
                    )
                obstacle.state.p_pos = position
                obstacle.state.p_vel = np.zeros(world.dim_p)
                placed_dynamic.append(obstacle)
            else:
                obstacle.state.p_pos = world.obstacles[i - world.num_obstacles].state.p_pos.copy()
                obstacle.state.p_vel = np.zeros(world.dim_p)

        for agent in world.agents:
            agent.reward_prev_target_distance = self._distance(
                target_landmark.state.p_pos,
                agent.state.p_pos,
            )
            agent.reward_prev_action = np.zeros(world.agent_action_dim)
            agent.reward_last_components = {}
            agent.reward_last_metrics = {}
            agent.reward_done_reason = 'none'
        #benchmark variables
        self.pre_error = np.ones(world.num_landmarks)
        self.agent_outofworld = 0
        self.landmark_collision = 0
        self.agent_collision = 0
        self.obstacle_collision = 0
        self.ob_pre_error = np.ones(world.num_obstacles)
        #tacke a random velocity
        for landmark in world.landmarks:
            if self.movement == 'static':
                landmark.landmark_vel = 0.0
            elif landmark.movable:
                if self.random_vel == True:
                    landmark.landmark_vel = float(np.random.uniform(
                        self.target_min_speed,
                        self.target_max_speed,
                    ))
                else:
                    landmark.landmark_vel = self.target_initial_speed
            else:
                landmark.landmark_vel = 0.
            landmark.max_speed = self.target_max_speed if landmark.movable else 0.0
            landmark.current_landmark_vel = landmark.landmark_vel
            landmark.target_escape_heading_bias = 0.0
            landmark.target_escape_last_distance = None
            landmark.target_escape_desired_heading = None
            landmark.target_escape_control_action = 'none'
            landmark.target_speed_intent = landmark.landmark_vel
            landmark.target_speed_hold_steps = 0
            landmark.target_nominal_heading = 0.0
            landmark.target_random_vector = np.zeros(world.dim_p, dtype=float)
            landmark.target_levy_hold_steps = 0
                
        #take a random direction
        for landmark in world.landmarks:
            landmark.ra = (np.random.rand(1)*np.pi*2.).item(0) #initial landmark direction
            landmark.target_nominal_heading = landmark.ra
            landmark.target_random_vector = self._heading_command(landmark.ra, 1.0)
            #take a random target depth
            landmark.landmark_depth = 0.
            # 给障碍物设置一个随机的速度
        for obstacle in world.obstacles:
            if getattr(obstacle, 'encounter_generated', False):
                # P0-1：会遇生成器已确定速度（构造几何），不覆盖
                obstacle.max_speed = obstacle.obstacle_vel
                continue
            if self.dynamic_obstacle_speed_range is not None:
                min_speed, max_speed = self.dynamic_obstacle_speed_range
                obstacle.obstacle_vel = float(np.random.uniform(min_speed, max_speed))
            elif self.random_vel == True:
                obstacle.obstacle_vel = np.random.rand(1).item(0) * self.max_vel
            else:
                obstacle.obstacle_vel = self.landmark_vel  #后续可以考虑给障碍物加一个配置属性，最大深度，最大速度，等等
            obstacle.max_speed = obstacle.obstacle_vel
        # 给障碍物设置一个随机的方向
        for obstacle in world.obstacles:
            if not getattr(obstacle, 'encounter_generated', False):
                # P0-1：会遇生成器已确定航向（构造几何），不覆盖
                obstacle.ra = (np.random.rand(1) * np.pi * 2.).item(0)  # initial obstacle direction
            obstacle.obstacle_depth = 0.
            obstacle.colregs_last_report = None
            obstacle.colregs_last_reports = ()
            obstacle.colregs_control_action = 'none'
            self._reset_geometric_avoidance(obstacle)

        #Initailize the landmark estimated positions
        world.landmarks_estimated = [Target(L_pos=world.landmarks[i].state.p_pos,V=[world.landmarks[i].landmark_vel*np.cos(world.landmarks[i].ra),world.landmarks[i].landmark_vel*np.sin(world.landmarks[i].ra)]) for i in range(world.num_landmarks)]
        # 定义障碍物估计位置
        world.obstacles_estimated = [Target(L_pos=world.obstacles[i].state.p_pos,V=[world.obstacles[i].obstacle_vel*np.cos(world.obstacles[i].ra),world.obstacles[i].obstacle_vel*np.sin(world.obstacles[i].ra)]) for i in range(world.num_obstacles)]
        #initialize the ocean current at random
        world.vel_ocean_current = np.random.rand(1).item(0)*self.max_vel_ocean_current #initial random strength
        world.angle_ocean_current = (np.random.rand(1)*np.pi*2.).item(0) #initial landmark direction
    
       
    def _reset_scaled_small_layout(self, world):
        """Generate the 600 m / 1 s scene without running legacy placement."""
        world.vel_ocean_current = np.random.rand() * self.max_vel_ocean_current
        world.angle_ocean_current = np.random.rand() * np.pi * 2.0
        self._set_entity_colors(world)

        for agent_i, agent in enumerate(world.agents):
            self._reset_agent_state(agent, world)
            if agent_i < len(world.angle):
                world.angle[agent_i] = float(np.random.uniform(-np.pi, np.pi))

        agent = world.agents[0]
        target = world.landmarks[0]
        agent_pos, target_pos = self._sample_agent_target_pair(world)
        agent.state.p_pos = agent_pos
        agent.state.p_pos_origin = agent_pos.copy()
        target.state.p_pos = target_pos
        target.state.p_vel = np.zeros(world.dim_p, dtype=float)
        target.action.u = np.zeros(world.dim_p, dtype=float)

        for i, landmark in enumerate(world.landmarks):
            if i < world.num_landmarks:
                if i > 0:
                    landmark.state.p_pos = self._sample_clear_position(
                        world,
                        landmark.size,
                        world.agents + world.landmarks[:i],
                        clearance=self.vessel_vessel_spawn_clearance,
                        margin=self.agent_target_spawn_margin,
                    )
                landmark.state.p_vel = np.zeros(world.dim_p, dtype=float)
            else:
                source = world.landmarks[i - world.num_landmarks]
                landmark.state.p_pos = source.state.p_pos.copy()
                landmark.state.p_vel = np.zeros(world.dim_p, dtype=float)

        fixed_entities = world.agents + world.landmarks[:world.num_landmarks]
        placed_static = []
        for obstacle in world.static_obstacles:
            obstacle.size = float(np.random.uniform(
                self.static_obstacle_min_size,
                self.static_obstacle_max_size,
            ))
            obstacle.state.p_pos = self._sample_clear_position(
                world,
                obstacle.size,
                fixed_entities + placed_static,
                clearance=self.static_obstacle_clearance,
                margin=self.static_obstacle_spawn_margin,
            )
            obstacle.state.p_vel = np.zeros(world.dim_p, dtype=float)
            obstacle.action.u = np.zeros(world.dim_p, dtype=float)
            placed_static.append(obstacle)

        agent_to_target = target.state.p_pos - agent.state.p_pos
        route_length = max(float(np.linalg.norm(agent_to_target)), 1.0e-9)
        route_normal = np.array([-agent_to_target[1], agent_to_target[0]]) / route_length
        lateral_limit = max(float(world.map_half_size) * 0.65, 1.0e-6)
        placed_dynamic = []
        for i, obstacle in enumerate(world.obstacles):
            if i < world.num_obstacles:
                obstacle.size = float(np.random.uniform(
                    self.dynamic_obstacle_min_size,
                    self.dynamic_obstacle_max_size,
                ))
                world.obstacles[i + world.num_obstacles].size = obstacle.size
                occupied = fixed_entities + world.static_obstacles + placed_dynamic
                position = None
                for _ in range(800):
                    along = np.random.uniform(0.15, 0.90)
                    lateral = np.random.uniform(-lateral_limit, lateral_limit)
                    candidate = agent.state.p_pos + agent_to_target * along + route_normal * lateral
                    if not self._inside_map(
                        candidate,
                        world,
                        radius=obstacle.size,
                        margin=self.dynamic_obstacle_spawn_margin,
                    ):
                        continue
                    if self._is_clear_of_entities(
                        candidate,
                        obstacle.size,
                        occupied,
                        self.vessel_static_clearance,
                    ):
                        position = candidate
                        break
                if position is None:
                    position = self._sample_clear_position(
                        world,
                        obstacle.size,
                        occupied,
                        clearance=self.vessel_vessel_spawn_clearance,
                        margin=self.dynamic_obstacle_spawn_margin,
                    )
                obstacle.state.p_pos = position
                obstacle.state.p_vel = np.zeros(world.dim_p, dtype=float)
                obstacle.action.u = np.zeros(world.dim_p, dtype=float)
                placed_dynamic.append(obstacle)
            else:
                source = world.obstacles[i - world.num_obstacles]
                obstacle.state.p_pos = source.state.p_pos.copy()
                obstacle.state.p_vel = np.zeros(world.dim_p, dtype=float)
                obstacle.action.u = np.zeros(world.dim_p, dtype=float)

        self.pre_error = np.ones(world.num_landmarks)
        self.ob_pre_error = np.ones(world.num_obstacles)
        self.agent_outofworld = 0
        self.landmark_collision = 0
        self.agent_collision = 0
        self.obstacle_collision = 0

        for landmark in world.landmarks:
            landmark.landmark_vel = 0.0 if self.movement == 'static' else self.target_initial_speed
            landmark.max_speed = 0.0 if self.movement == 'static' else self.target_max_speed
            landmark.current_landmark_vel = landmark.landmark_vel
            landmark.ra = float(np.random.uniform(-np.pi, np.pi))
            landmark.landmark_depth = 0.0
            landmark.target_escape_heading_bias = 0.0
            landmark.target_escape_last_distance = None
            landmark.target_escape_desired_heading = None
            landmark.target_escape_control_action = 'static' if self.movement == 'static' else 'none'
            landmark.target_speed_intent = landmark.landmark_vel
            landmark.target_speed_hold_steps = 0
            landmark.target_nominal_heading = landmark.ra
            landmark.target_random_vector = self._heading_command(landmark.ra, 1.0)
            landmark.target_levy_hold_steps = 0

        for obstacle in world.obstacles:
            if getattr(obstacle, 'encounter_generated', False):
                # P0-1：会遇生成器已确定速度与航向，保留构造几何
                obstacle.max_speed = obstacle.obstacle_vel
            else:
                if self.dynamic_obstacle_speed_range is not None:
                    min_speed, max_speed = self.dynamic_obstacle_speed_range
                    obstacle.obstacle_vel = float(np.random.uniform(min_speed, max_speed))
                else:
                    obstacle.obstacle_vel = float(self.landmark_vel)
                obstacle.max_speed = obstacle.obstacle_vel
                obstacle.ra = float(np.random.uniform(-np.pi, np.pi))
            obstacle.obstacle_depth = 0.0
            obstacle.colregs_last_report = None
            obstacle.colregs_last_reports = ()
            obstacle.colregs_control_action = 'none'
            self._reset_geometric_avoidance(obstacle)

        for agent in world.agents:
            agent.reward_prev_target_distance = self._distance(target.state.p_pos, agent.state.p_pos)
            agent.reward_prev_action = np.zeros(world.agent_action_dim, dtype=float)
            agent.reward_last_components = {}
            agent.reward_last_metrics = {}
            agent.reward_done_reason = 'none'

        world.landmarks_estimated = [
            Target(L_pos=landmark.state.p_pos, V=[0.0, 0.0])
            for landmark in world.landmarks[:world.num_landmarks]
        ]
        world.obstacles_estimated = [
            Target(
                L_pos=obstacle.state.p_pos,
                V=[
                    obstacle.obstacle_vel * np.cos(obstacle.ra),
                    obstacle.obstacle_vel * np.sin(obstacle.ra),
                ],
            )
            for obstacle in world.obstacles[:world.num_obstacles]
        ]

    def benchmark_data(self, agent, world):
        landmarks_real_p = []
        landmarks_predict_p = []
        obstacles_p = []
        entity_size = []
        for i in range(world.num_agents):
            entity_size.append(world.agents[i].size)
        for i in range(world.num_landmarks):
            landmarks_real_p.append(world.landmarks[i].state.p_pos)
            entity_size.append(world.landmarks[i].size)
        for i in range(world.num_landmarks):
            landmarks_predict_p.append(world.landmarks[i + world.num_landmarks].state.p_pos)
        for i in range(world.num_obstacles):
            obstacles_p.append(world.obstacles[i].state.p_pos)
            entity_size.append(world.obstacles[i].size)
        # return (rew, collisions, min_dists, occupied_landmarks,landmarks_real_p)
        return(self.pre_error,landmarks_real_p, self.agent_outofworld, self.landmark_collision, self.agent_collision,self.obstacle_collision,landmarks_predict_p,obstacles_p,entity_size) #后续要加入一个障碍物与智能体碰撞次数


    def benchmark_data_compact(self, agent, world):
        """返回固定长度 benchmark 数据，供环境 info 数组打包。"""
        errors = np.asarray(self.pre_error, dtype=np.float32).reshape(-1)
        errors = errors[:world.num_landmarks]
        if errors.size < world.num_landmarks:
            errors = np.pad(errors, (0, world.num_landmarks - errors.size))
        counters = np.asarray(
            [
                self.agent_outofworld,
                self.landmark_collision,
                self.agent_collision,
                self.obstacle_collision,
            ],
            dtype=np.float32,
        )
        return np.concatenate((errors, counters)).astype(np.float32, copy=False)

    def is_collision(self, agent1, agent2):
        dist = self._distance(agent1.state.p_pos, agent2.state.p_pos)
        dist_min = agent1.size + agent2.size
        return dist < dist_min

    def _target_for_agent(self, agent, world):
        for i, landmark in enumerate(world.landmarks[:world.num_landmarks]):
            if agent.name[-1] == landmark.name[-1]:
                return i, landmark
        return 0, world.landmarks[0]

    def _reward_obstacle_entities(self, agent, world, target):
        entities = []
        entities.extend(world.obstacles[:world.num_obstacles])
        entities.extend(getattr(world, 'static_obstacles', []))
        entities.extend([other for other in world.agents if other is not agent])
        entities.extend([
            landmark for landmark in world.landmarks[:world.num_landmarks]
            if landmark is not target
        ])
        return [entity for entity in entities if entity.state.p_pos is not None]

    def _minimum_clearance(self, agent, entities):
        min_clearance = np.inf
        closest_entity = None
        for entity in entities:
            clearance = (
                self._distance(agent.state.p_pos, entity.state.p_pos)
                - agent.size
                - entity.size
            )
            if clearance < min_clearance:
                min_clearance = float(clearance)
                closest_entity = entity
        return min_clearance, closest_entity

    def _boundary_clearance(self, agent, world):
        half_size = getattr(world, 'map_half_size', 2.0)
        return float(half_size - agent.size - np.max(np.abs(agent.state.p_pos)))

    def _agent_reward_action(self, agent, world):
        action = getattr(agent, 'action_executed_u', None)
        if action is None:
            action = agent.action.u
        if action is None:
            return np.zeros(world.agent_action_dim)
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.size < world.agent_action_dim:
            padded = np.zeros(world.agent_action_dim)
            padded[:action.size] = action
            return padded
        return action[:world.agent_action_dim]

    def _update_tracking_error_metric(self, agent, world):
        for i, landmark_estimated in enumerate(world.landmarks_estimated[:world.num_landmarks]):
            if agent.name[-1] != world.landmarks[i].name[-1]:
                continue
            try:
                if self.pre_method == 'PF':
                    estimated_pos = [landmark_estimated.pfxs[0], landmark_estimated.pfxs[2]]
                elif self.pre_method == 'LS' or self.pre_method == 'KM':
                    estimated_pos = [landmark_estimated.lsxs[-1][0], landmark_estimated.lsxs[-1][2]]
                else:
                    continue
                self.pre_error[i] = self._distance(estimated_pos, world.landmarks[i].state.p_pos)
            except (IndexError, AttributeError, TypeError):
                continue
    
    # Legacy reward kept only as a reference during reward-v2 migration.
    def _legacy_reward_unused(self, agent, world):
        global done_state
        done_state = False
        agent.done_state = False
        # Agents are rewarded based on landmarks_estimated covariance_vals, penalized for collisions
        rew = 0.

        if not self._inside_map(agent.state.p_pos, world, radius=agent.size):
            rew -= 100
            done_state = True
            agent.done_state = True
            self.agent_outofworld += 1
            return rew
        
        for i,o in enumerate(world.obstacles):
            if i < world.num_obstacles and self.is_collision(o, agent):
                self.obstacle_collision += 1
                rew -= 10.
                done_state = True
                agent.done_state = True
                return rew
            elif i < world.num_obstacles:
                dist = self._distance(world.obstacles[i].state.p_pos, agent.state.p_pos)
                if dist < self.rew_dis_th:
                    rew += np.log(abs(dist - world.obstacles[i].size))
        for obstacle in getattr(world, 'static_obstacles', []):
            if self.is_collision(obstacle, agent):
                self.obstacle_collision += 1
                rew -= 10.
                done_state = True
                agent.done_state = True
                return rew
            dist = self._distance(obstacle.state.p_pos, agent.state.p_pos)
            if dist < self.rew_dis_th:
                rew += np.log(abs(dist - obstacle.size))
        if agent.collide:
            for a in world.agents:
                if a is agent: continue
                if self.is_collision(a, agent):
                    rew -= 10.
                    self.agent_collision += 1
                    done_state = True
                    agent.done_state = True
                    return rew
        for i,l in enumerate(world.landmarks):
            if i < world.num_landmarks and self.is_collision(l, agent) and agent.name[-1] != l.name[-1]:
                self.landmark_collision += 1
                rew -= 10.
                done_state = True
                agent.done_state = True
                return rew
            elif i < world.num_landmarks and agent.name[-1] == l.name[-1]:
                estimated_pos = [world.landmarks_estimated[i].lsxs[-1][0], world.landmarks_estimated[i].lsxs[-1][2]]
                dist = self._distance(estimated_pos, agent.state.p_pos)
                true_dist = self._distance(world.landmarks[i].state.p_pos, agent.state.p_pos)
                if dist > 2 * self.set_max_range and true_dist > 2 * self.set_max_range:
                    rew -= 100
                    done_state = True
                    agent.done_state = True
                    self.agent_outofworld += 1
                    return rew
                elif dist < agent.size + l.size:
                    self.landmark_collision += 1
                    rew -= 10.
                    done_state = True
                    agent.done_state = True
                    return rew
                # ===== 捕获+保持型稀疏奖励 =====
                # 1. 捕获阶段：首次进入安全跟踪范围的奖励
                elif not agent.has_captured and (agent.size + l.size) <= dist <= self.rew_dis_th:
                    rew += 10.  # 一次性高奖励，激励捕获目标
                    agent.has_captured = True
                    agent.consecutive_hold = 0  # 重置保持计数
                
                # 2. 保持阶段：持续停留在安全范围内的奖励
                elif agent.has_captured and (agent.size + l.size) <= dist <= self.rew_dis_th:
                    # 连续保持在范围内的时间越长，奖励越高 (阶梯式稀疏奖励)
                    agent.consecutive_hold += 1
                    
                    # 每达到一定时间步长给予额外奖励
                    if agent.consecutive_hold % 10 == 0:  # 每10步
                        rew += 10.  # 保持奖励
                
                # 3. 脱离观测范围惩罚
                elif agent.has_captured and dist > world.ob_range:
                    # 脱离范围的惩罚与脱离时间相关
                    rew -= 10
                    agent.has_captured = False  # 重置捕获状态
        for i,l in enumerate(world.landmarks_estimated): #计算估计地标位置与真实地标位置关系，给予奖惩
            if agent.name[-1] == world.landmarks[i].name[-1]:
                if self.pre_method == 'PF':
                    self.pre_error[i] = self._distance([l.pfxs[0], l.pfxs[2]], world.landmarks[i].state.p_pos) #Error from PF
                elif self.pre_method == 'LS' or self.pre_method == 'KM':
                    self.pre_error[i] = self._distance([l.lsxs[-1][0], l.lsxs[-1][2]], world.landmarks[i].state.p_pos) #Error from LS
                if self.pre_error[i]<self.rew_err_th:
                    rew += 1
        return rew

    # Reward v2: task-driven pursuit reward. This is the active reward entry
    # and keeps each reward term easy to identify.
    def reward(self, agent, world):
        global done_state
        done_state = False
        agent.done_state = False
        components = {
            'success': 0.0,
            'progress': 0.0,
            'range': 0.0,
            'safety': 0.0,
            'boundary': 0.0,
            'smooth': 0.0,
            'yaw_rate': 0.0,
            'rudder_smooth': 0.0,
            'intervention': 0.0,
            'time': 0.0,
            'terminal': 0.0,
        }

        _, target = self._target_for_agent(agent, world)
        target_distance = self._distance(target.state.p_pos, agent.state.p_pos)
        target_collision_distance = agent.size + target.size
        capture_distance = self.rew_dis_th
        obstacle_entities = self._reward_obstacle_entities(agent, world, target)
        min_clearance, closest_entity = self._minimum_clearance(agent, obstacle_entities)
        boundary_clearance = self._boundary_clearance(agent, world)
        physical_state = self._agent_3dof_state(agent, world)
        physical_config = world.usv_3dof_model.config
        previous_rudder = getattr(agent, 'reward_prev_rudder', None)
        rudder_delta = (
            0.0
            if previous_rudder is None
            else float(physical_state.rudder) - float(previous_rudder)
        )
        metrics = {
            'target_distance': float(target_distance),
            'min_clearance': float(min_clearance) if np.isfinite(min_clearance) else float('inf'),
            'boundary_clearance': float(boundary_clearance),
            'capture_distance': float(capture_distance),
            'yaw_rate_abs_deg_s': float(np.rad2deg(abs(physical_state.r))),
            'rudder_angle_abs_deg': float(np.rad2deg(abs(physical_state.rudder))),
            'rudder_delta_abs_deg': float(np.rad2deg(abs(rudder_delta))),
            # 执行器诊断使用 3DOF 物理状态，而不是策略输出的归一化动作：
            # actual_thrust 保留物理推力单位，actual_rudder_deg 保留角度单位。
            'actual_thrust': float(physical_state.thrust),
            'actual_rudder_deg': float(np.rad2deg(physical_state.rudder)),
        }
        agent.reward_done_reason = 'none'

        # Reward 0: terminal boundary penalty. Leaving the map is unsafe and ends the episode.
        if boundary_clearance < 0.0 or getattr(agent, 'boundary_violation', False):
            components['terminal'] = -self.reward_out_of_bounds
            agent.reward_last_components = components
            agent.reward_last_metrics = metrics
            agent.reward_done_reason = 'out_of_bounds'
            self.agent_outofworld += 1
            done_state = True
            agent.done_state = True
            return float(sum(components.values()))

        # Reward 0: terminal collision penalty against dynamic/static obstacles or other entities.
        if min_clearance < 0.0:
            components['terminal'] = -self.reward_collision
            agent.reward_last_components = components
            agent.reward_last_metrics = metrics
            agent.reward_done_reason = 'collision'
            if closest_entity in world.agents:
                self.agent_collision += 1
            elif closest_entity in world.landmarks[:world.num_landmarks]:
                self.landmark_collision += 1
            else:
                self.obstacle_collision += 1
            done_state = True
            agent.done_state = True
            return float(sum(components.values()))

        # Reward 0: terminal collision penalty for getting too close to the target hull.
        if target_distance < target_collision_distance:
            components['terminal'] = -self.reward_collision
            agent.reward_last_components = components
            agent.reward_last_metrics = metrics
            agent.reward_done_reason = 'target_collision'
            self.landmark_collision += 1
            done_state = True
            agent.done_state = True
            return float(sum(components.values()))

        # Reward 1: task completion reward. One safe entry into the target neighborhood ends the episode.
        if target_distance <= capture_distance:
            components['success'] = self.reward_success
            agent.has_captured = True
            agent.consecutive_hold = 0
            # 任务阶段标记（接口预留）：transit / approach / capture
            agent.phase = 'transit'
            # per-agent 终止标记（多智能体隔离）
            agent.done_state = False
            agent.reward_done_reason = 'success'
            done_state = True
            agent.done_state = True

        # Reward 1b: reaching the configured horizon without success is a timeout failure.
        if (
            not done_state
            and bool(getattr(world, 'timeout_termination_enabled', False))
            and getattr(world, 'max_episode_steps', None) is not None
            and int(getattr(world, 'step_count', 0)) >= int(world.max_episode_steps)
        ):
            components['terminal'] = -self.reward_timeout
            agent.reward_last_components = components
            agent.reward_last_metrics = metrics
            agent.reward_done_reason = 'timeout'
            done_state = True
            agent.done_state = True
            return float(sum(components.values()))

        # Reward 2: progress reward. Positive when the USV reduces target distance this step.
        previous_distance = getattr(agent, 'reward_prev_target_distance', None)
        distance_scale = max(2.0 * getattr(world, 'map_half_size', 2.0), 1.0e-9)
        if previous_distance is not None:
            components['progress'] = (
                self.reward_progress_weight
                * (float(previous_distance) - float(target_distance))
                / distance_scale
            )

        # Reward 3: dense range penalty. It replaces the old per-step positive
        # range reward, so merely surviving no longer accumulates easy scores.
        clipped_distance = min(float(target_distance), distance_scale)
        components['range'] = -self.reward_range_weight * clipped_distance / distance_scale

        # Reward 4: continuous safety penalty. It activates before collision to teach early avoidance.
        if np.isfinite(min_clearance):
            safety_ratio = max(
                0.0,
                (self.reward_safe_clearance - min_clearance)
                / max(self.reward_safe_clearance, 1.0e-9),
            )
            components['safety'] = -self.reward_safety_weight * safety_ratio ** 2

        # Reward 5: boundary buffer penalty. It discourages hugging the map edge before going out.
        boundary_ratio = max(
            0.0,
            (self.reward_boundary_safe_distance - boundary_clearance)
            / max(self.reward_boundary_safe_distance, 1.0e-9),
        )
        components['boundary'] = -self.reward_boundary_weight * boundary_ratio ** 2

        # Reward 6: action smoothness penalty. It discourages abrupt thrust/rudder changes.
        current_action = self._agent_reward_action(agent, world)
        previous_action = getattr(agent, 'reward_prev_action', None)
        if previous_action is not None:
            previous_action = np.asarray(previous_action, dtype=float).reshape(-1)
            if previous_action.size == current_action.size:
                action_delta = current_action - previous_action
                components['smooth'] = -self.reward_action_smooth_weight * float(np.dot(action_delta, action_delta))

        # Reward 6b: normalized yaw-rate penalty. This acts on the real 3-DOF
        # state after integration, so it directly discourages sustained hull oscillation.
        max_yaw_rate = max(float(physical_config.max_yaw_rate), 1.0e-9)
        yaw_rate_ratio = float(np.clip(physical_state.r / max_yaw_rate, -1.0, 1.0))
        components['yaw_rate'] = -self.reward_yaw_rate_weight * yaw_rate_ratio ** 2

        # Reward 6c: actual-rudder smoothness. The actuator already rate-limits
        # delta; this light term teaches the policy not to reverse that physical
        # rudder motion unnecessarily from one environment step to the next.
        max_rudder_angle = max(float(physical_config.max_rudder_angle), 1.0e-9)
        rudder_delta_ratio = float(np.clip(rudder_delta / max_rudder_angle, -2.0, 2.0))
        components['rudder_smooth'] = -(
            self.reward_rudder_smooth_weight * rudder_delta_ratio ** 2
        )

        # Reward 7: rule intervention penalty. It discourages relying on the COLREGs action-set teacher.
        if bool(getattr(agent, 'action_filter_active', False)):
            constrained_action = getattr(agent, 'action_constrained_u', current_action)
            constrained_action = np.asarray(constrained_action, dtype=float).reshape(-1)
            # 规则干预只应度量 constrained -> rule。最终执行动作还会经过
            # EMA 平滑器，所以不能使用 current_action（smoothed_action）计算
            # 这里的惩罚，否则会把平滑器造成的差异错误归因给 COLREGs。
            rule_action = getattr(agent, 'action_rule_u', None)
            if rule_action is None:
                decision = getattr(agent, 'action_filter_decision', None)
                if isinstance(decision, dict):
                    rule_action = decision.get('rule_action')
                elif decision is not None:
                    rule_action = getattr(decision, 'rule_action', None)
            if rule_action is None:
                # 兼容旧控制路径：没有独立规则动作时，退化为约束动作，
                # 这样不会凭空制造规则干预惩罚。
                rule_action = constrained_action
            rule_action = np.asarray(rule_action, dtype=float).reshape(-1)
            if constrained_action.size == rule_action.size:
                correction = rule_action - constrained_action
                correction_energy = float(np.dot(correction, correction)) / max(current_action.size, 1)
                components['intervention'] = -(
                    self.reward_intervention_active_penalty
                    + self.reward_intervention_magnitude_weight * correction_energy
                )

        # Optional COLREGs reward guidance. It is deliberately folded into the
        # existing intervention component so the fixed info schema and legacy
        # reward layout remain unchanged when the option is disabled.
        if self.rule_reward_enabled:
            decision = getattr(agent, 'action_filter_decision', None)
            if isinstance(decision, dict):
                rule_action = decision.get('rule_action')
                raw_action = decision.get('raw_action')
                encounter_type = decision.get('encounter_type')
                dcpa = decision.get('dcpa')
                tcpa = decision.get('tcpa')
                give_way = decision.get('give_way_vessel')
            else:
                rule_action = getattr(decision, 'rule_action', None)
                raw_action = getattr(decision, 'raw_action', None)
                encounter_type = getattr(decision, 'encounter_type', None)
                dcpa = getattr(decision, 'dcpa', None)
                tcpa = getattr(decision, 'tcpa', None)
                give_way = getattr(decision, 'give_way_vessel', None)
            if rule_action is not None and raw_action is not None:
                alignment = np.asarray(rule_action, dtype=float).reshape(-1) - np.asarray(raw_action, dtype=float).reshape(-1)
                components['intervention'] -= self.rule_action_alignment_weight * float(np.dot(alignment, alignment))
            # DCPA/TCPA shaping is continuous and only applies to finite rule reports.
            if dcpa is not None and tcpa is not None and np.isfinite(dcpa) and np.isfinite(tcpa):
                cpa_risk = max(0.0, (self.reward_safe_clearance - float(dcpa)) / max(self.reward_safe_clearance, 1.0e-9))
                tcpa_risk = np.clip((self.world_colregs_risk_time_horizon - float(tcpa)) / max(self.world_colregs_risk_time_horizon, 1.0e-9), 0.0, 1.0) if hasattr(self, 'world_colregs_risk_time_horizon') else 0.0
                components['intervention'] -= self.rule_cpa_weight * float(cpa_risk * tcpa_risk)
            # 仅对明确的让路义务施加可选惩罚，避免把直航船动作误判为违规。
            if self.rule_obligation_penalty > 0.0 and give_way in (agent.name, 'both'):
                raw = np.asarray(raw_action if raw_action is not None else current_action, dtype=float).reshape(-1)
                if raw.size >= 2 and raw[1] >= 0.0:
                    components['intervention'] -= self.rule_obligation_penalty

        # Reward 8: per-step time pressure. A stronger value makes slow pursuit
        # clearly worse than completing the task in fewer environment steps.
        components['time'] = -self.reward_time_penalty

        agent.reward_prev_target_distance = float(target_distance)
        agent.reward_prev_action = current_action.copy()
        agent.reward_prev_rudder = float(physical_state.rudder)
        agent.reward_last_components = components
        agent.reward_last_metrics = metrics
        self._update_tracking_error_metric(agent, world)
        return float(sum(components.values()))

    def _update_task_phase(self, agent, world):
        """任务阶段标记（接口预留）：transit → approach → capture。

        approach 半径默认取 5 倍捕获距离（rew_dis_th）；进入捕获区间后由
        agent.has_captured 决定 capture 阶段。分阶段统计规则事件用。
        """
        if getattr(agent, 'has_captured', False):
            agent.phase = 'capture'
            return
        dist = None
        for i, landmark in enumerate(world.landmarks[:world.num_landmarks]):
            if str(agent.name)[-1] == str(landmark.name)[-1]:
                dist = self._distance(landmark.state.p_pos, agent.state.p_pos)
                break
        if dist is None:
            agent.phase = 'transit'
            return
        capture_dist = float(getattr(self, 'rew_dis_th', 0.1) or 0.1)
        agent.phase = 'approach' if dist <= 5.0 * capture_dist else 'transit'

    def observation(self, agent, world):
        self._update_task_phase(agent, world)
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        entity_range = []
        entity_depth = []
        for i, entity in enumerate(world.landmarks):
            if i < world.num_landmarks and agent.name[-1] == entity.name[-1]: 
                if self.target_observation_mode == 'direct_noisy':
                    # Current training uses a direct target-position sensor with
                    # small random measurement error. Legacy LS/KM/PF target
                    # estimators are kept below, but bypassed in this mode.
                    true_rel_pos = np.asarray(entity.state.p_pos - agent.state.p_pos, dtype=float)
                    if self.target_position_noise_std > 0.0:
                        noise = np.random.normal(
                            0.0,
                            self.target_position_noise_std,
                            size=world.dim_p,
                        )
                    else:
                        noise = np.zeros(world.dim_p, dtype=float)
                    measured_rel_pos = true_rel_pos + noise
                    measured_target_pos = np.asarray(agent.state.p_pos, dtype=float) + measured_rel_pos
                    world.landmarks[i + world.num_landmarks].state.p_pos = measured_target_pos
                    world.landmarks[i + world.num_landmarks].state.p_vel = np.asarray(entity.state.p_vel, dtype=float).copy()
                    self.pre_error[i] = self._distance(measured_target_pos, entity.state.p_pos)

                    physical_state = self._agent_3dof_state(agent, world)
                    entity_pos.append(world_to_body_vector(physical_state.psi, measured_rel_pos))
                    entity_range.append(float(np.linalg.norm(measured_rel_pos)))
                    self._move_target(entity, world)
                    continue

                #Update the landmarks_estiamted position using Particle Fileter
                #1:Compute radius between the agent and each landmark
                slant_range = self._distance(entity.state.p_pos, agent.state.p_pos)
                target_depth = entity.landmark_depth/1000. #normalize the target depth ###修改一下，把地标深度改为0，因为最小二乘法里面并没有对深度进行模拟  #  尝试完毕，效果不明显
                slant_range = np.sqrt(slant_range**2+target_depth**2) #add target depth to the range measurement
                # Add some systematic error in the measured range
                slant_range *= 1.01 # where 0.99 = 1% of sound speed difference = 1495 m/s
                # Add some noise in the measured range
                slant_range += np.random.uniform(-0.001, +0.001)
                # Return to a planar range
                slant_range = np.sqrt(abs(slant_range**2-target_depth**2))
                # 计算差值
                dx = entity.state.p_pos[0] - agent.state.p_pos[0]
                dy = entity.state.p_pos[1] - agent.state.p_pos[1]

                # 计算角度（弧度制）
                slant_angle = np.arctan2(dy, dx)
                #set a maximum range between target and agent where the measurement can not be conducted.
                if slant_range > self.set_max_range * 2 or np.random.rand() < self.range_dropping:
                    slant_range = -1.
                    new_range = False
                else:
                    new_range = True
                if self.pre_method == 'KM':
                    world.landmarks_estimated[i].updateEKF(dt=0.01, new_range=new_range, z=slant_range,
                                                                    L_info=[agent.state.p_pos[0],agent.state.p_pos[1]])
                elif self.pre_method == 'PF':
                #2:Update the PF
                    add_pos_error = False
                    if add_pos_error == True:
                        world.landmarks_estimated[i].updatePF(dt=30., new_range=new_range, z=slant_range, myobserver=[agent.state.p_pos[0]+np.random.randn(1).item(0)*3/1000.,0.,agent.state.p_pos[1]+np.random.randn(1).item(0)*3/1000.,0.], update=new_range)
                    else:
                        world.landmarks_estimated[i].updatePF(dt=30., new_range=new_range, z=slant_range, myobserver=[agent.state.p_pos[0],0.,agent.state.p_pos[1],0.], update=new_range)
                elif self.pre_method == 'LS':
                    #2b: Update the LS
                    add_pos_error = False
                    if add_pos_error == True:
                        world.landmarks_estimated[i].updateLS(dt=0.04, new_range=new_range, z=slant_range, myobserver=[agent.state.p_pos[0]+np.random.randn(1).item(0)*3/1000.,0.,agent.state.p_pos[1]+np.random.randn(1).item(0)*3/1000.,0.])
                    else:
                        world.landmarks_estimated[i].updateLS(dt=0.01, new_range=new_range, z=slant_range, myobserver=[agent.state.p_pos[0],0.,agent.state.p_pos[1],0.],L_pos = world.landmarks[i].state.p_pos) ##尝试更改步长，有一定效果但不明显
                else:
                    pass
                #3:Publish the new estimated position
                try:
                    if self.pre_method == 'PF':
                        world.landmarks[i+world.num_landmarks].state.p_pos = [world.landmarks_estimated[i].pfxs[0],world.landmarks_estimated[i].pfxs[2]] #Using PF
                    elif self.pre_method == 'LS' or self.pre_method == 'KM':
                        world.landmarks[i+world.num_landmarks].state.p_pos = [world.landmarks_estimated[i].lsxs[-1][0],world.landmarks_estimated[i].lsxs[-1][2]] #Using LS or KM #将LS或者KM计算的估计地表坐标录入  预测地标实体内
                except (IndexError, AttributeError, TypeError):
                    #An error will be produced if its the initial time and no good range measurement has been conducted yet. In this case, we supose that the target 
                    #is at the same position of the agent.
                    world.landmarks[i+world.num_landmarks].state.p_pos = world.landmarks[i].state.p_pos.copy() #原本是Agent位置，解释是说在初期没有进行良好的距离判断，所以无法得知地标位置，但暂时现在不考虑这个问题，假设地标探测一直良好，所以把agent改为对应的地标landmark(效果变好了一点)
                #Append the position of the landmark to generate the observation state
                #Using the true landmark position
                # entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                #Using the estimated landmark position  从此再无实际位置，实际位置有误差，完全依靠预测
                target_rel_world = world.landmarks[i].state.p_pos - agent.state.p_pos
                physical_state = self._agent_3dof_state(agent, world)
                entity_pos.append(world_to_body_vector(physical_state.psi, target_rel_world))
                #Using the estimated landmark position but without delating the agent position. so it has a global position.
                # entity_pos.append(world.landmarks[i+world.num_landmarks].state.p_pos)
                entity_range.append(slant_range) #将测量距离输入
                entity_depth.append(target_depth)
                error = self._distance(world.landmarks[i + world.num_landmarks].state.p_pos, world.landmarks[i].state.p_pos)
                self.pre_error[i] = error
                self._move_target(entity, world)
        # Split nearby observations by obstacle type.
        dynamic_inside = []
        static_inside = []
        for i,entity_o in enumerate(world.obstacles):
            if i < world.num_obstacles:
                self._add_nearby_dynamic_entity(dynamic_inside, entity_o, agent, world.ob_range)
        for entity_o in getattr(world, 'static_obstacles', []):
            self._add_nearby_entity(static_inside, entity_o, agent, world.ob_range)
        for i, obstacle in enumerate(world.obstacles):
            if i < world.num_obstacles:
                self._move_obstacle(obstacle, world)
        num_dynamic_slots = int(getattr(world, 'num_ob', 0))
        num_static_slots = int(getattr(world, 'num_static_ob_slots', 1))
        dynamic_feature_dim = DYNAMIC_OBS_FEATURE_DIM
        static_feature_dim = STATIC_OBS_FEATURE_DIM
        base_dim = BASE_OBS_DIM
        obs = np.zeros(
            observation_dim(num_dynamic_slots, num_static_slots),
            dtype=np.float32,
        )
        physical_state = self._agent_3dof_state(agent, world)
        physical_config = world.usv_3dof_model.config
        obs[BASE_INDEX['surge_u']] = np.clip(
            physical_state.u / max(float(physical_config.max_surge_speed), 1.0e-9),
            -1.0,
            1.0,
        )
        sway_scale = max(float(physical_config.max_sway_speed), 1.0e-9)
        obs[BASE_INDEX['sway_v']] = np.clip(physical_state.v / sway_scale, -1.0, 1.0)
        obs[BASE_INDEX['yaw_rate_r']] = np.clip(
            physical_state.r / max(float(physical_config.max_yaw_rate), 1.0e-9),
            -1.0,
            1.0,
        )
        obs[BASE_INDEX['sin_heading']] = np.sin(physical_state.psi)
        obs[BASE_INDEX['cos_heading']] = np.cos(physical_state.psi)
        obs[BASE_INDEX['actual_rudder']] = np.clip(
            physical_state.rudder / max(float(physical_config.max_rudder_angle), 1.0e-9),
            -1.0,
            1.0,
        )
        # 执行器状态必须和 3DOF 模型保持同一语义：actual_thrust 是物理
        # 推力经 [min_thrust, max_thrust] 映射到 [-1, 1]；两个 previous_* 是
        # 上一个环境步送入平滑器后的归一化动作，不能误写成当前 raw_action。
        thrust_min = float(getattr(physical_config, 'min_thrust', 0.0))
        thrust_max = float(getattr(physical_config, 'max_thrust', thrust_min + 1.0))
        thrust_span = max(thrust_max - thrust_min, 1.0e-9)
        obs[BASE_INDEX['actual_thrust']] = np.clip(
            2.0 * (float(physical_state.thrust) - thrust_min) / thrust_span - 1.0,
            -1.0,
            1.0,
        )
        previous_smoothed = np.asarray(
            getattr(agent, 'action_smoothed_u', np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if previous_smoothed.size > 0:
            obs[BASE_INDEX['previous_smoothed_thrust']] = np.clip(
                previous_smoothed[0], -1.0, 1.0
            )
        if previous_smoothed.size > 1:
            obs[BASE_INDEX['previous_smoothed_rudder']] = np.clip(
                previous_smoothed[1], -1.0, 1.0
            )

        # A square-map component spans at most one full side length, while its
        # Euclidean distance spans the diagonal. These scales avoid clipping
        # physically valid target observations near opposite corners.
        target_axis_scale = max(2.0 * float(world.map_half_size), 1.0e-9)
        target_distance_scale = max(np.sqrt(2.0) * target_axis_scale, 1.0e-9)
        if entity_pos:
            obs[BASE_INDEX['target_x_body']:BASE_INDEX['target_y_body'] + 1] = np.clip(
                np.asarray(entity_pos[0], dtype=float) / target_axis_scale,
                -1.0,
                1.0,
            )
        if entity_range:
            obs[BASE_INDEX['target_distance']] = float(
                np.clip(entity_range[0] / target_distance_scale, 0.0, 1.0)
            )
        obs[
            BASE_INDEX['boundary_clearance']:BASE_INDEX['boundary_risk_y_body'] + 1
        ] = self._boundary_observation(agent, world, physical_state.psi)

        offset = base_dim
        for slot_i, en in enumerate(dynamic_inside[:num_dynamic_slots]):
            start = offset + slot_i * dynamic_feature_dim
            obs[start:start + dynamic_feature_dim] = self._dynamic_observation_slot(en, agent, world)

        offset += num_dynamic_slots * dynamic_feature_dim
        for slot_i, en in enumerate(static_inside[:num_static_slots]):
            start = offset + slot_i * static_feature_dim
            obs[start:start + static_feature_dim] = self._static_observation_slot(en, agent, world)
        return obs

    def done(self, agent, world):
        # episodes are done based on the agents minimum distance from a landmark.
        # per-agent 终止标记优先：多 USV 场景下避免一个 agent 出局终止全体。
        if not hasattr(agent, 'done_state'):
            # 防腐：多智能体时必须使用 per-agent 标记，禁止回退到模块级全局量。
            if int(getattr(world, 'num_agents', 1)) > 1:
                raise RuntimeError(
                    'per-agent done_state 缺失：num_agents>1 时禁止使用模块级全局 done_state。'
                )
            global done_state
            return bool(done_state)
        return bool(agent.done_state)
        # 终止接口预留：保持达标即终止（默认关闭）
        if bool(getattr(world, 'capture_hold_termination_enabled', False)):
            hold_steps = int(getattr(world, 'capture_hold_success_steps', 30))
            if (getattr(agent, 'has_captured', False)
                    and int(getattr(agent, 'consecutive_hold', 0)) >= hold_steps):
                agent.reward_done_reason = 'hold_success'
                agent.phase = 'capture'
                return True
        return bool(done_state)
