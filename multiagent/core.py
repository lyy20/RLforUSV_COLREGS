import time
from dataclasses import replace

import numpy as np

from dynamics import USV3DOFAction, USV3DOFConfig, USV3DOFModel, USV3DOFState
from safety import (
    COLREGsActionSetConfig,
    COLREGsActionSetFilter,
    COLREGsConfig,
    COLREGsEngine,
    VesselState,
)

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        self.p_pos_origin = None
        # physical velocity
        self.p_vel = None
        self.p_vel_old = None
        self.a_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None
        self.usv_3dof = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        self.max_a_speed = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        # vessel metadata used by rule-level COLREGs evaluation
        self.is_vessel = False
        self.colregs_compliant = False
        self.vessel_role = 'generic'
        self.colregs_last_report = None
        self.colregs_last_reports = ()
        # COLREGs 持久化状态仅保存会遇义务和解除确认计数，不固定具体舵角。
        # 该字段在每回合 reset 时清空，避免不同场景之间串状态。
        self.colregs_rule_state = None
        self.colregs_rule_state_steps = 0
        self.colregs_rule_state_exit_steps = 0
        self.colregs_control_action = 'none'
        self.geometric_avoidance_active = False
        self.geometric_avoidance_target = None
        self.geometric_avoidance_distance = None
        self.action_raw_u = None
        self.action_constrained_u = None
        self.action_rule_u = None
        self.action_smoothed_u = None
        self.action_executed_u = None
        # 动作链四段差异的诊断量，数值均为归一化动作空间中的 L2 范数。
        self.action_clip_correction_norm = 0.0
        self.action_rule_correction_norm = 0.0
        self.action_smooth_correction_norm = 0.0
        self.action_total_correction_norm = 0.0
        self.action_filter_decision = None
        self.action_filter_active = False
        self.action_filter_mode = 'none'

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()
        # action
        self.action = Action()
        # physical motor noise amount
        self.u_noise = None
        # velocity,direction,depth
        self.landmark_vel = 0.
        self.current_landmark_vel = 0.
        self.ra = 0.
        self.landmark_depth = 0.
        self.target_escape_desired_heading = None
        self.target_escape_last_distance = None
        self.target_escape_control_action = 'none'
        self.target_escape_heading_bias = 0.

# properties of Obstacle entities,新增障碍物实体
class Obstacle(Entity):
     def __init__(self):
        super(Obstacle, self).__init__()
        # action
        self.action = Action()
        # physical motor noise amount
        self.u_noise = None
        # velocity,direction,depth
        self.obstacle_vel = 0.
        self.ra = 0.
        self.obstacle_depth = 0.

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # agent has_captured target
        self.has_captured = False
        # Time ( agent has_captured target )
        self.consecutive_hold = 0
# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.obstacles = []
        self.static_obstacles = []
        self.landmarks_estimated = []
        self.obstacles_estimated = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep (in seconds)
        #self.dt = 0.1
        self.dt = 30
        self.step_count = 0
        self.max_episode_steps = None
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # if world is collaborative
        self.num_agents = 3
        self.num_landmarks = 3
        self.num_obstacles = 5
        self.num_static_obstacles = 0
        self.num_static_ob_slots = 1
        self.map_half_size = 2.0
        self.ob_range = 1.
        self.num_ob = 3
        self.collaborative = True
        self.angle = []
        # Temporary baseline: one heading-rate action with fixed forward speed.
        self.control_model = 'heading_rate'
        self.agent_action_dim = 1
        self.agent_nominal_speed = 0.001
        self.heading_rate_gain = 0.3
        self.astar_heading_rate_gain = 1.0
        self.ocean_current_enabled = True
        self.usv_3dof_action_dim = 2
        self.usv_3dof_normalized_action = False
        # The environment stores positions in km and velocities in km/s,
        # while the 3-DOF vessel model uses m and m/s internally.
        self.usv_3dof_m_per_km = 1000.0
        self.usv_3dof_config = USV3DOFConfig()
        self.usv_3dof_model = USV3DOFModel(self.usv_3dof_config)
        self.colregs = COLREGsEngine(COLREGsConfig(risk_time_horizon=240.0, min_cpa_distance=0.20, safety_buffer=0.08))
        self.colregs_enabled = True
        self.colregs_last_reports = []
        # 仅当环境配置显式开启时，动作过滤器才会锁存会遇规则义务。
        self.colregs_persistent_state_enabled = False
        self.colregs_persistent_exit_steps = 3
        self.colregs_persistent_enter_dcpa = 0.20
        self.colregs_persistent_enter_tcpa = 240.0
        self.colregs_persistent_exit_dcpa = 0.24
        self.colregs_persistent_exit_tcpa = 270.0
        self.colregs_persistent_max_hold_steps = 0
        self.agent_colregs_action_filter_enabled = True
        self.agent_colregs_action_filter = COLREGsActionSetFilter()
        self.agent_action_smoothing_alpha = 1.0
        self.scripted_action_is_velocity = False
        self.motion_execution_safety_buffer = 0.0
        self.profile_timing = False
        self.last_step_timing = {}
        self._collision_pairs = None
        self._collision_entity_signature = None
        self._colregs_state_cache = {}
        self._colregs_report_cache = {}
        self._colregs_cache_signature = None
        # sea currents #洋流影响
        self.vel_ocean_current = 0.
        self.angle_ocean_current = 0.

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks + self.obstacles + self.static_obstacles

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self,AStar_now):
        timings = {}
        stage_start = time.perf_counter() if self.profile_timing else None
        self._invalidate_colregs_cache()
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        if self.profile_timing:
            timings['world_scripted_agents'] = float(time.perf_counter() - stage_start)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls # 将action转变为对实体施加的推力 ，当前没考虑任何噪音，所以p_force就等于entity.action.u
        stage_start = time.perf_counter() if self.profile_timing else None
        p_force = self.apply_action_force(p_force)
        if self.profile_timing:
            timings['world_action_force'] = float(time.perf_counter() - stage_start)
        # apply environment forces #环境施加力，基本为碰撞，暂时不考虑
        stage_start = time.perf_counter() if self.profile_timing else None
        p_force = self.apply_environment_force(p_force)
        if self.profile_timing:
            timings['world_environment_force'] = float(time.perf_counter() - stage_start)
        # integrate physical state
        stage_start = time.perf_counter() if self.profile_timing else None
        self.integrate_state(p_force,AStar_now)
        if self.profile_timing:
            timings['world_integrate_state'] = float(time.perf_counter() - stage_start)
        # update agent state
        stage_start = time.perf_counter() if self.profile_timing else None
        for agent in self.agents:
            self.update_agent_state(agent)
        if self.profile_timing:
            timings['world_update_agent_state'] = float(time.perf_counter() - stage_start)
        if self.colregs_enabled:
            stage_start = time.perf_counter() if self.profile_timing else None
            self.evaluate_all_colregs()
            if self.profile_timing:
                timings['world_colregs'] = float(time.perf_counter() - stage_start)
        self.last_step_timing = timings

    # gather agent action forces
    def _legacy_apply_action_force(self, p_force):
        # set applied forces
        for i,entity in enumerate(self.entities):
            if 'agent' in entity.name:
                if entity.movable:#给实体的运动增加噪音，代表着运动的偏差，但事实上现在实体还没加运动噪声，所以目前没效果！
                    noise = np.random.randn(*entity.action.u.shape) * entity.u_noise if entity.u_noise else 0.0
                    p_force[i] = entity.action.u + noise 
            if 'landmark' in entity.name or 'obstacle' in entity.name:
                if entity.movable:
                    noise = np.random.randn(*entity.action.u.shape) * entity.u_noise if entity.u_noise else 0.0
                    p_force[i] = entity.action.u + noise 
        return p_force

    def apply_action_force(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            if 'agent' in entity.name:
                action_u = self._filter_agent_action(entity)
            elif 'landmark' in entity.name or 'obstacle' in entity.name:
                action_u = entity.action.u
            else:
                continue
            noise = np.random.randn(*action_u.shape) * entity.u_noise if entity.u_noise else 0.0
            p_force[i] = action_u + noise
        return p_force

    def _filter_agent_action(self, agent):
        raw_action = np.asarray(agent.action.u, dtype=float).reshape(-1)
        if self.control_model != 'usv_3dof':
            # 非 3DOF 旧控制路径没有规则投影和平滑器，四个动作层级退化
            # 为同一个动作，同时显式初始化新字段，避免 info 中残留上一步数据。
            agent.action_raw_u = raw_action.copy()
            agent.action_constrained_u = raw_action.copy()
            agent.action_rule_u = raw_action.copy()
            agent.action_smoothed_u = raw_action.copy()
            agent.action_executed_u = raw_action.copy()
            agent.action_clip_correction_norm = 0.0
            agent.action_rule_correction_norm = 0.0
            agent.action_smooth_correction_norm = 0.0
            agent.action_total_correction_norm = 0.0
            agent.action_filter_decision = None
            agent.action_filter_active = False
            agent.action_filter_mode = 'pass_through'
            return raw_action

        decision = self.agent_colregs_action_filter.filter_action(self, agent, raw_action)
        # 动作执行链：raw -> constrained -> rule -> smoothed -> actuator。
        # ``rule_action`` 是规则层的纯输出，不能用最终执行动作替代，否则
        # 规则模仿损失会把平滑器行为错误地当成海事规则教师信号。
        rule_action = np.asarray(decision.rule_action, dtype=float)
        previous_command = getattr(agent, 'action_smoothed_u', None)
        alpha = float(np.clip(self.agent_action_smoothing_alpha, 0.0, 1.0))
        if previous_command is None or np.asarray(previous_command).size != rule_action.size:
            smoothed_action = rule_action.copy()
        else:
            previous_command = np.asarray(previous_command, dtype=float).reshape(-1)
            smoothed_action = alpha * rule_action + (1.0 - alpha) * previous_command
        constrained_action = np.asarray(decision.constrained_action, dtype=float)
        raw_action = np.asarray(decision.raw_action, dtype=float)
        agent.action_smoothed_u = smoothed_action.copy()
        agent.action_raw_u = np.asarray(decision.raw_action, dtype=float)
        agent.action_constrained_u = constrained_action.copy()
        agent.action_rule_u = rule_action.copy()
        agent.action_executed_u = smoothed_action.copy()
        # 这四个范数用于 TensorBoard 和论文诊断，不参与动作计算。
        agent.action_clip_correction_norm = float(np.linalg.norm(constrained_action - raw_action))
        agent.action_rule_correction_norm = float(np.linalg.norm(rule_action - constrained_action))
        agent.action_smooth_correction_norm = float(np.linalg.norm(smoothed_action - rule_action))
        agent.action_total_correction_norm = float(np.linalg.norm(smoothed_action - raw_action))
        # 过滤器返回的 decision 在过滤阶段还不知道 EMA 结果；把兼容字段
        # applied_action 更新为最终平滑动作，同时保留 rule_action 不变。
        agent.action_filter_decision = replace(
            decision,
            applied_action=tuple(float(value) for value in smoothed_action),
        )
        agent.action_filter_active = bool(decision.active)
        agent.action_filter_mode = decision.mode
        agent.action.u = smoothed_action.copy()
        return smoothed_action

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # 只缓存可能产生物理力的候选对，实际碰撞力公式保持不变。
        entities = self.entities
        for a, b in self._collision_candidate_pairs():
            entity_a = entities[a]
            entity_b = entities[b]
            [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
            if(f_a is not None):
                if(p_force[a] is None): p_force[a] = 0.0
                p_force[a] = f_a + p_force[a] 
            if(f_b is not None):
                if(p_force[b] is None): p_force[b] = 0.0
                p_force[b] = f_b + p_force[b]        
        return p_force

    def _collision_candidate_pairs(self):
        """缓存实体结构，跳过静态-静态等必然没有物理力的组合。"""
        entities = self.entities
        signature = tuple(
            (id(entity), bool(entity.collide), bool(entity.movable))
            for entity in entities
        )
        if signature != self._collision_entity_signature:
            self._collision_entity_signature = signature
            self._collision_pairs = tuple(
                (a, b)
                for a, entity_a in enumerate(entities)
                for b, entity_b in enumerate(entities[a + 1:], start=a + 1)
                if entity_a.collide
                and entity_b.collide
                and (entity_a.movable or entity_b.movable)
            )
        return self._collision_pairs or ()

    # integrate physical state
    def integrate_state(self, p_force,AStar_now):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)#阻尼，代表世界的阻力
            
            if 'landmark' in entity.name or 'obstacle' in entity.name:
                #if entity is a landmark / obstacle (x-y force applyied independently)
                if self.scripted_action_is_velocity and p_force[i] is not None:
                    command_velocity = np.asarray(p_force[i], dtype=float).reshape(-1)[:self.dim_p]
                    entity.state.p_vel = command_velocity.copy()
                elif (p_force[i] is not None):
                    entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
                if entity.max_speed is not None:
                    speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                    if speed > entity.max_speed:
                        entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(entity.state.p_vel[1])) * entity.max_speed
                candidate_pos = entity.state.p_pos + entity.state.p_vel * self.dt
                if (
                    getattr(entity, 'vessel_role', '') in {'dynamic_obstacle', 'target'}
                    and not self._scripted_motion_is_safe(entity, candidate_pos)
                ):
                    # Last-resort guard: proactive COLREGs/geometric avoidance
                    # should turn first; stopping here prevents hull overlap.
                    entity.state.p_vel = np.zeros(self.dim_p, dtype=float)
                    entity.action.u = np.zeros(self.dim_p, dtype=float)
                    entity.geometric_avoidance_active = True
                    entity.geometric_avoidance_target = 'motion_execution_guard'
                    entity.geometric_avoidance_distance = 0.0
                    candidate_pos = np.asarray(entity.state.p_pos, dtype=float).copy()
                entity.state.p_pos = candidate_pos
                self._keep_scripted_entity_inside_map(entity)
            
            if 'agent' in entity.name:
                self._integrate_agent_state(entity, i, p_force[i], AStar_now)

    def _scripted_motion_is_safe(self, entity, candidate_pos):
        if not self._position_inside_map(candidate_pos, entity.size):
            return False
        start = np.asarray(entity.state.p_pos, dtype=float)
        end = np.asarray(candidate_pos, dtype=float)
        segment = end - start
        segment_norm_sq = float(np.dot(segment, segment))
        ignored_roles = {'target_estimation', 'obstacle_estimation'}
        for other in self.entities:
            if other is entity or other.state.p_pos is None:
                continue
            if getattr(other, 'vessel_role', '') in ignored_roles:
                continue
            other_pos = np.asarray(other.state.p_pos, dtype=float)
            if segment_norm_sq <= 1.0e-15:
                closest = start
            else:
                fraction = float(np.clip(np.dot(other_pos - start, segment) / segment_norm_sq, 0.0, 1.0))
                closest = start + fraction * segment
            required = (
                float(entity.size)
                + float(other.size)
                + float(self.motion_execution_safety_buffer)
            )
            if np.linalg.norm(closest - other_pos) <= required:
                return False
        return True

    def _position_inside_map(self, position, radius=0.0):
        if not hasattr(self, 'map_half_size') or self.map_half_size is None:
            return True
        limit = max(float(self.map_half_size) - float(radius), 0.0)
        return bool(np.all(np.abs(np.asarray(position, dtype=float)) <= limit))

    def _keep_scripted_entity_inside_map(self, entity):
        if not hasattr(self, 'map_half_size') or self.map_half_size is None:
            entity.boundary_violation = False
            return
        limit = max(float(self.map_half_size) - float(entity.size), 0.0)
        if entity.state.p_pos is None or entity.state.p_vel is None:
            entity.boundary_violation = False
            return
        # Boundary clamp is intentionally disabled for training. Once an agent
        # crosses the valid map boundary, the scenario reward should terminate
        # the episode instead of silently pulling the state back inside.
        entity.boundary_violation = bool(np.any(np.abs(entity.state.p_pos) > limit))

        # Historical clamp logic kept as reference in case we later need a
        # non-terminal "soft wall" mode for scripted preview or curriculum.
        # clamped = False
        # for axis in range(self.dim_p):
        #     if entity.state.p_pos[axis] > limit:
        #         entity.state.p_pos[axis] = limit
        #         entity.state.p_vel[axis] = -abs(entity.state.p_vel[axis]) * 0.2
        #         clamped = True
        #     elif entity.state.p_pos[axis] < -limit:
        #         entity.state.p_pos[axis] = -limit
        #         entity.state.p_vel[axis] = abs(entity.state.p_vel[axis]) * 0.2
        #         clamped = True
        # if clamped and getattr(entity.state, 'usv_3dof', None) is not None:
        #     entity.state.usv_3dof.x = float(entity.state.p_pos[0])
        #     entity.state.usv_3dof.y = float(entity.state.p_pos[1])

    def _integrate_agent_state(self, entity, index, action_force, AStar_now):
        if self.control_model == 'heading_rate':
            self._integrate_heading_rate_agent(entity, index, action_force, AStar_now)
        elif self.control_model == 'usv_3dof':
            self._integrate_usv_3dof_agent(entity, index, action_force)
        else:
            raise NotImplementedError(f"Unsupported control model: {self.control_model}")
        self._keep_scripted_entity_inside_map(entity)

    def _integrate_heading_rate_agent(self, entity, index, action_force, AStar_now):
        if action_force is not None:
            turn_action = np.asarray(action_force).reshape(-1)[0]
            turn_gain = self.astar_heading_rate_gain if AStar_now else self.heading_rate_gain
            self.angle[index] = self._wrap_angle(self.angle[index] + turn_action * turn_gain)

        speed = max(self.agent_nominal_speed, 0.)
        heading_velocity = np.array([
            speed * np.cos(self.angle[index]),
            speed * np.sin(self.angle[index]),
        ])
        entity.state.p_pos += heading_velocity * self.dt
        entity.state.p_vel = heading_velocity
        if self.ocean_current_enabled:
            current_velocity = np.array([
                self.vel_ocean_current * np.cos(self.angle_ocean_current),
                self.vel_ocean_current * np.sin(self.angle_ocean_current),
            ])
            entity.state.p_pos += current_velocity * self.dt

    @staticmethod
    def _wrap_angle(angle):
        if angle > np.pi * 2.:
            angle -= np.pi * 2.
        if angle < -np.pi * 2:
            angle += np.pi * 2
        return angle

    def _integrate_usv_3dof_agent(self, entity, index, action_force):
        state = self._ensure_usv_3dof_state(entity, index)
        self._sync_usv_3dof_current()
        if self.usv_3dof_normalized_action:
            action = self._coerce_usv_3dof_action_array(action_force)
            normalized_action = True
        else:
            action = self._coerce_usv_3dof_action_array(action_force)
            normalized_action = False

        substeps = self._usv_3dof_substeps()
        next_state = self.usv_3dof_model.advance_substeps(
            state,
            action,
            normalized_action=normalized_action,
            substeps=substeps,
        )

        entity.state.usv_3dof = next_state
        scale = self.usv_3dof_m_per_km
        entity.state.p_pos = np.array([next_state.x, next_state.y]) / scale
        x_dot, y_dot, _ = self.usv_3dof_model.earth_fixed_velocity(
            next_state.psi,
            next_state.u,
            next_state.v,
            next_state.r,
        )
        entity.state.p_vel = np.array([x_dot, y_dot]) / scale
        self.angle[index] = next_state.psi

    def _ensure_usv_3dof_state(self, entity, index):
        state = entity.state.usv_3dof
        if state is None:
            # Gym queries observation dimensions while the environment is being
            # constructed, before scenario.reset_world() has assigned random
            # headings. Use an explicit zero-heading physical state for that
            # bootstrap observation; reset_world() replaces it before rollout.
            while len(self.angle) <= index:
                self.angle.append(0.0)
            psi = float(self.angle[index])
            scale = self.usv_3dof_m_per_km
            world_vel = np.asarray(entity.state.p_vel, dtype=float) * scale
            state = USV3DOFState(
                x=float(entity.state.p_pos[0]) * scale,
                y=float(entity.state.p_pos[1]) * scale,
                psi=psi,
                u=float(world_vel[0] * np.cos(psi) + world_vel[1] * np.sin(psi)),
                v=float(-world_vel[0] * np.sin(psi) + world_vel[1] * np.cos(psi)),
                r=0.0,
            )
            entity.state.usv_3dof = state
        return state

    def _usv_3dof_substeps(self):
        cfg_dt = max(float(self.usv_3dof_model.config.dt), 1.0e-9)
        return max(1, int(round(float(self.dt) / cfg_dt)))

    def _coerce_usv_3dof_action_array(self, action_force):
        if action_force is None:
            return np.zeros(self.usv_3dof_action_dim)
        data = np.asarray(action_force, dtype=float).reshape(-1)
        if data.size < self.usv_3dof_action_dim:
            padded = np.zeros(self.usv_3dof_action_dim)
            padded[:data.size] = data
            return padded
        return data[:self.usv_3dof_action_dim]

    def _coerce_usv_3dof_action(self, action_force):
        data = self._coerce_usv_3dof_action_array(action_force)
        return USV3DOFAction(thrust=float(data[0]), rudder=float(data[1]))

    def _sync_usv_3dof_current(self):
        cfg = self.usv_3dof_model.config
        cfg.current_speed = (
            self.vel_ocean_current * self.usv_3dof_m_per_km
            if self.ocean_current_enabled
            else 0.0
        )
        cfg.current_direction = self.angle_ocean_current

    def vessel_state_from_entity(self, entity):
        """Convert a world entity to a COLREGs vessel state."""
        cached = self._colregs_state_cache.get(id(entity))
        if cached is not None:
            return cached
        position = np.asarray(entity.state.p_pos, dtype=float)
        velocity = np.asarray(entity.state.p_vel, dtype=float)
        heading = self._entity_heading(entity, velocity)
        speed = self._entity_speed(entity, velocity)
        if np.linalg.norm(velocity) <= 1.0e-9 and speed > 0.0:
            velocity = np.array([speed * np.cos(heading), speed * np.sin(heading)])
        state = VesselState.from_kinematics(
            vessel_id=entity.name,
            position=position,
            velocity=velocity,
            heading=heading,
            speed=speed,
            radius=entity.size,
            role=getattr(entity, 'vessel_role', 'generic'),
            colregs_compliant=getattr(entity, 'colregs_compliant', False),
        )
        if self._colregs_cache_signature is not None:
            self._colregs_state_cache[id(entity)] = state
        return state

    def evaluate_colregs_between(self, ownship_entity, target_entity):
        self._prepare_colregs_cache([ownship_entity, target_entity])
        cache_key = (id(ownship_entity), id(target_entity))
        cached_report = self._colregs_report_cache.get(cache_key)
        if cached_report is not None:
            return cached_report
        ownship = self.vessel_state_from_entity(ownship_entity)
        target = self.vessel_state_from_entity(target_entity)
        report = self.colregs.evaluate(ownship, target)
        self._colregs_report_cache[cache_key] = report
        return report

    def evaluate_colregs_for_entity(self, ownship_entity, target_entities=None):
        if target_entities is None:
            target_entities = [entity for entity in self.entities if entity is not ownship_entity]
        vessel_targets = [
            entity for entity in target_entities
            if entity is not ownship_entity and getattr(entity, 'is_vessel', False)
        ]
        self._prepare_colregs_cache([ownship_entity] + vessel_targets)
        ownship = self.vessel_state_from_entity(ownship_entity)
        reports = []
        for target in vessel_targets:
            cache_key = (id(ownship_entity), id(target))
            report = self._colregs_report_cache.get(cache_key)
            if report is None:
                report = self.colregs.evaluate(
                    ownship,
                    self.vessel_state_from_entity(target),
                )
                self._colregs_report_cache[cache_key] = report
            reports.append(report)
        return tuple(reports)

    def evaluate_all_colregs(self, entities=None):
        vessel_entities = [
            entity for entity in (entities or self.entities)
            if getattr(entity, 'is_vessel', False) and entity.state.p_pos is not None
        ]
        self._prepare_colregs_cache(vessel_entities)
        reports = []
        for ownship in vessel_entities:
            reports.extend(self.evaluate_colregs_for_entity(ownship, vessel_entities))
        self.colregs_last_reports = tuple(reports)
        return self.colregs_last_reports

    def _invalidate_colregs_cache(self):
        self._colregs_state_cache = {}
        self._colregs_report_cache = {}
        self._colregs_cache_signature = None

    @staticmethod
    def _colregs_entity_signature(entities):
        signature = []
        for entity in entities:
            if entity is None or entity.state.p_pos is None:
                continue
            position = tuple(np.asarray(entity.state.p_pos, dtype=float).reshape(-1))
            velocity = tuple(
                np.asarray(
                    entity.state.p_vel if entity.state.p_vel is not None else np.zeros(2),
                    dtype=float,
                ).reshape(-1)
            )
            signature.append(
                (
                    id(entity),
                    position,
                    velocity,
                    float(getattr(entity, 'size', 0.0)),
                )
            )
        return tuple(signature)

    def _prepare_colregs_cache(self, entities):
        valid_entities = [
            entity for entity in entities
            if entity is not None
            and getattr(entity.state, 'p_pos', None) is not None
            and getattr(entity, 'is_vessel', False)
        ]
        signature = self._colregs_entity_signature(valid_entities)
        if signature != self._colregs_cache_signature:
            self._colregs_cache_signature = signature
            self._colregs_state_cache = {}
            self._colregs_report_cache = {}

    def dynamic_obstacle_vessels(self):
        return tuple(
            obstacle for obstacle in self.obstacles[:self.num_obstacles]
            if getattr(obstacle, 'is_vessel', False) and obstacle.movable
        )

    def _entity_heading(self, entity, velocity):
        if entity in self.agents:
            index = self.agents.index(entity)
            if index < len(self.angle):
                return float(self.angle[index])
        if hasattr(entity, 'ra'):
            return float(entity.ra)
        if np.linalg.norm(velocity) > 1.0e-9:
            return float(np.arctan2(velocity[1], velocity[0]))
        return 0.0

    @staticmethod
    def _entity_speed(entity, velocity):
        # Prefer the actual integrated velocity. Nominal obstacle/landmark
        # speeds can differ after COLREGs, acceleration and safety filtering.
        actual_speed = float(np.linalg.norm(np.asarray(velocity, dtype=float)))
        if actual_speed > 1.0e-9:
            return actual_speed
        if hasattr(entity, 'current_landmark_vel'):
            return float(entity.current_landmark_vel)
        if hasattr(entity, 'obstacle_vel'):
            return float(entity.obstacle_vel)
        if hasattr(entity, 'landmark_vel'):
            return float(entity.landmark_vel)
        return actual_speed

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities·
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
