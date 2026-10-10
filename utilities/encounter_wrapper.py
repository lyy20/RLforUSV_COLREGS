
# -*- coding: utf-8 -*-
"""P0-1：会遇生成包装器（在 env.reset() 之后重定位动态船，保证会遇）。

为什么用包装器而不是改场景内部：
    场景存在多条重置路径（_reset_world_once / _reset_scaled_small_layout），
    内部注入容易落到未执行分支；包装器作用在**最终 world 状态**上，稳定可靠，
    且与参考实现 RL_CORALL_extension 的 RandomEncounterEnv(gym.Wrapper) 思路一致。

数学（期望 TCPA = T、DCPA = d）：
    v_rel = v_o - v_s,  n ⟂ v_rel (单位法向)
    r = p_o(0) - p_s(0) = d·n - v_rel·T
    ⇒ TCPA = -(r·v_rel)/|v_rel|² = T,  DCPA = |r + v_rel·T| = d
"""
from __future__ import annotations
import numpy as np


class EncounterResetWrapper(object):
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = dict(cfg or {})

    # --- geometry construction -------------------------------------------------
    def _sample_state(self, p_s, psi_s, u_s, size, occupied):
        cfg = self.cfg
        w_ho = max(float(cfg.get("encounter_head_on_weight", 1.0)), 0.0)
        w_cr = max(float(cfg.get("encounter_crossing_weight", 1.0)), 0.0)
        w_ot = max(float(cfg.get("encounter_overtaking_weight", 1.0)), 0.0)
        w = np.array([w_ho, w_cr, w_ot], dtype=float)
        if w.sum() <= 0:
            w = np.ones(3)
        kind = int(np.random.choice(3, p=w / w.sum()))
        u_min = float(cfg.get("dynamic_obstacle_min_speed", 0.001))
        u_max = float(cfg.get("dynamic_obstacle_max_speed", 0.002))
        if kind == 0:      # 对遇
            psi_o = psi_s + np.pi + np.deg2rad(np.random.uniform(-15.0, 15.0))
            u_o = np.random.uniform(u_min, u_max)
        elif kind == 1:    # 交叉
            side = 1.0 if np.random.rand() < 0.5 else -1.0
            psi_o = psi_s + side * np.deg2rad(np.random.uniform(45.0, 135.0))
            u_o = np.random.uniform(u_min, u_max)
        else:              # 追越
            psi_o = psi_s + np.deg2rad(np.random.uniform(-20.0, 20.0))
            u_o = min(np.random.uniform(u_min, u_max), max(u_s * 0.8, u_min))
        v_s = np.array([np.cos(psi_s), np.sin(psi_s)]) * u_s
        v_o = np.array([np.cos(psi_o), np.sin(psi_o)]) * u_o
        v_rel = v_o - v_s
        vv = float(np.dot(v_rel, v_rel))
        if vv <= 1e-16:
            return None
        T = np.random.uniform(float(cfg.get("encounter_tcpa_min_s", 60.0)),
                              float(cfg.get("encounter_tcpa_max_s", 120.0)))
        d = np.random.uniform(float(cfg.get("encounter_dcpa_min_m", 10.0)),
                              float(cfg.get("encounter_dcpa_max_m", 60.0))) / 1000.0
        n_hat = np.array([-v_rel[1], v_rel[0]]) / np.sqrt(vv)
        if np.random.rand() < 0.5:
            n_hat = -n_hat
        p_o = np.asarray(p_s, float) + d * n_hat - v_rel * T
        half = float(cfg.get("map_half_size", 1.0))
        if np.any(np.abs(p_o) + size > half):
            return None
        clearance = float(cfg.get("vessel_static_clearance", 0.028))
        for (q, r) in occupied:
            if float(np.linalg.norm(p_o - q)) <= size + r + clearance:
                return None
        # M2 接受判据自检
        r_rel = p_o - np.asarray(p_s, float)
        tcpa = -float(np.dot(r_rel, v_rel)) / vv
        dcpa = float(np.linalg.norm(r_rel + v_rel * tcpa))
        if tcpa > float(cfg.get("encounter_accept_tcpa_max_s", 120.0)) + 1e-6:
            return None
        if dcpa * 1000.0 > float(cfg.get("encounter_accept_dcpa_max_m", 60.0)) + 1e-6:
            return None
        return p_o, psi_o, u_o

    # --- motion authority ------------------------------------------------------
    def _enforce_encounter_motion(self):
        """把被标记的会遇船速度写回设计值（场景的逐帧指令会覆盖它）。

        注意：这里只负责「保持设计速度」，运动方式（静止/逃逸/航点/随机/蛇形/绕行）
        后续在此处按 mode 扩展。
        """
        world = getattr(self.env, 'world', None)
        if world is None:
            return
        n_ob = int(getattr(world, 'num_obstacles', 0))
        for i in range(max(n_ob, 0)):
            ob = world.obstacles[i]
            if not getattr(ob, 'encounter_generated', False):
                continue
            mode = getattr(ob, 'encounter_motion_mode', 'constant')
            if mode == 'stationary':
                ob.state.p_vel = np.zeros(2)
                continue
            psi = float(getattr(ob, 'ra', 0.0))
            u = float(getattr(ob, 'obstacle_vel', 0.0))
            if u > 0.0:
                ob.state.p_vel = np.array([np.cos(psi), np.sin(psi)]) * u

    def step(self, action):
        out = self.env.step(action)
        self._enforce_encounter_motion()
        return out


    # --- env interface ---------------------------------------------------------
    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self.relocate()
        return out

    def relocate(self):
        world = self.env.world
        if world is None:
            return 0
        cfg = self.cfg
        n_ob = int(getattr(world, "num_obstacles", 0))
        if n_ob <= 0:
            return 0
        agent = world.agents[0]
        target = world.landmarks[0]
        p_s = np.asarray(agent.state.p_pos, float)
        route = np.asarray(target.state.p_pos, float) - p_s
        norm = float(np.linalg.norm(route))
        if norm < 1e-9:
            return 0
        psi_s = float(np.arctan2(route[1], route[0]))
        u_s = float(getattr(world, "agent_nominal_speed", 0.001))
        occupied = [(np.asarray(e.state.p_pos, float), float(e.size))
                    for e in list(world.agents) + list(world.landmarks[: world.num_landmarks])
                    + list(world.static_obstacles)]
        done = 0
        max_try = int(round(float(cfg.get("encounter_max_resample", 200.0))))
        for i in range(n_ob):
            ob = world.obstacles[i]
            sampled = None
            for _ in range(max(max_try, 1)):
                sampled = self._sample_state(p_s, psi_s, u_s, float(ob.size), occupied)
                if sampled is not None:
                    break
            if sampled is None:
                continue
            p_o, psi_o, u_o = sampled
            ob.state.p_pos = np.asarray(p_o, float)
            ob.ra = float(psi_o)
            ob.obstacle_vel = float(u_o)
            ob.max_speed = float(u_o)
            ob.state.p_vel = np.array([np.cos(psi_o), np.sin(psi_o)]) * float(u_o)
            ob.encounter_generated = True
            occupied.append((np.asarray(p_o, float), float(ob.size)))
            est = world.obstacles[i + n_ob] if len(world.obstacles) > i + n_ob else None
            if est is not None:
                est.state.p_pos = np.asarray(p_o, float)
                est.state.p_vel = ob.state.p_vel.copy()
                est.size = ob.size
            done += 1
        return done

    def __getattr__(self, item):
        return getattr(self.env, item)
