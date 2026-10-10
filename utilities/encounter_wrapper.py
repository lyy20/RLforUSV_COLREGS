
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

    # --- rollout verification (A) ----------------------------------------------
    def _snapshot_world(self):
        world = getattr(self.env, 'world', None)
        if world is None:
            return None
        ents = []
        for e in list(getattr(world, 'agents', []) or []) + list(getattr(world, 'landmarks', []) or []) + list(getattr(world, 'obstacles', []) or []) + list(getattr(world, 'static_obstacles', []) or []):
            item = {'e': e,
                    'pos': np.array(e.state.p_pos, float, copy=True),
                    'vel': np.array(e.state.p_vel, float, copy=True)}
            st = getattr(e.state, 'usv_3dof', None)
            if st is not None:
                item['st'] = dict(st.__dict__) if hasattr(st, '__dict__') else None
            ents.append(item)
        obs = []
        for o in list(getattr(world, 'obstacles', []) or []):
            obs.append({'o': o, 'ra': getattr(o, 'ra', None), 'u': getattr(o, 'obstacle_vel', None),
                        'mode': getattr(o, 'encounter_motion_mode', None),
                        'ms': getattr(o, 'encounter_mode_state', None)})
        return {'ents': ents, 'obs': obs, 't': getattr(self, '_t', 0),
                'step_count': int(getattr(world, 'step_count', 0)),
                'rng': np.random.get_state()}

    def _restore_world(self, snap):
        if not snap:
            return
        for item in snap['ents']:
            e = item['e']
            e.state.p_pos = np.array(item['pos'], float, copy=True)
            e.state.p_vel = np.array(item['vel'], float, copy=True)
            st = getattr(e.state, 'usv_3dof', None)
            if st is not None and item.get('st'):
                for k, v in item['st'].items():
                    try:
                        setattr(st, k, v)
                    except Exception:
                        pass
        for item in snap['obs']:
            o = item['o']
            if item['ra'] is not None:
                o.ra = item['ra']
            if item['u'] is not None:
                o.obstacle_vel = item['u']
            if item['mode'] is not None:
                o.encounter_motion_mode = item['mode']
            if item['ms'] is not None:
                o.encounter_mode_state = item['ms']
        self._t = snap.get('t', getattr(self, '_t', 0))
        try:
            world = getattr(self.env, 'world', None)
            if world is not None and 'step_count' in snap:
                world.step_count = int(snap['step_count'])
            if 'rng' in snap:
                np.random.set_state(snap['rng'])
        except Exception:
            pass

    def _extrapolated_cpa(self, world):
        agent = world.agents[0]
        ps = np.asarray(agent.state.p_pos, float)
        vs = np.asarray(agent.state.p_vel, float)
        out = {}
        n_ob = int(getattr(world, 'num_obstacles', 0))
        for i in range(max(n_ob, 0)):
            ob = world.obstacles[i]
            if not getattr(ob, 'encounter_generated', False):
                continue
            r = np.asarray(ob.state.p_pos, float) - ps
            v = np.asarray(ob.state.p_vel, float) - vs
            vv = float(np.dot(v, v))
            if vv <= 1e-16:
                out[i] = float(np.linalg.norm(r))
            else:
                t = max(0.0, -float(np.dot(r, v)) / vv)
                out[i] = float(np.linalg.norm(r + v * t))
        return out

    def _verify_placement(self):
        steps = int(self.cfg.get('encounter_verify_steps', 40))
        tol_rel = float(self.cfg.get('encounter_verify_tol_rel', 2.0))
        tol_abs = float(self.cfg.get('encounter_verify_tol_abs_m', 50.0)) / 1000.0
        world = getattr(self.env, 'world', None)
        if world is None or steps <= 0:
            return True
        snap = self._snapshot_world()
        world = getattr(self.env, 'world', None)
        try:
            if bool(self.cfg.get('encounter_verify_fast', True)):
                world._fast_verify_rollout = True
        except Exception:
            pass
        for _ in range(steps):
            own = world.agents[0]
            tgt = world.landmarks[0]
            v_now = np.asarray(own.state.p_vel, float)
            sp = float(np.linalg.norm(v_now))
            psi = float(np.arctan2(v_now[1], v_now[0])) if sp > 1e-9 else 0.0
            des = np.asarray(tgt.state.p_pos, float) - np.asarray(own.state.p_pos, float)
            err = (float(np.arctan2(des[1], des[0])) - psi + np.pi) % (2 * np.pi) - np.pi
            act = np.array([[1.0, float(np.clip(2.0 * err, -1.0, 1.0))]])
            try:
                self.env.step(act)
            except Exception:
                break
            self._t += 1
            self._enforce_encounter_motion()
        cpa = self._extrapolated_cpa(world)
        ok = True
        for i, val in cpa.items():
            ob = world.obstacles[i]
            design = float(getattr(ob, 'encounter_design_dcpa', 0.0))
            if design <= 0.0:
                continue
            ceiling = float(self.cfg.get('encounter_verify_max_cpa_m', 300.0)) / 1000.0
            if val > max(design * tol_rel + tol_abs, ceiling):
                ok = False
                break
        try:
            world._fast_verify_rollout = False
        except Exception:
            pass
        self._restore_world(snap)
        return ok

    # --- target motion modes ---------------------------------------------------
    MODES = ('stationary', 'constant', 'waypoints', 'escape', 'random', 'weaving', 'orbit')

    def _mode_weights(self):
        raw = ''
        try:
            raw = ','.join('%s:%.3f' % (m, float(self.cfg.get('target_motion_weight_' + m, 1.0))) for m in self.MODES)
        except Exception:
            raw = ''
        w = {}
        for part in raw.split(','):
            if ':' in part:
                k, v = part.split(':', 1)
                k = k.strip()
                try:
                    w[k] = max(float(v), 0.0)
                except ValueError:
                    pass
        if not w:
            w = {m: 1.0 for m in self.MODES}
        return w

    def _randomize_initial_heading(self):
        if not bool(self.cfg.get('initial_heading_random', False)):
            return
        world = getattr(self.env, 'world', None)
        if world is None or not getattr(world, 'agents', None):
            return
        agent = world.agents[0]
        st = getattr(agent.state, 'usv_3dof', None)
        psi = float(np.random.uniform(-np.pi, np.pi))
        if st is not None and hasattr(st, 'psi'):
            st.psi = psi
            if hasattr(st, 'u'):
                st.u = 0.0
            if hasattr(st, 'v'):
                st.v = 0.0
            if hasattr(st, 'r'):
                st.r = 0.0
        agent.state.p_vel = np.zeros(2)

    def _assign_motion_modes(self):
        world = getattr(self.env, 'world', None)
        if world is None:
            return
        n_ob = int(getattr(world, 'num_obstacles', 0))
        w = self._mode_weights()
        modes = [m for m in self.MODES if w.get(m, 0.0) > 0.0]
        probs = np.array([w[m] for m in modes], dtype=float)
        probs = probs / probs.sum()
        half = float(self.cfg.get('map_half_size', 1.0))
        base_u = float(self.cfg.get('dynamic_obstacle_min_speed', 0.0012))
        max_u = float(self.cfg.get('dynamic_obstacle_max_speed', 0.0022))
        for i in range(max(n_ob, 0)):
            ob = world.obstacles[i]
            if not getattr(ob, 'encounter_generated', False):
                continue
            mode = str(np.random.choice(modes, p=probs))
            ob.encounter_motion_mode = mode
            ob.obstacle_vel = float(np.clip(getattr(ob, 'obstacle_vel', base_u), base_u, max_u))
            st = {'base_psi': float(getattr(ob, 'ra', 0.0)),
                  'base_u': float(ob.obstacle_vel),
                  'u_min': base_u, 'u_max': max_u,
                  'omega': float(np.random.uniform(2.0, 6.0) * np.pi / 180.0) * (1.0 if np.random.rand() < 0.5 else -1.0),
                  'amp': float(np.random.uniform(10.0, 30.0) * np.pi / 180.0),
                  'period': float(np.random.uniform(15.0, 40.0)),
                  'next_change': float(np.random.uniform(5.0, 20.0)),
                  'wps': [], 'wp_i': 0}
            for _ in range(2):
                st['wps'].append(np.random.uniform(-half * 0.8, half * 0.8, size=2))
            ob.encounter_mode_state = st

    def _mode_velocity(self, ob, dt):
        st = getattr(ob, 'encounter_mode_state', None) or {}
        mode = getattr(ob, 'encounter_motion_mode', 'constant')
        psi = float(getattr(ob, 'ra', 0.0))
        u = float(getattr(ob, 'obstacle_vel', 0.0))
        if mode == 'stationary':
            return np.zeros(2)
        if mode == 'constant':
            return np.array([np.cos(psi), np.sin(psi)]) * u
        if mode == 'weaving':
            psi = st.get('base_psi', psi) + st.get('amp', 0.2) * np.sin(2.0 * np.pi * self._t * dt / max(st.get('period', 20.0), 1e-6))
            return np.array([np.cos(psi), np.sin(psi)]) * u
        if mode == 'orbit':
            psi = st.get('base_psi', psi) + st.get('omega', 0.05) * self._t * dt
            return np.array([np.cos(psi), np.sin(psi)]) * u
        if mode == 'escape':
            world = getattr(self.env, 'world', None)
            own = world.agents[0] if world is not None and getattr(world, 'agents', None) else None
            if own is not None:
                d = np.asarray(ob.state.p_pos, float) - np.asarray(own.state.p_pos, float)
                if float(np.linalg.norm(d)) > 1e-9:
                    psi = float(np.arctan2(d[1], d[0]))
            return np.array([np.cos(psi), np.sin(psi)]) * u
        if mode == 'waypoints':
            wps = st.get('wps') or []
            if wps:
                idx = int(st.get('wp_i', 0)) % len(wps)
                d = np.asarray(wps[idx], float) - np.asarray(ob.state.p_pos, float)
                if float(np.linalg.norm(d)) < 0.05:
                    idx = (idx + 1) % len(wps)
                    st['wp_i'] = idx
                    d = np.asarray(wps[idx], float) - np.asarray(ob.state.p_pos, float)
                if float(np.linalg.norm(d)) > 1e-9:
                    psi = float(np.arctan2(d[1], d[0]))
            return np.array([np.cos(psi), np.sin(psi)]) * u
        if mode == 'random':
            if self._t * dt >= float(st.get('next_change', 10.0)):
                st['base_psi'] = st.get('base_psi', psi) + float(np.random.uniform(-0.7, 0.7))
                st['base_u'] = float(np.clip(st.get('base_u', u) * float(np.random.uniform(0.7, 1.3)), st.get('u_min', 0.0012), st.get('u_max', 0.0022)))
                st['next_change'] = self._t * dt + float(np.random.uniform(5.0, 20.0))
                ob.obstacle_vel = float(st['base_u'])
            psi = st.get('base_psi', psi)
            return np.array([np.cos(psi), np.sin(psi)]) * float(st.get('base_u', u))
        return np.array([np.cos(psi), np.sin(psi)]) * u

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
        dt = float(getattr(world, 'dt', 1.0))
        for i in range(max(n_ob, 0)):
            ob = world.obstacles[i]
            if not getattr(ob, 'encounter_generated', False):
                continue
            ob.state.p_vel = self._mode_velocity(ob, dt)

    def step(self, action):
        out = self.env.step(action)
        self._enforce_encounter_motion()
        return out


    # --- env interface ---------------------------------------------------------
    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._t = 0
        self.verify_attempts = 0
        self.verify_accepts = 0
        self.verify_first_try_accepts = 0
        self._randomize_initial_heading()
        self.relocate()
        if bool(self.cfg.get('encounter_verify_rollout', False)):
            tries = int(self.cfg.get('encounter_verify_max_tries', 6))
            for _k in range(max(tries, 1)):
                self.verify_attempts += 1
                if self._verify_placement():
                    self.verify_accepts += 1
                    if _k == 0:
                        self.verify_first_try_accepts += 1
                    break
                self.relocate()
        self._assign_motion_modes()
        self._enforce_encounter_motion()
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
            try:
                _r = np.asarray(p_o, float) - np.asarray(p_s, float)
                _v = (np.array([np.cos(psi_o), np.sin(psi_o)]) * float(u_o)
                      - (route / max(norm, 1e-9) * u_s))
                _vv = float(np.dot(_v, _v))
                if _vv > 1e-16:
                    _t = max(0.0, -float(np.dot(_r, _v)) / _vv)
                    ob.encounter_design_dcpa = float(np.linalg.norm(_r + _v * _t))
                    ob.encounter_design_tcpa = _t
            except Exception:
                pass
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
