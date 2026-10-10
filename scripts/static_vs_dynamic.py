# -*- coding: utf-8 -*-
"""静态目标 vs 动态目标（逃逸）：捕获步数、会遇覆盖、教师可发声步比。"""
import sys, math, json
from pathlib import Path
from collections import Counter
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build
from safety.colregs import COLREGsEngine, COLREGsConfig, VesselState
engine = COLREGsEngine(COLREGsConfig())

def own_state(own, prev_pos):
    st = getattr(own.state, "usv_3dof", None)
    if st is not None and hasattr(st, "psi") and hasattr(st, "u"):
        psi = float(st.psi); u = float(st.u)
        return u * math.cos(psi), u * math.sin(psi), psi, abs(u)
    pos = np.asarray(own.state.p_pos, float)[:2]
    if prev_pos is not None:
        d = pos - prev_pos
        sp = float(np.linalg.norm(d))
        if sp > 1e-9:
            return float(d[0]), float(d[1]), math.atan2(float(d[1]), float(d[0])), sp
    v = np.asarray(own.state.p_vel, float)[:2]
    nrm = float(np.linalg.norm(v))
    return float(v[0]), float(v[1]), (math.atan2(float(v[1]), float(v[0])) if nrm > 1e-9 else 0.0), nrm
CFGS = {"动态目标(逃逸)": "V6_A2_S1.txt", "静态目标": "V6_STATIC_TARGET.txt"}
EP, MAXSTEPS = 10, 900
out = {}
for label, cfg in CFGS.items():
    env = build(ROOT / cfg)
    cap = []; dist = []; types = Counter(); step_any = 0; step_tot = 0; m2 = []
    for _ in range(EP):
        np.random.seed(21); env.reset()
        w = env.world; own = w.agents[0]
        obs_i = [(i, o) for i, o in enumerate(list(getattr(w, "obstacles", []) or []))]
        first = set()
        d0 = float(np.linalg.norm(np.asarray(w.landmarks[0].state.p_pos, float) - np.asarray(own.state.p_pos, float)))
        reached = None
        prev_pos = None
        for t in range(MAXSTEPS):
            op = np.asarray(own.state.p_pos, float)[:2]
            _ovx, _ovy, oh, osp = own_state(own, prev_pos)
            ov = np.array([_ovx, _ovy])
            any_risk = False
            for i, o in obs_i:
                tp = np.asarray(o.state.p_pos, float)[:2]; tv = np.asarray(o.state.p_vel, float)[:2]
                th = math.atan2(float(tv[1]), float(tv[0])) if float(np.linalg.norm(tv)) > 1e-9 else 0.0
                ovs = VesselState.from_kinematics(vessel_id="o", position=[0.0, 0.0], heading=oh, speed=osp)
                tvs = VesselState.from_kinematics(vessel_id="t", position=list(tp - op), heading=th, speed=float(np.linalg.norm(tv)))
                rep = engine.evaluate(ovs, tvs)
                et = str(getattr(rep, "encounter_type")).split(".")[-1].lower()
                if et != "safe_passage":
                    any_risk = True
                    if i not in first:
                        first.add(i); types[et] += 1
            step_any += 1 if any_risk else 0; step_tot += 1
            prev_pos = np.asarray(own.state.p_pos, float)[:2]
            _vx, _vy, psi, _sp = own_state(own, None)
            d = np.asarray(w.landmarks[0].state.p_pos, float)[:2] - np.asarray(own.state.p_pos, float)[:2]
            err = (math.atan2(float(d[1]), float(d[0])) - psi + math.pi) % (2*math.pi) - math.pi
            try:
                _, _, dones, _ = env.step(np.array([[1.0, float(np.clip(2.0*err, -1.0, 1.0))]]))
            except Exception:
                break
            dn = float(np.linalg.norm(np.asarray(w.landmarks[0].state.p_pos, float)[:2] - np.asarray(own.state.p_pos, float)[:2]))
            if reached is None and dn < 0.100:
                reached = t + 1
            if bool(np.any(dones)):
                break
        cap.append(reached if reached is not None else -1)
        dist.append(dn); m2.append(len(first))
    ok = [c for c in cap if c > 0]
    out[label] = {"episodes": EP, "start_dist_median_m": float(np.median(dist)*1000),
                  "capture_steps_median": float(np.median(ok)) if ok else None,
                  "capture_rate": len(ok)/EP, "M2_per_episode": float(np.mean(m2)),
                  "teacher_step_ratio": step_any/max(step_tot,1), "types": dict(types)}
    r = out[label]
    print("%-14s 起始距离中位=%.0fm 捕获率=%.0f%% 捕获步数中位=%s M2=%.2f 教师步比=%.1f%% 类型=%s"
          % (label, r["start_dist_median_m"], 100*r["capture_rate"],
             r["capture_steps_median"], r["M2_per_episode"], 100*r["teacher_step_ratio"], r["types"]))
(ROOT / "docs" / "static_vs_dynamic.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
