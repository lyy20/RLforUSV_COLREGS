# -*- coding: utf-8 -*-
"""短滚验证：滚 40 步后按当前相对状态外推 CPA，再看接受率。
同时输出障碍船速度是否被保住的检查结果。
"""
import sys, math
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build

env = build(ROOT / "SAC_3DOF_ENTITY_ENCODER_0911_DENSE_REAL_V5.txt")
EP, ROLL = 25, 40
rows = []; hold_err = []
for ep in range(EP):
    env.reset()
    w = env.world
    own = w.agents[0]; tgt = w.landmarks[0]
    ps = np.asarray(own.state.p_pos, dtype=float)[:2]
    to_t = np.asarray(tgt.state.p_pos, dtype=float)[:2] - ps
    n_t = float(np.linalg.norm(to_t)); u_nom = float(getattr(w, "agent_nominal_speed", 0.001))
    v_s_nom = (to_t / n_t * u_nom) if n_t > 1e-9 else np.zeros(2)
    marked = [(i, o) for i, o in enumerate(list(getattr(w, "obstacles", []) or [])) if getattr(o, "encounter_generated", False)]
    info = {}
    for i, o in marked:
        po = np.asarray(o.state.p_pos, dtype=float)[:2]; vo = np.asarray(o.state.p_vel, dtype=float)[:2]
        r = po - ps; v = vo - v_s_nom; vv = float(np.dot(v, v))
        if vv <= 1e-16: continue
        tcpa = -float(np.dot(r, v)) / vv
        info[i] = {"des": float(np.linalg.norm(r + v * tcpa)), "v0": vo.copy()}
    if not info: continue
    for _ in range(ROLL):
        v_now = np.asarray(own.state.p_vel, dtype=float)[:2]
        psi_now = math.atan2(float(v_now[1]), float(v_now[0])) if np.linalg.norm(v_now) > 1e-9 else 0.0
        des = np.asarray(tgt.state.p_pos, dtype=float)[:2] - np.asarray(own.state.p_pos, dtype=float)[:2]
        err = (math.atan2(float(des[1]), float(des[0])) - psi_now + math.pi) % (2*math.pi) - math.pi
        env.step(np.array([[1.0, float(np.clip(2.0*err, -1.0, 1.0))]]))
    ps = np.asarray(own.state.p_pos, dtype=float)[:2]; vs = np.asarray(own.state.p_vel, dtype=float)[:2]
    for i, o in marked:
        if i not in info: continue
        po = np.asarray(o.state.p_pos, dtype=float)[:2]; vo = np.asarray(o.state.p_vel, dtype=float)[:2]
        hold_err.append(float(np.linalg.norm(vo - info[i]["v0"])) * 1000.0)
        r = po - ps; v = vo - vs; vv = float(np.dot(v, v))
        if vv <= 1e-16:
            info[i]["ext"] = float(np.linalg.norm(r))
        else:
            t = -float(np.dot(r, v)) / vv
            t = max(t, 0.0)
            info[i]["ext"] = float(np.linalg.norm(r + v * t))
    for i in info:
        if "ext" in info[i]: rows.append((info[i]["des"], info[i]["ext"]))
des = np.array([r[0] for r in rows]); ext = np.array([r[1] for r in rows])
print("样本 =", len(rows), "| 短滚 =", ROLL, "步")
print("障碍船速度保持误差（初速-当前）中位 = %.2f m/s" % np.median(hold_err))
print("设计 CPA 中位 %.1f m | 短滚外推 CPA 中位 %.1f m | 比值中位 %.2f" % (np.median(des)*1000, np.median(ext)*1000, np.median(ext/des)))
for tol_abs, tol_rel in [(0.020, 0.5), (0.030, 1.0), (0.050, 2.0)]:
    ok = np.sum(ext <= des*tol_rel + tol_abs); p = ok/max(len(rows),1)
    print("  容差 设计x%.1f+%.0fm -> 接受率 %.0f%% (重试 %.1f)" % (tol_rel, tol_abs*1000, 100*p, 1.0/max(p,1e-6)))
