# -*- coding: utf-8 -*-
"""前向仿真验证：接受率测量（决定容差与重试上限）。

对每个 reset：
  1) 用生成器假设算 设计 DCPA/TCPA
  2) 用"追目标"占位策略真实推演 T 步，记录实际最近距离
  3) 统计在各种容差下的接受率
"""
import sys, math
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build

CFG = sys.argv[1] if len(sys.argv) > 1 else "SAC_3DOF_ENTITY_ENCODER_0911_DENSE_REAL_V5.txt"
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 25
ROLL = int(sys.argv[3]) if len(sys.argv) > 3 else 150
env = build(ROOT / CFG)
rows = []
for ep in range(EPISODES):
    env.reset()
    w = env.world
    own = w.agents[0]; tgt = w.landmarks[0]
    ps = np.asarray(own.state.p_pos, dtype=float)[:2]
    to_t = np.asarray(tgt.state.p_pos, dtype=float)[:2] - ps
    n_t = float(np.linalg.norm(to_t))
    u_nom = float(getattr(w, "agent_nominal_speed", 0.001))
    v_s_nom = (to_t / n_t * u_nom) if n_t > 1e-9 else np.zeros(2)
    marked = [(i, o) for i, o in enumerate(list(getattr(w, "obstacles", []) or []))
              if getattr(o, "encounter_generated", False)]
    if not marked:
        continue
    info = {}
    for i, o in marked:
        po = np.asarray(o.state.p_pos, dtype=float)[:2]
        vo = np.asarray(o.state.p_vel, dtype=float)[:2]
        r = po - ps; v = vo - v_s_nom
        vv = float(np.dot(v, v))
        if vv <= 1e-16: continue
        tcpa = -float(np.dot(r, v)) / vv
        dcpa = float(np.linalg.norm(r + v * tcpa))
        info[i] = {"des": dcpa, "min_d": float(np.linalg.norm(r))}
    for _ in range(ROLL):
        v_now = np.asarray(own.state.p_vel, dtype=float)[:2]
        psi_now = math.atan2(float(v_now[1]), float(v_now[0])) if float(np.linalg.norm(v_now)) > 1e-9 else 0.0
        des = np.asarray(tgt.state.p_pos, dtype=float)[:2] - np.asarray(own.state.p_pos, dtype=float)[:2]
        err = (math.atan2(float(des[1]), float(des[0])) - psi_now + math.pi) % (2*math.pi) - math.pi
        act = np.array([[1.0, float(np.clip(2.0*err, -1.0, 1.0))]])
        try:
            _, _, dones, _ = env.step(act)
        except Exception:
            break
        ps = np.asarray(own.state.p_pos, dtype=float)[:2]
        for i, o in marked:
            if i not in info: continue
            d = float(np.linalg.norm(np.asarray(o.state.p_pos, dtype=float)[:2] - ps))
            if d < info[i]["min_d"]: info[i]["min_d"] = d
        if bool(np.any(dones)): break
    for i, v in info.items():
        rows.append((v["des"], v["min_d"]))

des = np.array([r[0] for r in rows]); real = np.array([r[1] for r in rows])
print("样本 =", len(rows), "| 滚动 =", ROLL, "步")
print("设计 DCPA 中位 %.1f m | 实际最近中位 %.1f m | 比值中位 %.2f" % (np.median(des)*1000, np.median(real)*1000, np.median(real/des)))
for tol_abs, tol_rel in [(0.020, 0.5), (0.030, 1.0), (0.050, 2.0)]:
    ok = np.sum(real <= des*tol_rel + tol_abs)
    p = ok / max(len(rows), 1)
    print("  容差: 实际 <= 设计x%.1f + %.0fm  -> 接受率 %.1f%%  (期望重试 %.1f 次)" % (tol_rel, tol_abs*1000, 100*p, (1.0/max(p,1e-6))))
