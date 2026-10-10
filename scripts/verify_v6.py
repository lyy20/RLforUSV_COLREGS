# -*- coding: utf-8 -*-
"""V6 验收：随机朝向 + 运动方式覆盖 + 验证后的设计/实际 CPA。"""
import sys, math
from pathlib import Path
from collections import Counter
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build
env = build(ROOT / "SAC_3DOF_ENTITY_ENCODER_0911_V6.txt")
EP, ROLL = 20, 150
headings, modes, rows = [], Counter(), []
for ep in range(EP):
    env.reset()
    w = env.world
    own = w.agents[0]
    st = getattr(own.state, 'usv_3dof', None)
    if st is not None:
        headings.append(math.degrees(float(st.psi)))
    marked = [(i, o) for i, o in enumerate(list(getattr(w, "obstacles", []) or [])) if getattr(o, "encounter_generated", False)]
    for i, o in marked:
        modes[getattr(o, "encounter_motion_mode", "?")] += 1
    info = {}
    for i, o in marked:
        des = float(getattr(o, "encounter_design_dcpa", 0.0))
        po = np.asarray(o.state.p_pos, float)[:2]
        ps = np.asarray(own.state.p_pos, float)[:2]
        info[i] = {"des": des, "min": float(np.linalg.norm(po - ps))}
    for _ in range(ROLL):
        v = np.asarray(own.state.p_vel, float)[:2]
        psi = math.atan2(float(v[1]), float(v[0])) if float(np.linalg.norm(v)) > 1e-9 else 0.0
        d = np.asarray(w.landmarks[0].state.p_pos, float)[:2] - np.asarray(own.state.p_pos, float)[:2]
        err = (math.atan2(float(d[1]), float(d[0])) - psi + math.pi) % (2*math.pi) - math.pi
        try:
            _, _, dones, _ = env.step(np.array([[1.0, float(np.clip(2.0*err, -1.0, 1.0))]]))
        except Exception:
            break
        ps = np.asarray(own.state.p_pos, float)[:2]
        for i, o in marked:
            if i not in info: continue
            dd = float(np.linalg.norm(np.asarray(o.state.p_pos, float)[:2] - ps))
            if dd < info[i]["min"]: info[i]["min"] = dd
        if bool(np.any(dones)): break
    for i, v in info.items():
        if v["des"] > 0: rows.append((v["des"], v["min"]))
print("重置 %d 次 | 随机朝向样本 %d 个" % (EP, len(headings)))
if headings:
    hs = np.array(headings)
    print("  朝向: 最小 %.0f 最大 %.0f 标准差 %.1f deg (应接近均匀分布, sd~104)" % (hs.min(), hs.max(), hs.std()))
print("运动方式分布:", dict(modes))
des = np.array([r[0] for r in rows]); real = np.array([r[1] for r in rows])
if len(rows):
    print("设计 DCPA 中位 %.1f m | 实际最近中位 %.1f m | 比值中位 %.2f" % (np.median(des)*1000, np.median(real)*1000, np.median(real/des)))
    ok = np.sum(real <= des*2.0 + 0.050)
    print("  容差(设计x2+50m) 内比例 = %d/%d = %.0f%%" % (ok, len(rows), 100.0*ok/len(rows)))
