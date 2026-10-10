# -*- coding: utf-8 -*-
"""重置安全探针：设计 DCPA/TCPA vs 实际（每回合独立，遇 done 即停）。

设计值由初始状态独立重算（不依赖生成器内部变量）：
  v_rel = v_o - v_s ; r_rel = p_o - p_s
  TCPA = -(r·v_rel)/|v_rel|^2 ; DCPA = |r + v_rel*TCPA|
"""
import sys, math, json
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build

CFG = sys.argv[1] if len(sys.argv) > 1 else "SAC_3DOF_ENTITY_ENCODER_0911_DENSE_REAL_V4.txt"
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 60
MAX_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 600
env = build(ROOT / CFG)
rows = []
for ep in range(EPISODES):
    env.reset()
    w = env.world
    own = w.agents[0]
    ps = np.asarray(own.state.p_pos, dtype=float)[:2]
    vs = np.asarray(own.state.p_vel, dtype=float)[:2]
    marked = [(i, o) for i, o in enumerate(list(getattr(w, "obstacles", []) or []))
              if getattr(o, "encounter_generated", False)]
    if not marked:
        continue
    info = {}
    for i, o in marked:
        po = np.asarray(o.state.p_pos, dtype=float)[:2]
        vo = np.asarray(o.state.p_vel, dtype=float)[:2]
        r = po - ps; v = vo - vs
        vv = float(np.dot(v, v))
        if vv <= 1e-16:
            continue
        tcpa = -float(np.dot(r, v)) / vv
        dcpa = float(np.linalg.norm(r + v * tcpa))
        info[i] = {"des_tcpa": tcpa, "des_dcpa": dcpa, "min_d": float(np.linalg.norm(r)), "done": False}
    if not info:
        continue
    for t in range(MAX_STEPS):
        done = False
        try:
            _, _, dones, _ = env.step(np.zeros((1, 2)))
            done = bool(np.any(dones))
        except Exception:
            break
        ps = np.asarray(own.state.p_pos, dtype=float)[:2]
        for i, o in marked:
            if i not in info:
                continue
            po = np.asarray(o.state.p_pos, dtype=float)[:2]
            d = float(np.linalg.norm(po - ps))
            if d < info[i]["min_d"]:
                info[i]["min_d"] = d
        if done:
            for i in info:
                info[i]["done"] = True
            break
    for i, v in info.items():
        rows.append(v)

import statistics as st
def med(v):
    v = [x for x in v if x == x]
    return st.median(v) if v else float("nan")
print("episodes=%d  pairs=%d  (config=%s)" % (EPISODES, len(rows), CFG))
print("  设计 DCPA 中位 = %.1f m   (区间应为 10-60)" % (med([r["des_dcpa"] * 1000 for r in rows])))
print("  设计 TCPA 中位 = %.1f s   (区间应为 60-120)" % med([r["des_tcpa"] for r in rows]))
print("  实际最小距离中位 = %.1f m" % med([r["min_d"] * 1000 for r in rows]))
real = [r for r in rows if r["des_dcpa"] > 0]
ratio = [r["min_d"] / r["des_dcpa"] for r in real]
within = sum(1 for r in real if r["min_d"] <= r["des_dcpa"] * 1.5 + 0.010)
print("  实际/设计 比值中位 = %.2f  (1.0=完全实现)" % med(ratio))
print("  实际 <= 设计x1.5+10m 的比例 = %d/%d = %.0f%%" % (within, len(real), 100.0 * within / max(len(real), 1)))
print("  实际最小距离分布: p10=%.0f p50=%.0f p90=%.0f m" % (
    np.percentile([r["min_d"] * 1000 for r in rows], 10),
    np.percentile([r["min_d"] * 1000 for r in rows], 50),
    np.percentile([r["min_d"] * 1000 for r in rows], 90)))
Path(ROOT / "docs" / "encounter_realized.json").write_text(json.dumps(rows[:200], indent=1), encoding="utf-8")
