# -*- coding: utf-8 -*-
"""单步成本绝对分解：累计 env.last_step_timing（ms/步）。"""
import sys, time
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build
env = build(ROOT / "V6_A2_S1.txt")
inner = env.env
inner.profile_timing = True
env.reset()
inner.profile_timing = True
N = 300
acc = {}
t0 = time.perf_counter()
for t in range(N):
    v = np.asarray(inner.world.agents[0].state.p_vel, float)
    psi = np.arctan2(float(v[1]), float(v[0])) if float(np.linalg.norm(v)) > 1e-9 else 0.0
    d = np.asarray(inner.world.landmarks[0].state.p_pos, float) - np.asarray(inner.world.agents[0].state.p_pos, float)
    err = (np.arctan2(float(d[1]), float(d[0])) - psi + np.pi) % (2*np.pi) - np.pi
    env.step(np.array([[1.0, float(np.clip(2.0*err, -1.0, 1.0))]]))
    for k, val in (inner.last_step_timing or {}).items():
        acc[k] = acc.get(k, 0.0) + float(val)
    if inner.world.step_count >= 590:
        env.reset(); inner.profile_timing = True
total = time.perf_counter() - t0
print("=== %d 步：总 %6.2f s，平均 %.2f ms/步（含 Python 外层开销）===" % (N, total, 1000*total/N))
s = sum(acc.values())
print("被计时阶段合计 %.2f ms/步（占实测 %.0f%%）" % (1000*s/N, 100*s/(N*total/N)))
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print("   %-28s %6.3f ms/步  (%5.1f%%)" % (k, 1000*v/N, 100*v/s))
