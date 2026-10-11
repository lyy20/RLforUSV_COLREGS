# -*- coding: utf-8 -*-
"""诊断：长距离下本船到底怎么走的（距离/航向/速度/舵）。"""
import sys, math
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build
env = build(ROOT / "last_cfg" / "PROBE_STEPS.txt")
np.random.seed(100)
env.reset()
w = env.world; own = w.agents[0]
st = own.state.usv_3dof
print("初始: psi=%.1f deg u=%.3f m/s 位置=(%.3f,%.3f) 目标=(%.3f,%.3f)"
      % (math.degrees(st.psi), st.u, own.state.p_pos[0], own.state.p_pos[1],
         w.landmarks[0].state.p_pos[0], w.landmarks[0].state.p_pos[1]))
for t in range(400):
    d = np.asarray(w.landmarks[0].state.p_pos, float) - np.asarray(own.state.p_pos, float)
    dist = float(np.linalg.norm(d))
    bearing = math.atan2(float(d[1]), float(d[0]))
    err = (bearing - float(st.psi) + math.pi) % (2 * math.pi) - math.pi
    if t % 40 == 0:
        print("  t=%3d 距离=%6.1f m 航向=%7.1f deg 目标方位=%7.1f deg 误差=%6.1f deg u=%.2f r=%.3f 舵=%.3f 推力=%.2f"
              % (t, dist * 1000, math.degrees(st.psi), math.degrees(bearing), math.degrees(err),
                 st.u, st.r, st.rudder, st.thrust))
    act = np.array([[1.0, float(np.clip(1.2 * err, -1.0, 1.0))]])
    _, _, dones, _ = env.step(act)
    if bool(np.any(dones)):
        print("  终止于 t=%d，距离=%.1f m" % (t, dist * 1000)); break
