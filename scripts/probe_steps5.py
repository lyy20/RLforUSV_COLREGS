# -*- coding: utf-8 -*-
"""串级控制器（航向→转艏率→舵）测到达步数。"""
import sys, math
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build
env = build(ROOT / "last_cfg" / "PROBE_STEPS2.txt")
steps_need = []; d0s = []; maxr = []; maxu = []
for ep in range(20):
    np.random.seed(100 + ep)
    env.reset()
    w = env.world; own = w.agents[0]
    d0 = float(np.linalg.norm(np.asarray(w.landmarks[0].state.p_pos, float) - np.asarray(own.state.p_pos, float)))
    d0s.append(d0); reached = None
    for tt in range(1200):
        d = np.asarray(w.landmarks[0].state.p_pos, float) - np.asarray(own.state.p_pos, float)
        dist = float(np.linalg.norm(d))
        st = own.state.usv_3dof
        psi = float(st.psi); r = float(getattr(st, "r", 0.0)); u = float(getattr(st, "u", 0.0))
        maxr.append(abs(r)); maxu.append(u)
        if ep == 0 and tt % 60 == 0:
            print("  ep0 t=%3d 距离=%6.1f m psi=%7.1f u=%.2f r=%+.3f" % (tt, dist*1000, math.degrees(psi), u, r))
        err = (math.atan2(float(d[1]), float(d[0])) - psi + math.pi) % (2*math.pi) - math.pi
        r_des = float(np.clip(0.06 * err, -0.08, 0.08))
        rud = float(np.clip(4.0 * (r_des - r), -1.0, 1.0))
        try:
            _, _, dones, _ = env.step(np.array([[1.0, rud]]))
        except Exception:
            break
        if dist < 0.100:
            reached = tt + 1; break
        if bool(np.any(dones)):
            break
    steps_need.append(reached if reached is not None else -1)
ok = [x for x in steps_need if x > 0]
print("实测 |r| max=%.3f rad/s (%.1f deg/s) | u max=%.2f m/s" % (max(maxr), math.degrees(max(maxr)), max(maxu)))
print("起始距离中位 %.3f km | 到达 %d/20 | 中位 %s | p90 %s | 最大 %s"
      % (float(np.median(d0s)), len(ok), (int(np.median(ok)) if ok else "NA"),
         (int(np.percentile(ok, 90)) if ok else "NA"), (max(ok) if ok else "NA")))
print("阈值到达:", {k: "%d/20" % sum(1 for x in ok if x <= k) for k in (250, 300, 350, 400, 500)})
