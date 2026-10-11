# -*- coding: utf-8 -*-
"""修正版：每步重读 usv_3dof；控制器加阻尼。测长距离到达步数。"""
import sys, math
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build
env = build(ROOT / "last_cfg" / "PROBE_STEPS2.txt")

def get_st(own):
    st = getattr(own.state, "usv_3dof", None)
    if st is None:
        return None
    return st

steps_need = []; d0s = []
for ep in range(20):
    np.random.seed(100 + ep)
    env.reset()
    w = env.world; own = w.agents[0]
    d0 = float(np.linalg.norm(np.asarray(w.landmarks[0].state.p_pos, float) - np.asarray(own.state.p_pos, float)))
    d0s.append(d0)
    reached = None
    for tt in range(1500):
        d = np.asarray(w.landmarks[0].state.p_pos, float) - np.asarray(own.state.p_pos, float)
        dist = float(np.linalg.norm(d))
        st = get_st(own)
        psi = float(getattr(st, "psi", 0.0)) if st is not None else 0.0
        r = float(getattr(st, "r", 0.0)) if st is not None else 0.0
        if ep == 0 and tt % 60 == 0 and st is not None:
            print("  ep0 t=%3d 距离=%6.1f m psi=%7.1f u=%.2f r=%.3f" % (tt, dist*1000, math.degrees(psi), st.u, r))
        err = (math.atan2(float(d[1]), float(d[0])) - psi + math.pi) % (2*math.pi) - math.pi
        rud = float(np.clip(0.8 * err - 0.6 * r, -1.0, 1.0))
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
print("起始距离中位 %.3f km | 到达 %d/20 | 中位 %s | p90 %s | 最大 %s"
      % (float(np.median(d0s)), len(ok), (int(np.median(ok)) if ok else "NA"),
         (int(np.percentile(ok, 90)) if ok else "NA"), (max(ok) if ok else "NA")))
print("阈值到达:", {k: "%d/20" % sum(1 for x in ok if x <= k) for k in (250, 300, 350, 400, 500)})
