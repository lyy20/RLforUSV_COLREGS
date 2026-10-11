# -*- coding: utf-8 -*-
"""先标定舵的符号，再测到达步数。"""
import sys, math
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build
env = build(ROOT / "last_cfg" / "PROBE_STEPS2.txt")
np.random.seed(100); env.reset()
w = env.world; own = w.agents[0]
psi0 = float(own.state.usv_3dof.psi)
for t in range(40):
    env.step(np.array([[1.0, 0.3]]))
psi1 = float(own.state.usv_3dof.psi)
dpsi = (psi1 - psi0 + math.pi) % (2 * math.pi) - math.pi
print("标定：舵=+0.3 持续 40 步 => 航向变化 %+.1f deg  => 正舵使航向 %s"
      % (math.degrees(dpsi), "增大(左转/逆时针)" if dpsi > 0 else "减小(右转/顺时针)"))
sign = 1.0 if dpsi > 0 else -1.0
print("因此跟踪控制器应使用 rudder = %+.1f * err" % sign)
steps_need = []; d0s = []
for ep in range(20):
    np.random.seed(100 + ep)
    env.reset()
    w = env.world; own = w.agents[0]
    d0 = float(np.linalg.norm(np.asarray(w.landmarks[0].state.p_pos, float) - np.asarray(own.state.p_pos, float)))
    d0s.append(d0)
    reached = None
    for tt in range(1200):
        d = np.asarray(w.landmarks[0].state.p_pos, float) - np.asarray(own.state.p_pos, float)
        dist = float(np.linalg.norm(d))
        st = own.state.usv_3dof
        psi = float(st.psi); r = float(getattr(st, "r", 0.0))
        err = (math.atan2(float(d[1]), float(d[0])) - psi + math.pi) % (2*math.pi) - math.pi
        rud = float(np.clip(sign * (1.0 * err) - 0.8 * r, -1.0, 1.0))
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
