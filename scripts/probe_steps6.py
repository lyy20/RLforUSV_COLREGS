# -*- coding: utf-8 -*-
"""用位置差分估计航向（不依赖 usv_3dof 快照），重标定符号并测到达步数。"""
import sys, math
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build
env = build(ROOT / "last_cfg" / "PROBE_STEPS2.txt")

def est_psi(prev, cur, fallback):
    d = np.asarray(cur, float) - np.asarray(prev, float)
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return fallback
    return math.atan2(float(d[1]), float(d[0]))

# --- 标定 ---
np.random.seed(100); env.reset()
w = env.world; own = w.agents[0]
prev = np.asarray(own.state.p_pos, float).copy()
psi_prev = None
for t in range(30):
    env.step(np.array([[1.0, 0.3]]))
    cur = np.asarray(own.state.p_pos, float).copy()
    p = est_psi(prev, cur, 0.0); prev = cur
    if psi_prev is None: psi_prev = p
psi_last = p
dpsi = (psi_last - psi_prev + math.pi) % (2*math.pi) - math.pi
sign = 1.0 if dpsi > 0 else -1.0
print("标定（位置差分）：舵=+0.3 30 步 => 航向 %+.1f deg => 控制律用 rudder = %+.1f * err" % (math.degrees(dpsi), sign))

steps_need = []; d0s = []
for ep in range(20):
    np.random.seed(100 + ep)
    env.reset()
    w = env.world; own = w.agents[0]
    d0 = float(np.linalg.norm(np.asarray(w.landmarks[0].state.p_pos, float) - np.asarray(own.state.p_pos, float)))
    d0s.append(d0)
    prev = np.asarray(own.state.p_pos, float).copy(); psi_hat = None; pprev = None
    reached = None
    for tt in range(1200):
        d = np.asarray(w.landmarks[0].state.p_pos, float) - np.asarray(own.state.p_pos, float)
        dist = float(np.linalg.norm(d))
        if ep == 0 and tt % 40 == 0 and psi_hat is not None:
            print("  ep0 t=%3d 距离=%6.1f m 估计航向=%7.1f u_est=%.2f" %
                  (tt, dist*1000, math.degrees(psi_hat), float(np.linalg.norm(np.asarray(own.state.p_pos, float)-pprev))*1000))
        err = (math.atan2(float(d[1]), float(d[0])) - (psi_hat if psi_hat is not None else 0.0) + math.pi) % (2*math.pi) - math.pi
        rud = float(np.clip(sign * 0.5 * err, -1.0, 1.0))
        try:
            _, _, dones, _ = env.step(np.array([[1.0, rud]]))
        except Exception:
            break
        cur = np.asarray(own.state.p_pos, float).copy()
        psi_hat = est_psi((pprev if pprev is not None else prev), cur, psi_hat if psi_hat is not None else 0.0)
        pprev = prev; prev = cur
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
