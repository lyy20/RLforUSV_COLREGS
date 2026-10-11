# -*- coding: utf-8 -*-
"""修正探针配置生成（文本替换，保留键名大小写），再测到达步数。"""
import sys, math, re
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
src = (ROOT / "V6_A2_S1.txt").read_text(encoding="utf-8")
def setk(t, k, v):
    return re.sub(r"(?m)^\s*" + re.escape(k) + r"\s*=.*$", "%s = %s" % (k, v), t)
t = setk(src, "episode_length", "1500")
t = setk(t, "TRAINING_RUN_NAME", "PROBE_STEPS2")
t = setk(t, "POLICY_SAVE_INTERVAL", "100000")
t = setk(t, "TENSORBOARD_LOG_INTERVAL", "100000")
(ROOT / "last_cfg" / "PROBE_STEPS2.txt").write_text(t, encoding="utf-8")
from measure_encounter_density import build
env = build(ROOT / "last_cfg" / "PROBE_STEPS2.txt")
print("校验: episode_length =", env.cfg.get("max_episode_steps"), "| dmin/dmax =",
      env.cfg.get("landmark_initial_distance_min_km"), env.cfg.get("landmark_initial_distance_max_km"))
np.random.seed(100)
env.reset()
w = env.world; own = w.agents[0]; st = own.state.usv_3dof
print("初始 psi=%.1f deg u=%.3f 距离=%.1f m" % (math.degrees(st.psi), st.u,
      float(np.linalg.norm(np.asarray(w.landmarks[0].state.p_pos, float) - np.asarray(own.state.p_pos, float))) * 1000))
steps_need = []
for ep in range(20):
    np.random.seed(100 + ep)
    env.reset()
    w = env.world; own = w.agents[0]; st = own.state.usv_3dof
    reached = None
    for tt in range(1500):
        d = np.asarray(w.landmarks[0].state.p_pos, float) - np.asarray(own.state.p_pos, float)
        dist = float(np.linalg.norm(d))
        if ep == 0 and tt % 40 == 0:
            print("  ep0 t=%3d 距离=%6.1f m u=%.2f 舵=%.3f 推力=%.2f"
                  % (tt, dist * 1000, st.u, st.rudder, st.thrust))
        err = (math.atan2(float(d[1]), float(d[0])) - float(st.psi) + math.pi) % (2 * math.pi) - math.pi
        try:
            _, _, dones, _ = env.step(np.array([[1.0, float(np.clip(1.2 * err, -1.0, 1.0))]]))
        except Exception:
            break
        if dist < 0.100:
            reached = tt + 1; break
        if bool(np.any(dones)):
            break
    steps_need.append(reached if reached is not None else -1)
ok = [x for x in steps_need if x > 0]
print("到达 %d/20 | 中位 %s | p90 %s | 最大 %s" % (len(ok), (int(np.median(ok)) if ok else "NA"),
      (int(np.percentile(ok, 90)) if ok else "NA"), (max(ok) if ok else "NA")))
print("阈值到达比例:", {k: "%d/20" % sum(1 for x in ok if x <= k) for k in (250, 300, 350, 400)})
