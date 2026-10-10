# -*- coding: utf-8 -*-
"""诊断：静态目标回合里本船到底走近了多少（用于确定合理步长）。"""
import sys, math
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build
env = build(ROOT / "V6_STATIC_TARGET.txt")
np.random.seed(33); env.reset()
w = env.world; own = w.agents[0]
print("episode_length(配置) =", env.cfg.get("max_episode_steps"))
for t in range(700):
    dp = float(np.linalg.norm(np.asarray(w.landmarks[0].state.p_pos, float) - np.asarray(own.state.p_pos, float)))
    sp = float(np.linalg.norm(np.asarray(own.state.p_vel, float)))
    if t % 50 == 0:
        print("  t=%3d 距目标=%7.1f m 本船速度=%.2f m/s" % (t, dp*1000, sp))
    v = np.asarray(own.state.p_vel, float)
    psi = math.atan2(float(v[1]), float(v[0])) if float(np.linalg.norm(v)) > 1e-9 else 0.0
    d = np.asarray(w.landmarks[0].state.p_pos, float) - np.asarray(own.state.p_pos, float)
    err = (math.atan2(float(d[1]), float(d[0])) - psi + math.pi) % (2*math.pi) - math.pi
    obs, rew, dones, info = env.step(np.array([[1.0, float(np.clip(2.0*err, -1.0, 1.0))]]))
    if bool(np.any(dones)):
        print("  回合结束于 t=%d, 距目标=%.1f m, info=%s" % (t, dp*1000, str(info[0])[:160]))
        break
print("  最终: 距目标=%.1f m 速度=%.2f m/s" % (dp*1000, sp))
