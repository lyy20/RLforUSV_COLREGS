# -*- coding: utf-8 -*-
"""判决探针：不同行为下，规则触发（filter_active）随"回合内步序"的分布。"""
import sys, math, re
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
src = (ROOT / "V6_A1_S1.txt").read_text(encoding="utf-8")
src = re.sub(r"(?m)^\s*TRAINING_RUN_NAME\s*=.*$", "TRAINING_RUN_NAME = PROBE_FIRING", src)
src = re.sub(r"(?m)^\s*TENSORBOARD_LOG_INTERVAL\s*=.*$", "TENSORBOARD_LOG_INTERVAL = 100000", src)
src = re.sub(r"(?m)^\s*POLICY_SAVE_INTERVAL\s*=.*$", "POLICY_SAVE_INTERVAL = 100000", src)
(ROOT / "last_cfg" / "PROBE_FIRING.txt").write_text(src, encoding="utf-8")
from measure_encounter_density import build
env = build(ROOT / "last_cfg" / "PROBE_FIRING.txt")

def get_active(info):
    try:
        rows = info if isinstance(info, (list, tuple)) else [info]
        for r in rows:
            if isinstance(r, dict) and "filter_active" in r:
                v = np.asarray(r["filter_active"], float).reshape(-1)
                return bool(v.size and v[0] > 0)
    except Exception:
        pass
    return None

def run(policy, n_ep=12):
    phases = [0]*4; total = [0]*4; fires = 0; steps = 0; speeds = []
    for ep in range(n_ep):
        np.random.seed(700 + ep); env.reset()
        w = env.world; own = w.agents[0]
        prev = np.asarray(own.state.p_pos, float).copy(); spd = []
        for t in range(400):
            d = np.asarray(w.landmarks[0].state.p_pos, float) - np.asarray(own.state.p_pos, float)
            act = policy(own, d, t, spd)
            try:
                _, _, dones, info = env.step(np.array([act]))
            except Exception:
                break
            cur = np.asarray(own.state.p_pos, float).copy()
            v = float(np.linalg.norm(cur - prev)); prev = cur; spd.append(v)
            a = get_active(info)
            ph = min(3, t // 100)
            total[ph] += 1
            if a: phases[ph] += 1; fires += 1
            steps += 1
            if bool(np.any(dones)): break
        if spd: speeds.append(float(np.mean(spd)))
    rate = [phases[i] / total[i] if total[i] else 0.0 for i in range(4)]
    return rate, fires / max(1, steps), float(np.mean(speeds)) if speeds else 0.0, steps

POL = {
 "直行(舵=0,推力1.0)":       lambda o, d, t, s: [1.0, 0.0],
 "直行(舵=0,推力0.3)":       lambda o, d, t, s: [0.3, 0.0],
 "缓追(朝目标,推力1.0)":     None,
}
def pursuit(o, d, t, s):
    if len(s) >= 2:
        v = np.array([d[0], d[1]])
        pass
    # 用最近两步位置差分估计航向
    return None

print("%-24s %8s %8s %8s %8s %10s %9s" % ("行为", "步0-99", "100-199", "200-299", "300-399", "整回合", "平均速度m/步"))
for name, pol in [("直行(舵0,推力1.0)", lambda o,d,t,s: [1.0,0.0]),
                  ("直行(舵0,推力0.3)", lambda o,d,t,s: [0.3,0.0]),
                  ("半推力+小幅摆舵",  lambda o,d,t,s: [0.5, 0.4*math.sin(t/25.0)])]:
    r, overall, sp, st = run(lambda o,d,t,s,p=pol: p(o,d,t,s))
    print("%-24s %7.3f %8.3f %8.3f %8.3f %10.3f %9.2f" % (name, r[0], r[1], r[2], r[3], overall, sp))
