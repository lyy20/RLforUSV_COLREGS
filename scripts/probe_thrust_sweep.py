# -*- coding: utf-8 -*-
"""判决：本船速度扫描 -> 规则开火率。若"静止时高、巡航时 0"，则机制确证。"""
import sys
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build
env = build(ROOT / "last_cfg" / "PROBE_FIRING.txt")

def sweep(thrust, n_ep=12):
    fire = 0; steps = 0; ph = [0,0,0,0]; tot = [0,0,0,0]; spd = []
    for ep in range(n_ep):
        np.random.seed(900 + ep); env.reset()
        prev = np.asarray(env.world.agents[0].state.p_pos, float).copy()
        for t in range(400):
            _, _, dones, _ = env.step(np.array([[thrust, 0.0]]))
            own = env.world.agents[0]
            cur = np.asarray(own.state.p_pos, float).copy()
            spd.append(float(np.linalg.norm(cur - prev))); prev = cur
            a = bool(getattr(own, "action_filter_active", False))
            i = min(3, t // 100); tot[i] += 1
            if a: ph[i] += 1; fire += 1
            steps += 1
            if bool(np.any(dones)): break
    r = [ph[i] / tot[i] if tot[i] else 0.0 for i in range(4)]
    return fire / max(1, steps), r, float(np.mean(spd)) * 1000.0

print("%-14s %9s %8s %8s %8s %8s %12s" % ("推力", "整回合开火", "0-99", "100-199", "200-299", "300-399", "平均m/步"))
for thr in (0.0, 0.1, 0.3, 0.6, 1.0):
    ov, r, sp = sweep(thr)
    print("%-14.1f %9.4f %8.4f %8.4f %8.4f %8.4f %12.2f" % (thr, ov, r[0], r[1], r[2], r[3], sp))
