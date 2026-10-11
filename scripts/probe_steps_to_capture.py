# -*- coding: utf-8 -*-
"""测量：长距离(1.0~1.2 km)下，够格控制器的到达步数分布（决定 300 还是 400 步）。"""
import sys, math
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build
import configparser
cp = configparser.ConfigParser(); cp.read(str(ROOT / "V6_A2_S1.txt"), encoding="utf-8")
cp["hyperparam"]["episode_length"] = "1500"      # 放开上限，只为测所需步数
cp["hyperparam"]["TRAINING_RUN_NAME"] = "PROBE_STEPS"
cp["hyperparam"]["POLICY_SAVE_INTERVAL"] = "100000"
p = ROOT / "last_cfg" / "PROBE_STEPS.txt"
with open(p, "w", encoding="utf-8") as f:
    cp.write(f)
env = build(p)

def own_psi(own):
    st = getattr(own.state, "usv_3dof", None)
    if st is not None and hasattr(st, "psi"):
        return float(st.psi), abs(float(getattr(st, "u", 0.0)))
    v = np.asarray(own.state.p_vel, float)
    n = float(np.linalg.norm(v))
    return (math.atan2(float(v[1]), float(v[0])) if n > 1e-9 else 0.0), n

steps_need = []
d0s = []
for ep in range(20):
    np.random.seed(100 + ep)
    env.reset()
    w = env.world; own = w.agents[0]
    d0 = float(np.linalg.norm(np.asarray(w.landmarks[0].state.p_pos, float) - np.asarray(own.state.p_pos, float)))
    d0s.append(d0)
    reached = None
    for t in range(1500):
        psi, sp = own_psi(own)
        des = np.asarray(w.landmarks[0].state.p_pos, float) - np.asarray(own.state.p_pos, float)
        err = (math.atan2(float(des[1]), float(des[0])) - psi + math.pi) % (2 * math.pi) - math.pi
        act = np.array([[1.0, float(np.clip(1.2 * err, -1.0, 1.0))]])
        try:
            _, _, dones, _ = env.step(act)
        except Exception:
            break
        d = float(np.linalg.norm(np.asarray(w.landmarks[0].state.p_pos, float) - np.asarray(own.state.p_pos, float)))
        if d < 0.100:
            reached = t + 1
            break
        if bool(np.any(dones)):
            break
    steps_need.append(reached if reached is not None else -1)
ok = [x for x in steps_need if x > 0]
print("起始距离中位 = %.3f km" % float(np.median(d0s)))
print("到达步数：成功 %d/20 | 中位 %s | p90 %s | 最大 %s"
      % (len(ok), (int(np.median(ok)) if ok else "NA"),
         (int(np.percentile(ok, 90)) if ok else "NA"), (max(ok) if ok else "NA")))
print("按步数阈值统计到达比例：", {k: "%d/20" % sum(1 for x in ok if x <= k) for k in (200, 250, 300, 350, 400)})
