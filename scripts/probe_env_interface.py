# -*- coding: utf-8 -*-
import sys, traceback
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build
env = build(ROOT / "last_cfg" / "PROBE_FIRING.txt")
np.random.seed(700); o = env.reset()
print("reset 返回类型:", type(o), "| world.agents 数:", len(env.world.agents),
      "| landmarks:", len(env.world.landmarks))
own = env.world.agents[0]
print("初始位置:", np.asarray(own.state.p_pos, float))
try:
    a = np.array([[1.0, 0.0]])
    print("action shape:", a.shape)
    r = env.step(a)
    print("step 返回长度:", len(r), "| done:", r[2])
    print("step 后位置:", np.asarray(own.state.p_pos, float))
    print("位移:", float(np.linalg.norm(np.asarray(own.state.p_pos, float) - np.asarray(own.state.p_pos, float))))
    info = r[3]
    print("info 类型:", type(info))
    if isinstance(info, (list, tuple)):
        print("info[0]:", str(info[0])[:300])
    else:
        print("info:", str(info)[:300])
except Exception:
    traceback.print_exc()
