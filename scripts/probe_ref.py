# -*- coding: utf-8 -*-
"""判定：是"引用过期"还是"真被冻结"。"""
import sys
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build
env = build(ROOT / "last_cfg" / "PROBE_FIRING.txt")
np.random.seed(700); env.reset()
own_stale = env.world.agents[0]
for t in range(6):
    env.step(np.array([[1.0, 0.0]]))
    fresh = np.asarray(env.world.agents[0].state.p_pos, float)
    stale = np.asarray(own_stale.state.p_pos, float)
    print("t=%d  fresh=%s  stale=%s  同对象=%s" % (t, np.round(fresh, 4), np.round(stale, 4), fresh is stale or env.world.agents[0] is own_stale))
