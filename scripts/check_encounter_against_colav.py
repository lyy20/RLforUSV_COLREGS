# -*- coding: utf-8 -*-
"""交叉校验：我们生成的会遇，是否落在 colav-simulator 的方位/航向差范围内。

colav 基线（config/scenario_generator.yaml）：
  HO bearing +-3.0   course +-5.0
  OT bearing +-20.0  course +-15.0
  CR bearing 10.1-90.5  course +-10.0
"""
import sys, math, json
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build
from safety.colregs import COLREGsEngine, COLREGsConfig, VesselState

COLAV = {
    "head_on": {"bearing": (-3.0, 3.0), "course": (-5.0, 5.0)},
    "overtaking": {"bearing": (-20.0, 20.0), "course": (-15.0, 15.0)},
    "crossing": {"bearing": (10.1, 90.5), "course": (-10.0, 10.0)},
}
def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0

cfg = ROOT / "SAC_3DOF_ENTITY_ENCODER_0911_DENSE_REAL_V3.txt"
env = build(cfg)
engine = COLREGsEngine(COLREGsConfig())
rows = []
for _ in range(40):
    env.reset()
    w = env.world
    agents = list(getattr(w, "agents", []) or [])
    if not agents:
        continue
    own = agents[0]
    ostate = getattr(own, "state", None)
    op = np.asarray(ostate.p_pos, dtype=float)[:2]
    ov = np.asarray(ostate.p_vel, dtype=float)[:2]
    oh = math.atan2(float(ov[1]), float(ov[0])) if float(np.linalg.norm(ov)) > 1e-9 else 0.0
    for o in list(getattr(w, "obstacles", []) or []):
        if not getattr(o, "encounter_generated", False):
            continue
        tstate = getattr(o, "state", None)
        tp = np.asarray(tstate.p_pos, dtype=float)[:2]
        tv = np.asarray(tstate.p_vel, dtype=float)[:2]
        th = math.atan2(float(tv[1]), float(tv[0])) if float(np.linalg.norm(tv)) > 1e-9 else 0.0
        d = tp - op
        bearing = wrap180(math.degrees(math.atan2(d[1], d[0]) - oh))
        course = wrap180(math.degrees(th - oh))
        own_v = VesselState.from_kinematics(vessel_id="o", position=[0.0, 0.0], heading=oh, speed=float(np.linalg.norm(ov)))
        tgt_v = VesselState.from_kinematics(vessel_id="t", position=list(d), heading=th, speed=float(np.linalg.norm(tv)))
        rep = engine.evaluate(own_v, tgt_v)
        et = str(getattr(rep, "encounter_type")).split(".")[-1].lower()
        rows.append((et, bearing, course, float(np.linalg.norm(d))))

from collections import Counter, defaultdict
print("samples =", len(rows))
by = defaultdict(list)
for et, b, c, dist in rows:
    key = "head_on" if "head_on" in et else ("overtaking" if "overtak" in et else ("crossing" if "crossing" in et else et))
    by[key].append((b, c, dist))
for key, vals in sorted(by.items()):
    rng = COLAV.get(key)
    if not rng:
        print("  %-12s n=%d  (colav 无对应范围)" % (key, len(vals)))
        continue
    inb = sum(1 for b, c, d in vals if rng["bearing"][0] <= b <= rng["bearing"][1])
    inc = sum(1 for b, c, d in vals if rng["course"][0] <= c <= rng["course"][1])
    both = sum(1 for b, c, d in vals if rng["bearing"][0] <= b <= rng["bearing"][1] and rng["course"][0] <= c <= rng["course"][1])
    dists = [d for _, _, d in vals]
    print("  %-12s n=%3d | bearing 命中 %3d/%3d | course 命中 %3d/%3d | 两者同时 %3d (%.0f%%) | 初始距离 %.0f-%.0f m"
          % (key, len(vals), inb, len(vals), inc, len(vals), both, 100.0 * both / max(len(vals), 1), min(dists), max(dists)))
