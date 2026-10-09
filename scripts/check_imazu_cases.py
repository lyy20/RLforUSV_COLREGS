
# -*- coding: utf-8 -*-
"""P0-5 验收：Imazu 22 案例的会遇分类复现（含尺度归一，处理尺度不匹配）。

关键发现：Imazu 案例是"长时域"设定（16 km 图 / 700 s / 10 m/s），
本项目是缩尺场景（<=2 km / 1~3 m/s）。因此 t=0 直接比分类无意义——
需要把相对位置按比例 s 归一（保持速度，等价于缩短会遇时间尺度）。
"""
import sys
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT))
from curriculum.imazu_cases import load_cases, to_relative, TYPE_TO_ENGINE
from safety.colregs import COLREGsEngine, COLREGsConfig, VesselState

engine = COLREGsEngine(COLREGsConfig())
cases = load_cases()

def run(scale, rotate_sign=-1.0):
    per_label = {}; mism = []
    for c in cases:
        rel = to_relative(c, rotate_sign=rotate_sign)
        own = rel["ownship"]
        own_v = VesselState.from_kinematics(vessel_id="own", position=[0.0, 0.0],
                                            heading=own["heading_rad"], speed=own["speed_mps"], radius=10.0)
        for i, t in enumerate(rel["targets"]):
            tgt = VesselState.from_kinematics(vessel_id="t%d" % i,
                                              position=[t["x_m"] * scale, t["y_m"] * scale],
                                              heading=t["heading_rad"], speed=t["speed_mps"], radius=10.0)
            rep = engine.evaluate(own_v, tgt)
            got = str(getattr(rep, "encounter_type")).split(".")[-1].lower()
            want = TYPE_TO_ENGINE.get(rel["type"])
            if want is None:
                continue
            st = per_label.setdefault(rel["type"], {"n": 0, "ok": 0})
            st["n"] += 1; st["ok"] += int(got == want)
            if got != want:
                mism.append((rel["name"], rel["type"], want, got, round(t["x_m"] * scale, 1), round(t["y_m"] * scale, 1), round(t["speed_mps"], 2)))
    return per_label, mism

for sign in (-1.0, 1.0):
  for scale in (0.05, 0.02):
    per_label, mism = run(scale, sign)
    tot_n = sum(v["n"] for v in per_label.values()); tot_ok = sum(v["ok"] for v in per_label.values())
    detail = " ".join("%s:%d/%d" % (k, v["ok"], v["n"]) for k, v in sorted(per_label.items()))
    print("sign=%+.1f scale=%-6s TOTAL %d/%d = %3.0f%%  [%s]" % (sign, scale, tot_ok, tot_n, 100.0 * tot_ok / max(tot_n, 1), detail))
    for m in mism[:4]:
        print("     %-9s %-5s want=%-18s got=%-18s rel=(%.1f, %.1f) u=%.2f" % m)
