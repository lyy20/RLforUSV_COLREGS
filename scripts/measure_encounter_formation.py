# -*- coding: utf-8 -*-
"""会遇形成的 A/B 度量（同协议、同占位策略）。

指标：
  M1 会遇形成率 = 有 >=1 艘船触发非 safe 判定的回合占比
  M2 每回合形成会遇数（触发过非 safe 的船数）
  M3 类型分布（首次非 safe 的类型）
"""
import sys, math, json
from pathlib import Path
from collections import Counter
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build
from safety.colregs import COLREGsEngine, COLREGsConfig, VesselState

engine = COLREGsEngine(COLREGsConfig())
def run(cfg_name, episodes=8, steps=250):
    env = build(ROOT / cfg_name)
    ep_with = 0; per_ep = []; types = Counter(); total = 0
    for _ in range(episodes):
        env.reset()
        w = env.world
        own = w.agents[0]
        marked = [(i, o) for i, o in enumerate(list(getattr(w, "obstacles", []) or []))
                  if getattr(o, "encounter_generated", False)] or \
                 [(i, o) for i, o in enumerate(list(getattr(w, "obstacles", []) or [])[:int(getattr(w, 'num_obstacles', 0))])]
        first = {}
        for t in range(steps):
            op = np.asarray(own.state.p_pos, float)[:2]; ov = np.asarray(own.state.p_vel, float)[:2]
            oh = math.atan2(float(ov[1]), float(ov[0])) if float(np.linalg.norm(ov)) > 1e-9 else 0.0
            for i, o in marked:
                if i in first:
                    continue
                tp = np.asarray(o.state.p_pos, float)[:2]; tv = np.asarray(o.state.p_vel, float)[:2]
                th = math.atan2(float(tv[1]), float(tv[0])) if float(np.linalg.norm(tv)) > 1e-9 else 0.0
                ovs = VesselState.from_kinematics(vessel_id="o", position=[0.0, 0.0], heading=oh, speed=float(np.linalg.norm(ov)))
                tvs = VesselState.from_kinematics(vessel_id="t", position=list(tp - op), heading=th, speed=float(np.linalg.norm(tv)))
                rep = engine.evaluate(ovs, tvs)
                et = str(getattr(rep, "encounter_type")).split(".")[-1].lower()
                if et != "safe_passage":
                    first[i] = et
                    types[et] += 1
            v = np.asarray(own.state.p_vel, float)[:2]
            psi = math.atan2(float(v[1]), float(v[0])) if float(np.linalg.norm(v)) > 1e-9 else 0.0
            d = np.asarray(w.landmarks[0].state.p_pos, float)[:2] - np.asarray(own.state.p_pos, float)[:2]
            err = (math.atan2(float(d[1]), float(d[0])) - psi + math.pi) % (2*math.pi) - math.pi
            try:
                _, _, dones, _ = env.step(np.array([[1.0, float(np.clip(2.0*err, -1.0, 1.0))]]))
            except Exception:
                break
            if bool(np.any(dones)):
                break
        per_ep.append(len(first)); total += 1
        if first:
            ep_with += 1
    return {"episodes": total, "M1_formation_rate": ep_with / max(total, 1),
            "M2_formations_per_episode": float(np.mean(per_ep)) if per_ep else 0.0,
            "M3_types": dict(types)}

out = {}
for name in ["BASELINE_NO_ENCOUNTER_V6.txt", "SAC_3DOF_ENTITY_ENCODER_0911_V6.txt"]:
    out[name] = run(name)
    r = out[name]
    print("%-38s M1 形成率 %.0f%% | M2 每回合 %.2f 次 | M3 %s" % (name, 100*r["M1_formation_rate"], r["M2_formations_per_episode"], r["M3_types"]))
Path(ROOT / "docs" / "encounter_formation_ab.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
