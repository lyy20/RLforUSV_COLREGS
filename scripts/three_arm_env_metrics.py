# -*- coding: utf-8 -*-
"""三组对比：环境层指标（会遇覆盖 / 教师可发声步比例 / 接近度 / 设计兑现度）。"""
import sys, math, json
from pathlib import Path
from collections import Counter
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build
from safety.colregs import COLREGsEngine, COLREGsConfig, VesselState

engine = COLREGsEngine(COLREGsConfig())
CFGS = ["COST_Aoff_Boff.txt", "COST_Aon_Boff.txt", "COST_Aon_Bon.txt"]
LABEL = {"COST_Aoff_Boff.txt": "最简单", "COST_Aon_Boff.txt": "+CPA反推", "COST_Aon_Bon.txt": "+CPA反推+筛选"}
EP, STEPS = 8, 250
out = {}
for cfg in CFGS:
    env = build(ROOT / cfg)
    ep_with = 0; per_ep = []; types = Counter()
    step_any = 0; step_tot = 0; mind = []; ratio = []
    for _ in range(EP):
        np.random.seed(11); env.reset()
        w = env.world; own = w.agents[0]
        marked = [(i, o) for i, o in enumerate(list(getattr(w, "obstacles", []) or []))]
        first = {}; mins = {}
        for i, o in marked:
            mins[i] = float(np.linalg.norm(np.asarray(o.state.p_pos, float) - np.asarray(own.state.p_pos, float)))
            d0 = float(getattr(o, "encounter_design_dcpa", 0.0))
            if d0 > 0: ratio.append(d0)
        for t in range(STEPS):
            op = np.asarray(own.state.p_pos, float)[:2]; ov = np.asarray(own.state.p_vel, float)[:2]
            oh = math.atan2(float(ov[1]), float(ov[0])) if float(np.linalg.norm(ov)) > 1e-9 else 0.0
            any_risk = False
            for i, o in marked:
                tp = np.asarray(o.state.p_pos, float)[:2]; tv = np.asarray(o.state.p_vel, float)[:2]
                th = math.atan2(float(tv[1]), float(tv[0])) if float(np.linalg.norm(tv)) > 1e-9 else 0.0
                ovs = VesselState.from_kinematics(vessel_id="o", position=[0.0, 0.0], heading=oh, speed=float(np.linalg.norm(ov)))
                tvs = VesselState.from_kinematics(vessel_id="t", position=list(tp - op), heading=th, speed=float(np.linalg.norm(tv)))
                rep = engine.evaluate(ovs, tvs)
                et = str(getattr(rep, "encounter_type")).split(".")[-1].lower()
                if et != "safe_passage":
                    any_risk = True
                    if i not in first:
                        first[i] = et; types[et] += 1
            step_any += 1 if any_risk else 0; step_tot += 1
            v = np.asarray(own.state.p_vel, float)[:2]
            psi = math.atan2(float(v[1]), float(v[0])) if float(np.linalg.norm(v)) > 1e-9 else 0.0
            d = np.asarray(w.landmarks[0].state.p_pos, float)[:2] - np.asarray(own.state.p_pos, float)[:2]
            err = (math.atan2(float(d[1]), float(d[0])) - psi + math.pi) % (2*math.pi) - math.pi
            try:
                _, _, dones, _ = env.step(np.array([[1.0, float(np.clip(2.0*err, -1.0, 1.0))]]))
            except Exception:
                break
            ps = np.asarray(own.state.p_pos, float)[:2]
            for i, o in marked:
                dd = float(np.linalg.norm(np.asarray(o.state.p_pos, float)[:2] - ps))
                if dd < mins[i]: mins[i] = dd
            if bool(np.any(dones)):
                break
        per_ep.append(len(first))
        if first: ep_with += 1
        mind.extend(list(mins.values()))
    out[cfg] = {"label": LABEL[cfg], "M1_formation_rate": ep_with / EP,
                "M2_per_episode": float(np.mean(per_ep)), "M3_types": dict(types),
                "teacher_step_ratio": step_any / max(step_tot, 1),
                "min_dist_median_m": float(np.median(mind) * 1000.0),
                "design_dcpa_median_m": float(np.median(ratio) * 1000.0) if ratio else 0.0}
    r = out[cfg]
    print("%-18s M1=%.0f%% M2=%.2f 教师步比=%.1f%% 最近中位=%.0fm 设计中位=%.0fm 类型=%s"
          % (LABEL[cfg], 100*r["M1_formation_rate"], r["M2_per_episode"],
             100*r["teacher_step_ratio"], r["min_dist_median_m"], r["design_dcpa_median_m"], r["M3_types"]))
(ROOT / "docs" / "three_arm_env_metrics.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
