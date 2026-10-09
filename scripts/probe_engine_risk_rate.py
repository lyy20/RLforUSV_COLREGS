
# -*- coding: utf-8 -*-
"""P0-3 代理测量：在 DENSE_WIDE 下，COLREGs 引擎在 reset 时刻判定"有风险"的比例。
这是 rule_filter_active_ratio（教师发声密度）的静态上界代理。"""
import sys, configparser
from pathlib import Path
import numpy as np
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from measure_encounter_density import build
from utilities.scenario_config import read_environment_config
from utilities.envs import wrap_encounter_if_enabled

cfg_path = ROOT / "SAC_3DOF_ENTITY_ENCODER_0911_DENSE_WIDE.txt"
cp = configparser.ConfigParser(); cp.read(str(cfg_path), encoding="utf-8")
env_cfg = dict(read_environment_config(cp))
env_cfg["map_half_size"] = cp["hyperparam"].getfloat("map_half_size")
print("thresholds: horizon=%.0f min_cpa=%.3f km emergency=%.3f km"
      % (env_cfg["colregs_risk_time_horizon"], env_cfg["colregs_min_cpa_distance"],
         env_cfg["colregs_emergency_distance"]))
env = wrap_encounter_if_enabled(build(cfg_path), env_cfg, env_cfg["map_half_size"])
print("wrapped =", type(env).__name__)

def risk_flag(rep):
    for name in ("has_collision_risk", "risk_of_collision", "in_risk", "collision_risk"):
        if hasattr(rep, name):
            return bool(getattr(rep, name)), name
    et = getattr(rep, "encounter_type", None)
    return (str(et) not in ("EncounterType.SAFE_PASSAGE", "safe_passage"), "encounter_type")

n = 40; flagged = 0; total = 0; printed = False; ev = 0
for i in range(n):
    np.random.seed(12000 + i)
    env.reset()
    w = env.world
    a = w.agents[0]
    hit = 0
    for ob in w.obstacles[: w.num_obstacles]:
        rep = w.evaluate_colregs_between(a, ob)
        if not printed:
            names = [x for x in dir(rep) if not x.startswith("_")]
            print("  report fields:", names[:18])
            f, key = risk_flag(rep)
            print("  using flag key:", key)
            printed = True
        f, _ = risk_flag(rep)
        total += 1
        if f:
            flagged += 1; hit += 1
    if hit:
        ev += 1
print("resets=%d  pairs=%d  engine_risk_rate=%.1f%%  resets_with_risk=%.1f%%"
      % (n, total, 100.0 * flagged / max(total, 1), 100.0 * ev / n))
