
# -*- coding: utf-8 -*-
"""用 main.py 自带的 filter_active_from_info 提取过滤器发声率（rollout 实测）。"""
import sys, configparser
from pathlib import Path
import numpy as np
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from measure_encounter_density import build
from utilities.scenario_config import read_environment_config
from utilities.envs import wrap_encounter_if_enabled
from main import filter_active_from_info

def probe(cfg_name, episodes=3, steps=200):
    cfg_path = ROOT / cfg_name
    cp = configparser.ConfigParser(); cp.read(str(cfg_path), encoding="utf-8")
    env_cfg = dict(read_environment_config(cp)); env_cfg["map_half_size"] = cp["hyperparam"].getfloat("map_half_size")
    env = wrap_encounter_if_enabled(build(cfg_path), env_cfg, env_cfg["map_half_size"])
    w = env.world; n_ag = int(w.num_agents); adim = int(getattr(w, "agent_action_dim", 2))
    total = 0; active = 0; shown = False
    for ep in range(episodes):
        np.random.seed(555 + ep)
        env.reset()
        for _ in range(steps):
            acts = np.zeros((n_ag, adim), dtype=float)
            res = env.step([acts[i] for i in range(n_ag)])
            info = res[-1]
            if not shown:
                print("  info.shape =", getattr(info, "shape", None), "dtype =", getattr(info, "dtype", None))
                shown = True
            fa = filter_active_from_info(info, acts)
            for v in np.asarray(fa, dtype=float).reshape(-1)[:n_ag]:
                total += 1
                active += int(v > 0.0)
    rate = 100.0 * active / max(total, 1)
    print("  %-46s steps=%d  filter_active_rate=%.2f%%" % (cfg_name, total, rate))
    return rate

print("== 过滤器发声率（三臂，rollout 实测） ==")
a = probe("SAC_3DOF_ENTITY_ENCODER_ARM_A_PURE.txt")
b = probe("SAC_3DOF_ENTITY_ENCODER_ARM_B_FILTER.txt")
c = probe("SAC_3DOF_ENTITY_ENCODER_ARM_C_IL.txt")
print("summary: A_pure=%.2f%%  B_filter=%.2f%%  C_il=%.2f%%" % (a, b, c))