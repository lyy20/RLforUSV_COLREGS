
# -*- coding: utf-8 -*-
"""诊断 + 测量：先看清 env.step 的 info 结构，再提取 filter_active。"""
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
env_cfg = dict(read_environment_config(cp)); env_cfg["map_half_size"] = cp["hyperparam"].getfloat("map_half_size")
env = wrap_encounter_if_enabled(build(cfg_path), env_cfg, env_cfg["map_half_size"])
w = env.world
adim = int(getattr(w, "agent_action_dim", 2))
out = env.reset()
res = env.step([np.zeros(adim, dtype=float) for _ in range(int(w.num_agents))])
info = res[-1] if isinstance(res, (list, tuple)) else None
print("len(out) =", len(res) if isinstance(res, (list, tuple)) else "n/a")
print("type(info) =", type(info))
if isinstance(info, dict):
    print("dict keys =", list(info.keys())[:14])
    fa = info.get("filter_active")
    print("  filter_active type =", type(fa), "value =", np.asarray(fa).reshape(-1)[:4] if fa is not None else None)
elif isinstance(info, np.ndarray):
    print("ndarray dtype names =", info.dtype.names)
else:
    print("repr(info)[:300] =", repr(info)[:300])
