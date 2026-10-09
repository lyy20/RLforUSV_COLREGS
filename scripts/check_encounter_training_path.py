
# -*- coding: utf-8 -*-
"""验收：训练链路所用 helper（envs.wrap_encounter_if_enabled）是否真的产生会遇。"""
import sys, configparser
from pathlib import Path
import numpy as np
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from measure_encounter_density import build, geometry
from utilities.scenario_config import read_environment_config
from utilities.envs import wrap_encounter_if_enabled

cfg_path = ROOT / "SAC_3DOF_ENTITY_ENCODER_0911_DENSE.txt"
cp = configparser.ConfigParser(); cp.read(str(cfg_path), encoding="utf-8")
env_cfg = dict(read_environment_config(cp))
env_cfg["map_half_size"] = cp["hyperparam"].getfloat("map_half_size")
env = build(cfg_path)
env = wrap_encounter_if_enabled(env, env_cfg, env_cfg["map_half_size"])
print("wrapped type =", type(env).__name__)
wide = 0; n_reset = 60; with_ev = 0
for i in range(n_reset):
    np.random.seed(9000 + i)
    env.reset()
    w = env.world
    a = w.agents[0]; t = w.landmarks[0]
    to_t = np.asarray(t.state.p_pos, float) - np.asarray(a.state.p_pos, float)
    n = float(np.linalg.norm(to_t))
    v_s = to_t / n * float(getattr(w, "agent_nominal_speed", 0.001))
    hit = 0
    for ob in w.obstacles[: w.num_obstacles]:
        closing, tcpa, dcpa = geometry(a.state.p_pos, v_s, ob.state.p_pos, np.asarray(ob.state.p_vel, float))
        if closing and dcpa < 0.060 and tcpa < 120.0:
            wide += 1; hit += 1
    if hit:
        with_ev += 1
print("resets=%d  wide_encounters/reset=%.4f  resets_with_event=%.2f%%" % (n_reset, wide / float(n_reset), 100.0 * with_ev / n_reset))
assert with_ev / float(n_reset) >= 0.8, "training path produced too few encounters"
assert wide / float(n_reset) >= 1.0, "density below target"
print("encounter_training_path_check = PASS")
