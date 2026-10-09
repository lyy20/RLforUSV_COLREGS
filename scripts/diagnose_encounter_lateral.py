
# -*- coding: utf-8 -*-
"""诊断：动态船相对于“本船→目标”航路的横向偏移与速度，解释为何会遇密度为 0。"""
import sys, configparser
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from measure_encounter_density import build  # noqa

env = build(ROOT / "SAC_3DOF_ENTITY_ENCODER_0911.txt")
laterals, speeds, dists = [], [], []
scenario = getattr(env.reset_callback, "__self__", None)
for i in range(60):
    np.random.seed(12345 + i)
    env.reset()
    w = env.world
    a = w.agents[0]; t = w.landmarks[0]
    route = np.asarray(t.state.p_pos, float) - np.asarray(a.state.p_pos, float)
    L = np.linalg.norm(route)
    u = route / L
    nrm = np.array([-u[1], u[0]])
    for ob in w.obstacles[: w.num_obstacles]:
        rel = np.asarray(ob.state.p_pos, float) - np.asarray(a.state.p_pos, float)
        laterals.append(float(np.dot(rel, nrm)) * 1000.0)
        dists.append(float(np.linalg.norm(rel)) * 1000.0)
        v = scenario._entity_world_velocity(ob) if scenario is not None else ob.state.p_vel
        speeds.append(float(np.linalg.norm(v)) * 1000.0)
laterals = np.asarray(laterals); speeds = np.asarray(speeds); dists = np.asarray(dists)
print("samples=%d  route_len~%.0f m" % (laterals.size, L * 1000))
print("|lateral| m: mean=%.0f  p10=%.0f  p50=%.0f  p90=%.0f  max=%.0f"
      % (np.abs(laterals).mean(), np.percentile(np.abs(laterals), 10), np.percentile(np.abs(laterals), 50),
         np.percentile(np.abs(laterals), 90), np.abs(laterals).max()))
print("fraction with |lateral| <= 60 m : %.3f" % float(np.mean(np.abs(laterals) <= 60.0)))
print("fraction with |lateral| <= 150 m: %.3f" % float(np.mean(np.abs(laterals) <= 150.0)))
print("obstacle speed m/s: mean=%.2f  p50=%.2f  min=%.2f  max=%.2f  zero=%.2f%%"
      % (speeds.mean(), np.percentile(speeds, 50), speeds.min(), speeds.max(), 100.0 * float(np.mean(speeds < 1e-6))))
print("obstacle distance m: p10=%.0f p50=%.0f p90=%.0f" % (np.percentile(dists, 10), np.percentile(dists, 50), np.percentile(dists, 90)))
