# -*- coding: utf-8 -*-
"""重构等价性探针（确定性版）：固定 PYTHONHASHSEED/random/numpy/torch 后跑 K 步，
输出轨迹摘要。两处 digest 相同 ⇒ 行为数值级一致。
"""
import os, sys, time, random, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("PYTHONHASHSEED", "0")
import numpy as np
import torch
from measure_encounter_density import build

cfg = sys.argv[1] if len(sys.argv) > 1 else "V6_A2_S1.txt"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 400
SEED = 20261010

def reseed(s=SEED):
    random.seed(s); np.random.seed(s)
    try: torch.manual_seed(s)
    except Exception: pass

reseed()
env = build(ROOT / cfg)
reseed()
env.reset()
h = hashlib.sha256()
def feed(arr):
    h.update(np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).tobytes())
t0 = time.perf_counter()
for t in range(K):
    w = env.world; own = w.agents[0]
    v = np.asarray(own.state.p_vel, float)
    psi = np.arctan2(float(v[1]), float(v[0])) if float(np.linalg.norm(v)) > 1e-9 else 0.0
    d = np.asarray(w.landmarks[0].state.p_pos, float) - np.asarray(own.state.p_pos, float)
    err = (np.arctan2(float(d[1]), float(d[0])) - psi + np.pi) % (2 * np.pi) - np.pi
    act = np.array([[1.0, float(np.clip(2.0 * err, -1.0, 1.0))]])
    feed(act)
    obs, rew, done, info = env.step(act)
    feed(obs[0]); feed(rew)
    for e in list(w.agents) + list(w.obstacles) + list(w.landmarks):
        feed(e.state.p_pos); feed(e.state.p_vel)
    if bool(np.any(done)):
        reseed(SEED + t + 1); env.reset()
dt = (time.perf_counter() - t0) / K
print("cfg=%-18s steps=%d  hashseed=%s  digest=%s  %.3f ms/step"
      % (cfg, K, os.environ.get("PYTHONHASHSEED"), h.hexdigest()[:32], 1000 * dt))
