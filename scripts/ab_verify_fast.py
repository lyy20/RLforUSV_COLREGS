# -*- coding: utf-8 -*-
"""A/B：验证快路径是否改变接受决策（固定随机种子，消除跨进程差异）。"""
import sys, time
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build
N = 30
for cfg in ["V6_A2_S1.txt", "FASTOFF_PROBE.txt"]:
    env = build(ROOT / cfg)
    att = acc = first = 0
    np.random.seed(12345)
    env.reset()
    np.random.seed(12345)
    t0 = time.perf_counter()
    for _ in range(N):
        env.reset()
        att += int(getattr(env, "verify_attempts", 0))
        acc += int(getattr(env, "verify_accepts", 0))
        first += int(getattr(env, "verify_first_try_accepts", 0))
    dt = (time.perf_counter() - t0) / N
    fast = bool(env.cfg.get("encounter_verify_fast", True))
    print("%-20s fast=%-5s | 平均尝试 %.2f | 首次接受 %.0f%% | 限内接受 %.0f%% | reset %.3f s"
          % (cfg, fast, att / N, 100.0 * first / N, 100.0 * acc / N, dt))
