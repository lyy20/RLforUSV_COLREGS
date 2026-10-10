# -*- coding: utf-8 -*-
"""验证成本/接受率测量（读 wrapper 计数器）。"""
import sys, time
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from measure_encounter_density import build
for cfg in ["SPEED_VERIFY40.txt", "SPEED_VERIFY20.txt"]:
    env = build(ROOT / cfg)
    steps = int(getattr(env, "cfg", {}).get("encounter_verify_steps", 0))
    tries = int(getattr(env, "cfg", {}).get("encounter_verify_max_tries", 0))
    N = 30
    att = acc = first = 0
    t0 = time.perf_counter()
    for _ in range(N):
        env.reset()
        att += int(getattr(env, "verify_attempts", 0))
        acc += int(getattr(env, "verify_accepts", 0))
        first += int(getattr(env, "verify_first_try_accepts", 0))
    dt = time.perf_counter() - t0
    print("%-22s steps=%2d tries=%d | 每 reset 平均尝试 %.2f 次 | 首次接受率 %.0f%% | 限内接受率 %.0f%% | reset 墙钟 %.3f s"
          % (cfg, steps, tries, att / max(N, 1), 100.0 * first / max(N, 1), 100.0 * acc / max(N, 1), dt / N))
