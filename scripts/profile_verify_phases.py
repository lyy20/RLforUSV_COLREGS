# -*- coding: utf-8 -*-
"""验证流程分阶段计时（干净环境，单进程）。"""
import sys, time, math
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build
from utilities.encounter_wrapper import EncounterResetWrapper

ACC = {"snapshot": 0.0, "steps": 0.0, "restore": 0.0, "cpa": 0.0, "enforce": 0.0,
       "relocate": 0.0, "mode_assign": 0.0, "n_step_calls": 0, "n_attempts": 0, "n_relocate": 0}

def wrap(name, key):
    orig = getattr(EncounterResetWrapper, name)
    def inner(self, *a, **k):
        t = time.perf_counter()
        r = orig(self, *a, **k)
        ACC[key] += time.perf_counter() - t
        return r
    setattr(EncounterResetWrapper, name, inner)

for nm, key in [("_snapshot_world", "snapshot"), ("_restore_world", "restore"),
                ("_extrapolated_cpa", "cpa"), ("_enforce_encounter_motion", "enforce"),
                ("_assign_motion_modes", "mode_assign")]:
    wrap(nm, key)
orig_reloc = EncounterResetWrapper.relocate
def timed_reloc(self, *a, **k):
    t = time.perf_counter(); r = orig_reloc(self, *a, **k)
    ACC["relocate"] += time.perf_counter() - t; ACC["n_relocate"] += 1
    return r
EncounterResetWrapper.relocate = timed_reloc

env = build(ROOT / "V6_A2_S1.txt")
inner = env.env
orig_step = inner.step
def timed_step(*a, **k):
    t = time.perf_counter(); r = orig_step(*a, **k)
    ACC["steps"] += time.perf_counter() - t; ACC["n_step_calls"] += 1
    return r
inner.step = timed_step

N = 20
np.random.seed(999)
env.reset()
np.random.seed(999)
t0 = time.perf_counter()
for _ in range(N):
    env.reset()
total = time.perf_counter() - t0
print("=== %d 次 reset，总耗时 %.2f s，单次 %.3f s ===" % (N, total, total / N))
print("验证尝试次数 %d（%.2f 次/reset），relocate 调用 %d 次" % (ACC["n_attempts"], ACC["n_attempts"]/N, ACC["n_relocate"]))
print("env.step 调用 %d 次（%.1f 步/reset）" % (ACC["n_step_calls"], ACC["n_step_calls"]/N))
for k in ["steps", "relocate", "snapshot", "restore", "cpa", "enforce", "mode_assign"]:
    v = ACC[k]
    print("  %-14s %6.2f s  (%.1f%%)  单次平均 %.1f ms" % (k, v, 100*v/total, 1000*v/max(N,1)))
resid = total - sum(ACC[k] for k in ["steps","relocate","snapshot","restore","cpa","enforce","mode_assign"])
print("  %-14s %6.2f s  (%.1f%%)" % ("其他/未计", resid, 100*resid/total))
if ACC["n_step_calls"]:
    print("平均每步 %.2f ms（验证步与训练步混在一起统计）" % (1000*ACC["steps"]/ACC["n_step_calls"]))
