
# -*- coding: utf-8 -*-
"""验收：phase 接口 + capture/hold 终止接口（默认关闭）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib
from types import SimpleNamespace
import numpy as np

mod = importlib.import_module("multiagent.scenarios.usv_tracking_base")
mod.done_state = False
cls = None
for _n, _o in vars(mod).items():
    if isinstance(_o, type) and "_update_task_phase" in _o.__dict__:
        cls = _o
        break
assert cls is not None, "phase helper not found"

self = cls.__new__(cls)
self.rew_dis_th = 0.1                      # 100 m 捕获距离 -> approach 半径 500 m

def agent_stub():
    return SimpleNamespace(name="agent 0", state=SimpleNamespace(p_pos=np.zeros(2)),
                           has_captured=False, consecutive_hold=0,
                           reward_done_reason="none", phase="transit")

def world_stub(lm, enabled=False, steps=30):
    return SimpleNamespace(num_landmarks=1, landmarks=[lm],
                           capture_hold_termination_enabled=enabled,
                           capture_hold_success_steps=steps)

lm = SimpleNamespace(name="landmark 0", state=SimpleNamespace(p_pos=np.array([0.05, 0.0])))
a = agent_stub(); w = world_stub(lm)
self._update_task_phase(a, w); print("near  ->", a.phase); assert a.phase == "approach"

lm.state.p_pos = np.array([0.9, 0.0])
self._update_task_phase(a, w); print("far   ->", a.phase); assert a.phase == "transit"

a.has_captured = True
self._update_task_phase(a, w); print("captured ->", a.phase); assert a.phase == "capture"

# 终止接口：默认关闭
a.consecutive_hold = 50
w = world_stub(lm, enabled=False)
print("hold_success OFF -> done=%s reason=%s" % (self.done(a, w), a.reward_done_reason))
assert self.done(a, w) is False

# 终止接口：打开后达标即终止
w = world_stub(lm, enabled=True, steps=30)
res = self.done(a, w)
print("hold_success ON  -> done=%s reason=%s phase=%s" % (res, a.reward_done_reason, a.phase))
assert res is True and a.reward_done_reason == "hold_success"

# 未达标不终止
a.consecutive_hold = 10
a.reward_done_reason = "none"
assert self.done(a, w) is False
print("role_phase_termination_check = PASS")
