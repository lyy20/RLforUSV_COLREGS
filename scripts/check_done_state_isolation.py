
# -*- coding: utf-8 -*-
"""验收：per-agent done_state + 多智能体防腐门。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib
from types import SimpleNamespace

mod = importlib.import_module("multiagent.scenarios.usv_tracking_base")
cls = None
for _n, _o in vars(mod).items():
    if isinstance(_o, type) and "done" in _o.__dict__ and "reward" in _o.__dict__:
        cls = _o; break
assert cls is not None
self = cls.__new__(cls)

a = SimpleNamespace(done_state=False)
w1 = SimpleNamespace(num_agents=1)
mod.done_state = True                       # 旧全局量置 True，验证不会污染 per-agent
print("per-agent False (global True) ->", self.done(a, w1)); assert self.done(a, w1) is False
a.done_state = True
print("per-agent True  ->", self.done(a, w1)); assert self.done(a, w1) is True

# 防腐：num_agents>1 且缺 per-agent 字段 -> 快速失败
b = SimpleNamespace()
w2 = SimpleNamespace(num_agents=2)
try:
    self.done(b, w2)
    raise AssertionError("应当抛出 RuntimeError")
except RuntimeError as exc:
    print("guard(num_agents=2, missing field) -> RuntimeError OK")

# 单智能体且缺字段 -> 允许回退到全局量（向后兼容）
c = SimpleNamespace()
w3 = SimpleNamespace(num_agents=1)
print("fallback(num_agents=1) ->", self.done(c, w3)); assert self.done(c, w3) is True
print("done_state_isolation_check = PASS")
