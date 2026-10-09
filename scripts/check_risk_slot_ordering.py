
# -*- coding: utf-8 -*-
"""W1 验收：动态槽位必须按“风险优先”而不是“距离优先”排序。

反例构造：A 船 60 m 正前方但同速同向（正在远离/无会遇）；B 船 180 m 正前方对向驶来（DCPA≈10 m）。
旧逻辑（按距离）会把 A 排在 B 前面；新逻辑必须把 B 排在前面。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib
from types import SimpleNamespace
import numpy as np

mod = importlib.import_module("multiagent.scenarios.usv_tracking_base")
cls = None
for _n, _o in vars(mod).items():
    if isinstance(_o, type) and "_dynamic_entity_risk_key" in _o.__dict__:
        cls = _o
        break
assert cls is not None, "risk-key method not found"

self = cls.__new__(cls)

def ent(pos, vel):
    return SimpleNamespace(state=SimpleNamespace(p_pos=np.array(pos, dtype=float),
                                                p_vel=np.array(vel, dtype=float)))

agent = ent([0.0, 0.0], [0.0014, 0.0])                    # 1.4 m/s = 0.0014 km/s（与实体同单位）
near_opening = ent([0.060, 0.005], [0.0014, 0.0])          # 60 m ahead, same course/speed
far_closing = ent([0.180, 0.010], [-0.0014, 0.0])          # 180 m ahead, head-on
mid_closing = ent([0.120, 0.004], [-0.0014, 0.0])          # 120 m ahead, head-on

order = []
for e in (near_opening, far_closing, mid_closing):
    self._add_nearby_dynamic_entity(order, e, agent, 0.2)

names = []
for e in order:
    names.append("near_opening" if e is near_opening else "far_closing" if e is far_closing else "mid_closing")
print("ordered =", names)
print("keys    =", [tuple(round(v, 5) for v in self._dynamic_entity_risk_key(e, agent)) for e in order])

assert names[0] == "mid_closing", "最近的交会船应排第一（TCPA 最小）"
assert names[1] == "far_closing", "较远的交会船应排在“更近但正远离”的船之前"
assert names[2] == "near_opening", "无会遇（远离）的船优先级最低"
print("risk_slot_ordering_check = PASS")
