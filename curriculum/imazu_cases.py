
# -*- coding: utf-8 -*-
"""P0-5：Imazu 22 标准会遇案例接口（解析 colav-simulator 的 YAML，转为相对几何）。

每个案例给出：name / type(HO|CRGW|CRSO|OTGW|OTSO) / ownship(x,y,U,chi) / 其他船相对几何。
本模块只做"读 + 归一化"，不依赖 yaml 库（正则解析所需字段）。
"""
from __future__ import annotations
import math
import re
from pathlib import Path

CASE_DIR_DEFAULT = Path(r"D:\DSH_USV_Learning\_refs\colav-simulator\scenarios\imazu_cases")
TYPE_TO_ENGINE = {
    "HO": "head_on",
    "CRGW": "crossing_give_way",
    "CR_GW": "crossing_give_way",
    "CRSO": "crossing_stand_on",
    "CR_SO": "crossing_stand_on",
    "OTGW": "overtaking_give_way",
    "OT_ing": "overtaking_give_way",
    "OTSO": "overtaken_stand_on",
}


def _parse_case(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    m_type = re.search(r"^type:\s*(\S+)", text, re.M)
    ctype = m_type.group(1) if m_type else "?"
    states = []
    for m in re.finditer(r"csog_state:\s*\[([^\]]+)\]", text):
        parts = [p.strip() for p in m.group(1).split(",")]
        try:
            x, y, u, chi = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
            states.append((x, y, u, chi))
        except (ValueError, IndexError):
            continue
    if not states:
        return None
    return {"name": path.stem, "type": ctype, "ships": states}


def load_cases(case_dir=None):
    d = Path(case_dir) if case_dir else CASE_DIR_DEFAULT
    cases = []
    for f in sorted(d.glob("imazu*.yaml")):
        c = _parse_case(f)
        if c:
            cases.append(c)
    return cases


def to_relative(case, rotate_sign=-1.0):
    """以 ship0 为本船做归一化。

    坐标/航向约定修正（关键）：YAML 的 csog_state = [northing, easting, U, chi]，
    且 chi 为**航海约定**（自正北顺时针）。需转到数学约定：
        east = yaml[1], north = yaml[0]
        psi_math = radians(90 - chi)
    """
    own = case['ships'][0]
    n0, e0, u0, chi0 = own
    psi0 = math.radians(90.0 - chi0)
    c, s = math.cos(-psi0), math.sin(-psi0)
    targets = []
    for (n, e, u, chi) in case['ships'][1:]:
        d_east = e - e0
        d_north = n - n0
        rx = d_east * c - d_north * s
        ry = d_east * s + d_north * c
        psi = math.radians(90.0 - chi) - psi0
        targets.append({'x_m': rx, 'y_m': ry, 'speed_mps': u, 'heading_rad': psi})
    return {'name': case['name'], 'type': case['type'],
            'ownship': {'x_m': 0.0, 'y_m': 0.0, 'speed_mps': u0, 'heading_rad': 0.0},
            'targets': targets}
