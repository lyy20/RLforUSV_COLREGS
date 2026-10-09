
# -*- coding: utf-8 -*-
"""四臂汇总：A_PURE / B_FILTER / C_IL_STAGE1050 / D_W20 —— M13 六项 + A0 指标 + 事件明细。
用法: python scripts/summarize_arms.py [--out docs/arm_summary.md]
"""
import sys, json, argparse
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT / "evaluation")); sys.path.insert(0, str(ROOT / "scripts"))
import joint_report as jr

BASE = Path(r"D:\USV\logs")
ARMS = [
    ("A_PURE",           "SAC_3DOF_BODY_FRAME_ENTITY_ENCODER_ARM_A_PURE",           "A0_ATTRIB_ARM_A"),
    ("B_FILTER",         "SAC_3DOF_BODY_FRAME_ENTITY_ENCODER_ARM_B_FILTER",         "A0_ATTRIB_ARM_B"),
    ("C_IL_STAGE1050",   "SAC_3DOF_BODY_FRAME_ENTITY_ENCODER_ARM_C_IL_STAGE1050",   "A0_ATTRIB_ARM_C1050"),
    ("C_IL_old2085",     "SAC_3DOF_BODY_FRAME_ENTITY_ENCODER_ARM_C_IL",             "A0_ATTRIB_ARM_C"),
    ("D_W20",            "SAC_3DOF_BODY_FRAME_ENTITY_ENCODER_ARM_D_W20",            "A0_ATTRIB_ARM_D"),
]
KEYS = [("a0_compliance_ratio", "A0合规比"), ("a0_anti_rule_ratio", "A0反向比"),
        ("internalization_rate", "内化成功率"), ("zero_event_ratio", "零事件占比"),
        ("success_rate", "任务成功率"), ("timeout_rate", "超时率"), ("filter_active_rate", "过滤介入率")]
rows, detail = {}, {}
for tag, run, evname in ARMS:
    ev = BASE / run / "evaluation" / evname
    if not ev.exists():
        print("[skip] %s: no eval dir" % tag); continue
    latest = sorted([p for p in ev.iterdir() if p.is_dir()], key=lambda p: p.name)[-1]
    try:
        res = jr.build_joint_report(latest, "head_on")
    except Exception as exc:
        print("[err] %s: %s" % (tag, exc)); continue
    rows[tag] = res; detail[tag] = latest.name
print("")
print("| 指标 | " + " | ".join(rows) + " |")
print("| --- |" + " --- |" * len(rows))
for k, label in KEYS:
    cells = []
    for t in rows:
        v = rows[t].get(k)
        cells.append("%.4f" % v if isinstance(v, float) and v == v else "N/A")
    print("| %s | %s |" % (label, " | ".join(cells)))
print("")
print("eval dirs: " + json.dumps(detail, ensure_ascii=False))
out = ROOT / "docs" / "arm_summary.json"
out.write_text(json.dumps({"rows": rows, "dirs": detail}, indent=1, ensure_ascii=False), encoding="utf-8")
print("saved:", out)
