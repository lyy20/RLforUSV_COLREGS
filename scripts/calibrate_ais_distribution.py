
# -*- coding: utf-8 -*-
"""P0-6：AIS 真实会遇分布 vs 本项目生成器分布（Moss/Horten 2022 已识别会遇）。"""
import csv, json, statistics as st
from pathlib import Path

SRC = Path(r"D:\DSH_USV_Learning\_refs\collision_avoidance_identifier\Data\Output_data\Identified_encounters\MossHorten2022_example")
TYPES = ["HO", "CRGW", "CRSO", "OTGW", "OTSO"]
M_PER_KN = 0.514444


def num(row, key):
    try:
        return float(row.get(key, "") or "nan")
    except (TypeError, ValueError):
        return float("nan")


def med(vals):
    vals = [v for v in vals if v == v]
    return st.median(vals) if vals else float("nan")


report = {}
print("== AIS 真实会遇分布（Moss/Horten 2022，已识别会遇） ==")
print("  %-6s %5s %10s %10s %10s %12s %12s %10s" % ("类型", "n", "r_cpa(m)", "本船速(m/s)", "他船速(m/s)", "本船长(m)", "他船长(m)", "CPA/船长"))
all_rows = []
for t in TYPES:
    p = SRC / ("%s.csv" % t)
    if not p.exists():
        print("  %-6s  (缺文件)" % t); continue
    rows = list(csv.DictReader(p.open("r", encoding="utf-8-sig", errors="ignore", newline=""), delimiter=";"))
    all_rows += rows
    if not rows:
        print("  %-6s  (空)" % t); continue
    r_cpa = [num(r, "r_cpa") for r in rows]
    own_s = [num(r, "own_speed") * M_PER_KN for r in rows]
    ob_s = [num(r, "obst_speed") * M_PER_KN for r in rows]
    own_l = [num(r, "own_length") for r in rows]
    ob_l = [num(r, "obst_length") for r in rows]
    ratio = [a / b for a, b in zip(r_cpa, ob_l) if a == a and b == b and b > 0]
    report[t] = {"n": len(rows), "r_cpa_med_m": med(r_cpa), "own_speed_med_mps": med(own_s),
                 "obst_speed_med_mps": med(ob_s), "own_len_med_m": med(own_l), "obst_len_med_m": med(ob_l),
                 "cpa_over_length_med": med(ratio)}
    print("  %-6s %5d %10.1f %10.2f %10.2f %12.1f %12.1f %10.2f"
          % (t, len(rows), med(r_cpa), med(own_s), med(ob_s), med(own_l), med(ob_l), med(ratio)))

print("")
print("== 与本项目生成器的对照（按船长归一） ==")
ours_dcpa_m = 35.0      # 生成器 DCPA 采样中位数 (10~60 m)
ours_len_m = 20.0       # 本船直径参考（半径 10 m）
print("  本项目：DCPA∈[10,60] m（中位 %.0f m）/ 船长 %.0f m -> %.2f 倍船长" % (ours_dcpa_m, ours_len_m, ours_dcpa_m / ours_len_m))
real_med = med([v["cpa_over_length_med"] for v in report.values() if v["cpa_over_length_med"] == v["cpa_over_length_med"]])
print("  真实交通：CPA/他船长 中位 = %.2f 倍" % real_med)
print("  结论：真实交通的最小会遇距离约为 %.1f~%.1f 倍船长；本项目 DCPA 中位为 %.2f 倍船长。"
      % (real_med * 0.5, real_med * 1.5, ours_dcpa_m / ours_len_m))
out = Path(r"D:\DSH_USV_Learning\_mainline\refs\ais_calibration.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"real": report, "ours": {"dcpa_med_m": ours_dcpa_m, "len_m": ours_len_m}}, indent=1, ensure_ascii=False), encoding="utf-8")
print("artifact:", out)
