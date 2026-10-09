
# -*- coding: utf-8 -*-
"""A0: designed-encounter analysis.

Recomputes rule-internalization metrics from the per-step COLREGs audit log,
using a DESIGNED event window instead of the engine's dynamic risk trip-wire.

Usage:
  python scripts/analyze_designed_encounter.py <step_log.csv> --encounter head_on [--peer "obstacle 0"]
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path

RUDDER_MIN = 0.15          # 归一化舵量阈值：超过才认为“有明确转向动作”
HEADING_MIN_DEG = 10.0     # 或累计航向改变达到该角度
ANTI_THRESHOLD = 0.15      # 反向动作阈值（明显朝错误方向）


def load(path: Path):
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(row, key, default=float("nan")):
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def pick_peer(rows):
    """目标船（encounter peer）= 全程最小净距最小的动态实体。"""
    best, best_val = None, float("inf")
    for row in rows:
        tid = row.get("target_id", "")
        if not tid.startswith("obstacle"):
            continue
        c = fnum(row, "clearance_m")
        if not math.isnan(c) and c < best_val:
            best, best_val = tid, c
    return best, best_val


def analyse(rows, peer, encounter):
    seq = [r for r in rows if r.get("target_id") == peer]
    seq.sort(key=lambda r: int(float(r["step"])))
    if not seq:
        raise SystemExit("no rows for peer %s" % peer)
    dist = [fnum(r, "distance_m") for r in seq]
    idx_cpa = min(range(len(dist)), key=lambda i: dist[i])
    window = seq[: idx_cpa + 1]
    head0 = fnum(window[0], "heading_rad")
    risks = [int(float(r.get("risk_of_collision", 0) or 0)) for r in window]
    comp_audit = [int(float(r.get("action_compliant", 0) or 0)) for r in window]

    def compliant(r):
        rud = fnum(r, "applied_rudder")
        dhead = math.degrees(fnum(r, "heading_rad") - head0)
        if encounter == "head_on":
            return (rud <= -RUDDER_MIN) or (dhead <= -HEADING_MIN_DEG)
        if encounter == "crossing_give_way":
            return (rud <= -RUDDER_MIN) or (dhead <= -HEADING_MIN_DEG)
        if encounter == "crossing_stand_on":
            return abs(rud) <= RUDDER_MIN and abs(dhead) <= HEADING_MIN_DEG
        if encounter == "overtaking":
            return (rud <= -RUDDER_MIN) or (dhead <= -HEADING_MIN_DEG)
        return False

    def anti(r):
        rud = fnum(r, "applied_rudder")
        dhead = math.degrees(fnum(r, "heading_rad") - head0)
        if encounter == "crossing_stand_on":
            return abs(rud) >= ANTI_THRESHOLD or abs(dhead) >= HEADING_MIN_DEG
        return (rud >= ANTI_THRESHOLD) or (dhead >= HEADING_MIN_DEG)

    comp = [1 if compliant(r) else 0 for r in window]
    anti_seq = [1 if anti(r) else 0 for r in window]
    first = next((i for i, c in enumerate(comp) if c), None)
    peak_star = min([fnum(r, "applied_rudder") for r in window] or [0.0])
    peak_port = max([fnum(r, "applied_rudder") for r in window] or [0.0])
    net_head = math.degrees(fnum(window[-1], "heading_rad") - head0)

    summary = {
        "encounter_designed": encounter,
        "peer": peer,
        "window_steps": len(window),
        "dcpa_at_start_m": round(fnum(window[0], "dcpa_m"), 1),
        "tcpa_at_start_s": round(fnum(window[0], "tcpa_s"), 1),
        "min_distance_m": round(dist[idx_cpa], 1),
        "min_clearance_m": round(min([fnum(r, "clearance_m") for r in window]), 1),
        "risk_tripwire_fired": int(max(risks)) if risks else 0,
        "risk_steps": int(sum(risks)),
        "audit_action_compliant_steps": int(sum(comp_audit)),
        "designed_first_compliance_step": (first + 1) if first is not None else None,
        "designed_compliance_ratio": round(sum(comp) / float(len(window)), 3),
        "anti_rule_ratio": round(sum(anti_seq) / float(len(window)), 3),
        "peak_starboard_rudder": round(peak_star, 3),
        "peak_port_rudder": round(peak_port, 3),
        "net_heading_change_deg": round(net_head, 1),
    }
    if summary["risk_tripwire_fired"] == 0:
        verdict = "BLIND_SPOT_NO_RISK_EVENT"
    elif first is None:
        verdict = "VIOLATION_NO_COMPLIANT_ACTION"
    elif first <= max(1, int(0.3 * len(window))):
        verdict = "EARLY_COMPLIANT"
    else:
        verdict = "LATE_COMPLIANT"
    if summary["anti_rule_ratio"] >= 0.5 and first is None:
        verdict = "ANTI_RULE_ACTION"
    summary["verdict"] = verdict
    trace = [
        {"step": int(float(r["step"])), "distance_m": round(fnum(r, "distance_m"), 1),
         "rudder": round(fnum(r, "applied_rudder"), 3), "dir": r.get("policy_rudder_direction", ""),
         "heading_deg": round(math.degrees(fnum(r, "heading_rad")), 1),
         "compliant": compliant(r), "anti": anti(r)}
        for r in window[:: max(1, len(window) // 12)]
    ]
    return summary, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step_log")
    ap.add_argument("--encounter", required=True,
                    choices=["head_on", "crossing_give_way", "crossing_stand_on", "overtaking"])
    ap.add_argument("--peer", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    rows = load(Path(args.step_log))
    peer = args.peer or pick_peer(rows)[0]
    summary, trace = analyse(rows, peer, args.encounter)
    print("== designed encounter summary ==")
    for k, v in summary.items():
        print("  %-32s %s" % (k, v))
    print("== trace (sampled) ==")
    print("  %6s %10s %9s %6s %10s %5s %5s" % ("step", "dist_m", "rudder", "dir", "heading", "ok", "anti"))
    for t in trace:
        print("  %6d %10.1f %9.3f %6s %10.1f %5s %5s" % (t["step"], t["distance_m"], t["rudder"], t["dir"], t["heading_deg"], int(t["compliant"]), int(t["anti"])))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"summary": summary, "trace": trace}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
