
# -*- coding: utf-8 -*-
"""A2: specification-robustness sweep for the designed-encounter metric.

Re-runs the designed-encounter analysis across a grid of compliance thresholds and
reports whether the verdict (early-compliant / late / never / anti-rule) is stable.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_designed_encounter as ade

RUDDER_GRID = [0.05, 0.10, 0.15, 0.25, 0.35]
HEADING_GRID = [5.0, 10.0, 15.0, 20.0, 30.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step_log")
    ap.add_argument("--encounter", required=True)
    ap.add_argument("--peer", default=None)
    ap.add_argument("--csv-out", default=None)
    args = ap.parse_args()

    rows = ade.load(Path(args.step_log))
    peer = args.peer or ade.pick_peer(rows)[0]
    print("sweep on %s  peer=%s  encounter=%s" % (Path(args.step_log).name, peer, args.encounter))
    print("  %7s %8s %10s %10s %10s %9s" % ("rud_min", "hdg_min", "first", "ratio", "anti", "verdict"))
    out = []
    for rmin in RUDDER_GRID:
        for hmin in HEADING_GRID:
            ade.RUDDER_MIN = rmin
            ade.HEADING_MIN_DEG = hmin
            s, _ = ade.analyse(rows, peer, args.encounter)
            row = (rmin, hmin, s["designed_first_compliance_step"], s["designed_compliance_ratio"],
                   s["anti_rule_ratio"], s["verdict"])
            out.append(row)
            print("  %7.2f %8.1f %10s %10.3f %10.3f %9s" % row)
    ratios = [r[3] for r in out]
    firsts = [r[2] for r in out]
    print("-- stability --")
    print("  compliance_ratio  min=%.3f max=%.3f spread=%.3f" % (min(ratios), max(ratios), max(ratios) - min(ratios)))
    print("  first_compliance  value_set=%s" % sorted({str(f) for f in firsts})[:6])
    print("  verdict_set       %s" % sorted({r[5] for r in out}))
    if args.csv_out:
        Path(args.csv_out).write_text("rudder_min,heading_min,first_compliance,compliance_ratio,anti_ratio,verdict\n" +
                                      "\n".join("%.2f,%.1f,%s,%.3f,%.3f,%s" % r for r in out), encoding="utf-8")


if __name__ == "__main__":
    main()
