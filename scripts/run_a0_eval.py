
# -*- coding: utf-8 -*-
"""Run a custom evaluation and immediately report A0 designed-encounter metrics.

Usage:
  python scripts/run_a0_eval.py <custom_config.txt> --encounter head_on
         [--variants sac,sac_colregs] [--output-dir DIR] [--episodes N] [--max-steps N]
"""
from __future__ import annotations
import argparse, subprocess, sys, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, cwd):
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd], cwd=str(cwd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("custom_config")
    ap.add_argument("--encounter", required=True,
                    choices=["head_on", "crossing_give_way", "crossing_stand_on", "overtaking"])
    ap.add_argument("--variants", default="sac")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    cfg = Path(args.custom_config)
    out_dir = Path(args.output_dir) if args.output_dir else (ROOT / "curriculum" / (cfg.stem + "_a0"))
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [args.python, "see_trained_custom.py", str(cfg), "--variants", args.variants,
           "--output-dir", str(out_dir)]
    if args.episodes:
        cmd += ["--episodes", str(args.episodes)]
    if args.max_steps:
        cmd += ["--max-steps", str(args.max_steps)]
    code = run(cmd, ROOT)

    step_log = out_dir / "colregs_step_log.csv"
    if not step_log.exists():
        print("[A0] step log not found: %s" % step_log)
        return code
    print("\n===== A0 designed-encounter metrics =====", flush=True)
    rc = run([args.python, "scripts/analyze_designed_encounter.py", str(step_log),
              "--encounter", args.encounter,
              "--json-out", str(out_dir / "designed_encounter.json")], ROOT)
    if rc != 0:
        return rc
    print("\n[A0] artifacts:", flush=True)
    for name in ("designed_encounter.json", "designed_encounter_summary.csv", "colregs_step_log.csv"):
        p = out_dir / name
        if p.exists():
            print("   %s  (%d bytes)" % (p.name, p.stat().st_size))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
