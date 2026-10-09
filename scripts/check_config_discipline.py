
# -*- coding: utf-8 -*-
"""配置纪律门：换 run 名后若 PRE_TRAINED=true 而目标 run 无 checkpoint，直接报错。"""
import sys, configparser
from pathlib import Path
LOGS = Path(r"D:\USV\logs")
def check(cfg_path):
    cp = configparser.ConfigParser(); cp.read(str(cfg_path), encoding="utf-8")
    s = cp["hyperparam"]
    run = s.get("TRAINING_RUN_NAME")
    pre = s.getboolean("PRE_TRAINED", fallback=False)
    md = LOGS / run / "model_dir"
    has_ckpt = md.exists() and any(md.glob("episode_*.pt"))
    ok = (not pre) or has_ckpt
    print("%-58s PRE_TRAINED=%-5s ckpt=%-5s %s" % (Path(cfg_path).name, pre, has_ckpt, "OK" if ok else "FAIL"))
    return ok
if __name__ == "__main__":
    import glob, os
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\DSH_USV_Learning\USV_COLREGS")
    bad = 0
    for f in sorted(root.glob("*0911*.txt")):
        try:
            if not check(f): bad += 1
        except Exception as exc:
            print("%-58s SKIP (%s)" % (f.name, exc))
    print("config_discipline_check:", "PASS" if bad == 0 else "FAIL(%d)" % bad)
    sys.exit(0 if bad == 0 else 1)
