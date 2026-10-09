
# -*- coding: utf-8 -*-
"""参数现实性门（无量纲对照）：把配置参数换算成"船长倍数/惯量比"，与实测区间比较。

依据：
  R1 = min_cpa / L      : AIS 实测 CPA/船长 4.5~9.9（本仓库 calibrate_ais_distribution.py 计算）
  R2 = 探测距离 / L      : 同类实现 rlmpc 1000 m / L=8 m = 125；工程下限取 10
  R3 = 速度 / sqrt(gL)   : 小型船 Froude 数典型 0.15~0.6
  R4 = horizon 内航程 / L : 至少要覆盖若干船长（>=5）
"""
import configparser, math, sys
from pathlib import Path

ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
G = 9.81


def check(cfg_name):
    cp = configparser.ConfigParser(); cp.read(str(ROOT / cfg_name), encoding="utf-8")
    s = cp["hyperparam"]
    L = 2.0 * s.getfloat("agent_radius") * 1000.0          # km->m，直径
    v = s.getfloat("usv_max_surge_speed")
    ob = s.getfloat("ob_range") * 1000.0
    cpa = s.getfloat("colregs_min_cpa_distance") * 1000.0
    horizon = s.getfloat("colregs_risk_time_horizon")
    r1 = cpa / L
    r2 = ob / L
    r3 = v / math.sqrt(G * L)
    r4 = horizon * v / L
    ok1 = 4.0 <= r1 <= 10.0
    ok2 = r2 >= 10.0
    ok3 = 0.15 <= r3 <= 0.6
    ok4 = r4 >= 5.0
    print("== %s ==" % cfg_name)
    print("  L=%.0f m  v=%.2f m/s  ob_range=%.0f m  min_cpa=%.0f m  horizon=%.0f s" % (L, v, ob, cpa, horizon))
    print("  R1 CPA/L      = %5.2f   [现实 4.0~10.0]  %s" % (r1, "OK" if ok1 else "FAIL"))
    print("  R2 探测/L      = %5.2f   [>=10]           %s" % (r2, "OK" if ok2 else "FAIL"))
    print("  R3 Froude     = %5.3f   [0.15~0.60]      %s" % (r3, "OK" if ok3 else "FAIL"))
    print("  R4 horizon航程/L= %5.2f   [>=5]            %s" % (r4, "OK" if ok4 else "FAIL"))
    return all([ok1, ok2, ok3, ok4])


if __name__ == "__main__":
    names = sys.argv[1:] or ["SAC_3DOF_ENTITY_ENCODER_0911.txt",
                             "SAC_3DOF_ENTITY_ENCODER_0911_DENSE_WIDE.txt"]
    results = [check(n) for n in names]
    print("param_realism_check:", "PASS" if all(results) else "FAIL")
