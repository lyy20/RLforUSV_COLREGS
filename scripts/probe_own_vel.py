# -*- coding: utf-8 -*-
"""实测：会遇设计读到的本船速度 vs 本船真实速度（3DOF）。"""
import sys
from pathlib import Path
ROOT = Path(r"D:\DSH_USV_Learning\USV_COLREGS")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from measure_encounter_density import build
env = build(ROOT / "last_cfg" / "PROBE_FIRING.txt")
for ep in range(3):
    np.random.seed(700 + ep); env.reset()
    w = env.world; own = w.agents[0]
    def u3():
        st = getattr(own.state, "usv_3dof", None)
        return (float(getattr(st, "u", float("nan"))), float(getattr(st, "psi", float("nan")))) if st is not None else (float("nan"), float("nan"))
    u0, psi0 = u3()
    des = [float(getattr(ob, "encounter_design_dcpa", 0.0)) * 1000 for ob in w.obstacles]
    print("ep%d reset: p_vel=%s | usv_3dof.u=%.3f psi=%.1f | 设计DCPA(m)=%s | 障碍数=%d"
          % (ep, np.round(np.asarray(own.state.p_vel, float), 4), u0, np.degrees(psi0), [round(x,1) for x in des], len(w.obstacles)))
    for t in range(200):
        env.step(np.array([[1.0, 0.0]]))
        if t in (49, 99, 199):
            u, psi = u3()
            print("   t=%3d: p_vel=%s | usv_3dof.u=%.3f (真实航速) | psi=%.1f"
                  % (t, np.round(np.asarray(own.state.p_vel, float), 4), u, np.degrees(psi)))
