"""船舶身份参数集：加载、推导（3DOF 可执行约束）与按 AIS 比例抽样。

数据源：configs/vessel_identities.json（人读版为同目录 .yaml）。
依据分级 A 权威标准 / B 实测数据 / C 同行实现 / D 任务设定。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'vessel_identities.json'


def load_identities(path=None) -> dict:
    p = Path(path) if path else DEFAULT_PATH
    with p.open('r', encoding='utf-8') as fh:
        return json.load(fh)


def _rudder_rate_for(length_m: float) -> float:
    """舵速：>=50 m 用 SOLAS 35°→35°/28 s ≈ 2.5 °/s；小艇取 15 °/s（C 级，待补引用）。"""
    return 2.5 if float(length_m) >= 50.0 else 15.0


def _accel_for(identity: dict) -> float:
    """加速度：优先用身份自带；否则按船长分档（商船 0.05、中型 0.5、小艇 1.5）。"""
    a = identity.get('max_accel_mps2')
    if a:
        return float(a)
    L = float(identity['length_m'])
    if L >= 100.0:
        return 0.05
    if L >= 30.0:
        return 0.5
    return 1.5


def derive(identity: dict, scene: dict | None = None) -> dict:
    """把身份推导成本项目环境可直接使用的约束量。"""
    scene = scene or {}
    scale = float(scene.get('length_scale', 1.0) or 1.0)
    L = float(identity['length_m']) * scale
    v = identity.get('speed_max_mps') or identity.get('speed_median_mps')
    v = float(v) if v else 3.0
    delta_max = math.radians(float(identity.get('max_rudder_angle_deg', 35.0)))
    r_imo_bound = 2.0 * v / (5.0 * L)          # rad/s，IMO 回转直径 <= 5L 对应的稳态转艏率下限
    r_identity = identity.get('max_turn_rate_deg_s')
    r_ss = math.radians(float(r_identity)) if r_identity else r_imo_bound
    rudder_rate = _rudder_rate_for(L)
    k_index = r_ss / delta_max if delta_max > 0 else float('nan')
    yaw_accel = k_index * math.radians(rudder_rate)
    imo_bound_deg_s = math.degrees(r_imo_bound)
    impl_cap_deg_s = math.degrees(r_ss)
    # 现实区间：下限=IMO 能力要求（回转直径<=5L），上限=实现速率饱和；
    # 推荐值取上限，但不超过下限的 3 倍（避免超出任何真实小艇的敏捷度）
    rec_deg_s = min(impl_cap_deg_s, max(imo_bound_deg_s, 3.0 * imo_bound_deg_s)) if impl_cap_deg_s > 0 else imo_bound_deg_s
    return {
        'id': identity['id'],
        'grade': identity.get('grade', 'D'),
        'length_m': L,
        'beam_m': float(identity.get('beam_m') or 0.0) * scale,
        'radius_km': 0.5 * L / 1000.0,
        'speed_max_mps': v,
        'max_accel_mps2': _accel_for(identity),
        'max_rudder_angle_deg': float(identity.get('max_rudder_angle_deg', 35.0)),
        'rudder_rate_deg_s': rudder_rate,
        'max_yaw_rate_deg_s': math.degrees(r_ss),
        'yaw_max_imo_bound_deg_s': imo_bound_deg_s,
        'yaw_max_impl_cap_deg_s': impl_cap_deg_s,
        'yaw_max_recommended_deg_s': rec_deg_s,
        'yaw_accel_deg_s2': math.degrees(yaw_accel),
        'k_index_1_s': k_index,
        'turning_diameter_L': 2.0 * v / r_ss / L if r_ss > 0 else float('inf'),
    }


def sample_identity(rng, data: dict | None = None) -> dict:
    """按 sampling.weights 抽样一个身份。"""
    data = data or load_identities()
    ids = [i['id'] for i in data['identities']]
    w = np.array([float(data['sampling']['weights'].get(i, 0.0)) for i in ids], dtype=float)
    if w.sum() <= 0:
        raise ValueError('vessel identity weights must sum > 0')
    pick = int(rng.choice(len(ids), p=w / w.sum()))
    return data['identities'][pick]


def apply_to_environment_config(env_cfg: dict, identity: dict, scene: dict | None = None) -> dict:
    """把身份写进环境配置（动态船尺寸/速度/转向能力）。返回被修改的副本。"""
    d = derive(identity, scene)
    out = dict(env_cfg)
    out['dynamic_obstacle_min_size'] = round(d['radius_km'] * 0.9, 6)
    out['dynamic_obstacle_max_size'] = round(d['radius_km'] * 1.1, 6)
    out['dynamic_obstacle_max_turn_rate_deg_s'] = round(d['max_yaw_rate_deg_s'], 4)
    out['dynamic_obstacle_max_accel_mps2'] = round(d['max_accel_mps2'], 4)
    return out


def self_test() -> None:
    data = load_identities()
    print('identities =', len(data['identities']), '| sampling enabled =', data['sampling']['enabled'])
    print('%-18s %6s %6s %8s %8s %8s %8s %8s %8s' % ('id', 'L(m)', 'v(m/s)', 'accel', 'rudRate', 'yawIMO', 'yawCap', 'yawRec', 'turnD/L'))
    for ident in data['identities']:
        d = derive(ident, data['scene'])
        print('%-18s %6.2f %6.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f'
              % (d['id'], d['length_m'], d['speed_max_mps'], d['max_accel_mps2'],
                 d['rudder_rate_deg_s'], d['yaw_max_imo_bound_deg_s'], d['yaw_max_impl_cap_deg_s'],
                 d['yaw_max_recommended_deg_s'], d['turning_diameter_L']))
    rng = np.random.default_rng(0)
    counts = {}
    for _ in range(2000):
        i = sample_identity(rng, data)['id']
        counts[i] = counts.get(i, 0) + 1
    print('sampled(2000):', {k: round(v / 2000.0, 3) for k, v in sorted(counts.items())})


if __name__ == '__main__':
    self_test()