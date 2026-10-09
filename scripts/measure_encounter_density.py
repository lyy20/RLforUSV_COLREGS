
# -*- coding: utf-8 -*-
"""测量“会遇密度”：在给定配置下采样 N 次 reset，统计动态船与本船形成会遇的比例。

会遇判据（两条，与 filter / B 模块口径一致）：
  窄：DCPA < 30 m 且 TCPA < 60 s
  宽：DCPA < 60 m 且 TCPA < 120 s
本船名义速度方向 = 指向目标；速度 = world.agent_nominal_speed（默认 1 m/s）。
"""
import argparse, configparser, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utilities import envs  # noqa: E402
from utilities.scenario_config import read_environment_config  # noqa: E402
from utilities.encounter_wrapper import EncounterResetWrapper  # noqa: E402

def build(config_path):
    cfg = configparser.ConfigParser()
    cfg.read(str(config_path), encoding="utf-8")
    s = cfg["hyperparam"]
    env_cfg = read_environment_config(cfg)
    env_cfg["max_episode_steps"] = s.getint("episode_length")
    env = envs.make_env(
        s.get("SCENARIO"), num_agents=s.getint("num_agents"), num_landmarks=s.getint("num_landmarks"),
        num_obstacles=s.getint("num_obstacles"), num_static_obstacles=s.getint("num_static_obstacles"),
        map_half_size=s.getfloat("map_half_size"),
        static_obstacle_min_size=s.getfloat("static_obstacle_min_size"),
        static_obstacle_max_size=s.getfloat("static_obstacle_max_size"),
        ob_range=s.getfloat("ob_range"), num_ob=s.getint("num_ob"),
        num_static_ob_slots=s.getint("num_static_ob_slots"), landmark_depth=s.getfloat("landmark_depth"),
        landmark_movable=s.getboolean("landmark_movable"), obstacle_movable=s.getboolean("obstacle_movable"),
        landmark_vel=s.getfloat("landmark_vel"), max_vel=s.getfloat("max_vel"),
        random_vel=s.getboolean("random_vel"), movement=s.get("movement"),
        obstacle_movement=s.get("obstacle_movement"), pre_method=s.get("pre_method"),
        rew_err_th=s.getfloat("rew_err_th"), rew_dis_th=s.getfloat("rew_dis_th"),
        max_range=s.getfloat("max_range"), max_current_vel=s.getfloat("max_current_vel"),
        range_dropping=s.getfloat("range_dropping"), control_model=s.get("control_model"),
        target_observation_mode=s.get("target_observation_mode"),
        target_position_noise_std=s.getfloat("target_position_noise_std"),
        scenario_profile=s.get("SCENARIO_PROFILE"),
        dynamic_obstacle_min_speed=s.getfloat("dynamic_obstacle_min_speed"),
        dynamic_obstacle_max_speed=s.getfloat("dynamic_obstacle_max_speed"),
        environment_config=env_cfg, benchmark=True,
    )
    return env


def geometry(p_self, v_self, p_other, v_other):
    p_rel = np.asarray(p_other, float) - np.asarray(p_self, float)
    v_rel = np.asarray(v_other, float) - np.asarray(v_self, float)
    vv = float(np.dot(v_rel, v_rel))
    if vv <= 1e-12:
        return False, float("inf"), float(np.linalg.norm(p_rel))
    tcpa = -float(np.dot(p_rel, v_rel)) / vv
    if tcpa <= 0:
        return False, float("inf"), float(np.linalg.norm(p_rel))
    return True, tcpa, float(np.linalg.norm(p_rel + v_rel * tcpa))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--resets", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=12345)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    cfg_path = Path(args.config) if Path(args.config).is_absolute() else (ROOT / args.config)
    env = build(cfg_path)
    _cfgp = configparser.ConfigParser()
    _cfgp.read(str(cfg_path), encoding='utf-8')
    _ec = dict(read_environment_config(_cfgp))
    _ec['map_half_size'] = _cfgp['hyperparam'].getfloat('map_half_size')
    if float(_ec.get('encounter_generation_mode', 0.0)) >= 0.5:
        env = EncounterResetWrapper(env, _ec)
        print('  [encounter wrapper ENABLED]')
    rng = np.random.RandomState(args.seed0)
    n_narrow = n_wide = n_resets_with_wide = 0
    total_ob = 0
    in_range_steps = []
    scenario = getattr(env.reset_callback, "__self__", None)
    for i in range(args.resets):
        np.random.seed(args.seed0 + i)
        env.reset()
        w = env.world
        a = w.agents[0]; t = w.landmarks[0]
        to_t = np.asarray(t.state.p_pos, float) - np.asarray(a.state.p_pos, float)
        n = np.linalg.norm(to_t)
        if n < 1e-9:
            continue
        v_self = to_t / n * float(getattr(w, "agent_nominal_speed", 0.001))
        found_wide = 0
        for ob in w.obstacles[: w.num_obstacles]:
            if scenario is not None:
                v_ob = scenario._entity_world_velocity(ob)
            else:
                v_ob = ob.state.p_vel
            closing, tcpa, dcpa = geometry(a.state.p_pos, v_self, ob.state.p_pos, v_ob)
            total_ob += 1
            if closing:
                if dcpa < 0.030 and tcpa < 60.0:
                    n_narrow += 1
                if dcpa < 0.060 and tcpa < 120.0:
                    n_wide += 1; found_wide += 1
        if found_wide:
            n_resets_with_wide += 1
    r = float(args.resets)
    print("== encounter density: %s ==" % (args.label or cfg_path.name))
    print("  resets                     %d" % args.resets)
    print("  dynamic-vessel samples     %d" % total_ob)
    print("  narrow (DCPA<30m, TCPA<60s) per-reset  %.4f" % (n_narrow / r))
    print("  wide   (DCPA<60m, TCPA<120s) per-reset %.4f" % (n_wide / r))
    print("  resets with >=1 wide encounter         %.2f%%" % (100.0 * n_resets_with_wide / r))


if __name__ == "__main__":
    main()
