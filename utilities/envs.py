from utilities.env_wrapper import SubprocVecEnv, DummyVecEnv
import numpy as np
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv

DEFAULT_NUM_AGENTS = 1
DEFAULT_NUM_LANDMARKS = 1
DEFAULT_NUM_OBSTACLES = 5
DEFAULT_NUM_STATIC_OBSTACLES = 8
DEFAULT_MAP_HALF_SIZE = 2.0
DEFAULT_STATIC_OBSTACLE_MIN_SIZE = 0.04
DEFAULT_STATIC_OBSTACLE_MAX_SIZE = 0.10
DEFAULT_OB_RANGE = 0.8
DEFAULT_NUM_OB = 3
DEFAULT_NUM_STATIC_OB_SLOTS = 1
DEFAULT_LANDMARK_DEPTH = 0.
DEFAULT_LANDMARK_MOVABLE = True
DEFAULT_OBSTACLE_MOVABLE = True
DEFAULT_LANDMARK_VEL = 0.0003
DEFAULT_MAX_VEL = 0.0003
DEFAULT_RANDOM_VEL = False
DEFAULT_MOVEMENT = 'escape'
DEFAULT_OBSTACLE_MOVEMENT = 'linear'
DEFAULT_PRE_METHOD = 'LS'
DEFAULT_REW_ERR_TH = 0.03
DEFAULT_REW_DIS_TH = 0.3
DEFAULT_MAX_RANGE = 2.0
DEFAULT_MAX_CURRENT_VEL = 0.0003
DEFAULT_RANGE_DROPPING = 0.01
DEFAULT_CONTROL_MODEL = 'usv_3dof'
DEFAULT_TARGET_OBSERVATION_MODE = 'direct_noisy'
DEFAULT_TARGET_POSITION_NOISE_STD = 0.005
DEFAULT_SCENARIO_PROFILE = 'DEFAULT'
DEFAULT_DYNAMIC_OBSTACLE_MIN_SPEED = None
DEFAULT_DYNAMIC_OBSTACLE_MAX_SPEED = None
DEFAULT_REWARD_CONFIG = None
DEFAULT_ENVIRONMENT_CONFIG = None


def wrap_encounter_if_enabled(env, environment_config=None, map_half_size=None):
    """当 encounter_generation_mode >= 0.5 时套上会遇生成包装器（P0-1/P0-2）。"""
    cfg = dict(environment_config or {})
    if map_half_size is not None:
        cfg['map_half_size'] = float(map_half_size)
    try:
        mode = float(cfg.get('encounter_generation_mode', 0.0) or 0.0)
    except (TypeError, ValueError):
        mode = 0.0
    if mode < 0.5:
        return env
    from utilities.encounter_wrapper import EncounterResetWrapper
    return EncounterResetWrapper(env, cfg)

def make_parallel_env(
    n_rollout_threads,
    scenario,
    seed=1,
    num_agents=DEFAULT_NUM_AGENTS,
    num_landmarks=DEFAULT_NUM_LANDMARKS,
    num_obstacles=DEFAULT_NUM_OBSTACLES,
    num_static_obstacles=DEFAULT_NUM_STATIC_OBSTACLES,
    map_half_size=DEFAULT_MAP_HALF_SIZE,
    static_obstacle_min_size=DEFAULT_STATIC_OBSTACLE_MIN_SIZE,
    static_obstacle_max_size=DEFAULT_STATIC_OBSTACLE_MAX_SIZE,
    ob_range=DEFAULT_OB_RANGE,
    num_ob=DEFAULT_NUM_OB,
    num_static_ob_slots=DEFAULT_NUM_STATIC_OB_SLOTS,
    landmark_depth=DEFAULT_LANDMARK_DEPTH,
    landmark_movable=DEFAULT_LANDMARK_MOVABLE,
    obstacle_movable=DEFAULT_OBSTACLE_MOVABLE,
    landmark_vel=DEFAULT_LANDMARK_VEL,
    max_vel=DEFAULT_MAX_VEL,
    random_vel=DEFAULT_RANDOM_VEL,
    movement=DEFAULT_MOVEMENT,
    obstacle_movement=DEFAULT_OBSTACLE_MOVEMENT,
    pre_method=DEFAULT_PRE_METHOD,
    rew_err_th=DEFAULT_REW_ERR_TH,
    rew_dis_th=DEFAULT_REW_DIS_TH,
    max_range=DEFAULT_MAX_RANGE,
    max_current_vel=DEFAULT_MAX_CURRENT_VEL,
    range_dropping=DEFAULT_RANGE_DROPPING,
    control_model=DEFAULT_CONTROL_MODEL,
    target_observation_mode=DEFAULT_TARGET_OBSERVATION_MODE,
    target_position_noise_std=DEFAULT_TARGET_POSITION_NOISE_STD,
    scenario_profile=DEFAULT_SCENARIO_PROFILE,
    dynamic_obstacle_min_speed=DEFAULT_DYNAMIC_OBSTACLE_MIN_SPEED,
    dynamic_obstacle_max_speed=DEFAULT_DYNAMIC_OBSTACLE_MAX_SPEED,
    reward_config=DEFAULT_REWARD_CONFIG,
    environment_config=DEFAULT_ENVIRONMENT_CONFIG,
    benchmark=False,
    profile_timing=False,
):
    #print('Make parallel env')
    def get_env_fn(rank):
        #print('Get env fn')
        def init_env():
            #print('Init env')
            # env = make_env("simple_adversary")
            env = make_env(
                scenario,
                num_agents=num_agents,
                num_landmarks=num_landmarks,
                num_obstacles=num_obstacles,
                num_static_obstacles=num_static_obstacles,
                map_half_size=map_half_size,
                static_obstacle_min_size=static_obstacle_min_size,
                static_obstacle_max_size=static_obstacle_max_size,
                ob_range=ob_range,
                num_ob=num_ob,
                num_static_ob_slots=num_static_ob_slots,
                landmark_depth=landmark_depth,
                landmark_movable=landmark_movable,
                obstacle_movable=obstacle_movable,
                landmark_vel=landmark_vel,
                max_vel=max_vel,
                random_vel=random_vel,
                movement=movement,
                obstacle_movement=obstacle_movement,
                pre_method=pre_method,
                rew_err_th=rew_err_th,
                rew_dis_th=rew_dis_th,
                max_range=max_range,
                max_current_vel=max_current_vel,
                range_dropping=range_dropping,
                control_model=control_model,
                target_observation_mode=target_observation_mode,
                target_position_noise_std=target_position_noise_std,
                scenario_profile=scenario_profile,
                dynamic_obstacle_min_speed=dynamic_obstacle_min_speed,
                dynamic_obstacle_max_speed=dynamic_obstacle_max_speed,
                reward_config=reward_config,
                environment_config=environment_config,
                benchmark=benchmark,
                profile_timing=profile_timing,
            )
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            env = wrap_encounter_if_enabled(env, environment_config, map_half_size)
            return env
        return init_env
#    if n_rollout_threads == 1:
#        return DummyVecEnv([get_env_fn(0)])
#    else:
    return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def make_env(
    scenario_name,
    num_agents=DEFAULT_NUM_AGENTS,
    num_landmarks=DEFAULT_NUM_LANDMARKS,
    num_obstacles=DEFAULT_NUM_OBSTACLES,
    num_static_obstacles=DEFAULT_NUM_STATIC_OBSTACLES,
    map_half_size=DEFAULT_MAP_HALF_SIZE,
    static_obstacle_min_size=DEFAULT_STATIC_OBSTACLE_MIN_SIZE,
    static_obstacle_max_size=DEFAULT_STATIC_OBSTACLE_MAX_SIZE,
    ob_range=DEFAULT_OB_RANGE,
    num_ob=DEFAULT_NUM_OB,
    num_static_ob_slots=DEFAULT_NUM_STATIC_OB_SLOTS,
    landmark_depth=DEFAULT_LANDMARK_DEPTH,
    landmark_movable=DEFAULT_LANDMARK_MOVABLE,
    obstacle_movable=DEFAULT_OBSTACLE_MOVABLE,
    landmark_vel=DEFAULT_LANDMARK_VEL,
    max_vel=DEFAULT_MAX_VEL,
    random_vel=DEFAULT_RANDOM_VEL,
    movement=DEFAULT_MOVEMENT,
    obstacle_movement=DEFAULT_OBSTACLE_MOVEMENT,
    pre_method=DEFAULT_PRE_METHOD,
    rew_err_th=DEFAULT_REW_ERR_TH,
    rew_dis_th=DEFAULT_REW_DIS_TH,
    max_range=DEFAULT_MAX_RANGE,
    max_current_vel=DEFAULT_MAX_CURRENT_VEL,
    range_dropping=DEFAULT_RANGE_DROPPING,
    control_model=DEFAULT_CONTROL_MODEL,
    target_observation_mode=DEFAULT_TARGET_OBSERVATION_MODE,
    target_position_noise_std=DEFAULT_TARGET_POSITION_NOISE_STD,
    scenario_profile=DEFAULT_SCENARIO_PROFILE,
    dynamic_obstacle_min_speed=DEFAULT_DYNAMIC_OBSTACLE_MIN_SPEED,
    dynamic_obstacle_max_speed=DEFAULT_DYNAMIC_OBSTACLE_MAX_SPEED,
    reward_config=DEFAULT_REWARD_CONFIG,
    environment_config=DEFAULT_ENVIRONMENT_CONFIG,
    benchmark=False,
    profile_timing=False,
):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.
    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)
    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    
    # load scenario from script
    #print('load scenario')
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    #print('Create world')
    world = scenario.make_world(
        num_agents=num_agents,
        num_landmarks=num_landmarks,
        num_obstacles=num_obstacles,
        num_static_obstacles=num_static_obstacles,
        map_half_size=map_half_size,
        static_obstacle_min_size=static_obstacle_min_size,
        static_obstacle_max_size=static_obstacle_max_size,
        ob_range=ob_range,
        num_ob=num_ob,
        num_static_ob_slots=num_static_ob_slots,
        landmark_depth=landmark_depth,
        landmark_movable=landmark_movable,
        obstacle_movable=obstacle_movable,
        landmark_vel=landmark_vel,
        max_vel=max_vel,
        random_vel=random_vel,
        movement=movement,
        obstacle_movement=obstacle_movement,
        pre_method=pre_method,
        rew_err_th=rew_err_th,
        rew_dis_th=rew_dis_th,
        max_range=max_range,
        max_current_vel=max_current_vel,
        range_dropping=range_dropping,
        control_model=control_model,
        target_observation_mode=target_observation_mode,
        target_position_noise_std=target_position_noise_std,
        scenario_profile=scenario_profile,
        dynamic_obstacle_min_speed=dynamic_obstacle_min_speed,
        dynamic_obstacle_max_speed=dynamic_obstacle_max_speed,
        reward_config=reward_config,
        environment_config=environment_config,
    )
    # create multiagent environment
    #print('Create multiagent environment')
    if benchmark:
        env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            info_callback=scenario.benchmark_data_compact,
            done_callback=scenario.done,
            profile_timing=profile_timing,
        )
        # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
        #                     info_callback=scenario.benchmark_data, done_callback = scenario.done)
    else:
        env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            done_callback=scenario.done,
            profile_timing=profile_timing,
        )

    return env
