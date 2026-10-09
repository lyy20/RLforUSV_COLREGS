from __future__ import annotations

import configparser
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utilities.observation_schema import observation_dim

DYNAMIC_OB_SLOTS = 3
STATIC_OB_SLOTS = 1
# 观测长度必须由唯一 schema 计算，不能在自检脚本中重复维护旧公式。
# 这样实体槽数量或基础状态字段变化后，自检会和训练网络保持同一契约。
OBS_DIM = observation_dim(DYNAMIC_OB_SLOTS, STATIC_OB_SLOTS)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_control_model() -> str:
    config = configparser.ConfigParser()
    config.read(ROOT / "CTIPSAC_T.txt")
    return config.get("hyperparam", "control_model", fallback="heading_rate")


def check_dynamics() -> None:
    from dynamics.usv_3dof import USV3DOFConfig, USV3DOFModel, USV3DOFState

    model = USV3DOFModel(USV3DOFConfig())
    state = USV3DOFState()
    next_state = model.step(state, [1.0, 0.0], normalized_action=True)

    require(next_state.as_array().shape == (8,), "USV state vector must contain 8 values.")
    require(next_state.thrust > 0.0, "Normalized thrust command did not increase thrust.")
    require(next_state.u > 0.0, "USV did not gain surge speed after thrust.")
    print("PASS dynamics: state_dim=8 thrust=%.3f surge=%.6f" % (next_state.thrust, next_state.u))


def check_environment(action_dim: int) -> None:
    from utilities.envs import make_env

    env = make_env(
        "usv_tracking_base",
        num_agents=1,
        num_landmarks=1,
        num_obstacles=5,
        num_ob=DYNAMIC_OB_SLOTS,
        num_static_ob_slots=STATIC_OB_SLOTS,
        control_model="usv_3dof",
    )
    obs = env.reset()
    action_space = env.action_space[0]

    require(env.world.control_model == "usv_3dof", "World control_model is not usv_3dof.")
    require(env.world.agent_action_dim == action_dim, "World action dimension mismatch.")
    require(action_space.shape == (action_dim,), "Gym action space shape mismatch.")
    require(np.asarray(obs[0]).shape == (OBS_DIM,), "Observation shape changed unexpectedly.")

    before = env.world.agents[0].state.p_pos.copy()
    next_obs, rewards, dones, _ = env.step([np.array([1.0, 0.0], dtype=float)])
    after = env.world.agents[0].state.p_pos.copy()
    usv_state = env.world.agents[0].state.usv_3dof.as_array()

    require(np.asarray(next_obs[0]).shape == (OBS_DIM,), "Next observation shape changed unexpectedly.")
    require(usv_state.shape == (8,), "Integrated USV state vector must contain 8 values.")
    require(not np.allclose(before, after), "USV position did not change after environment step.")
    print(
        "PASS env: action_shape=%s obs_shape=%s reward=%.3f done=%s"
        % (action_space.shape, np.asarray(obs[0]).shape, float(rewards[0]), bool(dones[0]))
    )


def check_agent_colregs_action_filter(action_dim: int) -> None:
    from main import (
        constrained_actions_from_info,
        executed_actions_from_info,
        filter_active_from_info,
        raw_actions_from_info,
        rule_actions_from_info,
        smoothed_actions_from_info,
    )
    from utilities.envs import make_env

    env = make_env(
        "usv_tracking_base",
        num_agents=1,
        num_landmarks=1,
        num_obstacles=1,
        num_ob=1,
        num_static_ob_slots=STATIC_OB_SLOTS,
        control_model="usv_3dof",
    )
    env.reset()
    agent = env.world.agents[0]
    target = env.world.landmarks[0]
    obstacle = env.world.obstacles[0]

    target.is_vessel = False
    target.state.p_pos = np.array([3.0, 3.0], dtype=float)
    target.state.p_vel = np.zeros(2, dtype=float)

    agent.state.p_pos = np.array([0.0, 0.0], dtype=float)
    agent.state.p_vel = np.zeros(2, dtype=float)
    agent.state.usv_3dof = None
    env.world.angle[0] = 0.0

    obstacle.state.p_pos = np.array([0.20, 0.0], dtype=float)
    obstacle.state.p_vel = np.array([-0.0003, 0.0], dtype=float)
    obstacle.size = 0.04
    obstacle.obstacle_vel = 0.0003
    obstacle.max_speed = 0.0003
    obstacle.ra = np.pi
    obstacle.is_vessel = True
    obstacle.colregs_compliant = True

    raw_action = np.array([0.9, 0.9], dtype=float)
    _, _, _, info = env.step([raw_action])
    executed = executed_actions_from_info([info], raw_action.reshape(1, 1, action_dim))
    raw_logged = raw_actions_from_info([info], raw_action.reshape(1, 1, action_dim))
    constrained_logged = constrained_actions_from_info(
        [info], raw_action.reshape(1, 1, action_dim)
    )
    rule_logged = rule_actions_from_info(
        [info], raw_action.reshape(1, 1, action_dim)
    )
    smoothed_logged = smoothed_actions_from_info(
        [info], raw_action.reshape(1, 1, action_dim)
    )
    filter_active = filter_active_from_info([info], raw_action.reshape(1, 1, action_dim))
    applied = np.asarray(executed[0, 0], dtype=float)
    decision = agent.action_filter_decision
    decision_dict = decision.as_dict() if decision is not None else {}

    require(decision_dict.get("active"), "Agent COLREGs action filter did not activate in head-on risk.")
    require(decision_dict.get("encounter_type") == "head_on", "Agent action filter did not classify the encounter as head-on.")
    require(applied.shape == (action_dim,), "Applied action shape mismatch.")
    require(applied[1] < 0.0, "COLREGs head-on action set should force a starboard rudder command.")
    require(not np.allclose(raw_action, applied), "Applied action was not replaced by the COLREGs action set.")
    reward_components = env.world.agents[0].reward_last_components
    require(reward_components.get("intervention", 0.0) < 0.0, "Reward did not include a negative intervention term.")

    require(np.allclose(executed[0, 0], applied), "Training action extraction did not return applied action.")
    require(np.allclose(raw_logged[0, 0], raw_action), "Training raw-action extraction did not preserve raw action.")
    require(
        np.allclose(constrained_logged[0, 0], agent.action_constrained_u),
        "Training constrained-action extraction did not preserve boundary-constrained action.",
    )
    require(
        np.allclose(rule_logged[0, 0], agent.action_rule_u),
        "Training rule-action extraction did not preserve pure COLREGs action.",
    )
    require(
        np.allclose(smoothed_logged[0, 0], agent.action_smoothed_u),
        "Training smoothed-action extraction did not preserve EMA action.",
    )
    require(filter_active.shape == (1, 1, 1), "Filter-active extraction shape mismatch.")
    require(filter_active[0, 0, 0] == 1.0, "Filter-active extraction did not mark the COLREGs intervention.")
    print(
        "PASS agent_colregs_filter mode=%s raw=[%.3f, %.3f] rule=[%.3f, %.3f] applied=[%.3f, %.3f]"
        % (
            decision_dict.get("mode"),
            raw_action[0],
            raw_action[1],
            rule_logged[0, 0, 0],
            rule_logged[0, 0, 1],
            applied[0],
            applied[1],
        )
    )


def check_parallel_environment(action_dim: int) -> None:
    from main import (
        constrained_actions_from_info,
        executed_actions_from_info,
        filter_active_from_info,
        raw_actions_from_info,
        rule_actions_from_info,
        smoothed_actions_from_info,
    )
    from utilities.envs import make_parallel_env

    env = make_parallel_env(
        1,
        "usv_tracking_base",
        seed=1,
        num_agents=1,
        num_landmarks=1,
        num_obstacles=5,
        num_ob=DYNAMIC_OB_SLOTS,
        num_static_ob_slots=STATIC_OB_SLOTS,
        control_model="usv_3dof",
    )
    try:
        obs = env.reset()
        actions = np.zeros((1, 1, action_dim), dtype=float)
        next_obs, rewards, dones, infos = env.step(actions)

        require(obs.shape == (1, 1, OBS_DIM), "Parallel reset observation shape mismatch.")
        require(next_obs.shape == (1, 1, OBS_DIM), "Parallel step observation shape mismatch.")
        require(rewards.shape == (1, 1), "Parallel reward shape mismatch.")
        require(dones.shape == (1, 1), "Parallel done shape mismatch.")
        applied = executed_actions_from_info(infos, actions)
        raw_logged = raw_actions_from_info(infos, actions)
        filter_active = filter_active_from_info(infos, actions)
        require(applied.shape == actions.shape, "Parallel applied-action diagnostics shape mismatch.")
        require(raw_logged.shape == actions.shape, "Parallel raw-action diagnostics shape mismatch.")
        require(filter_active.shape == (1, 1, 1), "Parallel filter-active diagnostics shape mismatch.")
        print(
            "PASS parallel_env: obs_shape=%s reward_shape=%s done_shape=%s"
            % (obs.shape, rewards.shape, dones.shape)
        )
    finally:
        env.close()


def check_masac(action_dim: int) -> None:
    from algorithms.sac.masac import MASAC

    model = MASAC(num_agents=1, num_ob=DYNAMIC_OB_SLOTS, num_static_ob_slots=STATIC_OB_SLOTS, rnn=False, action_dim=action_dim)
    obs = torch.zeros(4, OBS_DIM)
    history = torch.zeros(4, 5, OBS_DIM + action_dim)
    actions = model.act([history], [obs], noise=0.0)
    action_tensor = actions[0]

    require(tuple(action_tensor.shape) == (4, action_dim), "MASAC actor output shape mismatch.")
    require(float(action_tensor.min()) >= -1.0, "MASAC action lower bound violated.")
    require(float(action_tensor.max()) <= 1.0, "MASAC action upper bound violated.")
    require(model.masac_agent[0].target_entropy == -float(action_dim), "SAC target entropy mismatch.")
    print(
        "PASS masac: action_shape=%s range=[%.6f, %.6f] target_entropy=%.1f coverage_store=%s"
        % (
            tuple(action_tensor.shape),
            float(action_tensor.min()),
            float(action_tensor.max()),
            model.masac_agent[0].target_entropy,
            type(model.masac_agent[0].tracker.visited_states).__name__,
        )
    )


def check_policy_environment_loop(action_dim: int) -> None:
    from algorithms.sac.masac import MASAC
    from utilities.envs import make_parallel_env

    env = make_parallel_env(
        1,
        "usv_tracking_base",
        seed=3,
        num_agents=1,
        num_landmarks=1,
        num_obstacles=5,
        num_ob=DYNAMIC_OB_SLOTS,
        num_static_ob_slots=STATIC_OB_SLOTS,
        control_model="usv_3dof",
    )
    model = MASAC(num_agents=1, num_ob=DYNAMIC_OB_SLOTS, num_static_ob_slots=STATIC_OB_SLOTS, rnn=False, action_dim=action_dim)
    history = torch.zeros(1, 5, OBS_DIM + action_dim)

    try:
        obs = env.reset()
        total_reward = 0.0
        last_action = None
        for _ in range(5):
            obs_tensor = torch.as_tensor(obs[:, 0, :], dtype=torch.float32)
            action = model.act([history], [obs_tensor], noise=0.0)[0].detach().numpy()
            require(action.shape == (1, action_dim), "Closed-loop action shape mismatch.")
            obs, rewards, dones, _ = env.step(action.reshape(1, 1, action_dim))
            require(np.isfinite(obs).all(), "Closed-loop observation contains non-finite values.")
            require(np.isfinite(rewards).all(), "Closed-loop reward contains non-finite values.")
            total_reward += float(rewards[0, 0])
            last_action = action[0]

        print(
            "PASS policy_loop: steps=5 total_reward=%.3f last_action=[%.6f, %.6f]"
            % (total_reward, float(last_action[0]), float(last_action[1]))
        )
    finally:
        env.close()


class NoOpLogger:
    def add_scalars(self, *args, **kwargs) -> None:
        pass


def check_sac_update(action_dim: int) -> None:
    from algorithms.sac.masac import MASAC
    from main import (
        constrained_actions_from_info,
        executed_actions_from_info,
        filter_active_from_info,
        raw_actions_from_info,
        rule_actions_from_info,
        smoothed_actions_from_info,
    )
    from utilities.buffer import ReplayBuffer_SummTree
    from utilities.envs import make_parallel_env
    from utilities.utilities import transpose_list

    parallel_envs = 1
    num_agents = 1
    history_length = 5
    buffer = ReplayBuffer_SummTree(64, seed=7)
    insert_priority = np.ones(parallel_envs)
    env = make_parallel_env(
        parallel_envs,
        "usv_tracking_base",
        seed=7,
        num_agents=num_agents,
        num_landmarks=1,
        num_obstacles=5,
        num_ob=DYNAMIC_OB_SLOTS,
        num_static_ob_slots=STATIC_OB_SLOTS,
        control_model="usv_3dof",
    )
    model = MASAC(
        num_agents=num_agents,
        num_ob=DYNAMIC_OB_SLOTS,
        num_static_ob_slots=STATIC_OB_SLOTS,
        rnn=False,
        action_dim=action_dim,
        alpha=0.05,
        automatic_entropy_tuning=False,
        CTIP_active=True,
    )

    try:
        all_obs = env.reset()
        obs = transpose_list(np.rollaxis(all_obs, 1))
        obs_size = obs[0][0].size
        history = [[np.zeros((history_length, obs_size), dtype=float)]]
        history_a = np.zeros((parallel_envs, num_agents, history_length, action_dim), dtype=float)

        for _ in range(8):
            actions = np.random.uniform(-1.0, 1.0, (parallel_envs, num_agents, action_dim))
            next_obs, rewards, dones, infos = env.step(actions)
            applied_actions = executed_actions_from_info(infos, actions)
            raw_actions = raw_actions_from_info(infos, actions)
            constrained_actions = constrained_actions_from_info(infos, raw_actions)
            rule_actions = rule_actions_from_info(infos, applied_actions)
            smoothed_actions = smoothed_actions_from_info(infos, applied_actions)
            filter_active = filter_active_from_info(infos, actions)
            # 使用正式的 12 字段 transition 合同：主动作字段为 raw_action，
            # 其余四项保存动作链审计信息，和 main.py 的训练路径保持一致。
            transition = (
                history,
                history_a,
                obs,
                raw_actions,
                rewards,
                next_obs,
                dones,
                raw_actions,
                constrained_actions,
                rule_actions,
                smoothed_actions,
                filter_active,
            )
            buffer.push(transition, insert_priority)
            obs = next_obs

        samples, indexes = buffer.sample(4)
        new_priorities = model.update(samples, 0, NoOpLogger())
        buffer.update(indexes, new_priorities)
        insert_priority = np.full(parallel_envs, max(float(np.max(new_priorities)), 1.0))
        model.update_targets()

        require(new_priorities.shape == (4,), "SAC update priority shape mismatch.")
        require(np.isfinite(new_priorities).all(), "SAC update returned non-finite priorities.")
        require(np.isfinite(insert_priority).all(), "PER insert priority contains non-finite values.")
        require(model.iter == 1, "Target update counter did not advance.")
        print(
            "PASS sac_update: priorities=[%.6f, %.6f] insert_priority=%.6f target_updates=%d"
            % (
                float(np.min(new_priorities)),
                float(np.max(new_priorities)),
                float(insert_priority[0]),
                model.iter,
            )
        )
    finally:
        env.close()


def main() -> None:
    control_model = read_control_model()
    action_dims = {"heading_rate": 1, "usv_3dof": 2}
    require(control_model in action_dims, "Unsupported control_model: %s" % control_model)
    action_dim = action_dims[control_model]
    require(control_model == "usv_3dof", "Expected CTIPSAC_T.txt to default to usv_3dof.")

    print("CHECK control_model=%s action_dim=%d" % (control_model, action_dim))
    check_dynamics()
    check_environment(action_dim)
    check_agent_colregs_action_filter(action_dim)
    check_parallel_environment(action_dim)
    check_masac(action_dim)
    check_policy_environment_loop(action_dim)
    check_sac_update(action_dim)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
