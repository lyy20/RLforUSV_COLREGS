import os
import time
import copy
import random
import pickle
import sys
from configparser import ConfigParser

try:
    import imageio
except ImportError:
    imageio = None
import numpy as np
from utilities.teacher_query import should_query, query_confidence
import torch
from tensorboardX import SummaryWriter
import glob
from torch.optim import Adam

from algorithms.sac.ICM import ICM
from algorithms.sac.masac import MASAC
from utilities import envs
from utilities.buffer import ReplayBuffer, ReplayBuffer_SummTree
from utilities.rule_replay_buffer import RuleReplayBuffer
from utilities.rule_metadata import (
    encode_encounter_type,
    encode_filter_mode,
    rule_type_mapping,
)
from utilities.paths import (
    training_benchmark_dir,
    training_before_dir,
    training_log_dir,
    training_model_dir,
    training_run_root,
    usv_log_root,
)
from utilities.utilities import transpose_list, transpose_to_tensor
from utilities.info_schema import (
    DONE_REASON_NAMES,
    REWARD_COMPONENT_KEYS as INFO_REWARD_COMPONENT_KEYS,
    REWARD_METRIC_KEYS as INFO_REWARD_METRIC_KEYS,
    decode_done_reason,
    field_slice,
    has_rule_metadata,
    infer_num_landmarks,
    is_fixed_info_array,
)
from utilities.scenario_config import read_environment_config
from utilities.observation_schema import (
    BASE_OBS_DIM,
    DYNAMIC_OBS_FEATURE_DIM,
    OBSERVATION_LAYOUT_VERSION,
    STATIC_OBS_FEATURE_DIM,
    observation_dim,
)

SUPPORTED_SCENARIO = 'usv_tracking_base'
SUPPORTED_DNN = 'MASAC'
SUPPORTED_TARGET_MOVEMENTS = {'linear', 'random', 'levy', 'escape', 'static'}
SUPPORTED_OBSTACLE_MOVEMENTS = {'linear', 'random', 'levy'}
SUPPORTED_CONTROL_MODELS = {'heading_rate': 1, 'usv_3dof': 2}
DEFAULT_CONTROL_MODEL = 'usv_3dof'
TRAINING_STATE_LAST = 'training_state_last.pt'
METRIC_WINDOWS_LAST = 'metric_windows_last.file'
TRAINING_STATE_VERSION = 4
# 断点中的动作语义版本；改变 raw/rule/smoothed 的定义时必须开启新实验。
ACTION_SEMANTIC_VERSION = 'raw_policy_rule_smoothed_execution_v1'

# 奖励分量指标：用于拆开观察奖励函数中每一项到底在训练中贡献了多少。
REWARD_COMPONENT_DESCRIPTIONS = {
    'success': '成功进入目标周边获得的任务完成奖励，反映最终追踪任务是否达成。',
    'progress': '距离接近奖励，反映本回合内智能体是否持续向目标靠近。',
    'range': '归一化距离惩罚，反映智能体离目标越远时付出的持续代价，避免仅靠存活刷分。',
    'safety': '障碍物安全距离惩罚，反映智能体是否提前远离动态船或静态障碍。',
    'boundary': '边界安全距离惩罚，反映智能体是否避免贴边或出界风险。',
    'smooth': '动作平滑惩罚，反映推力和舵角变化是否过于突兀。',
    'yaw_rate': '横摆角速度惩罚，直接约束 3DOF 船体的实际转动强度。',
    'rudder_smooth': '实际舵角平滑惩罚，反映相邻环境步的物理舵角变化是否平顺。',
    'intervention': '规则干预惩罚，反映策略是否依赖 COLREGs 动作过滤器接管。',
    'time': '每步时间压力，反映智能体是否倾向于用更少环境步完成追踪任务。',
    'terminal': '终止惩罚，反映出界、碰撞等失败终止带来的代价。',
}
REWARD_COMPONENT_KEYS = tuple(REWARD_COMPONENT_DESCRIPTIONS.keys())

# 奖励辅助指标：不是奖励本身，而是解释奖励变化原因的物理量或任务量。
REWARD_METRIC_DESCRIPTIONS = {
    'target_distance': '智能体到目标船的距离均值，越低表示追踪越接近成功。',
    'min_clearance': '智能体到最近危险实体的净空距离均值，越高表示安全裕度越大。',
    'boundary_clearance': '智能体到地图边界的净空距离均值，越高表示越不容易出界。',
    'capture_distance': '任务成功判定半径，用于对照当前目标距离是否已进入目标周边。',
    'yaw_rate_abs_deg_s': '实际横摆角速度绝对值，单位 deg/s，用于判断船体摆动强度。',
    'rudder_angle_abs_deg': '实际舵角绝对值，单位 deg，用于判断舵机使用幅度。',
    'rudder_delta_abs_deg': '相邻环境步实际舵角变化绝对值，单位 deg，用于判断转向平滑性。',
    'actual_thrust': '3DOF 执行器实际推力状态，用于核对动作是否受到执行器速率限制。',
    'actual_rudder_deg': '3DOF 执行器实际舵角，单位 deg，用于核对动作链最终执行结果。',
}
REWARD_METRIC_KEYS = tuple(REWARD_METRIC_DESCRIPTIONS.keys())

# 回合终止原因：用于区分成功完成、出界、普通碰撞和目标船船体碰撞。
DONE_REASON_DESCRIPTIONS = {
    'success': '进入目标周边并安全完成任务的次数。',
    'out_of_bounds': '智能体触发地图出界终止的次数。',
    'collision': '智能体与动态船、静态障碍或其他实体碰撞终止的次数。',
    'target_collision': '智能体直接撞到目标船船体导致失败终止的次数。',
    'timeout': '达到最大环境步数但未完成任务的次数。',
}
DONE_REASON_KEYS = tuple(DONE_REASON_DESCRIPTIONS.keys())

# 智能体级窗口指标：每个 TensorBoard 标量都来自最近 N 个回合的窗口统计。
AGENT_WINDOW_METRIC_SPECS = (
    {
        'key': 'episode_reward',
        'tag': 'mean_episode_rewards',
        'agg': 'mean',
        'window': 'reward',
        'description': '最近奖励窗口内的平均回合总奖励，用于观察总体学习趋势。',
    },
    {
        'key': 'agent_outofworld',
        'tag': 'agent_outofworld_episode',
        'agg': 'sum',
        'window': 'safety',
        'description': '最近安全窗口内的出界次数，用于观察边界安全性。',
    },
    {
        'key': 'landmark_collision',
        'tag': 'landmark_collision_episode',
        'agg': 'sum',
        'window': 'safety',
        'description': '最近安全窗口内的目标船碰撞次数，用于观察追踪末端安全性。',
    },
    {
        'key': 'agent_collision',
        'tag': 'agent_collision_episode',
        'agg': 'sum',
        'window': 'safety',
        'description': '最近安全窗口内的智能体间碰撞次数，单 USV 任务中通常为 0。',
    },
    {
        'key': 'obstacle_collision',
        'tag': 'obstacle_collision_episode',
        'agg': 'sum',
        'window': 'safety',
        'description': '最近安全窗口内的动态船或静态障碍碰撞次数，用于观察避障安全性。',
    },
    {
        'key': 'state_coverage',
        'tag': 'state_coverages_episode',
        'agg': 'mean',
        'window': 'safety',
        'description': '最近安全窗口内每回合结束时的状态覆盖数均值，用于观察探索范围。',
    },
    {
        'key': 'rule_filter_active_rate',
        'tag': 'rule_filter_active_rate_episode',
        'agg': 'mean',
        'window': 'safety',
        'description': '最近安全窗口内 COLREGs 动作过滤器触发比例，越低表示策略越少依赖规则接管。',
    },
    {
        'key': 'action_correction_norm_mean',
        'tag': 'action_correction_norm_mean_episode',
        'agg': 'mean',
        'window': 'safety',
        'description': '最近安全窗口内规则动作修正幅度均值，用于观察策略输出与规则动作的平均差距。',
    },
    {
        'key': 'action_correction_norm_max',
        'tag': 'action_correction_norm_max_episode',
        'agg': 'max',
        'window': 'safety',
        'description': '最近安全窗口内规则动作修正幅度最大值，用于捕捉最严重的一次动作修正。',
    },
    {
        'key': 'action_chain_raw_to_constrained',
        'tag': 'action_chain_raw_to_constrained_mean_episode',
        'agg': 'mean',
        'window': 'safety',
        'description': '最近安全窗口内原始动作到边界约束动作的平均差异，用于观察动作是否频繁触碰控制边界。',
    },
    {
        'key': 'action_chain_constrained_to_rule',
        'tag': 'action_chain_constrained_to_rule_mean_episode',
        'agg': 'mean',
        'window': 'safety',
        'description': '最近安全窗口内边界约束动作到纯 COLREGs 规则动作的平均差异，用于观察规则修正幅度。',
    },
    {
        'key': 'action_chain_rule_to_smoothed',
        'tag': 'action_chain_rule_to_smoothed_mean_episode',
        'agg': 'mean',
        'window': 'safety',
        'description': '最近安全窗口内纯规则动作到最终平滑动作的平均差异，用于观察平滑器对规则动作的改变程度。',
    },
    {
        'key': 'action_chain_raw_to_smoothed',
        'tag': 'action_chain_raw_to_smoothed_mean_episode',
        'agg': 'mean',
        'window': 'safety',
        'description': '最近安全窗口内原始动作到最终执行动作的平均总差异，用于衡量整条动作链的总修正量。',
    },
)

# info schema 中与动作链一一对应的四个范数字段。
ACTION_CHAIN_NORM_FIELDS = (
    'raw_to_constrained_norm',
    'constrained_to_rule_norm',
    'rule_to_smoothed_norm',
    'raw_to_smoothed_norm',
)

# SAC 更新链指标：这些标量来自一次 gradient update，而不是最近 N 回合窗口。
# 统一记录 Q/TD、策略分布、熵温度和有限值保护状态，便于区分数值异常与策略效果问题。
SAC_UPDATE_METRIC_DESCRIPTIONS = {
    'current_q1_mean': '当前 Critic Q1 输出均值，检查价值估计的尺度和稳定性。',
    'current_q2_mean': '当前 Critic Q2 输出均值，检查双 Q 估计的尺度和稳定性。',
    'target_q_mean': 'Target Critic 构造出的 SAC TD 目标均值。',
    'td_error_abs_mean': '当前 Q 与 TD 目标绝对差的均值，反映单次更新的拟合误差。',
    'td_error_squared_mean': '用于 PER 优先级的平方 TD 误差均值。',
    'policy_action_mean': '当前 actor 采样 raw action 的均值，动作范围为归一化控制域。',
    'policy_log_prob_mean': '当前策略采样动作的 log probability 均值。',
    'actor_mean_mean': 'actor 未经 tanh 的高斯均值输出，用于观察策略中心位置。',
    'actor_log_std_mean': 'actor 高斯 log 标准差均值，用于观察探索噪声尺度。',
    'critic_loss': '双 Critic 的均方 TD 损失。',
    'actor_loss': '包含 SAC 项和可选规则模仿项的 actor 总损失。',
    'sac_actor_loss': '不含规则模仿项的原始 SAC actor 损失。',
    'alpha': 'SAC 熵温度系数，自动熵调节时由 log_alpha 更新。',
    'target_entropy': 'SAC 目标熵；固定动作维度时通常为负的动作维数。',
    'alpha_loss': '自动熵调节的温度损失；关闭自动熵调节时记为 0。',
    'actor_update_applied': '本次 update 是否执行 actor/alpha 更新，1 表示执行。',
    'new_priority_mean': '本次返回给 PER 的新优先级均值。',
    'finite_guard_pass': 'SAC 更新链有限值检查通过标记；1 表示张量、梯度、参数和优先级均通过检查。',
}


def torch_load_checkpoint(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def latest_checkpoint_episode(model_dir):
    marker_paths = glob.glob(os.path.join(model_dir, 'episode_last_*.pt'))
    episodes = []
    for marker_path in marker_paths:
        marker_name = os.path.splitext(os.path.basename(marker_path))[0]
        try:
            episodes.append(int(marker_name.rsplit('_', 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(episodes) if episodes else None


def load_training_state(model_dir, device):
    state_path = os.path.join(model_dir, TRAINING_STATE_LAST)
    if not os.path.exists(state_path):
        return None
    state = torch_load_checkpoint(state_path, map_location=device)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid training state file: {state_path}")
    return state


def validate_resume_metadata(
    resume_state,
    obs_dim,
    action_dim,
    target_observation_mode,
    automatic_entropy_tuning=None,
    network_contract=None,
    action_semantic_version=None,
    curriculum_stage=None,
    scenario_profile=None,
):
    """Fail early when a checkpoint was produced by an incompatible observation/action layout."""
    if not isinstance(resume_state, dict):
        return

    saved_action_semantic = resume_state.get('action_semantic_version')
    if saved_action_semantic is None:
        saved_contract = resume_state.get('network_contract')
        if isinstance(saved_contract, dict):
            saved_action_semantic = saved_contract.get('action_semantic_version')
    if (
        saved_action_semantic is not None
        and action_semantic_version is not None
        and str(saved_action_semantic) != str(action_semantic_version)
    ):
        raise ValueError(
            'Checkpoint action semantic mismatch: saved=%s current=%s. '
            'Start a new run when action-layer definitions change.'
            % (saved_action_semantic, action_semantic_version)
        )

    saved_obs_dim = resume_state.get('obs_dim')
    if saved_obs_dim is not None and int(saved_obs_dim) != int(obs_dim):
        raise ValueError(
            "Checkpoint obs_dim mismatch: saved obs_dim = %d, current obs_dim = %d. "
            "Start a new run whenever the observation layout, entity-slot count, or "
            "entity-encoder contract changes."
            % (int(saved_obs_dim), int(obs_dim))
        )

    saved_action_dim = resume_state.get('action_dim')
    if saved_action_dim is not None and int(saved_action_dim) != int(action_dim):
        raise ValueError(
            "Checkpoint action_dim mismatch: saved action_dim = %d, current action_dim = %d."
            % (int(saved_action_dim), int(action_dim))
        )

    saved_auto_entropy = resume_state.get('automatic_entropy_tuning')
    if (
        saved_auto_entropy is not None
        and automatic_entropy_tuning is not None
        and bool(saved_auto_entropy) != bool(automatic_entropy_tuning)
    ):
        raise ValueError(
            "Checkpoint automatic entropy setting mismatch: saved=%s current=%s. "
            "Start a new run when switching between fixed-alpha SAC and automatic-entropy SAC."
            % (bool(saved_auto_entropy), bool(automatic_entropy_tuning))
        )

    saved_target_mode = resume_state.get('target_observation_mode')
    if saved_target_mode is not None:
        saved_target_mode = str(saved_target_mode).lower()
        current_target_mode = str(target_observation_mode).lower()
        if saved_target_mode != current_target_mode:
            print(
                "checkpoint_target_observation_mode_warning = saved:%s current:%s"
                % (saved_target_mode, current_target_mode)
            )

    saved_contract = resume_state.get('network_contract')
    if isinstance(saved_contract, dict) and isinstance(network_contract, dict):
        mismatches = []
        for key, current_value in network_contract.items():
            if key in saved_contract and saved_contract[key] != current_value:
                mismatches.append(
                    '%s: saved=%r current=%r' % (key, saved_contract[key], current_value)
                )
        if mismatches:
            raise ValueError(
                'Checkpoint network contract mismatch. Curriculum continuation may only '
                'change scene parameters; keep the network and observation layout unchanged. '
                + '; '.join(mismatches)
            )

    saved_stage = resume_state.get('curriculum_stage')
    saved_profile = resume_state.get('scenario_profile')
    if saved_stage is not None or saved_profile is not None:
        print(
            'curriculum_resume_transition = stage:%s->%s profile:%s->%s'
            % (saved_stage, curriculum_stage, saved_profile, scenario_profile)
        )


def capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    python_state = state.get('python')
    numpy_state = state.get('numpy')
    torch_state = state.get('torch')
    cuda_state = state.get('torch_cuda')
    if python_state is not None:
        random.setstate(python_state)
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    if torch_state is not None:
        torch_state = torch.as_tensor(torch_state, dtype=torch.uint8, device='cpu')
        torch.set_rng_state(torch_state)
    if cuda_state is not None and torch.cuda.is_available():
        cuda_states = [
            torch.as_tensor(item, dtype=torch.uint8, device='cpu')
            for item in cuda_state
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def priority_snapshot(priority):
    if priority is None:
        return None
    return np.asarray(priority, dtype=float).reshape(-1).tolist()


def restore_priority(priority_state, parallel_envs):
    if priority_state is None:
        return np.ones(parallel_envs)
    priority_arr = np.asarray(priority_state, dtype=float).reshape(-1)
    if priority_arr.size == parallel_envs:
        return priority_arr
    if priority_arr.size > 0:
        return np.full(parallel_envs, float(priority_arr[-1]))
    return np.ones(parallel_envs)


_RUNTIME_TEACHER_QUERY_STATS = {'rate': 0.0, 'coverage': 0.0}


class ScalarWhitelistWriter:
    """只允许白名单内的标量写入 TensorBoard，减少 I/O 与存储。

    匹配规则：标签最后一段（或完整标签）命中白名单即放行；白名单为空表示不过滤。
    """

    def __init__(self, writer, whitelist):
        self._writer = writer
        self._allow = set(str(x).strip() for x in (whitelist or []) if str(x).strip())

    def _permitted(self, tag):
        if not self._allow:
            return True
        tag = str(tag)
        return (tag in self._allow) or (tag.split('/')[-1] in self._allow)

    def add_scalar(self, tag, *args, **kwargs):
        if self._permitted(tag):
            return self._writer.add_scalar(tag, *args, **kwargs)
        return None

    def add_scalars(self, tag, *args, **kwargs):
        if not self._allow:
            return self._writer.add_scalars(tag, *args, **kwargs)
        scalars = args[0] if args else kwargs.get('scalar_dict', {})
        kept = dict((k, v) for k, v in (scalars or {}).items() if str(k) in self._allow)
        if not kept:
            return None
        rest = list(args[1:]) if len(args) > 1 else []
        return self._writer.add_scalars(tag, kept, *rest)

    def __getattr__(self, item):
        return getattr(self._writer, item)


DEFAULT_SCALAR_WHITELIST = [
    'mean_episode_rewards', 'agent_collision_episode', 'agent_outofworld_episode',
    'done_reason_success_episode', 'target_distance_mean_episode',
    'capture_distance_mean_episode', 'min_clearance_mean_episode',
    'rule_filter_active_rate_episode', 'action_chain_raw_to_constrained_mean_episode',
    'action_chain_constrained_to_rule_mean_episode',
    'action_chain_rule_to_smoothed_mean_episode', 'size', 'fill_ratio',
    'teacher_query_rate', 'teacher_query_coverage', 'teacher_confidence_mean',
    'teacher_valid_rate', 'unknown_type_rate', 'rudder_angle_abs_deg_mean_episode',
    'yaw_rate_abs_deg_s_mean_episode', 'reward_component_progress_episode',
    'reward_component_safety_episode', 'reward_component_success_episode',
    'reward_component_intervention_episode', 'mean_episode_error',
    'actor_loss_episode', 'critic_loss_episode', 'alpha_loss_episode',
    'rule_imitation_loss_episode',
]


def build_scalar_whitelist(config):
    if config.has_option('hyperparam', 'TENSORBOARD_SCALAR_WHITELIST'):
        raw = config.get('hyperparam', 'TENSORBOARD_SCALAR_WHITELIST')
        return [x.strip() for x in raw.split(',') if x.strip()]
    return list(DEFAULT_SCALAR_WHITELIST)


class NoOpSummaryWriter:
    """非 TensorBoard 记录回合使用的空 logger，保证训练更新不被日志频率影响。"""

    def add_scalar(self, *args, **kwargs):
        return None

    def add_scalars(self, *args, **kwargs):
        return None

    def flush(self):
        return None


PERFORMANCE_STAGE_DESCRIPTIONS = {
    'env_reset': '环境 reset 与场景初始化耗时。',
    'policy_input': '把观测、历史动作转换为策略网络输入张量的耗时。',
    'policy_act': 'actor 根据当前观测输出动作的耗时。',
    'env_step': '环境推进耗时，包含 3DOF 动力学、COLREGs 规则过滤、障碍船运动和碰撞检查。',
    'env_action_setup': '子进程环境内动作解析与动作设置阶段耗时，不包含网络推理。',
    'env_world_step': '子进程环境调用 world.step 的总耗时。',
    'world_scripted_agents': '环境中脚本实体生成动作的耗时。',
    'world_action_force': '动作过滤、动作转物理输入及噪声处理耗时。',
    'world_environment_force': '碰撞候选对遍历和碰撞力计算耗时。',
    'world_integrate_state': '3DOF/运动学状态积分和实体位置更新耗时。',
    'world_update_agent_state': '通信状态等 agent 状态更新耗时。',
    'world_colregs': '环境内 COLREGs 状态评估和报告生成耗时。',
    'env_observation_reward': '观测生成、奖励计算和 done 判定耗时。',
    'env_info_pack': '固定宽度 info 数组打包耗时。',
    'env_reset_worker': '子进程内部 reset 场景生成和初始观测耗时。',
    'info_extract': '从 info 中提取 applied_action、raw_action、filter_active 等训练附加信息的耗时。',
    'icm_update': 'ICM 内在奖励与 ICM 网络更新耗时，仅在 ICM=True 时出现。',
    'buffer_push': '经验写入 ReplayBuffer 或 PER SumTree 的耗时。',
    'history_update': 'RNN 历史观测/动作缓存更新耗时，仅在 RNN=True 时出现。',
    'benchmark': 'benchmark 与回合诊断统计耗时。',
    'network_update': 'SAC 采样、critic/actor/alpha 更新以及 target 软更新耗时。',
    'checkpoint_save': '策略、训练状态、经验池和指标窗口保存耗时。',
}
PERFORMANCE_STAGE_KEYS = tuple(PERFORMANCE_STAGE_DESCRIPTIONS.keys())


def resolve_training_device(device_name):
    """根据配置选择训练设备；auto 会在 CUDA 可用时使用 GPU，否则回退到 CPU。"""
    requested = str(device_name or 'cpu').strip().lower()
    if requested in {'auto', 'cuda_if_available'}:
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    if requested.startswith('cuda'):
        if torch.cuda.is_available():
            return torch.device(requested)
        print(f"Requested DEVICE={device_name}, but CUDA is not available. Fall back to CPU.")
        return torch.device('cpu')
    return torch.device('cpu')


def _device_is_cuda(device):
    try:
        return torch.device(device).type == 'cuda'
    except (TypeError, RuntimeError):
        return False


class TrainingProfiler:
    """轻量训练耗时统计器，只记录耗时，不改变训练流程。"""

    def __init__(self, enabled=False, device='cpu'):
        self.enabled = bool(enabled)
        self.device = device
        self.reset_window()
        self._episode_start = None

    def _sync_cuda(self):
        if self.enabled and _device_is_cuda(self.device):
            torch.cuda.synchronize(self.device)

    def tic(self):
        if not self.enabled:
            return None
        self._sync_cuda()
        return time.perf_counter()

    def toc(self, key, start_time, count=1):
        if not self.enabled or start_time is None:
            return
        self._sync_cuda()
        elapsed = time.perf_counter() - start_time
        if key not in self.stage_totals:
            self.stage_totals[key] = 0.0
            self.stage_counts[key] = 0
        self.stage_totals[key] += elapsed
        self.stage_counts[key] += max(int(count), 1)

    def add_worker_timings(self, timing_list, average=True):
        """吸收子进程环境 timing；默认记录并行环境的平均单步耗时。"""
        if not self.enabled or not timing_list:
            return
        totals = {}
        counts = {}
        for timing in timing_list:
            if not isinstance(timing, dict):
                continue
            for key, value in timing.items():
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(value):
                    continue
                totals[key] = totals.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1
        divisor = max(len(timing_list), 1) if average else 1
        for key, total in totals.items():
            elapsed = total / divisor
            self.stage_totals[key] = self.stage_totals.get(key, 0.0) + elapsed
            self.stage_counts[key] = self.stage_counts.get(key, 0) + 1

    def begin_episode_batch(self):
        if self.enabled:
            self._episode_start = time.perf_counter()

    def finish_episode_batch(self, parallel_envs, env_steps, update_calls):
        if not self.enabled or self._episode_start is None:
            return
        wall_time = time.perf_counter() - self._episode_start
        self.wall_time += wall_time
        self.episode_batches += 1
        self.parallel_episodes += int(parallel_envs)
        self.env_step_calls += int(env_steps)
        self.env_transitions += int(env_steps) * int(parallel_envs)
        self.update_calls += int(update_calls)
        self._episode_start = None

    def reset_window(self):
        self.stage_totals = {key: 0.0 for key in PERFORMANCE_STAGE_KEYS}
        self.stage_counts = {key: 0 for key in PERFORMANCE_STAGE_KEYS}
        self.wall_time = 0.0
        self.episode_batches = 0
        self.parallel_episodes = 0
        self.env_step_calls = 0
        self.env_transitions = 0
        self.update_calls = 0

    def snapshot(self, reset=True):
        if not self.enabled:
            return None
        wall_time = max(float(self.wall_time), 1.0e-12)
        stage_ratios = {
            key: float(value) / wall_time
            for key, value in self.stage_totals.items()
        }
        stage_avg_ms = {
            key: (1000.0 * self.stage_totals[key] / self.stage_counts[key])
            if self.stage_counts[key] > 0 else 0.0
            for key in self.stage_totals
        }
        data = {
            'wall_time_sec': float(self.wall_time),
            'episode_batches': int(self.episode_batches),
            'parallel_episodes': int(self.parallel_episodes),
            'env_step_calls': int(self.env_step_calls),
            'env_transitions': int(self.env_transitions),
            'update_calls': int(self.update_calls),
            'episodes_per_sec': float(self.parallel_episodes) / wall_time,
            'env_transitions_per_sec': float(self.env_transitions) / wall_time,
            'updates_per_sec': float(self.update_calls) / wall_time,
            'stage_totals': copy.deepcopy(self.stage_totals),
            'stage_ratios': stage_ratios,
            'stage_avg_ms': stage_avg_ms,
        }
        if reset:
            self.reset_window()
        return data


def log_training_profiler(logger, profiler, episode):
    """把最近一个 TensorBoard 间隔内的耗时统计写入 performance 分组。"""
    snapshot = profiler.snapshot(reset=True)
    if not snapshot:
        return
    logger.add_scalar('performance/wall_time_sec', snapshot['wall_time_sec'], episode)
    logger.add_scalar('performance/episodes_per_sec', snapshot['episodes_per_sec'], episode)
    logger.add_scalar('performance/env_transitions_per_sec', snapshot['env_transitions_per_sec'], episode)
    logger.add_scalar('performance/updates_per_sec', snapshot['updates_per_sec'], episode)
    logger.add_scalar('performance/env_step_calls', snapshot['env_step_calls'], episode)
    logger.add_scalar('performance/update_calls', snapshot['update_calls'], episode)
    for key in PERFORMANCE_STAGE_KEYS:
        logger.add_scalar(f'performance/{key}_total_sec', snapshot['stage_totals'].get(key, 0.0), episode)
        logger.add_scalar(f'performance/{key}_ratio', snapshot['stage_ratios'].get(key, 0.0), episode)
        logger.add_scalar(f'performance/{key}_avg_ms', snapshot['stage_avg_ms'].get(key, 0.0), episode)


def interval_due(episode, interval, parallel_envs, number_of_episodes=None):
    """判断当前回合是否到达配置的记录/保存间隔，并保证训练结束前最后一次会落盘。"""
    interval = max(int(interval), 1)
    due = episode % interval < parallel_envs
    final_episode = (
        number_of_episodes is not None
        and episode >= number_of_episodes - parallel_envs
    )
    return bool(due or final_episode)


def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # as reproducibility docs
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # as reproducibility docs
    torch.backends.cudnn.deterministic = True  # as reproducibility docs
def pre_process(entity, batchsize):
    processed_entity = []
    for j in range(3):
        list = []
        for i in range(batchsize):
            b = entity[i][j]
            list.append(b)
        c = torch.Tensor(list)
        processed_entity.append(c)
    return processed_entity

def action_dim_for_control_model(control_model):
    try:
        return SUPPORTED_CONTROL_MODELS[control_model]
    except KeyError as exc:
        raise ValueError(f"Unsupported control model '{control_model}'. Use one of {sorted(SUPPORTED_CONTROL_MODELS)}.") from exc


def _env_info_rows(info):
    """统一旧字典 info、固定数组 info 和 tuple(list) 兼容格式。"""
    if info is None:
        return []
    if isinstance(info, np.ndarray):
        if info.ndim == 3:
            return list(info)
        if info.ndim == 2:
            return [info]
    if isinstance(info, dict):
        return [info]
    return list(info)


def _info_num_landmarks(env_info):
    if is_fixed_info_array(env_info):
        return infer_num_landmarks(
            env_info.shape[1],
            has_rule_metadata(env_info),
        )
    return 1


def _fixed_action_from_info(env_info, field, fallback):
    result = np.asarray(fallback, dtype=float).copy()
    if not is_fixed_info_array(env_info):
        return result
    # 新动作字段只存在于带 schema marker 的 info 中。旧数组没有这些
    # 偏移，必须回退到调用方提供的动作，不能越界切片。
    if field in {
        'rule_action',
        'smoothed_action',
        'raw_to_constrained_norm',
        'constrained_to_rule_norm',
        'rule_to_smoothed_norm',
        'raw_to_smoothed_norm',
        'encounter_type_code',
        'filter_mode_code',
    } and not has_rule_metadata(env_info):
        return result
    values = np.asarray(env_info[:, field_slice(field, _info_num_landmarks(env_info))], dtype=float)
    for agent_i in range(min(result.shape[0], values.shape[0])):
        flat = values[agent_i].reshape(-1)
        target = result[agent_i].reshape(-1)
        target[:min(flat.size, target.size)] = flat[:target.size]
        result[agent_i] = target.reshape(result[agent_i].shape)
    return result


def _assign_action_matrix(target, values):
    """把字典 info 中的动作矩阵安全复制到固定形状的目标数组。"""
    if values is None:
        return False
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return False
    if values.shape == target.shape:
        target[...] = values
        return True
    if values.size == int(np.prod(target.shape)):
        target[...] = values.reshape(target.shape)
        return True
    return False


def _actions_from_info(info, fallback_actions, field, dict_key):
    """读取固定数组/旧字典 info 中指定动作层级，保留零动作语义。"""
    actions = np.asarray(fallback_actions, dtype=float).copy()
    if info is None:
        return actions
    env_infos = _env_info_rows(info)
    for env_i, env_info in enumerate(env_infos):
        if env_i >= actions.shape[0]:
            continue
        if is_fixed_info_array(env_info):
            actions[env_i] = _fixed_action_from_info(
                env_info,
                field,
                actions[env_i],
            )
            continue
        if not isinstance(env_info, dict):
            continue
        if _assign_action_matrix(actions[env_i], env_info.get(dict_key)):
            continue
        # 旧字典 info 没有统一矩阵字段时，从每个过滤器决策中读取动作。
        # 这里用 ``is not None``，确保 [0, 0] 规则动作不会被当作缺失。
        decisions = env_info.get('action_filter') or []
        for agent_i, decision in enumerate(decisions[:actions.shape[1]]):
            if not isinstance(decision, dict) or decision.get(field) is None:
                continue
            value = np.asarray(decision[field], dtype=float).reshape(-1)
            target = actions[env_i, agent_i].reshape(-1)
            target[:min(value.size, target.size)] = value[:target.size]
            actions[env_i, agent_i] = target.reshape(actions[env_i, agent_i].shape)
    return actions


def executed_actions_from_info(info, fallback_actions):
    """最终平滑执行动作；用于历史状态、ICM/CTIP 和物理诊断。"""
    return _actions_from_info(info, fallback_actions, 'applied_action', 'applied_actions')


def raw_actions_from_info(info, fallback_actions):
    """Actor 的原始归一化动作；用于主 SAC Critic 的当前动作字段。"""
    return _actions_from_info(info, fallback_actions, 'raw_action', 'raw_actions')


def constrained_actions_from_info(info, fallback_actions):
    """动作边界约束后的动作；用于规则修正幅度计算。"""
    return _actions_from_info(info, fallback_actions, 'constrained_action', 'constrained_actions')


def rule_actions_from_info(info, fallback_actions):
    """COLREGs 纯规则动作，不包含后续平滑器修正。"""
    return _actions_from_info(info, fallback_actions, 'rule_action', 'rule_actions')


def smoothed_actions_from_info(info, fallback_actions):
    """规则动作经 EMA 平滑后的最终执行动作。"""
    return _actions_from_info(info, fallback_actions, 'smoothed_action', 'smoothed_actions')


def filter_active_from_info(info, fallback_actions):
    fallback_actions = np.asarray(fallback_actions, dtype=float)
    filter_active = np.zeros(fallback_actions.shape[:2] + (1,), dtype=float)
    if info is None:
        return filter_active
    env_infos = _env_info_rows(info)
    for env_i, env_info in enumerate(env_infos):
        if env_i >= filter_active.shape[0]:
            continue
        if is_fixed_info_array(env_info):
            values = env_info[:, field_slice('filter_active', _info_num_landmarks(env_info))]
            count = min(values.shape[0], filter_active.shape[1])
            filter_active[env_i, :count, 0] = (values[:count] > 0.0).astype(float)
            continue
        if not isinstance(env_info, dict):
            continue
        active_values = env_info.get('filter_active')
        if active_values is not None:
            active_values = np.asarray(active_values, dtype=float).reshape(-1)
            for agent_i, active in enumerate(active_values[:filter_active.shape[1]]):
                filter_active[env_i, agent_i, 0] = float(active > 0.0)
            continue
        for agent_i, decision in enumerate((env_info.get('action_filter') or [])[:filter_active.shape[1]]):
            if isinstance(decision, dict):
                filter_active[env_i, agent_i, 0] = float(bool(decision.get('active', False)))
    return filter_active


def rule_metadata_from_info(info, fallback_actions):
    """Extract stable encounter/mode codes from fixed or legacy ``info``.

    The function is only called when the optional rule replay buffer is
    enabled.  Unknown or legacy rows deliberately receive code zero instead
    of inventing a class label.
    """
    fallback_actions = np.asarray(fallback_actions)
    shape = fallback_actions.shape[:2]
    encounter_codes = np.zeros(shape, dtype=np.int16)
    mode_codes = np.zeros(shape, dtype=np.int16)
    if info is None:
        return encounter_codes, mode_codes

    env_infos = _env_info_rows(info)
    for env_i, env_info in enumerate(env_infos):
        if env_i >= shape[0]:
            continue
        if is_fixed_info_array(env_info) and has_rule_metadata(env_info):
            landmark_count = _info_num_landmarks(env_info)
            encounter_values = env_info[:, field_slice(
                'encounter_type_code', landmark_count
            )].reshape(-1)
            mode_values = env_info[:, field_slice(
                'filter_mode_code', landmark_count
            )].reshape(-1)
            count = min(shape[1], encounter_values.size)
            encounter_codes[env_i, :count] = np.rint(
                encounter_values[:count]
            ).astype(np.int16)
            count = min(shape[1], mode_values.size)
            mode_codes[env_i, :count] = np.rint(
                mode_values[:count]
            ).astype(np.int16)
            continue

        if not isinstance(env_info, dict):
            continue
        decisions = env_info.get('action_filter') or []
        for agent_i, decision in enumerate(decisions[:shape[1]]):
            if not isinstance(decision, dict):
                continue
            encounter_codes[env_i, agent_i] = encode_encounter_type(
                decision.get('encounter_type')
            )
            mode_codes[env_i, agent_i] = encode_filter_mode(
                decision.get('mode')
            )
    return encounter_codes, mode_codes


def rule_geometry_from_info(info, fallback_actions):
    """读取 DCPA/TCPA 规则几何量；旧 info 使用无穷大兼容值。"""
    shape = np.asarray(fallback_actions).shape[:2]
    dcpa = np.full(shape, np.inf, dtype=np.float32)
    tcpa = np.full(shape, np.inf, dtype=np.float32)
    for env_i, env_info in enumerate(_env_info_rows(info)):
        if env_i >= shape[0]:
            continue
        if is_fixed_info_array(env_info) and has_rule_metadata(env_info):
            n = _info_num_landmarks(env_info)
            for agent_i in range(min(shape[1], env_info.shape[0])):
                dcpa[env_i, agent_i] = float(env_info[agent_i, field_slice('dcpa', n)])
                tcpa[env_i, agent_i] = float(env_info[agent_i, field_slice('tcpa', n)])
        elif isinstance(env_info, dict):
            for agent_i, decision in enumerate((env_info.get('action_filter') or [])[:shape[1]]):
                if isinstance(decision, dict):
                    dcpa[env_i, agent_i] = float(decision.get('dcpa', np.inf))
                    tcpa[env_i, agent_i] = float(decision.get('tcpa', np.inf))
    return dcpa, tcpa


def _blank_agent_windows(num_agents):
    return [[] for _ in range(num_agents)]


def _blank_landmark_windows(num_landmarks):
    return [[] for _ in range(num_landmarks)]


def _normalize_window_list(raw_windows, expected_count):
    normalized = [[] for _ in range(expected_count)]
    if not isinstance(raw_windows, (list, tuple)):
        return normalized
    for i in range(min(expected_count, len(raw_windows))):
        raw_series = raw_windows[i]
        if isinstance(raw_series, np.ndarray):
            normalized[i] = raw_series.astype(float).reshape(-1).tolist()
        elif isinstance(raw_series, (list, tuple)):
            normalized[i] = list(raw_series)
    return normalized


def init_metric_windows(num_agents, num_landmarks):
    """初始化所有 TensorBoard 回合级指标的最近 N 回合窗口。"""
    windows = {
        'version': 1,
        'episode_reward': _blank_agent_windows(num_agents),
        'agent_outofworld': _blank_agent_windows(num_agents),
        'landmark_collision': _blank_agent_windows(num_agents),
        'agent_collision': _blank_agent_windows(num_agents),
        'obstacle_collision': _blank_agent_windows(num_agents),
        'state_coverage': _blank_agent_windows(num_agents),
        'rule_filter_active_rate': _blank_agent_windows(num_agents),
        'action_correction_norm_mean': _blank_agent_windows(num_agents),
        'action_correction_norm_max': _blank_agent_windows(num_agents),
        # 动作链四段差异的最近 N 回合窗口。
        'action_chain_raw_to_constrained': _blank_agent_windows(num_agents),
        'action_chain_constrained_to_rule': _blank_agent_windows(num_agents),
        'action_chain_rule_to_smoothed': _blank_agent_windows(num_agents),
        'action_chain_raw_to_smoothed': _blank_agent_windows(num_agents),
        'reward_components': {
            key: _blank_agent_windows(num_agents)
            for key in REWARD_COMPONENT_KEYS
        },
        'reward_metrics': {
            key: _blank_agent_windows(num_agents)
            for key in REWARD_METRIC_KEYS
        },
        'done_reasons': {
            key: _blank_agent_windows(num_agents)
            for key in DONE_REASON_KEYS
        },
        'landmark_error': _blank_landmark_windows(num_landmarks),
    }
    return windows


def normalize_metric_windows(metric_windows, num_agents, num_landmarks):
    """恢复断点后补齐缺失字段，保证旧断点和新指标结构兼容。"""
    normalized = init_metric_windows(num_agents, num_landmarks)
    if not isinstance(metric_windows, dict):
        return normalized

    for spec in AGENT_WINDOW_METRIC_SPECS:
        key = spec['key']
        normalized[key] = _normalize_window_list(metric_windows.get(key), num_agents)

    for key in REWARD_COMPONENT_KEYS:
        normalized['reward_components'][key] = _normalize_window_list(
            (metric_windows.get('reward_components') or {}).get(key),
            num_agents,
        )
    for key in REWARD_METRIC_KEYS:
        normalized['reward_metrics'][key] = _normalize_window_list(
            (metric_windows.get('reward_metrics') or {}).get(key),
            num_agents,
        )
    for key in DONE_REASON_KEYS:
        normalized['done_reasons'][key] = _normalize_window_list(
            (metric_windows.get('done_reasons') or {}).get(key),
            num_agents,
        )

    normalized['landmark_error'] = _normalize_window_list(
        metric_windows.get('landmark_error'),
        num_landmarks,
    )
    return normalized


def metric_windows_from_legacy(
    num_agents,
    num_landmarks,
    agents_reward,
    landmark_error_episode,
    agent_outofworld_episode,
    agent_collision_episode,
    landmark_collision_episode,
    obstacle_collision_episode,
):
    """把老版本分散保存的窗口列表迁移到统一指标窗口中。"""
    metric_windows = init_metric_windows(num_agents, num_landmarks)
    metric_windows['episode_reward'] = _normalize_window_list(agents_reward, num_agents)
    metric_windows['landmark_error'] = _normalize_window_list(landmark_error_episode, num_landmarks)
    metric_windows['agent_outofworld'] = _normalize_window_list(agent_outofworld_episode, num_agents)
    metric_windows['agent_collision'] = _normalize_window_list(agent_collision_episode, num_agents)
    metric_windows['landmark_collision'] = _normalize_window_list(landmark_collision_episode, num_agents)
    metric_windows['obstacle_collision'] = _normalize_window_list(obstacle_collision_episode, num_agents)
    return metric_windows


def bind_legacy_metric_aliases(metric_windows):
    """保留旧变量名，方便旧保存逻辑和新窗口结构共用同一份数据。"""
    return (
        metric_windows['episode_reward'],
        metric_windows['landmark_error'],
        metric_windows['agent_outofworld'],
        metric_windows['agent_collision'],
        metric_windows['landmark_collision'],
        metric_windows['obstacle_collision'],
    )


def load_metric_windows(model_dir, resume_state, num_agents, num_landmarks):
    if isinstance(resume_state, dict) and isinstance(resume_state.get('metric_windows'), dict):
        return normalize_metric_windows(resume_state['metric_windows'], num_agents, num_landmarks)

    metric_path = os.path.join(model_dir, METRIC_WINDOWS_LAST)
    if os.path.exists(metric_path):
        with open(metric_path, "rb") as f:
            return normalize_metric_windows(pickle.load(f), num_agents, num_landmarks)
    return None


def metric_window_sizes(reward_window, landmark_error_window, safety_window):
    return {
        'reward': max(int(reward_window), 1),
        'landmark_error': max(int(landmark_error_window), 1),
        'safety': max(int(safety_window), 1),
    }


def _finite_scalar(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _append_window_value(series, value, window_size):
    value = _finite_scalar(value)
    if value is None:
        return
    series.append(value)
    if len(series) > window_size:
        del series[:-window_size]


def _window_stat(series, agg):
    values = np.asarray([v for v in series if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return 0.0
    if agg == 'sum':
        return float(values.sum())
    if agg == 'max':
        return float(values.max())
    if agg == 'std':
        return float(values.std())
    return float(values.mean())


def init_episode_diagnostics(parallel_envs, num_agents):
    return {
        'decision_count': np.zeros((parallel_envs, num_agents), dtype=float),
        'filter_active_count': np.zeros((parallel_envs, num_agents), dtype=float),
        'action_correction_norm_sum': np.zeros((parallel_envs, num_agents), dtype=float),
        'action_correction_norm_max': np.zeros((parallel_envs, num_agents), dtype=float),
        # 动作链各阶段差异的回合累计值；回合结束时除以决策步数，
        # 再写入最近 N 回合窗口，避免不同回合长度造成不可比的总量偏差。
        **{
            '%s_sum' % key: np.zeros((parallel_envs, num_agents), dtype=float)
            for key in (
                'action_chain_raw_to_constrained',
                'action_chain_constrained_to_rule',
                'action_chain_rule_to_smoothed',
                'action_chain_raw_to_smoothed',
            )
        },
        'reward_components': {
            key: np.zeros((parallel_envs, num_agents), dtype=float)
            for key in REWARD_COMPONENT_KEYS
        },
        'reward_metric_sum': {
            key: np.zeros((parallel_envs, num_agents), dtype=float)
            for key in REWARD_METRIC_KEYS
        },
        'reward_metric_count': {
            key: np.zeros((parallel_envs, num_agents), dtype=float)
            for key in REWARD_METRIC_KEYS
        },
        'done_reason_count': {
            key: np.zeros((parallel_envs, num_agents), dtype=float)
            for key in DONE_REASON_KEYS
        },
    }


def select_episode_diagnostics(diagnostics, env_indices):
    """Return an owned diagnostics batch for completed vector environments."""
    indices = np.asarray(env_indices, dtype=int).reshape(-1)
    selected = {
        'decision_count': diagnostics['decision_count'][indices].copy(),
        'filter_active_count': diagnostics['filter_active_count'][indices].copy(),
        'action_correction_norm_sum': diagnostics['action_correction_norm_sum'][indices].copy(),
        'action_correction_norm_max': diagnostics['action_correction_norm_max'][indices].copy(),
        **{
            '%s_sum' % key: diagnostics['%s_sum' % key][indices].copy()
            for key in (
                'action_chain_raw_to_constrained',
                'action_chain_constrained_to_rule',
                'action_chain_rule_to_smoothed',
                'action_chain_raw_to_smoothed',
            )
        },
        'reward_components': {},
        'reward_metric_sum': {},
        'reward_metric_count': {},
        'done_reason_count': {},
    }
    for key in REWARD_COMPONENT_KEYS:
        selected['reward_components'][key] = diagnostics['reward_components'][key][indices].copy()
    for key in REWARD_METRIC_KEYS:
        selected['reward_metric_sum'][key] = diagnostics['reward_metric_sum'][key][indices].copy()
        selected['reward_metric_count'][key] = diagnostics['reward_metric_count'][key][indices].copy()
    for key in DONE_REASON_KEYS:
        selected['done_reason_count'][key] = diagnostics['done_reason_count'][key][indices].copy()
    return selected


def reset_episode_diagnostics_rows(diagnostics, env_indices):
    """Clear per-environment episode accumulators after their metrics are saved."""
    indices = np.asarray(env_indices, dtype=int).reshape(-1)
    diagnostics['decision_count'][indices] = 0.0
    diagnostics['filter_active_count'][indices] = 0.0
    diagnostics['action_correction_norm_sum'][indices] = 0.0
    diagnostics['action_correction_norm_max'][indices] = 0.0
    for key in (
        'action_chain_raw_to_constrained',
        'action_chain_constrained_to_rule',
        'action_chain_rule_to_smoothed',
        'action_chain_raw_to_smoothed',
    ):
        diagnostics['%s_sum' % key][indices] = 0.0
    for key in REWARD_COMPONENT_KEYS:
        diagnostics['reward_components'][key][indices] = 0.0
    for key in REWARD_METRIC_KEYS:
        diagnostics['reward_metric_sum'][key][indices] = 0.0
        diagnostics['reward_metric_count'][key][indices] = 0.0
    for key in DONE_REASON_KEYS:
        diagnostics['done_reason_count'][key][indices] = 0.0


def select_benchmark_episode_stats(benchmark_stats, env_indices):
    """Slice benchmark counters using the same completed-environment ordering."""
    indices = np.asarray(env_indices, dtype=int).reshape(-1)
    return {
        key: np.asarray(values, dtype=float)[indices].copy()
        for key, values in benchmark_stats.items()
    }


def update_episode_diagnostics(diagnostics, info, num_agents):
    if info is None:
        return
    env_infos = _env_info_rows(info)
    for env_i, env_info in enumerate(env_infos):
        if env_i >= diagnostics['decision_count'].shape[0]:
            continue
        if is_fixed_info_array(env_info):
            landmark_count = _info_num_landmarks(env_info)
            for agent_i in range(min(num_agents, env_info.shape[0])):
                row = env_info[agent_i]
                diagnostics['decision_count'][env_i, agent_i] += 1.0
                if row[field_slice('filter_active', landmark_count)] > 0.0:
                    diagnostics['filter_active_count'][env_i, agent_i] += 1.0
                correction = float(row[field_slice('action_correction_norm', landmark_count)])
                if np.isfinite(correction):
                    diagnostics['action_correction_norm_sum'][env_i, agent_i] += correction
                    diagnostics['action_correction_norm_max'][env_i, agent_i] = max(
                        diagnostics['action_correction_norm_max'][env_i, agent_i],
                        correction,
                    )
                # 记录 raw -> constrained -> rule -> smoothed 四段动作链，
                # 每段均为固定 info 中的归一化动作 L2 范数。
                for key in ACTION_CHAIN_NORM_FIELDS:
                    value = float(row[field_slice(key, landmark_count)])
                    if np.isfinite(value):
                        diagnostics['action_chain_%s_sum' % key.replace('_norm', '')][env_i, agent_i] += value
                component_values = row[field_slice('reward_components', landmark_count)]
                for key, value in zip(INFO_REWARD_COMPONENT_KEYS, component_values):
                    if np.isfinite(value):
                        diagnostics['reward_components'][key][env_i, agent_i] += float(value)
                metric_values = row[field_slice('reward_metrics', landmark_count)]
                for key, value in zip(INFO_REWARD_METRIC_KEYS, metric_values):
                    if np.isfinite(value):
                        diagnostics['reward_metric_sum'][key][env_i, agent_i] += float(value)
                        diagnostics['reward_metric_count'][key][env_i, agent_i] += 1.0
                reason = decode_done_reason(row[field_slice('done_reason', landmark_count)])
                if reason in diagnostics['done_reason_count']:
                    diagnostics['done_reason_count'][reason][env_i, agent_i] += 1.0
            continue
        if not isinstance(env_info, dict):
            continue
        filter_values = env_info.get('filter_active') or []
        correction_values = env_info.get('action_correction_norm') or []
        reward_components = env_info.get('reward_components') or []
        reward_metrics = env_info.get('reward_metrics') or []
        done_reasons = env_info.get('done_reason') or []
        # 旧字典 info 也尽量恢复动作链指标；缺少新字段时保持为 0，
        # 不凭空把 applied_action 解释成纯规则动作。
        dict_actions = {}
        for action_key in (
            'raw_actions',
            'constrained_actions',
            'rule_actions',
            'smoothed_actions',
        ):
            values = env_info.get(action_key)
            if values is not None:
                try:
                    dict_actions[action_key] = np.asarray(values, dtype=float)
                except (TypeError, ValueError):
                    pass
        for agent_i in range(num_agents):
            diagnostics['decision_count'][env_i, agent_i] += 1.0
            if agent_i < len(filter_values) and bool(filter_values[agent_i]):
                diagnostics['filter_active_count'][env_i, agent_i] += 1.0
            if agent_i < len(correction_values):
                correction = float(correction_values[agent_i])
                if np.isfinite(correction):
                    diagnostics['action_correction_norm_sum'][env_i, agent_i] += correction
                    diagnostics['action_correction_norm_max'][env_i, agent_i] = max(
                        diagnostics['action_correction_norm_max'][env_i, agent_i],
                        correction,
                    )
            if all(key in dict_actions for key in (
                'raw_actions',
                'constrained_actions',
                'rule_actions',
                'smoothed_actions',
            )):
                try:
                    raw = dict_actions['raw_actions'][agent_i].reshape(-1)
                    constrained = dict_actions['constrained_actions'][agent_i].reshape(-1)
                    rule = dict_actions['rule_actions'][agent_i].reshape(-1)
                    smoothed = dict_actions['smoothed_actions'][agent_i].reshape(-1)
                    values = (
                        np.linalg.norm(constrained - raw),
                        np.linalg.norm(rule - constrained),
                        np.linalg.norm(smoothed - rule),
                        np.linalg.norm(smoothed - raw),
                    )
                    for key, value in zip(
                        ('raw_to_constrained', 'constrained_to_rule',
                         'rule_to_smoothed', 'raw_to_smoothed'),
                        values,
                    ):
                        if np.isfinite(value):
                            diagnostics['action_chain_%s_sum' % key][env_i, agent_i] += float(value)
                except (IndexError, ValueError):
                    pass
            if agent_i < len(reward_components) and isinstance(reward_components[agent_i], dict):
                for key in REWARD_COMPONENT_KEYS:
                    value = float(reward_components[agent_i].get(key, 0.0))
                    if np.isfinite(value):
                        diagnostics['reward_components'][key][env_i, agent_i] += value
            if agent_i < len(reward_metrics) and isinstance(reward_metrics[agent_i], dict):
                for key in REWARD_METRIC_KEYS:
                    value = float(reward_metrics[agent_i].get(key, np.nan))
                    if np.isfinite(value):
                        diagnostics['reward_metric_sum'][key][env_i, agent_i] += value
                        diagnostics['reward_metric_count'][key][env_i, agent_i] += 1.0
            if agent_i < len(done_reasons):
                reason = str(done_reasons[agent_i])
                if reason in diagnostics['done_reason_count']:
                    diagnostics['done_reason_count'][reason][env_i, agent_i] += 1.0


def collect_benchmark_episode_stats(info, parallel_envs, num_agents):
    """从环境 info 中提取每个并行回合的安全事件计数。"""
    stats = {
        'agent_outofworld': np.zeros((parallel_envs, num_agents), dtype=float),
        'landmark_collision': np.zeros((parallel_envs, num_agents), dtype=float),
        'agent_collision': np.zeros((parallel_envs, num_agents), dtype=float),
        'obstacle_collision': np.zeros((parallel_envs, num_agents), dtype=float),
    }
    if info is None:
        return stats
    env_infos = _env_info_rows(info)
    for env_i, env_info in enumerate(env_infos[:parallel_envs]):
        if is_fixed_info_array(env_info):
            landmark_count = _info_num_landmarks(env_info)
            counts = env_info[:, field_slice('benchmark_counts', landmark_count)]
            for agent_i in range(min(num_agents, counts.shape[0])):
                stats['agent_outofworld'][env_i, agent_i] = float(counts[agent_i, 0])
                stats['landmark_collision'][env_i, agent_i] = float(counts[agent_i, 1])
                stats['agent_collision'][env_i, agent_i] = float(counts[agent_i, 2])
                stats['obstacle_collision'][env_i, agent_i] = float(counts[agent_i, 3])
            continue
        if not isinstance(env_info, dict):
            continue
        benchmark_rows = env_info.get('n') or []
        for agent_i in range(min(num_agents, len(benchmark_rows))):
            row = benchmark_rows[agent_i]
            if len(row) >= 6:
                stats['agent_outofworld'][env_i, agent_i] = float(row[2])
                stats['landmark_collision'][env_i, agent_i] = float(row[3])
                stats['agent_collision'][env_i, agent_i] = float(row[4])
                stats['obstacle_collision'][env_i, agent_i] = float(row[5])
    return stats


def collect_landmark_error_values(landmark_error, num_landmarks):
    """记录最近一个回合内目标状态估计误差的均值。"""
    values = []
    for landmark_i in range(num_landmarks):
        if landmark_i >= len(landmark_error) or len(landmark_error[landmark_i]) == 0:
            values.append(None)
            continue
        values.append(float(np.asarray(landmark_error[landmark_i], dtype=float).mean()))
    return values


def append_episode_metric_windows(
    metric_windows,
    reward_this_episode,
    episode_diagnostics,
    state_coverage,
    benchmark_stats,
    landmark_error_values,
    window_sizes,
):
    """把本轮并行回合的指标写入最近 N 回合窗口。"""
    reward_this_episode = np.asarray(reward_this_episode, dtype=float)
    parallel_envs, num_agents = reward_this_episode.shape

    for env_i in range(parallel_envs):
        for agent_i in range(num_agents):
            # 总奖励使用奖励窗口，衡量策略整体表现是否逐步变好。
            _append_window_value(
                metric_windows['episode_reward'][agent_i],
                reward_this_episode[env_i, agent_i],
                window_sizes['reward'],
            )

            decision_count = max(float(episode_diagnostics['decision_count'][env_i, agent_i]), 1.0)
            filter_rate = float(episode_diagnostics['filter_active_count'][env_i, agent_i]) / decision_count
            correction_mean = float(episode_diagnostics['action_correction_norm_sum'][env_i, agent_i]) / decision_count
            correction_max = float(episode_diagnostics['action_correction_norm_max'][env_i, agent_i])

            # 规则接管类指标使用安全窗口，衡量策略是否逐渐少依赖 COLREGs 过滤器。
            _append_window_value(
                metric_windows['rule_filter_active_rate'][agent_i],
                filter_rate,
                window_sizes['safety'],
            )
            _append_window_value(
                metric_windows['action_correction_norm_mean'][agent_i],
                correction_mean,
                window_sizes['safety'],
            )
            _append_window_value(
                metric_windows['action_correction_norm_max'][agent_i],
                correction_max,
                window_sizes['safety'],
            )

            # 动作链指标采用每个已完成回合的“每步平均差异”，再进入最近 N
            # 回合窗口，避免长回合天然得到更大的累计值。
            for key in (
                'action_chain_raw_to_constrained',
                'action_chain_constrained_to_rule',
                'action_chain_rule_to_smoothed',
                'action_chain_raw_to_smoothed',
            ):
                chain_mean = float(
                    episode_diagnostics['%s_sum' % key][env_i, agent_i]
                    / decision_count
                )
                _append_window_value(
                    metric_windows[key][agent_i],
                    chain_mean,
                    window_sizes['safety'],
                )

            for key in REWARD_COMPONENT_KEYS:
                # 奖励分量记录的是最近奖励窗口内每一项奖励的平均贡献。
                _append_window_value(
                    metric_windows['reward_components'][key][agent_i],
                    episode_diagnostics['reward_components'][key][env_i, agent_i],
                    window_sizes['reward'],
                )

            for key in REWARD_METRIC_KEYS:
                count = float(episode_diagnostics['reward_metric_count'][key][env_i, agent_i])
                metric_value = 0.0
                if count > 0.0:
                    metric_value = float(episode_diagnostics['reward_metric_sum'][key][env_i, agent_i]) / count
                # 物理辅助量记录最近奖励窗口均值，用于解释奖励变化背后的距离和安全裕度。
                _append_window_value(
                    metric_windows['reward_metrics'][key][agent_i],
                    metric_value,
                    window_sizes['reward'],
                )

            for key in DONE_REASON_KEYS:
                # 终止原因记录最近安全窗口内累计次数，用于直接观察成功率和失败类型。
                _append_window_value(
                    metric_windows['done_reasons'][key][agent_i],
                    episode_diagnostics['done_reason_count'][key][env_i, agent_i],
                    window_sizes['safety'],
                )

            for key in ('agent_outofworld', 'landmark_collision', 'agent_collision', 'obstacle_collision'):
                _append_window_value(
                    metric_windows[key][agent_i],
                    benchmark_stats[key][env_i, agent_i],
                    window_sizes['safety'],
                )

    for agent_i, coverage in enumerate(state_coverage[:num_agents]):
        # 状态覆盖数记录最近安全窗口内的覆盖水平，用于辅助判断探索是否充分。
        _append_window_value(
            metric_windows['state_coverage'][agent_i],
            coverage,
            window_sizes['safety'],
        )

    for landmark_i, error_value in enumerate(landmark_error_values):
        # 目标误差记录最近误差窗口均值，用于观察目标/地标估计质量。
        _append_window_value(
            metric_windows['landmark_error'][landmark_i],
            error_value,
            window_sizes['landmark_error'],
        )


def log_trainer_losses(logger, trainer, episode, num_agents):
    """把训练器最近一次更新的关键损失写入 TensorBoard（此前被 NoOp 日志吞掉）。"""
    stats = getattr(trainer, 'last_rule_update_stats', None) or {}
    for agent_i in range(int(num_agents)):
        row = stats.get(agent_i) or stats.get(str(agent_i)) or {}
        for key in ('actor_loss', 'critic_loss', 'alpha_loss', 'rule_imitation_loss'):
            if key in row:
                logger.add_scalar('agent%i/%s_episode' % (agent_i, key), float(row[key]), episode)


def log_metric_windows(logger, metric_windows, episode, num_agents, num_landmarks):
    """统一向 TensorBoard 写入最近 N 回合窗口统计。"""
    for agent_i in range(num_agents):
        # 1. 智能体总体指标：奖励、安全事件、规则干预率、动作修正幅度等宏观训练状态。
        for spec in AGENT_WINDOW_METRIC_SPECS:
            logger.add_scalar(
                'agent%i/%s' % (agent_i, spec['tag']),
                _window_stat(metric_windows[spec['key']][agent_i], spec['agg']),
                episode,
            )

        # 2. 奖励分量：逐项记录 reward() 中每个奖励/惩罚来源的最近窗口均值。
        for key in REWARD_COMPONENT_KEYS:
            logger.add_scalar(
                'agent%i/reward_component_%s_episode' % (agent_i, key),
                _window_stat(metric_windows['reward_components'][key][agent_i], 'mean'),
                episode,
            )

        # 3. 奖励辅助物理量：不是奖励值本身，用于解释奖励变化来自距离、安全裕度还是边界风险。
        for key in REWARD_METRIC_KEYS:
            logger.add_scalar(
                'agent%i/%s_mean_episode' % (agent_i, key),
                _window_stat(metric_windows['reward_metrics'][key][agent_i], 'mean'),
                episode,
            )

        # 4. 回合终止原因：统计成功、出界、碰撞等事件，直接支撑安全性与有效性分析。
        for key in DONE_REASON_KEYS:
            logger.add_scalar(
                'agent%i/done_reason_%s_episode' % (agent_i, key),
                _window_stat(metric_windows['done_reasons'][key][agent_i], 'sum'),
                episode,
            )

    # 5. 目标估计误差：当前 direct_noisy 模式下仍保留接口，便于未来接入轨迹/意图预测模块。
    for landmark_i in range(num_landmarks):
        logger.add_scalar(
            'landmark%i/mean_episode_error' % landmark_i,
            _window_stat(metric_windows['landmark_error'][landmark_i], 'mean'),
            episode,
        )
        logger.add_scalar(
            'landmark%i/std_episode_error' % landmark_i,
            _window_stat(metric_windows['landmark_error'][landmark_i], 'std'),
            episode,
        )
    if hasattr(logger, 'flush'):
        logger.flush()


def log_rule_buffer_metrics(logger, rule_buffer, episode, num_agents):
    """写入规则池最近状态；禁用规则池时不创建任何 TensorBoard 标签。"""
    if rule_buffer is None:
        return
    stats = rule_buffer.stats()
    # 规则池容量/填充率：判断规则经验是否已经达到可训练规模。
    logger.add_scalar('rule_buffer/size', stats['size'], episode)
    logger.add_scalar('rule_buffer/fill_ratio', stats['fill_ratio'], episode)
    logger.add_scalar('rule_buffer/inserted_total', stats['inserted_total'], episode)
    logger.add_scalar('rule_buffer/sampled_total', stats['sampled_total'], episode)
    logger.add_scalar('rule_buffer/unknown_type_rate', stats['unknown_type_rate'], episode)
    logger.add_scalar('rule_buffer/teacher_valid_rate', stats['teacher_valid_rate'], episode)
    logger.add_scalar('rule_buffer/teacher_confidence_mean', stats['teacher_confidence_mean'], episode)
    for agent_i in range(num_agents):
        logger.add_scalar(
            'rule_buffer/agent%i_size' % agent_i,
            stats['per_agent_size'].get(agent_i, 0),
            episode,
        )


def save_metric_windows(metric_windows, model_dir, suffix='last', before_dir=None, episode=None):
    """保存最近 N 回合指标窗口，保证断点续训时 TensorBoard 统计口径连续。"""
    with open(os.path.join(model_dir, 'episode_metric_windows_%s.file' % suffix), "wb") as f:
        pickle.dump(metric_windows, f)
    if suffix == 'last':
        with open(os.path.join(model_dir, METRIC_WINDOWS_LAST), "wb") as f:
            pickle.dump(metric_windows, f)
    if before_dir is not None and episode is not None:
        with open(os.path.join(before_dir, 'metric_windows_{}.file'.format(episode)), "wb") as f:
            pickle.dump(metric_windows, f)


def validate_clean_config(
    SCENARIO,
    DNN,
    num_agents,
    num_landmarks,
    movement,
    obstacle_movement,
    landmark_movable,
    control_model,
    START_POLICY,
    RNN,
):
    if SCENARIO != SUPPORTED_SCENARIO:
        raise ValueError(f"Unsupported scenario '{SCENARIO}'. This clean copy is focused on '{SUPPORTED_SCENARIO}'.")
    if DNN != SUPPORTED_DNN:
        raise ValueError(f"Unsupported DNN '{DNN}'. This clean copy keeps only {SUPPORTED_DNN}.")
    if num_agents != 1:
        raise ValueError("This clean copy is currently scoped to a single-USV task. Set num_agents = 1.")
    if num_landmarks != 1:
        raise ValueError("This clean copy is currently scoped to one pursuit target. Set num_landmarks = 1.")
    if movement not in SUPPORTED_TARGET_MOVEMENTS:
        raise ValueError(f"Unsupported target movement '{movement}'. Use one of {sorted(SUPPORTED_TARGET_MOVEMENTS)}.")
    if obstacle_movement not in SUPPORTED_OBSTACLE_MOVEMENTS:
        raise ValueError(f"Unsupported obstacle movement '{obstacle_movement}'. Use one of {sorted(SUPPORTED_OBSTACLE_MOVEMENTS)}.")
    if movement == 'escape' and not landmark_movable:
        raise ValueError("movement = escape requires landmark_movable = True.")
    action_dim_for_control_model(control_model)
    if control_model == 'usv_3dof' and START_POLICY == 'AStar':
        raise ValueError("START_POLICY = AStar is still a heading-rate helper and is not compatible with usv_3dof.")
    if control_model == 'usv_3dof' and RNN:
        raise ValueError("RNN=True is not enabled for the new 2D usv_3dof action path yet.")


# --- B 模块：覆盖感知的主动规则教师（teacher query）参数 ---
# 触发条件 DCPA < K_D * D_SAFE 或 TCPA < K_T * T_REACT；置信度按 DCPA 余量线性衰减。
TEACHER_QUERY_K_D = 3.0            # DCPA 余量倍数
TEACHER_QUERY_K_T = 2.0            # TCPA 余量倍数
TEACHER_QUERY_D_SAFE = 0.020       # 安全净距 (km)；K_D×D_SAFE = 60 m 触发距离
TEACHER_QUERY_REACT_S = 60.0       # 反应窗 (s)；K_T×T_REACT = 120 s 触发窗

def main():
    # Read config file argument if its necessary
    if len(sys.argv) > 1:
        configFile = sys.argv[1]
    else:
        configFile = 'CTIPSAC_T'
    #print('Configuration File   =  ', configFile + '.txt')

    config = ConfigParser()
    config_path = configFile if configFile.endswith('.txt') else configFile + '.txt'
    # 配置文件统一按 UTF-8 读取；旧实验文件若仍是本机 GBK 编码，则回退
    # 到 GBK。直接调用 ConfigParser.read() 会使用 Windows 系统编码，
    # 在 UTF-8 中文注释配置上触发 UnicodeDecodeError。
    try:
        loaded_files = config.read(config_path, encoding='utf-8')
    except UnicodeDecodeError:
        config = ConfigParser()
        loaded_files = config.read(config_path, encoding='gbk')
    if not loaded_files:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    configFile = os.path.splitext(os.path.basename(config_path))[0]
    training_run_name = config.get(
        'hyperparam',
        'TRAINING_RUN_NAME',
        fallback=configFile,
    ).strip()
    if (
        not training_run_name
        or training_run_name in {'.', '..'}
        or os.path.basename(training_run_name) != training_run_name
    ):
        raise ValueError('TRAINING_RUN_NAME must be a single directory name, not a path.')

    BUFFER_SIZE = config.getint('hyperparam', 'BUFFER_SIZE')
    BATCH_SIZE = config.getint('hyperparam', 'BATCH_SIZE')
    GAMMA = config.getfloat('hyperparam', 'GAMMA')
    TAU = config.getfloat('hyperparam', 'TAU')
    LR_ACTOR = config.getfloat('hyperparam', 'LR_ACTOR')
    LR_CRITIC = config.getfloat('hyperparam', 'LR_CRITIC')
    LR_ALPHA = config.getfloat('hyperparam', 'LR_ALPHA', fallback=LR_ACTOR)
    WEIGHT_DECAY = config.getfloat('hyperparam', 'WEIGHT_DECAY')
    UPDATE_EVERY = config.getint('hyperparam', 'UPDATE_EVERY')
    UPDATE_TIMES = config.getint('hyperparam', 'UPDATE_TIMES')
    SEED = config.getint('hyperparam', 'SEED')
    BENCHMARK = config.getboolean('hyperparam', 'BENCHMARK')
    EXP_REP_BUF = config.getboolean('hyperparam', 'EXP_REP_BUF')
    PRE_TRAINED = config.getboolean('hyperparam', 'PRE_TRAINED')
    # When enabled, a resumed experiment uses the configured batch size instead
    # of silently restoring an old checkpoint value.
    RESUME_OVERRIDE_BATCH_SIZE = config.getboolean(
        'hyperparam',
        'RESUME_OVERRIDE_BATCH_SIZE',
        fallback=False,
    )

    # Optional dual-batch rule imitation.  These settings do not alter the
    # main SAC replay stream; they only control the independent actor-only
    # RuleReplayBuffer path.
    RULE_BUFFER_ENABLED = config.getboolean(
        'hyperparam',
        'RULE_BUFFER_ENABLED',
        fallback=False,
    )
    RULE_BUFFER_SIZE = config.getint(
        'hyperparam',
        'RULE_BUFFER_SIZE',
        fallback=100000,
    )
    RULE_BUFFER_MIN_SIZE = config.getint(
        'hyperparam',
        'RULE_BUFFER_MIN_SIZE',
        fallback=1000,
    )
    RULE_ACTOR_BATCH_SIZE = config.getint(
        'hyperparam',
        'RULE_ACTOR_BATCH_SIZE',
        fallback=32,
    )
    RULE_SAMPLE_MODE = config.get(
        'hyperparam',
        'RULE_SAMPLE_MODE',
        fallback='stratified',
    ).strip().lower()
    # ``RULE_IMMITATION_WEIGHT`` is retained as a compatibility fallback for
    # earlier configuration files containing that spelling.
    DAGGER_ENABLED = config.getboolean('hyperparam', 'DAGGER_ENABLED', fallback=False)
    DAGGER_TEACHER_WEIGHT = config.getfloat('hyperparam', 'DAGGER_TEACHER_WEIGHT', fallback=1.0)
    DAGGER_NON_INTERVENTION_CONFIDENCE = config.getfloat(
        'hyperparam', 'DAGGER_NON_INTERVENTION_CONFIDENCE', fallback=0.25
    )
    # DAgger 当前仅作为可选教师数据标记，默认关闭，不改变主 SAC 数据流。
    RULE_IMITATION_WEIGHT = config.getfloat(
        'hyperparam',
        'RULE_IMITATION_WEIGHT',
        fallback=config.getfloat(
            'hyperparam',
            'RULE_IMMITATION_WEIGHT',
            fallback=0.15,
        ),
    )
    if RULE_BUFFER_SIZE <= 0:
        raise ValueError('RULE_BUFFER_SIZE must be positive.')
    if RULE_BUFFER_MIN_SIZE <= 0:
        raise ValueError('RULE_BUFFER_MIN_SIZE must be positive.')
    if RULE_BUFFER_MIN_SIZE > RULE_BUFFER_SIZE:
        raise ValueError('RULE_BUFFER_MIN_SIZE cannot exceed RULE_BUFFER_SIZE.')
    if RULE_ACTOR_BATCH_SIZE <= 0:
        raise ValueError('RULE_ACTOR_BATCH_SIZE must be positive.')
    if RULE_SAMPLE_MODE not in {'uniform', 'stratified'}:
        raise ValueError("RULE_SAMPLE_MODE must be 'uniform' or 'stratified'.")
    if RULE_IMITATION_WEIGHT < 0.0:
        raise ValueError('RULE_IMITATION_WEIGHT must be non-negative.')
    if DAGGER_TEACHER_WEIGHT < 0.0:
        raise ValueError('DAGGER_TEACHER_WEIGHT must be non-negative.')
    if not 0.0 <= DAGGER_NON_INTERVENTION_CONFIDENCE <= 1.0:
        raise ValueError('DAGGER_NON_INTERVENTION_CONFIDENCE must be in [0, 1].')
    # Scenario used to train the networks
    SCENARIO = config.get('hyperparam', 'SCENARIO')
    SCENARIO_PROFILE = config.get(
        'hyperparam',
        'SCENARIO_PROFILE',
        fallback='DEFAULT',
    ).strip()
    OBSERVATION_LAYOUT = config.get(
        'hyperparam',
        'OBSERVATION_LAYOUT',
        fallback=OBSERVATION_LAYOUT_VERSION,
    ).strip()
    if OBSERVATION_LAYOUT != OBSERVATION_LAYOUT_VERSION:
        raise ValueError(
            "Unsupported OBSERVATION_LAYOUT '%s'. This project requires '%s'."
            % (OBSERVATION_LAYOUT, OBSERVATION_LAYOUT_VERSION)
        )
    CURRICULUM_STAGE = config.getint('hyperparam', 'CURRICULUM_STAGE', fallback=0)
    RENDER = config.getboolean('hyperparam', 'RENDER')
    PROGRESS_BAR = config.getboolean('hyperparam', 'PROGRESS_BAR')
    RNN = config.getboolean('hyperparam', 'RNN')
    ICM_Active = config.getboolean('hyperparam', 'ICM')
    ICM_GAMMA = config.getfloat('hyperparam', 'ICM_GAMMA')
    HISTORY_LENGTH = config.getint('hyperparam', 'HISTORY_LENGTH')
    DNN = config.get('hyperparam', 'DNN')
    START_STEPS = config.getint('hyperparam', 'START_STEPS')
    START_POLICY = config.get('hyperparam', 'START_POLICY')
    REWARD_WINDOWS = config.getint('hyperparam', 'REWARD_WINDOWS')
    LANDMARK_ERROR_WINDOWS = config.getint('hyperparam', 'LANDMARK_ERROR_WINDOWS')
    COLLISION_OUTWORLD_WINDOWS = config.getint('hyperparam', 'COLLISION_OUTWORLD_WINDOWS')
    window_sizes = metric_window_sizes(
        REWARD_WINDOWS,
        LANDMARK_ERROR_WINDOWS,
        COLLISION_OUTWORLD_WINDOWS,
    )
    ALPHA = config.getfloat('hyperparam', 'ALPHA')
    AUTOMATIC_ENTROPY = config.getboolean('hyperparam', 'AUTOMATIC_ENTROPY')
    DIM_1 = config.getint('hyperparam', 'DIM_1')
    DIM_2 = config.getint('hyperparam', 'DIM_2')
    # Optional lightweight entity encoder:
    # shared dynamic-vessel encoder + masked max pooling + static encoder.
    ENTITY_ENCODER_ENABLED = config.getboolean(
        'hyperparam',
        'ENTITY_ENCODER_ENABLED',
        fallback=False,
    )
    ENTITY_ENCODER_LATENT_DIM = config.getint(
        'hyperparam',
        'ENTITY_ENCODER_LATENT_DIM',
        fallback=32,
    )
    if ENTITY_ENCODER_LATENT_DIM <= 0:
        raise ValueError('ENTITY_ENCODER_LATENT_DIM must be positive.')
    # number of parallel agents
    parallel_envs = config.getint('hyperparam', 'parallel_envs')
    # number of agents per environment
    num_agents = config.getint('hyperparam', 'num_agents')
    # number of landmarks (or targets) per environment
    num_landmarks = config.getint('hyperparam', 'num_landmarks')
    # number of obstacles per environment
    num_obstacles = config.getint('hyperparam', 'num_obstacles')
    num_static_obstacles = config.getint('hyperparam', 'num_static_obstacles', fallback=8)
    map_half_size = config.getfloat('hyperparam', 'map_half_size', fallback=2.0)
    static_obstacle_min_size = config.getfloat('hyperparam', 'static_obstacle_min_size', fallback=0.04)
    static_obstacle_max_size = config.getfloat('hyperparam', 'static_obstacle_max_size', fallback=0.10)
    dynamic_obstacle_min_speed = (
        config.getfloat('hyperparam', 'dynamic_obstacle_min_speed')
        if config.has_option('hyperparam', 'dynamic_obstacle_min_speed')
        else None
    )
    dynamic_obstacle_max_speed = (
        config.getfloat('hyperparam', 'dynamic_obstacle_max_speed')
        if config.has_option('hyperparam', 'dynamic_obstacle_max_speed')
        else None
    )
    # range of obsevation
    ob_range = config.getfloat('hyperparam', 'ob_range')
    # num of obsevation (obstacles)
    num_ob = config.getint('hyperparam', 'num_ob')
    num_static_ob_slots = config.getint('hyperparam', 'num_static_ob_slots', fallback=1)
    landmark_depth = config.getfloat('hyperparam', 'landmark_depth')
    landmark_movable = config.getboolean('hyperparam', 'landmark_movable')
    obstacle_movable = config.getboolean('hyperparam', 'obstacle_movable')
    landmark_vel = config.getfloat('hyperparam', 'landmark_vel')
    movement = config.get('hyperparam', 'movement')
    obstacle_movement = config.get('hyperparam', 'obstacle_movement')
    pre_method = config.get('hyperparam', 'pre_method')
    rew_err_th = config.getfloat('hyperparam', 'rew_err_th')
    rew_dis_th = config.getfloat('hyperparam', 'rew_dis_th')
    max_range = config.getfloat('hyperparam', 'max_range')
    max_current_vel = config.getfloat('hyperparam', 'max_current_vel')
    range_dropping = config.getfloat('hyperparam', 'range_dropping')
    target_observation_mode = config.get(
        'hyperparam',
        'target_observation_mode',
        fallback='direct_noisy',
    )
    target_position_noise_std = config.getfloat(
        'hyperparam',
        'target_position_noise_std',
        fallback=0.005,
    )
    environment_config = read_environment_config(config)
    # 奖励配置：所有奖励项均可在配置文件中调整；fallback 保持旧实验的数值不变。
    reward_config = {
        'success': config.getfloat('hyperparam', 'REWARD_SUCCESS', fallback=100.0),
        'collision': config.getfloat('hyperparam', 'REWARD_COLLISION', fallback=120.0),
        'out_of_bounds': config.getfloat('hyperparam', 'REWARD_OUT_OF_BOUNDS', fallback=100.0),
        'timeout': config.getfloat('hyperparam', 'REWARD_TIMEOUT', fallback=25.0),
        'progress_weight': config.getfloat('hyperparam', 'REWARD_PROGRESS_WEIGHT', fallback=12.0),
        'range_weight': config.getfloat('hyperparam', 'REWARD_RANGE_WEIGHT', fallback=0.5),
        'safety_weight': config.getfloat('hyperparam', 'REWARD_SAFETY_WEIGHT', fallback=20.0),
        'boundary_weight': config.getfloat('hyperparam', 'REWARD_BOUNDARY_WEIGHT', fallback=8.0),
        'action_smooth_weight': config.getfloat(
            'hyperparam',
            'REWARD_ACTION_SMOOTH_WEIGHT',
            fallback=0.02,
        ),
        'yaw_rate_weight': config.getfloat(
            'hyperparam',
            'REWARD_YAW_RATE_WEIGHT',
            fallback=0.05,
        ),
        'rudder_smooth_weight': config.getfloat(
            'hyperparam',
            'REWARD_RUDDER_SMOOTH_WEIGHT',
            fallback=0.10,
        ),
        'intervention_active_penalty': config.getfloat(
            'hyperparam',
            'REWARD_INTERVENTION_ACTIVE_PENALTY',
            fallback=0.05,
        ),
        'intervention_magnitude_weight': config.getfloat(
            'hyperparam',
            'REWARD_INTERVENTION_MAGNITUDE_WEIGHT',
            fallback=0.25,
        ),
        'time_penalty': config.getfloat('hyperparam', 'REWARD_TIME_PENALTY', fallback=0.05),
        'safe_clearance': config.getfloat('hyperparam', 'REWARD_SAFE_CLEARANCE', fallback=0.15),
        'boundary_safe_distance': config.getfloat(
            'hyperparam',
            'REWARD_BOUNDARY_SAFE_DISTANCE',
            fallback=0.25,
        ),
        'rule_reward_enabled': config.getboolean(
            'hyperparam', 'RULE_REWARD_ENABLED', fallback=False
        ),
        'rule_action_alignment_weight': config.getfloat(
            'hyperparam', 'RULE_ACTION_ALIGNMENT_WEIGHT', fallback=0.0
        ),
        'rule_cpa_weight': config.getfloat(
            'hyperparam', 'RULE_CPA_WEIGHT', fallback=0.0
        ),
        'rule_obligation_penalty': config.getfloat(
            'hyperparam', 'RULE_OBLIGATION_PENALTY', fallback=0.0
        ),
    }

    # number of training episodes.
    # change this to higher number to experiment. say 30000.
    number_of_episodes = config.getint('hyperparam', 'number_of_episodes')
    episode_length = config.getint('hyperparam', 'episode_length')
    environment_config['max_episode_steps'] = int(episode_length)
    # POLICY_SAVE_INTERVAL：每隔多少回合保存一次策略/断点，旧字段 save_interval 仅作为兼容回退。
    policy_save_interval = config.getint(
        'hyperparam',
        'POLICY_SAVE_INTERVAL',
        fallback=config.getint('hyperparam', 'save_interval', fallback=10000),
    )
    # TENSORBOARD_LOG_INTERVAL：每隔多少回合向 TensorBoard 写入一次训练指标。
    tensorboard_log_interval = config.getint(
        'hyperparam',
        'TENSORBOARD_LOG_INTERVAL',
        fallback=100,
    )
    policy_save_interval = max(int(policy_save_interval), 1)
    tensorboard_log_interval = max(int(tensorboard_log_interval), 1)
    # DEVICE：auto 自动使用 CUDA，否则可手动写 cpu / cuda / cuda:0。
    device_config = config.get('hyperparam', 'DEVICE', fallback='cpu')
    # PROFILE_TRAINING：只做耗时统计，不改变动作、奖励、采样和网络更新。
    profile_training = config.getboolean('hyperparam', 'PROFILE_TRAINING', fallback=False)
    # TensorBoard 后台写队列配置，减少主训练线程被磁盘写入反压阻塞。
    tensorboard_max_queue = max(
        config.getint('hyperparam', 'TENSORBOARD_MAX_QUEUE', fallback=100),
        1,
    )
    tensorboard_flush_secs = max(
        config.getint('hyperparam', 'TENSORBOARD_FLUSH_SECS', fallback=30),
        1,
    )
    # amplitude of OU noise
    # this slowly decreases to 0
    noise = config.getfloat('hyperparam', 'noise')
    noise_reduction = config.getfloat('hyperparam', 'noise_reduction')
    fol_in = int(np.random.rand() * 1000)
    max_vel = config.getfloat('hyperparam', 'max_vel', fallback=0.)
    random_vel = config.getboolean('hyperparam', 'random_vel', fallback=False)
    if dynamic_obstacle_min_speed is not None or dynamic_obstacle_max_speed is not None:
        min_dynamic_speed = 0.0 if dynamic_obstacle_min_speed is None else dynamic_obstacle_min_speed
        max_dynamic_speed = min_dynamic_speed if dynamic_obstacle_max_speed is None else dynamic_obstacle_max_speed
        if min_dynamic_speed < 0.0 or max_dynamic_speed < min_dynamic_speed:
            raise ValueError(
                'dynamic_obstacle speed range must satisfy 0 <= min_speed <= max_speed.'
            )
    CTIP_active = config.getboolean('hyperparam', 'CTIP_active', fallback=False)
    CONTROL_MODEL = config.get('hyperparam', 'control_model', fallback=DEFAULT_CONTROL_MODEL)
    ACTION_DIM = action_dim_for_control_model(CONTROL_MODEL)
    validate_clean_config(
        SCENARIO,
        DNN,
        num_agents,
        num_landmarks,
        movement,
        obstacle_movement,
        landmark_movable,
        CONTROL_MODEL,
        START_POLICY,
        RNN,
    )
    # # 检查CUDA是否可用
    # if torch.cuda.is_available():
    #     print("CUDA is available!")
    #     print(f"CUDA device count: {torch.cuda.device_count()}")
    #     print(f"Current CUDA device: {torch.cuda.current_device()}")
    #     print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    # else:
    #     print("CUDA is not available.")

    # # Chose device
    DEVICE = resolve_training_device(device_config)
    print(f"Using device: {DEVICE}")

    ##################################################################################################

    log_root = usv_log_root()
    run_root = training_run_root(training_run_name)
    log_path = training_log_dir(training_run_name)
    model_dir = training_model_dir(training_run_name)
    before_dir = training_before_dir(training_run_name)
    # print(log_path)
    # print(model_dir)
    os.makedirs(log_root, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(before_dir, exist_ok=True)
    resume_state = load_training_state(model_dir, DEVICE) if PRE_TRAINED else None
    if PRE_TRAINED:
        if resume_state is not None and resume_state.get('episode') is not None:
            PRE_TRAINED_EP = int(resume_state['episode'])
        else:
            latest_episode = latest_checkpoint_episode(model_dir)
            if latest_episode is None:
                raise FileNotFoundError(
                    f"PRE_TRAINED=True but no checkpoint was found in {model_dir}"
                )
            PRE_TRAINED_EP = latest_episode
    else:
        PRE_TRAINED_EP = 0

    # print hyperparameters
    print('Hyperparameters:')
    print('DEVICE_CONFIG        =  ', device_config)
    print('DEVICE               =  ', DEVICE)
    if _device_is_cuda(DEVICE):
        print('CUDA_DEVICE_NAME     =  ', torch.cuda.get_device_name(DEVICE))
    print('BUFFER_SIZE          =  ', BUFFER_SIZE)
    print('BATCH_SIZE           =  ', BATCH_SIZE)
    print('GAMMA                =  ', GAMMA)
    print('TAU                  =  ', TAU)
    print('LR_ACTOR             =  ', LR_ACTOR)
    print('LR_CRITIC            =  ', LR_CRITIC)
    print('LR_ALPHA             =  ', LR_ALPHA)
    print('WEIGHT_DECAY         =  ', WEIGHT_DECAY)
    print('UPDATE_EVERY         =  ', UPDATE_EVERY)
    print('UPDATE_TIMES         =  ', UPDATE_TIMES)
    print('SEED                 =  ', SEED)
    print('BENCHMARK            =  ', BENCHMARK)
    print('EXP_REP_BUF          =  ', EXP_REP_BUF)
    print('RESUME_OVERRIDE_BATCH_SIZE =  ', RESUME_OVERRIDE_BATCH_SIZE)
    print('DAGGER_ENABLED        =  ', DAGGER_ENABLED)
    print('DAGGER_TEACHER_WEIGHT =  ', DAGGER_TEACHER_WEIGHT)
    print('RULE_BUFFER_ENABLED  =  ', RULE_BUFFER_ENABLED)
    print('RULE_BUFFER_SIZE     =  ', RULE_BUFFER_SIZE)
    print('RULE_BUFFER_MIN_SIZE =  ', RULE_BUFFER_MIN_SIZE)
    print('RULE_ACTOR_BATCH_SIZE=  ', RULE_ACTOR_BATCH_SIZE)
    print('RULE_SAMPLE_MODE     =  ', RULE_SAMPLE_MODE)
    print('RULE_IMITATION_WEIGHT= ', RULE_IMITATION_WEIGHT)
    print('COLREGS_PERSISTENT_STATE_ENABLED =  ', environment_config.get('colregs_persistent_state_enabled', False))
    print('COLREGS_PERSISTENT_EXIT_STEPS    =  ', environment_config.get('colregs_persistent_exit_steps', 3))
    print('COLREGS_PERSISTENT_MAX_HOLD_STEPS =  ', environment_config.get('colregs_persistent_max_hold_steps', 0))
    print('START_POLICY         =  ', START_POLICY)
    print('PRE_TRAINED          =  ', PRE_TRAINED)
    print('PRE_TRAINED_EP       =  ', PRE_TRAINED_EP)
    print('SCENARIO             =  ', SCENARIO)
    print('SCENARIO_PROFILE     =  ', SCENARIO_PROFILE)
    print('OBSERVATION_LAYOUT   =  ', OBSERVATION_LAYOUT)
    print('CURRICULUM_STAGE     =  ', CURRICULUM_STAGE)
    print('TRAINING_RUN_NAME    =  ', training_run_name)
    print('RNN activated        =  ', RNN)
    print('HISTORY_LENGTH       =  ', HISTORY_LENGTH)
    print('RENDER               =  ', RENDER)
    print('PROGRESS_BAR         =  ', PROGRESS_BAR)
    print('reward_window        =  ', window_sizes['reward'])
    print('landmark_err_window  =  ', window_sizes['landmark_error'])
    print('safety_window        =  ', window_sizes['safety'])
    print('DEVICE               =  ', DEVICE)
    print('parallel_envs        =  ', parallel_envs)
    print('num_agents           =  ', num_agents)
    print('num_landmarks        =  ', num_landmarks)
    print('num_obstacles        =  ', num_obstacles)
    print('num_static_obstacles =  ', num_static_obstacles)
    print('map_half_size        =  ', map_half_size)
    print('static_obstacle_size =  ', (static_obstacle_min_size, static_obstacle_max_size))
    print('dynamic_speed_range =  ', (dynamic_obstacle_min_speed, dynamic_obstacle_max_speed))
    print('ob_range             =  ', ob_range)
    print('num_ob               =  ', num_ob)
    print('num_static_ob_slots  =  ', num_static_ob_slots)
    print('landmark_depth       =  ', landmark_depth)
    print('landmark_movable     =  ', landmark_movable)
    print('target_movement      =  ', movement)
    print('landmark_velocity    =  ', landmark_vel)
    print('target_max_velocity  =  ', max_vel)
    print('obstacle_movable     =  ', obstacle_movable)
    print('obstacle_movement    =  ', obstacle_movement)
    print('random_velocity      =  ', random_vel)
    print('max_current_vel      =  ', max_current_vel)
    print('range_dropping       =  ', range_dropping)
    print('pre_method           =  ', pre_method)
    print('target_obs_mode      =  ', target_observation_mode)
    print('target_pos_noise_std =  ', target_position_noise_std)
    print('environment_config   =  ', environment_config)
    print('map_size_m           =  ', 2.0 * map_half_size * 1000.0)
    print('environment_dt_s     =  ', environment_config['environment_dt'])
    print('usv_substeps         =  ', environment_config['usv_substeps'])
    print('reward_config        =  ', reward_config)
    print('number_of_episodes   =  ', number_of_episodes)
    print('episode_length       =  ', episode_length)
    print('policy_save_interval =  ', policy_save_interval)
    print('tensorboard_interval =  ', tensorboard_log_interval)
    print('profile_training     =  ', profile_training)
    print('tensorboard_max_queue =  ', tensorboard_max_queue)
    print('tensorboard_flush_secs=  ', tensorboard_flush_secs)
    print('noise                =  ', noise)
    print('noise_reduction      =  ', noise_reduction)
    print('DNN architecture     =  ', DNN)
    print('entity_encoder       =  ', ENTITY_ENCODER_ENABLED)
    print('entity_encoder_latent=  ', ENTITY_ENCODER_LATENT_DIM)
    print('control_model        =  ', CONTROL_MODEL)
    print('action_dim           =  ', ACTION_DIM)
    print('observation_dim      =  ', observation_dim(num_ob, num_static_ob_slots))
    print('Alpha temperature    =  ', ALPHA)
    print('DNN Layer 1 size     =  ', DIM_1)
    print('DNN Layer 2 size     =  ', DIM_2)
    print('Log root             =  ', log_root)
    print('Run directory        =  ', run_root)
    print('Model directory      =  ', model_dir)
    print('TIMESTAMP            =  ', time.strftime("%m%d%y_%H%M%S"))

    # Start the each seed
    seeding(seed=SEED + PRE_TRAINED_EP)
    t = 0

    if BENCHMARK:
        benchmark_dir = training_benchmark_dir(training_run_name)
        os.makedirs(benchmark_dir, exist_ok=True)

        # initialize environment
    print('Initialize the number of parallel envs in torch')
    torch.set_num_threads(parallel_envs)
    print('Initialize the environments')
    env = envs.make_parallel_env(parallel_envs, SCENARIO, seed=SEED + PRE_TRAINED_EP, num_agents=num_agents,
                                 num_landmarks=num_landmarks, num_obstacles=num_obstacles,
                                 num_static_obstacles=num_static_obstacles, map_half_size=map_half_size,
                                 static_obstacle_min_size=static_obstacle_min_size,
                                 static_obstacle_max_size=static_obstacle_max_size,
                                 ob_range = ob_range,num_ob = num_ob,
                                 num_static_ob_slots=num_static_ob_slots,
                                 landmark_depth=landmark_depth, landmark_movable=landmark_movable,
                                 obstacle_movable=obstacle_movable, landmark_vel=landmark_vel, max_vel=max_vel,
                                 random_vel=random_vel, movement=movement, obstacle_movement=obstacle_movement,
                                 pre_method=pre_method, rew_err_th=rew_err_th, rew_dis_th=rew_dis_th, max_range=max_range,
                                 max_current_vel=max_current_vel, range_dropping=range_dropping,
                                 target_observation_mode=target_observation_mode,
                                 target_position_noise_std=target_position_noise_std,
                                 scenario_profile=SCENARIO_PROFILE,
                                 dynamic_obstacle_min_speed=dynamic_obstacle_min_speed,
                                 dynamic_obstacle_max_speed=dynamic_obstacle_max_speed,
                                 reward_config=reward_config,
                                 environment_config=environment_config,
                                 control_model=CONTROL_MODEL, benchmark=BENCHMARK,
                                 profile_timing=profile_training)

    # initialize replay buffer
    if EXP_REP_BUF == False:
        buffer = ReplayBuffer(int(BUFFER_SIZE))
    else:
        buffer = ReplayBuffer_SummTree(int(BUFFER_SIZE), SEED + PRE_TRAINED_EP)  # Experienced replay buffer
        priority = np.ones(parallel_envs)  # insertion priority for new parallel-environment transitions

    # initialize policy and critic
    print('Initialize the Actor-Critic networks')
    if DNN != SUPPORTED_DNN:
        raise ValueError(f"Unsupported DNN '{DNN}'. This clean copy keeps only {SUPPORTED_DNN}.")
    maddpg = MASAC(num_agents=num_agents, num_landmarks=num_landmarks, landmark_depth=landmark_depth,
                   num_ob=num_ob, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR,
                   lr_critic=LR_CRITIC, lr_alpha=LR_ALPHA, weight_decay=WEIGHT_DECAY,
                   device=DEVICE, rnn=RNN, alpha=ALPHA,
                   automatic_entropy_tuning=AUTOMATIC_ENTROPY, dim_1=DIM_1, dim_2=DIM_2,
                   CTIP_active=CTIP_active, action_dim=ACTION_DIM,
                   num_static_ob_slots=num_static_ob_slots,
                   entity_encoder_enabled=ENTITY_ENCODER_ENABLED,
                   entity_encoder_latent_dim=ENTITY_ENCODER_LATENT_DIM,
                   rule_buffer_enabled=RULE_BUFFER_ENABLED,
                   rule_imitation_weight=RULE_IMITATION_WEIGHT)
    logger = SummaryWriter(
        log_dir=str(log_path),
        max_queue=tensorboard_max_queue,
        flush_secs=tensorboard_flush_secs,
    )

    _scalar_whitelist = build_scalar_whitelist(config)
    logger = ScalarWhitelistWriter(logger, _scalar_whitelist)
    print('scalar_whitelist     =  ', len(_scalar_whitelist) if _scalar_whitelist else 'ALL')
    null_logger = NoOpSummaryWriter()
    profiler = TrainingProfiler(enabled=profile_training, device=DEVICE)
    obs_dim = observation_dim(num_ob, num_static_ob_slots)
    runtime_obs_dim = int(env.observation_space[0].shape[0])
    if runtime_obs_dim != obs_dim:
        raise ValueError(
            'Environment observation width mismatch: expected=%d actual=%d.'
            % (obs_dim, runtime_obs_dim)
        )

    # This pool is intentionally created only when explicitly enabled.  A
    # disabled experiment allocates no rule storage and follows the legacy
    # main-buffer-only path exactly.
    rule_buffer = None
    if RULE_BUFFER_ENABLED:
        rule_buffer = RuleReplayBuffer(
            capacity=RULE_BUFFER_SIZE,
            history_length=HISTORY_LENGTH,
            obs_dim=obs_dim,
            action_dim=ACTION_DIM,
            num_agents=num_agents,
            seed=SEED + PRE_TRAINED_EP + 7919,
        )
        print('rule_buffer_initialized = True')
    else:
        print('rule_buffer_initialized = False')
    network_contract = {
        'policy_algorithm': str(DNN),
        'control_model': str(CONTROL_MODEL),
        'num_agents': int(num_agents),
        'num_landmarks': int(num_landmarks),
        'dynamic_observation_slots': int(num_ob),
        'static_observation_slots': int(num_static_ob_slots),
        'observation_layout': OBSERVATION_LAYOUT_VERSION,
        'base_observation_dim': int(BASE_OBS_DIM),
        'dynamic_observation_feature_dim': int(DYNAMIC_OBS_FEATURE_DIM),
        'static_observation_feature_dim': int(STATIC_OBS_FEATURE_DIM),
        'obs_dim': int(obs_dim),
        'action_dim': int(ACTION_DIM),
        'hidden_dim_1': int(DIM_1),
        'hidden_dim_2': int(DIM_2),
        'rnn': bool(RNN),
        'history_length': int(HISTORY_LENGTH),
        'automatic_entropy_tuning': bool(AUTOMATIC_ENTROPY),
        'action_semantic_version': ACTION_SEMANTIC_VERSION,
        'action_smoothing_alpha': float(environment_config['action_smoothing_alpha']),
        'rule_filter_enabled': bool(environment_config.get('agent_colregs_action_filter_enabled', True)),
        'entity_encoder_enabled': bool(ENTITY_ENCODER_ENABLED),
        'entity_encoder_latent_dim': int(ENTITY_ENCODER_LATENT_DIM),
        'ctip_active': bool(CTIP_active),
        'icm_active': bool(ICM_Active),
        # DAgger is metadata-only and optional; it does not alter network shape.
        'dagger_enabled': bool(DAGGER_ENABLED),
    }
    in_actor = obs_dim + 1
    in_rnn = obs_dim - 1

    # 定义分箱参数
    bin_params = {
        'q_r': (0.0, 2.0, 200),
        'q_theta': (0.0, 2 * np.pi, 24),
    }
    for i in range(num_ob):
        bin_params[f'obstacle_dist_{i}'] = (0.0, 1.0, 10)
        bin_params[f'obstacle_angle_{i}'] = (0.0, 2 * np.pi, 8)
        bin_params[f'obstacle_size_{i}'] = (0.0, 1.0, 15)
    weight_dict = {
        'q_r': 0.3,
        'q_theta': 0.3,
    }
    for i in range(num_ob):
        weight_dict[f'obstacle_dist_{i}'] = 0.1
        weight_dict[f'obstacle_angle_{i}'] = 0.1
        weight_dict[f'obstacle_size_{i}'] = 0.1
    # 定义有效取值范围（关键改进）
    valid_ranges = {
        'q_r': (0.0, 2.0),
    }
    for i in range(num_ob):
        valid_ranges[f'obstacle_dist_{i}'] = (0.0, 1.0)

    # tracker = StateCoverageTracker(bin_params, lsh_epsilon=0.01, weight_dict=weight_dict,
    #                             valid_ranges=valid_ranges, prune_invalid=True, num_obstacles=num_ob)
    # ICM
    if ICM_Active:
        # 初始化 ICM 模块
        state_dim = in_actor - 1  
        action_dim = ACTION_DIM
        hidden_dim = 256
        icm = ICM(state_dim, action_dim, hidden_dim).to(DEVICE)
        icm_optimizer = Adam(icm.parameters(), lr=LR_ACTOR)
    metric_windows = init_metric_windows(num_agents, num_landmarks)
    (
        agents_reward,
        landmark_error_episode,
        agent_outofworld_episode,
        agent_collision_episode,
        landmark_collision_episode,
        obstacle_collision_episode,
    ) = bind_legacy_metric_aliases(metric_windows)

    counter = 0
    avg_rewards_best = -1000.

    validate_resume_metadata(
        resume_state,
        obs_dim,
        ACTION_DIM,
        target_observation_mode,
        automatic_entropy_tuning=AUTOMATIC_ENTROPY,
        network_contract=network_contract,
        action_semantic_version=ACTION_SEMANTIC_VERSION,
        curriculum_stage=CURRICULUM_STAGE,
        scenario_profile=SCENARIO_PROFILE,
    )

    if PRE_TRAINED == True:
        # Load the pretrained agent's weights
        trained_checkpoint = str(model_dir / 'episode')
        aux = torch_load_checkpoint(trained_checkpoint + '_last.pt', map_location=DEVICE)
        if AUTOMATIC_ENTROPY:
            with open(trained_checkpoint + '_target_entropy_last.file', "rb") as f:
                target_entropy_aux = pickle.load(f)
            with open(trained_checkpoint + '_log_alpha_last.file', "rb") as f:
                log_alpha_aux = pickle.load(f)
            with open(trained_checkpoint + '_alpha_last.file', "rb") as f:
                alpha_aux = pickle.load(f)
        for i in range(num_agents):
            # load the weights from file
            if ICM_Active:
                profile_start = profiler.tic()
                icm.load_state_dict(aux[i]['icm_state_dict'])
                icm_optimizer.load_state_dict(aux[i]['icm_optimizer_state_dict'])
            if DNN == 'MASAC':
                if AUTOMATIC_ENTROPY:
                    maddpg.masac_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                    maddpg.masac_agent[i].critic.load_state_dict(aux[i]['critic_params'])
                    maddpg.masac_agent[i].target_critic.load_state_dict(aux[i]['target_critic_params'])
                    maddpg.masac_agent[i].actor_optimizer.load_state_dict(aux[i]['actor_optim_params'])
                    maddpg.masac_agent[i].critic_optimizer.load_state_dict(aux[i]['critic_optim_params'])
                    maddpg.masac_agent[i].alpha_optimizer.load_state_dict(aux[i]['alpha_optim_params'])
                    # load agents alpha parameters
                    maddpg.masac_agent[i].target_entropy = float(target_entropy_aux[i])
                    loaded_log_alpha = torch.as_tensor(
                        log_alpha_aux[i],
                        dtype=maddpg.masac_agent[i].log_alpha.dtype,
                        device=maddpg.masac_agent[i].log_alpha.device,
                    )
                    maddpg.masac_agent[i].log_alpha.data.copy_(loaded_log_alpha)
                    if torch.is_tensor(alpha_aux[i]):
                        maddpg.masac_agent[i].alpha = alpha_aux[i].to(
                            device=maddpg.masac_agent[i].log_alpha.device
                        )
                    else:
                        maddpg.masac_agent[i].alpha = torch.as_tensor(
                            alpha_aux[i],
                            dtype=maddpg.masac_agent[i].log_alpha.dtype,
                            device=maddpg.masac_agent[i].log_alpha.device,
                        )
                else:
                    maddpg.masac_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                    maddpg.masac_agent[i].critic.load_state_dict(aux[i]['critic_params'])
                    maddpg.masac_agent[i].target_critic.load_state_dict(aux[i]['target_critic_params'])
                    maddpg.masac_agent[i].actor_optimizer.load_state_dict(aux[i]['actor_optim_params'])
                    maddpg.masac_agent[i].critic_optimizer.load_state_dict(aux[i]['critic_optim_params'])
                if CTIP_active:
                    if 'icm_state_dict' in aux[i] and 'icm_optimizer_state_dict' in aux[i]:
                        maddpg.masac_agent[i].icm.load_state_dict(aux[i]['icm_state_dict'])
                        maddpg.masac_agent[i].icm_optimizer.load_state_dict(aux[i]['icm_optimizer_state_dict'])
                    else:
                        print('checkpoint_missing_ctip_icm =  True')
        # reload the replay buffer
        # import pdb; pdb.set_trace()
        buffer.reload(trained_checkpoint + r'_last.file')
        if rule_buffer is not None:
            rule_buffer_path = os.path.join(model_dir, 'rule_buffer_last.file')
            if os.path.exists(rule_buffer_path):
                rule_buffer.reload(rule_buffer_path)
                print('rule_buffer_resume_loaded = True')
            else:
                # This is expected when a new optional module is enabled from
                # a legacy checkpoint; the main SAC checkpoint remains usable.
                print('rule_buffer_resume_loaded = False (starting empty)')
        print('next')
        # reload agents reward
        with open(trained_checkpoint + r'_reward_last.file', "rb") as f:
            agents_reward = pickle.load(f)
        # reload landmark error
        with open(trained_checkpoint + r'_lerror_last.file', "rb") as f:
            landmark_error_episode = pickle.load(f)
        # reload agent out of world
        with open(trained_checkpoint + r'_outworld_last.file', "rb") as f:
            agent_outofworld_episode = pickle.load(f)
        # reload landmark collision
        with open(trained_checkpoint + r'_landcoll_last.file', "rb") as f:
            landmark_collision_episode = pickle.load(f)
        # 增加障碍物碰撞记录文件
        with open(trained_checkpoint + r'_obstaclecoll_last.file', "rb") as f:
            obstacle_collision_episode = pickle.load(f)
        agent_collision_path = trained_checkpoint + r'_agentcoll_last.file'
        if os.path.exists(agent_collision_path):
            with open(agent_collision_path, "rb") as f:
                agent_collision_episode = pickle.load(f)

        loaded_metric_windows = load_metric_windows(model_dir, resume_state, num_agents, num_landmarks)
        if loaded_metric_windows is None:
            metric_windows = metric_windows_from_legacy(
                num_agents,
                num_landmarks,
                agents_reward,
                landmark_error_episode,
                agent_outofworld_episode,
                agent_collision_episode,
                landmark_collision_episode,
                obstacle_collision_episode,
            )
            print('metric_windows_loaded =  legacy_files')
        else:
            metric_windows = loaded_metric_windows
            print('metric_windows_loaded =  unified_checkpoint')
        (
            agents_reward,
            landmark_error_episode,
            agent_outofworld_episode,
            agent_collision_episode,
            landmark_collision_episode,
            obstacle_collision_episode,
        ) = bind_legacy_metric_aliases(metric_windows)

        # Batch size is a deliberate experiment parameter.  Legacy checkpoints
        # remain reproducible by default, while a config can explicitly replace
        # their saved value for a new continuation experiment.
        if resume_state is not None:
            t = int(resume_state.get('global_step', t))
            counter = int(resume_state.get('counter', counter))
            saved_batch_size = int(resume_state.get('batch_size', BATCH_SIZE))
            if RESUME_OVERRIDE_BATCH_SIZE:
                print('resume_batch_size_source = config')
            else:
                BATCH_SIZE = saved_batch_size
                print('resume_batch_size_source = checkpoint')
            noise = float(resume_state.get('noise', noise))
            avg_rewards_best = resume_state.get('avg_rewards_best', avg_rewards_best)
            maddpg.iter = int(resume_state.get('maddpg_iter', maddpg.iter))
            maddpg.iter_delay = int(resume_state.get('maddpg_iter_delay', maddpg.iter_delay))
            if EXP_REP_BUF:
                priority = restore_priority(resume_state.get('priority'), parallel_envs)
            restore_rng_state(resume_state.get('rng_state'))
            print('resume_state_loaded =  True')
            print('resume_global_step  = ', t)
            print('resume_masac_iter   = ', maddpg.iter)
            print('resume_batch_size   = ', BATCH_SIZE)
            print('resume_noise        = ', noise)
        else:
            print('resume_state_loaded =  False')
            print('configured_batch_size = ', BATCH_SIZE)

    print('Starting iterations... \r\n')
    start_episode = PRE_TRAINED_EP + parallel_envs if PRE_TRAINED else 0
    if PRE_TRAINED and resume_state is None:
        noise *= noise_reduction ** (int(PRE_TRAINED_EP / parallel_envs))
    if start_episode >= number_of_episodes:
        print(
            'No training episodes left: start_episode = %d, number_of_episodes = %d'
            % (start_episode, number_of_episodes)
        )
        env.close()
        logger.close()
        return

    # show progress bar
    if PROGRESS_BAR == True:
        import tqdm
        # initializing progress bar object
        timer_bar = tqdm.tqdm(total=number_of_episodes - start_episode, desc='\r\n Episode', position=0)

    # Vector workers are initialized once.  Thereafter, only workers with a
    # terminal transition are reset, so every unfinished trajectory remains
    # continuous across collection batches.
    profile_start = profiler.tic()
    all_obs = env.reset()
    profiler.add_worker_timings(getattr(env, 'last_reset_timings', []))
    profiler.toc('env_reset', profile_start)
    for i in range(num_agents):
        maddpg.masac_agent[i].noise.reset()

    obs_roll = np.rollaxis(all_obs, 1)
    obs = transpose_list(obs_roll)
    obs_size = obs[0][0].size
    history = copy.deepcopy(obs)
    for n in range(parallel_envs):
        for m in range(num_agents):
            history[n][m] = np.zeros((HISTORY_LENGTH, obs_size), dtype=float)
    history_a = np.zeros(
        (parallel_envs, num_agents, HISTORY_LENGTH, ACTION_DIM),
        dtype=float,
    )
    reward_this_episode = np.zeros((parallel_envs, num_agents), dtype=float)
    episode_diagnostics = init_episode_diagnostics(parallel_envs, num_agents)
    landmark_error = [[] for _ in range(num_landmarks)]
    frames = []
    tmax = 0
    if RENDER == True:
        frames.append(env.render('rgb_array'))

    episode = int(start_episode)
    completed_since_update = 0
    last_tensorboard_episode = int(start_episode)
    last_policy_save_episode = int(start_episode)
    while episode < number_of_episodes:
        profiler.begin_episode_batch()
        env_steps_this_episode = 0
        update_calls_this_episode = 0
        completed_this_batch = 0
        # next_history = copy.deepcopy(history)
        his = []

        for episode_t in range(episode_length):
            # get actions
            # explore = only explore for a certain number of episodes
            # action input needs to be transposed
            # actions = maddpg.act(transpose_to_tensor(obs), noise=noise)
            profile_start = profiler.tic()
            his = []
            # Convert each policy input once per environment step. The
            # previous code rebuilt these tensors inside the agent loop.
            policy_obs = None
            if episode >= START_STEPS:
                history_obs_tensors = transpose_to_tensor(history)
                history_action_tensors = transpose_to_tensor(history_a)
                policy_obs = transpose_to_tensor(obs)
                for i in range(num_agents):
                    his.append(
                        torch.cat(
                            (history_obs_tensors[i], history_action_tensors[i]),
                            dim=2,
                        )
                    )

            profiler.toc('policy_input', profile_start)
            profile_start = profiler.tic()
            if episode < START_STEPS:
                # Uniform random steps at the begining as suggested by https://spinningup.openai.com/en/latest/algorithms/ddpg.html
                # actions_array = np.random.uniform(-1,1,(1,parallel_envs,num_agents))
                # 原代码此处只定义了方向的变换范围在（-1，1）之间，原因在于不想让智能体转动的速度过快，局限于现实船只无法快速转向，并且是随机给值
                if START_POLICY == "random":
                    actions_array = np.random.uniform(-1, 1, (num_agents, parallel_envs, ACTION_DIM))
                # 但此处进行简化处理，方向变换定在在（-np.pi，np.pi）之间，并采用A * 算法计算当前应当的朝向,并且假设只有一个窗口一个智能体
                elif START_POLICY == "AStar":
                    actions_array = np.full((num_agents, parallel_envs, ACTION_DIM), 100.)  # 100作为暗号，在处理动作时有相应动作
                # actions_array = np.zeros((num_agents, parallel_envs, ACTION_DIM))
                # A_Star_Info = env.A_star_Info() #[parallel_env, Info_size] Info包括（  智能体数量，障碍物数量，目标数量，智能体基础信息，障碍物基础信息，目标基础信息（位置，大小），智能体朝向角集合 ）
                # for i in range(len(A_Star_Info)):#每一个场景分开
                #     for j in range(num_agents):  # 每一个智能体分开
                #         actions_array[j][i][0] = A_Star(A_Star_Info[i],j,0) #暂定第一目标是第一个地标
            else:
                actions = maddpg.act(his, policy_obs, noise=noise)
                actions_array = torch.stack(actions).detach().cpu().numpy()
            profiler.toc('policy_act', profile_start)

            # transpose the list of list
            # flip the first two indices
            # input to step requires the first index to correspond to number of parallel agents
            actions_for_env = np.rollaxis(actions_array, 1)

            # environment step
            # step forward one frame
            # next_obs, next_obs_full, rewards, dones, info = env.step(actions_for_env)
            profile_start = profiler.tic()
            next_obs, rewards, dones, info = env.step(actions_for_env)
            profiler.toc('env_step', profile_start)
            profiler.add_worker_timings(getattr(env, 'last_step_timings', []))
            env_steps_this_episode += 1
            profile_start = profiler.tic()
            # 从 info 中拆出完整动作链。主 transition 的动作字段使用
            # raw_action；实际执行动作只进入历史、ICM/CTIP 和物理诊断。
            executed_actions_for_env = executed_actions_from_info(info, actions_for_env)
            raw_actions_for_env = raw_actions_from_info(info, actions_for_env)
            constrained_actions_for_env = constrained_actions_from_info(
                info,
                raw_actions_for_env,
            )
            rule_actions_for_env = rule_actions_from_info(
                info,
                executed_actions_for_env,
            )
            smoothed_actions_for_env = smoothed_actions_from_info(
                info,
                executed_actions_for_env,
            )
            filter_active_for_env = filter_active_from_info(info, actions_for_env)
            if rule_buffer is not None and (DAGGER_ENABLED or np.any(filter_active_for_env > 0.0)):
                encounter_codes_for_env, filter_mode_codes_for_env = rule_metadata_from_info(
                    info,
                    actions_for_env,
                )
                dcpa_for_env, tcpa_for_env = rule_geometry_from_info(info, actions_for_env)
                rule_records = []
                for env_i in range(parallel_envs):
                    for agent_i in range(num_agents):
                        is_filter_active = bool(
                            filter_active_for_env[env_i, agent_i, 0] > 0.0
                        )
                        # 0904/0906 兼容路径：未启用 DAgger 时仍只保存过滤器
                        # 实际介入的样本，避免改变旧实验的规则池分布。
                        if not DAGGER_ENABLED and not is_filter_active:
                            continue
                        raw_action = np.asarray(
                            raw_actions_for_env[env_i, agent_i],
                            dtype=np.float32,
                        ).copy()
                        constrained_action = np.asarray(
                            constrained_actions_for_env[env_i, agent_i],
                            dtype=np.float32,
                        ).copy()
                        rule_action = np.asarray(
                            rule_actions_for_env[env_i, agent_i],
                            dtype=np.float32,
                        ).copy()
                        smoothed_action = np.asarray(
                            smoothed_actions_for_env[env_i, agent_i],
                            dtype=np.float32,
                        ).copy()
                        _q_dcpa = float(dcpa_for_env[env_i, agent_i])
                        _q_tcpa = float(tcpa_for_env[env_i, agent_i])
                        _q_closing = bool(np.isfinite(_q_tcpa) and _q_tcpa > 0.0)
                        _q_hit = bool(should_query(_q_dcpa, _q_tcpa, _q_closing,
                                                 TEACHER_QUERY_D_SAFE, TEACHER_QUERY_REACT_S,
                                                 TEACHER_QUERY_K_D, TEACHER_QUERY_K_T))
                        _q_conf = float(query_confidence(_q_dcpa, TEACHER_QUERY_D_SAFE,
                                                         _q_closing, TEACHER_QUERY_K_D))
                        rule_records.append({
                            'history_obs': np.asarray(
                                history[env_i][agent_i],
                                dtype=np.float32,
                            ).copy(),
                            'history_action': np.asarray(
                                history_a[env_i, agent_i],
                                dtype=np.float32,
                            ).copy(),
                            'observation': np.asarray(
                                obs[env_i][agent_i],
                                dtype=np.float32,
                            ).copy(),
                            'raw_action': raw_action,
                            'constrained_action': constrained_action,
                            'rule_action': rule_action,
                            # 兼容字段 applied_action 在 v2 中明确表示最终平滑执行动作。
                            'applied_action': smoothed_action,
                            'correction_norm': float(
                                np.linalg.norm(rule_action - constrained_action)
                            ),
                            'encounter_type_code': int(
                                encounter_codes_for_env[env_i, agent_i]
                            ),
                            'filter_mode_code': int(
                                filter_mode_codes_for_env[env_i, agent_i]
                            ),
                            'agent_index': agent_i,
                            'episode_id': int(episode + env_i),
                            'environment_step': int(episode_t),
                            'filter_active': is_filter_active,
                            # DAgger 的 shadow teacher 在每个访问状态上都产生
                            # 一个候选动作，但只有存在有效会遇风险时才作为规则
                            # 教师标签参与模仿；无风险访问样本保留在池中用于
                            # 分布聚合；非介入样本通过较低 confidence 降低模仿权重。
                            'teacher_valid': bool(is_filter_active or DAGGER_ENABLED),
                            'teacher_confidence': float(
                                np.clip(DAGGER_TEACHER_WEIGHT, 0.0, 1.0)
                                * (1.0 if is_filter_active else DAGGER_NON_INTERVENTION_CONFIDENCE)
                                if (is_filter_active or DAGGER_ENABLED) else 0.0
                            ),
                            'teacher_query': _q_hit,
                            'teacher_query_confidence': _q_conf,
                            'dcpa': float(dcpa_for_env[env_i, agent_i]),
                            'tcpa': float(tcpa_for_env[env_i, agent_i]),
                            'dagger_iteration': int(episode) if DAGGER_ENABLED else 0,
                        })
                if rule_records:
                    rule_buffer.push_batch(rule_records)
                    _q_rows = [r for r in rule_records if r.get('teacher_query')]
                    _q_valid = [r for r in _q_rows if r.get('teacher_valid')]
                    _RUNTIME_TEACHER_QUERY_STATS['rate'] = float(len(_q_rows)) / max(float(len(rule_records)), 1.0)
                    _RUNTIME_TEACHER_QUERY_STATS['coverage'] = float(len(_q_valid)) / max(float(len(_q_rows)), 1.0)
            update_episode_diagnostics(episode_diagnostics, info, num_agents)
            profiler.toc('info_extract', profile_start)

            # rewards_sum += np.mean(rewards)
            
            reward_this_episode += rewards
                
                
            if ICM_Active:
                # 在调用ICM模型之前添加
                obs_full = torch.cat(transpose_to_tensor(obs), dim=1)
                next_obs_full = torch.cat(transpose_to_tensor(next_obs), dim=1)
                actions_for_env_full = torch.cat(
                    transpose_to_tensor(executed_actions_for_env),
                    dim=1,
                )
                obs_full = obs_full.to(DEVICE)
                next_obs_full = next_obs_full.to(DEVICE)
                actions_for_env_full = actions_for_env_full.to(DEVICE)
                # current_batch = t // 100000  # t 是全局步数
                # lambda_t = 1.0 * np.exp(-0.1 * current_batch)
                # 计算内在奖励
                predicted_next_state_encoded, predicted_action = icm(obs_full, next_obs_full, actions_for_env_full)
                next_state_encoded = icm.encoder(next_obs_full)
                intrinsic_reward = 0.5 * torch.sum((predicted_next_state_encoded - next_state_encoded) ** 2, dim=-1, keepdim=True)
                # rewards = rewards + (intrinsic_reward * lambda_t * ICM_GAMMA).tolist()
                rewards = rewards + (intrinsic_reward * ICM_GAMMA).tolist()
                # 更新ICM网络
                ICM_UPDATE_FREQ = 5
                if episode_t % ICM_UPDATE_FREQ == 0 and episode_t > 0:
                    forward_loss = 0.5 * torch.sum((predicted_next_state_encoded - next_state_encoded) ** 2, dim=-1).mean()
                    inverse_loss = torch.nn.functional.mse_loss(predicted_action, actions_for_env_full)
                    icm_loss = forward_loss + inverse_loss
                    icm_optimizer.zero_grad()
                    icm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(icm.parameters(), 0.5)
                    icm_optimizer.step()
                profiler.toc('icm_update', profile_start)
            # collect experience
            # add data to buffer
            # transition = (obs, obs_full, actions_for_env, rewards, next_obs, next_obs_full, dones)
            # transition = (obs, actions_for_env, rewards, next_obs, dones)
            # transition = (history, actions_for_env, rewards, next_history, dones)
            # ReplayBuffer stores per-environment NumPy row views.  Keep an
            # owned copy here because completed workers are reset below and
            # their live next_obs entries are then replaced in place.
            transition_next_obs = np.asarray(next_obs, dtype=float).copy()
            transition = (
                history,
                history_a,
                obs,
                # 主 SAC Critic 估计的是 Q(s, raw_action)，因此这里不能
                # 写入被规则过滤器/平滑器修改后的动作。
                raw_actions_for_env,
                rewards,
                transition_next_obs,
                dones,
                # 兼容字段：后续附加字段不参与主 Critic 动作选择。
                raw_actions_for_env,
                constrained_actions_for_env,
                rule_actions_for_env,
                smoothed_actions_for_env,
                filter_active_for_env,
            )
            if EXP_REP_BUF == False:
                profile_start = profiler.tic()
                buffer.push(transition)
            else:
                profile_start = profiler.tic()
                buffer.push(transition, priority)
            profiler.toc('buffer_push', profile_start)

            # Update history buffers
            if RNN:
                profile_start = profiler.tic()
                # Add obs to the history buffer
                for n in range(parallel_envs):
                    for m in range(num_agents):
                        aux = obs[n][m].reshape(1, obs_size)
                        history[n][m] = np.concatenate((history[n][m], aux), axis=0)
                        history[n][m] = np.delete(history[n][m], 0, 0)
                # Add actions to the history buffer
                # 历史动作必须记录真正送入执行器的平滑动作，而不是
                # Critic 使用的 raw_action；这保持 RNN 状态与环境转移一致。
                history_a = np.concatenate((history_a, smoothed_actions_for_env.reshape(parallel_envs, num_agents, 1, ACTION_DIM)),
                                           axis=2)
                history_a = np.delete(history_a, 0, 2)
                profiler.toc('history_update', profile_start)

            # obs, obs_full = next_obs, next_obs_full
            obs = next_obs

            # increment global step counter
            t += parallel_envs

            # save gif frame
            if RENDER == True:
                frames.append(env.render('rgb_array'))
                tmax += 1

            # for benchmarking learned policies
            if BENCHMARK:
                profile_start = profiler.tic()
                error_mean = np.zeros(num_landmarks)
                for inf in _env_info_rows(info):
                    if is_fixed_info_array(inf):
                        landmark_count = _info_num_landmarks(inf)
                        errors = inf[0, field_slice('landmark_error', landmark_count)]
                        error_mean[:min(num_landmarks, errors.size)] += errors[:num_landmarks]
                    elif isinstance(inf, dict):
                        for l in range(num_landmarks):
                            error_mean[l] = np.add(error_mean[l], (inf['n'][0][0][l]))
                error_mean /= parallel_envs
                for i in range(num_landmarks):
                    landmark_error[i].append(error_mean[i])
                profiler.toc('benchmark', profile_start)
                # for e, inf in enumerate(info):
                #     for a in range(num_agents):
                #         agent_info[a] = np.add(agent_info[a],(inf['n'][a]))

            # A completed worker must be reset independently.  The terminal
            # transition above still contains its true terminal next_obs and
            # done flag; only the observation used for the *next* policy step
            # is replaced with a fresh scene.  This avoids truncating the other
            # parallel trajectories whenever one vessel succeeds or fails.
            done_env_mask = np.asarray(dones, dtype=bool).reshape(parallel_envs, -1).any(axis=1)
            if np.any(done_env_mask):
                done_indices = np.flatnonzero(done_env_mask)
                benchmark_stats = collect_benchmark_episode_stats(info, parallel_envs, num_agents)
                state_coverage = []
                for agent_i in range(num_agents):
                    coverage = maddpg.masac_agent[agent_i].tracker.calculate_coverage()
                    state_coverage.append(coverage['visited_states_count'])
                landmark_error_values = (
                    collect_landmark_error_values(landmark_error, num_landmarks)
                    if BENCHMARK else [None] * num_landmarks
                )

                # Metrics are appended only for episodes that actually ended.
                # This replaces the former mixed window of one finished worker
                # plus several prematurely truncated parallel trajectories.
                append_episode_metric_windows(
                    metric_windows,
                    reward_this_episode[done_indices].copy(),
                    select_episode_diagnostics(episode_diagnostics, done_indices),
                    state_coverage,
                    select_benchmark_episode_stats(benchmark_stats, done_indices),
                    landmark_error_values,
                    window_sizes,
                )
                completed_count = int(done_indices.size)
                episode += completed_count
                completed_this_batch += completed_count
                completed_since_update += completed_count
                if PROGRESS_BAR == True:
                    timer_bar.update(min(completed_count, number_of_episodes - (episode - completed_count)))

                reward_this_episode[done_indices] = 0.0
                reset_episode_diagnostics_rows(episode_diagnostics, done_indices)
                profile_start = profiler.tic()
                reset_indices, reset_obs = env.reset_done(done_env_mask)
                profiler.add_worker_timings(getattr(env, 'last_reset_timings', []))
                profiler.toc('env_reset', profile_start)

                for reset_pos, env_i in enumerate(reset_indices):
                    env_i = int(env_i)
                    next_obs[env_i] = reset_obs[reset_pos]
                    for agent_i in range(num_agents):
                        history[env_i][agent_i] = np.zeros(
                            (HISTORY_LENGTH, obs_size),
                            dtype=float,
                        )
                    history_a[env_i] = 0.0

                if episode >= number_of_episodes:
                    break

        # Reduce the quantity of noise added to the action
        noise *= noise_reduction

        # 提前生成下一回合场景；reset 在子进程中运行，可与本回合的网络更新、
        # TensorBoard 写入和 checkpoint 处理并行。下一轮开始时再 reset_wait。
        # UPDATE_EVERY retains the old parallel-group meaning: with eight
        # workers and UPDATE_EVERY=1, update after eight true episode endings.
        completed_group_size = max(int(UPDATE_EVERY) * parallel_envs, 1)
        update_groups = completed_since_update // completed_group_size
        tensorboard_log_this_episode = (
            completed_this_batch > 0
            and (
                episode - last_tensorboard_episode >= tensorboard_log_interval
                or episode >= number_of_episodes
            )
        )
        if len(buffer) > BATCH_SIZE and update_groups > 0:
            profile_start = profiler.tic()
            update_logger = logger if tensorboard_log_this_episode else null_logger
            for _ in range(UPDATE_TIMES * int(update_groups)):
                for a_i in range(num_agents):
                    if EXP_REP_BUF == False:
                        samples = buffer.sample(BATCH_SIZE)
                        rule_samples = None
                        # DAgger 可额外保存无风险访问状态，但 Actor 更新只采样
                        # 有效规则教师标签，避免零权重记录稀释规则模仿批次。
                        if rule_buffer is not None and rule_buffer.count(
                            a_i, teacher_valid_only=True
                        ) >= RULE_BUFFER_MIN_SIZE:
                            rule_samples = rule_buffer.sample(
                                RULE_ACTOR_BATCH_SIZE,
                                agent_index=a_i,
                                mode=RULE_SAMPLE_MODE,
                                teacher_valid_only=True,
                            )
                        maddpg.update(
                            samples,
                            a_i,
                            update_logger,
                            rule_samples=rule_samples,
                        )
                        update_calls_this_episode += 1
                    else:
                        samples, indexes = buffer.sample(BATCH_SIZE)
                        rule_samples = None
                        if rule_buffer is not None and rule_buffer.count(
                            a_i, teacher_valid_only=True
                        ) >= RULE_BUFFER_MIN_SIZE:
                            rule_samples = rule_buffer.sample(
                                RULE_ACTOR_BATCH_SIZE,
                                agent_index=a_i,
                                mode=RULE_SAMPLE_MODE,
                                teacher_valid_only=True,
                            )
                        new_priorities = maddpg.update(
                            samples,
                            a_i,
                            update_logger,
                            rule_samples=rule_samples,
                        )
                        update_calls_this_episode += 1
                        buffer.update(indexes, new_priorities)
                        priority = np.full(parallel_envs, max(float(np.max(new_priorities)), 1.0))
                maddpg.update_targets()  # soft update the target network towards the actual networks
            profiler.toc('network_update', profile_start, count=max(update_calls_this_episode, 1))
            completed_since_update %= completed_group_size
            # print("\n",buffer.tree.write,"总数",buffer.tree.total())
        avg_rewards = [
            _window_stat(metric_windows['episode_reward'][n], 'mean')
            for n in range(num_agents)
        ]
        if tensorboard_log_this_episode:
            log_metric_windows(logger, metric_windows, episode, num_agents, num_landmarks)
            log_trainer_losses(logger, maddpg, episode, num_agents)
            logger.add_scalar('rule_buffer/teacher_query_rate', _RUNTIME_TEACHER_QUERY_STATS['rate'], episode)
            logger.add_scalar('rule_buffer/teacher_query_coverage', _RUNTIME_TEACHER_QUERY_STATS['coverage'], episode)
            log_rule_buffer_metrics(logger, rule_buffer, episode, num_agents)
            last_tensorboard_episode = int(episode)

        if PROGRESS_BAR == True:
            progress_info = {'avg_rew': float(np.mean(avg_rewards))}
            if BENCHMARK:
                progress_info['avg_error'] = _window_stat(metric_windows['landmark_error'][0], 'mean')
            timer_bar.set_postfix(progress_info)

        # saving model
        save_info = (
            episode - last_policy_save_episode >= policy_save_interval
            or episode >= number_of_episodes - parallel_envs
        )
        save_dict_list = []
        target_entropy_list = []
        log_alpha_list = []
        alpha_list = []
        if save_info:
            profile_start = profiler.tic()
            for i in range(num_agents):

                if DNN == 'MASAC':
                    if AUTOMATIC_ENTROPY:
                        if ICM_Active:
                            save_dict = {'actor_params': maddpg.masac_agent[i].actor.state_dict(),
                                        'actor_optim_params': maddpg.masac_agent[i].actor_optimizer.state_dict(),
                                        'critic_params': maddpg.masac_agent[i].critic.state_dict(),
                                        'target_critic_params': maddpg.masac_agent[i].target_critic.state_dict(),
                                        'critic_optim_params': maddpg.masac_agent[i].critic_optimizer.state_dict(),
                                        'alpha_optim_params': maddpg.masac_agent[i].alpha_optimizer.state_dict(),
                                        'icm_state_dict': icm.state_dict(),
                                        'icm_optimizer_state_dict': icm_optimizer.state_dict()}
                        else:
                            if CTIP_active:
                                save_dict = {'actor_params': maddpg.masac_agent[i].actor.state_dict(),
                                        'actor_optim_params': maddpg.masac_agent[i].actor_optimizer.state_dict(),
                                        'critic_params': maddpg.masac_agent[i].critic.state_dict(),
                                        'target_critic_params': maddpg.masac_agent[i].target_critic.state_dict(),
                                        'critic_optim_params': maddpg.masac_agent[i].critic_optimizer.state_dict(),
                                        'alpha_optim_params': maddpg.masac_agent[i].alpha_optimizer.state_dict(),
                                        'icm_state_dict': maddpg.masac_agent[i].icm.state_dict(),
                                        'icm_optimizer_state_dict': maddpg.masac_agent[i].icm_optimizer.state_dict()}
                            else:
                                save_dict = {'actor_params': maddpg.masac_agent[i].actor.state_dict(),
                                            'actor_optim_params': maddpg.masac_agent[i].actor_optimizer.state_dict(),
                                            'critic_params': maddpg.masac_agent[i].critic.state_dict(),
                                            'target_critic_params': maddpg.masac_agent[i].target_critic.state_dict(),
                                            'critic_optim_params': maddpg.masac_agent[i].critic_optimizer.state_dict(),
                                            'alpha_optim_params': maddpg.masac_agent[i].alpha_optimizer.state_dict()}
                        # Append agents alpha parameters
                        target_entropy_list.append(maddpg.masac_agent[i].target_entropy)
                        log_alpha_list.append(maddpg.masac_agent[i].log_alpha)
                        alpha_list.append(maddpg.masac_agent[i].alpha)
                    else:
                        save_dict = {'actor_params': maddpg.masac_agent[i].actor.state_dict(),
                                     'actor_optim_params': maddpg.masac_agent[i].actor_optimizer.state_dict(),
                                     'critic_params': maddpg.masac_agent[i].critic.state_dict(),
                                     'target_critic_params': maddpg.masac_agent[i].target_critic.state_dict(),
                                     'critic_optim_params': maddpg.masac_agent[i].critic_optimizer.state_dict()}
                        if CTIP_active:
                            save_dict.update({
                                'icm_state_dict': maddpg.masac_agent[i].icm.state_dict(),
                                'icm_optimizer_state_dict': maddpg.masac_agent[i].icm_optimizer.state_dict()
                            })
                save_dict_list.append(save_dict)

            # 记录最后回合的所有信息
            #save num episode
            torch.save([], 
                       os.path.join(model_dir, 'episode_last_{}.pt'.format(episode)))
            # save dict_list
            torch.save(save_dict_list,
                       os.path.join(model_dir, 'episode_last.pt'))
            # save the replay buffer
            buffer.save(os.path.join(model_dir, 'episode_last.file'))
            if rule_buffer is not None:
                # The rule pool has its own format and lifecycle; it is never
                # merged into the main replay file.
                rule_buffer.save(os.path.join(model_dir, 'rule_buffer_last.file'))
            # save agents reward
            with open(os.path.join(model_dir, 'episode_reward_last.file'), "wb") as f:
                pickle.dump(agents_reward, f)
            # save landmark error
            with open(os.path.join(model_dir, 'episode_lerror_last.file'), "wb") as f:
                pickle.dump(landmark_error_episode, f)
            # reload agent out of world
            with open(os.path.join(model_dir, 'episode_outworld_last.file'), "wb") as f:
                pickle.dump(agent_outofworld_episode, f)
            # reload obstacle collisions
            with open(os.path.join(model_dir, 'episode_obstaclecoll_last.file'), "wb") as f:
                pickle.dump(obstacle_collision_episode, f)
            # reload landmark collisions
            with open(os.path.join(model_dir, 'episode_landcoll_last.file'), "wb") as f:
                pickle.dump(landmark_collision_episode, f)
            # save agent collision windows
            with open(os.path.join(model_dir, 'episode_agentcoll_last.file'), "wb") as f:
                pickle.dump(agent_collision_episode, f)
            save_metric_windows(metric_windows, model_dir, suffix='last', before_dir=before_dir, episode=episode)
            # save agents alpha parameters
            with open(os.path.join(model_dir, 'episode_target_entropy_last.file'), "wb") as f:
                pickle.dump(target_entropy_list, f)
            with open(os.path.join(model_dir, 'episode_log_alpha_last.file'), "wb") as f:
                pickle.dump(log_alpha_list, f)
            with open(os.path.join(model_dir, 'episode_alpha_last.file'), "wb") as f:
                pickle.dump(alpha_list, f)
            #记录当前阶段参数配置，用于时间回溯
            # save dict_list
            torch.save(save_dict_list,
                       os.path.join(before_dir, 'episode{}.pt'.format(episode)))
            # save the replay buffer
            buffer.save(os.path.join(before_dir, 'episode{}.file'.format(episode)))
            if rule_buffer is not None:
                rule_buffer.save(
                    os.path.join(before_dir, 'rule_buffer_{}.file'.format(episode))
                )
            # save agents alpha parameters
            with open(os.path.join(before_dir, 'episode_target_entropy{}.file').format(episode), "wb") as f:
                pickle.dump(target_entropy_list, f)
            with open(os.path.join(before_dir, 'episode_log_alpha{}.file').format(episode), "wb") as f:
                pickle.dump(log_alpha_list, f)
            with open(os.path.join(before_dir, 'episode_alpha{}.file').format(episode), "wb") as f:
                pickle.dump(alpha_list, f)
            
            # 最佳得分阶段记录
            if np.mean(avg_rewards) > np.mean(avg_rewards_best):
                # SAVE BEST VALUES
                # save num episode
                torch.save([],
                           os.path.join(model_dir, 'episode_best_{}.pt'.format(episode)))
                # save dict_list
                torch.save(save_dict_list,
                           os.path.join(model_dir, 'episode_best.pt'))
                # save the replay buffer
                buffer.save(os.path.join(model_dir, 'episode_best.file'))
                if rule_buffer is not None:
                    rule_buffer.save(os.path.join(model_dir, 'rule_buffer_best.file'))
                # save agents reward
                with open(os.path.join(model_dir, 'episode_reward_best.file'), "wb") as f:
                    pickle.dump(agents_reward, f)
                # save landmark error
                with open(os.path.join(model_dir, 'episode_lerror_best.file'), "wb") as f:
                    pickle.dump(landmark_error_episode, f)
                # reload agent out of world
                with open(os.path.join(model_dir, 'episode_outworld_best.file'), "wb") as f:
                    pickle.dump(agent_outofworld_episode, f)
                # reload obstacle collisions
                with open(os.path.join(model_dir, 'episode_obstaclecoll_best.file'), "wb") as f:
                    pickle.dump(obstacle_collision_episode, f)
                # reload landmark collisions
                with open(os.path.join(model_dir, 'episode_landcoll_best.file'), "wb") as f:
                    pickle.dump(landmark_collision_episode, f)
                # save agent collision windows
                with open(os.path.join(model_dir, 'episode_agentcoll_best.file'), "wb") as f:
                    pickle.dump(agent_collision_episode, f)
                save_metric_windows(metric_windows, model_dir, suffix='best')
                # save agents alpha parameters
                with open(os.path.join(model_dir, 'episode_target_entropy_best.file'), "wb") as f:
                    pickle.dump(target_entropy_list, f)
                with open(os.path.join(model_dir, 'episode_log_alpha_best.file'), "wb") as f:
                    pickle.dump(log_alpha_list, f)
                with open(os.path.join(model_dir, 'episode_alpha_best.file'), "wb") as f:
                    pickle.dump(alpha_list, f)
                # update avg_rewards_best
                try:
                    avg_rewards_best = avg_rewards.copy()
                except NameError:
                    pass

            training_state = {
                'version': TRAINING_STATE_VERSION,
                'saved_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                'config_file': configFile,
                'config_path': os.path.abspath(config_path),
                'episode': int(episode),
                'global_step': int(t),
                'maddpg_iter': int(maddpg.iter),
                'maddpg_iter_delay': int(maddpg.iter_delay),
                'noise': float(noise),
                'batch_size': int(BATCH_SIZE),
                'resume_override_batch_size': bool(RESUME_OVERRIDE_BATCH_SIZE),
                'counter': int(counter),
                'avg_rewards_best': copy.deepcopy(avg_rewards_best),
                'metric_windows': copy.deepcopy(metric_windows),
                'metric_window_sizes': copy.deepcopy(window_sizes),
                'metric_descriptions': {
                    'agent_window_metrics': copy.deepcopy(AGENT_WINDOW_METRIC_SPECS),
                    'reward_components': copy.deepcopy(REWARD_COMPONENT_DESCRIPTIONS),
                    'reward_metrics': copy.deepcopy(REWARD_METRIC_DESCRIPTIONS),
                    'done_reasons': copy.deepcopy(DONE_REASON_DESCRIPTIONS),
                    'performance': copy.deepcopy(PERFORMANCE_STAGE_DESCRIPTIONS),
                    'sac_update': copy.deepcopy(SAC_UPDATE_METRIC_DESCRIPTIONS),
                },
                'priority': priority_snapshot(priority if EXP_REP_BUF else None),
                'rng_state': capture_rng_state(),
                'config_file': str(configFile),
                'training_run_name': str(training_run_name),
                'scenario_profile': str(SCENARIO_PROFILE),
                'curriculum_stage': int(CURRICULUM_STAGE),
                'policy_algorithm': DNN,
                'scenario': SCENARIO,
                'control_model': CONTROL_MODEL,
                'action_semantic_version': ACTION_SEMANTIC_VERSION,
                'network_contract': copy.deepcopy(network_contract),
                'action_dim': int(ACTION_DIM),
                'automatic_entropy_tuning': bool(AUTOMATIC_ENTROPY),
                'entity_encoder_enabled': bool(ENTITY_ENCODER_ENABLED),
                'entity_encoder_latent_dim': int(ENTITY_ENCODER_LATENT_DIM),
                'lr_alpha': float(LR_ALPHA),
                'parallel_envs': int(parallel_envs),
                'num_agents': int(num_agents),
                'num_landmarks': int(num_landmarks),
                'num_obstacles': int(num_obstacles),
                'num_static_obstacles': int(num_static_obstacles),
                'map_half_size': float(map_half_size),
                'static_obstacle_min_size': float(static_obstacle_min_size),
                'static_obstacle_max_size': float(static_obstacle_max_size),
                'dynamic_obstacle_min_speed': dynamic_obstacle_min_speed,
                'dynamic_obstacle_max_speed': dynamic_obstacle_max_speed,
                'ob_range': float(ob_range),
                'num_ob': int(num_ob),
                'num_static_ob_slots': int(num_static_ob_slots),
                'obs_dim': int(obs_dim),
                'target_observation_mode': str(target_observation_mode),
                'target_position_noise_std': float(target_position_noise_std),
                'reward_config': copy.deepcopy(reward_config),
                'environment_config': copy.deepcopy(environment_config),
                # Optional dual-batch rule-learning contract.  These fields
                # are metadata only; the actual rule samples live in the
                # separate rule_buffer_last.file when enabled.
                'rule_buffer_enabled': bool(RULE_BUFFER_ENABLED),
                'rule_buffer_size': int(RULE_BUFFER_SIZE),
                'rule_buffer_min_size': int(RULE_BUFFER_MIN_SIZE),
                'rule_actor_batch_size': int(RULE_ACTOR_BATCH_SIZE),
                'rule_sample_mode': str(RULE_SAMPLE_MODE),
                'rule_imitation_weight': float(RULE_IMITATION_WEIGHT),
                'rule_buffer_current_size': int(
                    len(rule_buffer) if rule_buffer is not None else 0
                ),
                'rule_inserted_total': int(
                    rule_buffer.stats()['inserted_total'] if rule_buffer is not None else 0
                ),
                'rule_sampled_total': int(
                    rule_buffer.stats()['sampled_total'] if rule_buffer is not None else 0
                ),
                'rule_type_mapping': rule_type_mapping(),
                'rule_buffer_format_version': int(
                    rule_buffer.stats()['format_version'] if rule_buffer is not None else 2
                ),
                'dagger_enabled': bool(DAGGER_ENABLED),
            'dagger_teacher_weight': float(DAGGER_TEACHER_WEIGHT),
            'dagger_non_intervention_confidence': float(
                DAGGER_NON_INTERVENTION_CONFIDENCE
            ),
                'update_every': int(UPDATE_EVERY),
                'update_times': int(UPDATE_TIMES),
                'policy_save_interval': int(policy_save_interval),
                'tensorboard_log_interval': int(tensorboard_log_interval),
                'tensorboard_max_queue': int(tensorboard_max_queue),
                'tensorboard_flush_secs': int(tensorboard_flush_secs),
                'device_config': str(device_config),
                'device': str(DEVICE),
                'profile_training': bool(profile_training),
            }
            torch.save(training_state, os.path.join(model_dir, TRAINING_STATE_LAST))
            torch.save(training_state, os.path.join(before_dir, 'training_state_{}.pt'.format(episode)))
            profiler.toc('checkpoint_save', profile_start)
            last_policy_save_episode = int(episode)

            if RENDER == True:
                # save gif files
                if imageio is None:
                    print('imageio is not installed; skip gif saving.')
                else:
                    imageio.mimsave(os.path.join(model_dir, 'episode-{}.gif'.format(episode)),
                                    frames, duration=.04)

            # save benchmark
            # if BENCHMARK:
            #     file1 = open(benchmark_dir+r"\episode-{}.txt".format(episode),"w")#append mode
            #     file1.write(str(np.array(agent_info)/t))
            #     file1.close()

        profiler.finish_episode_batch(
            parallel_envs=completed_this_batch,
            env_steps=env_steps_this_episode,
            update_calls=update_calls_this_episode,
        )
        if tensorboard_log_this_episode:
            log_training_profiler(logger, profiler, episode)

    env.close()
    if PROGRESS_BAR == True:
        timer_bar.close()
    logger.close()
    # timer.finish()


if __name__ == '__main__':
    print('Start main')
    main()



