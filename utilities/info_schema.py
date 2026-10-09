"""固定宽度的环境诊断信息布局。

环境 worker 之间通过 Pipe 传递的 ``info`` 只保留训练和 TensorBoard
真正需要的数值，避免每一步构造多层 dict、list 和 benchmark 大对象。
布局使用 float32，字段顺序固定，便于主进程快速切片和向后兼容解码。
"""

import numpy as np

from .rule_metadata import encode_encounter_type, encode_filter_mode


REWARD_COMPONENT_KEYS = (
    'success',
    'progress',
    'range',
    'safety',
    'boundary',
    'smooth',
    'yaw_rate',
    'rudder_smooth',
    'intervention',
    'time',
    'terminal',
)

REWARD_METRIC_KEYS = (
    'target_distance',
    'min_clearance',
    'boundary_clearance',
    'capture_distance',
    'yaw_rate_abs_deg_s',
    'rudder_angle_abs_deg',
    'rudder_delta_abs_deg',
    'actual_thrust',
    'actual_rudder_deg',
)

DONE_REASON_CODES = {
    'none': 0,
    'success': 1,
    'out_of_bounds': 2,
    'collision': 3,
    'target_collision': 4,
    'timeout': 5,
}
DONE_REASON_NAMES = {value: key for key, value in DONE_REASON_CODES.items()}

# 固定前缀与目标数量无关。前 8 个位置保留原有布局，保证旧工具仍能
# 解码 raw/constrained/applied/filter/correction；新的动作语义字段放在
# benchmark 字段之后，避免改变旧字段的偏移。
ACTION_DIAGNOSTIC_DIM = 8
INFO_FIXED_PREFIX = (
    ACTION_DIAGNOSTIC_DIM
    + len(REWARD_COMPONENT_KEYS)
    + len(REWARD_METRIC_KEYS)
    + 1
)
BENCHMARK_COUNT_DIM = 4
ACTION_DIM_IN_INFO = 2

# 新动作语义扩展：纯规则动作、最终平滑动作，以及四段动作差异的范数。
# 这些字段只用于诊断、规则教师数据和 TensorBoard，不作为观测输入。
ACTION_EXTENSION_DIM = 8

# Rule metadata is appended after all legacy fields.  The marker makes it
# possible to distinguish the new width from an old info array without
# changing any pre-existing field offsets.
RULE_INFO_METADATA_DIM = 6
RULE_INFO_SCHEMA_VERSION = 3
RULE_INFO_SCHEMA_MAGIC = 9731.0


def info_width(num_landmarks):
    """返回每个 agent 的固定 info 行宽。"""
    return (
        INFO_FIXED_PREFIX
        + int(num_landmarks)
        + BENCHMARK_COUNT_DIM
        + ACTION_EXTENSION_DIM
        + RULE_INFO_METADATA_DIM
    )


def infer_num_landmarks(info_row_width, has_rule_metadata=False):
    """Infer target count from a fixed info row width.

    ``has_rule_metadata`` defaults to the legacy layout for callers that only
    have a width.  Callers holding the complete row should use
    :func:`has_rule_metadata` first.
    """
    metadata_dim = (
        RULE_INFO_METADATA_DIM + ACTION_EXTENSION_DIM
        if has_rule_metadata else 0
    )
    return max(
        int(info_row_width)
        - INFO_FIXED_PREFIX
        - BENCHMARK_COUNT_DIM
        - metadata_dim,
        0,
    )


def has_rule_metadata(info_row):
    """Return whether a row/array carries the appended rule metadata."""
    if not isinstance(info_row, np.ndarray) or info_row.ndim not in (1, 2):
        return False
    if (
        (info_row.ndim == 2 and info_row.shape[0] <= 0)
        or info_row.shape[-1]
        < INFO_FIXED_PREFIX + BENCHMARK_COUNT_DIM + RULE_INFO_METADATA_DIM
    ):
        return False
    # 元数据尾部顺序为 encounter、mode、schema_version、magic、DCPA、TCPA；
    # DCPA/TCPA 位于最后两列，因此不能直接用 ``[-1]``/``[-2]`` 判断标记。
    marker = info_row[-3] if info_row.ndim == 1 else info_row[0, -3]
    version = info_row[-4] if info_row.ndim == 1 else info_row[0, -4]
    # 版本号一并校验，避免把旧 v2 数组误解析为新增 DCPA/TCPA 布局。
    return bool(
        np.isclose(float(marker), RULE_INFO_SCHEMA_MAGIC)
        and np.isclose(float(version), RULE_INFO_SCHEMA_VERSION)
    )


def _field_slice(start, width):
    return slice(start, start + width)


def field_slice(field, num_landmarks=1):
    """返回固定数组中某字段的 slice 或整数索引。"""
    num_landmarks = int(num_landmarks)
    reward_component_start = ACTION_DIAGNOSTIC_DIM
    reward_metric_start = reward_component_start + len(REWARD_COMPONENT_KEYS)
    done_reason_index = reward_metric_start + len(REWARD_METRIC_KEYS)
    landmark_start = done_reason_index + 1
    fields = {
        'raw_action': _field_slice(0, 2),
        'constrained_action': _field_slice(2, 2),
        'applied_action': _field_slice(4, 2),
        'filter_active': 6,
        'action_correction_norm': 7,
        'reward_components': _field_slice(reward_component_start, len(REWARD_COMPONENT_KEYS)),
        'reward_metrics': _field_slice(reward_metric_start, len(REWARD_METRIC_KEYS)),
        'done_reason': done_reason_index,
        'landmark_error': _field_slice(landmark_start, num_landmarks),
        'benchmark_counts': _field_slice(landmark_start + num_landmarks, BENCHMARK_COUNT_DIM),
        # 新字段位于 benchmark 之后；只有带 schema marker 的当前 info 才有这些字段。
        'rule_action': _field_slice(
            landmark_start + num_landmarks + BENCHMARK_COUNT_DIM, ACTION_DIM_IN_INFO
        ),
        'smoothed_action': _field_slice(
            landmark_start + num_landmarks + BENCHMARK_COUNT_DIM + 2,
            ACTION_DIM_IN_INFO,
        ),
        'raw_to_constrained_norm': landmark_start + num_landmarks + BENCHMARK_COUNT_DIM + 4,
        'constrained_to_rule_norm': landmark_start + num_landmarks + BENCHMARK_COUNT_DIM + 5,
        'rule_to_smoothed_norm': landmark_start + num_landmarks + BENCHMARK_COUNT_DIM + 6,
        'raw_to_smoothed_norm': landmark_start + num_landmarks + BENCHMARK_COUNT_DIM + 7,
        'encounter_type_code': _field_slice(
            landmark_start + num_landmarks + BENCHMARK_COUNT_DIM + ACTION_EXTENSION_DIM, 1
        ),
        'filter_mode_code': _field_slice(
            landmark_start + num_landmarks + BENCHMARK_COUNT_DIM + ACTION_EXTENSION_DIM + 1, 1
        ),
        'dcpa': landmark_start + num_landmarks + BENCHMARK_COUNT_DIM + ACTION_EXTENSION_DIM + 4,
        'tcpa': landmark_start + num_landmarks + BENCHMARK_COUNT_DIM + ACTION_EXTENSION_DIM + 5,
        'info_schema_version': _field_slice(
            landmark_start + num_landmarks + BENCHMARK_COUNT_DIM + ACTION_EXTENSION_DIM + 2, 1
        ),
        'info_schema_magic': _field_slice(
            landmark_start + num_landmarks + BENCHMARK_COUNT_DIM + ACTION_EXTENSION_DIM + 3, 1
        ),
    }
    # Short aliases are kept for downstream tools that use the names from
    # the rule-buffer data contract.
    fields['schema_version'] = fields['info_schema_version']
    fields['magic_marker'] = fields['info_schema_magic']
    return fields[field]


def is_fixed_info_array(value, num_landmarks=None):
    """判断一个环境 info 是否为 ``(num_agents, info_width)`` 数组。"""
    if not isinstance(value, np.ndarray) or value.ndim != 2:
        return False
    if value.dtype.kind not in 'fiu':
        return False
    if num_landmarks is None:
        return value.shape[1] >= INFO_FIXED_PREFIX + BENCHMARK_COUNT_DIM
    expected_legacy = INFO_FIXED_PREFIX + int(num_landmarks) + BENCHMARK_COUNT_DIM
    expected_current = info_width(num_landmarks)
    return value.shape[1] in (expected_legacy, expected_current)


def _action2(value):
    result = np.zeros(ACTION_DIM_IN_INFO, dtype=np.float32)
    if value is None:
        return result
    data = np.asarray(value, dtype=np.float32).reshape(-1)
    result[:min(data.size, ACTION_DIM_IN_INFO)] = data[:ACTION_DIM_IN_INFO]
    return result


def _benchmark_values(benchmark_row, num_landmarks):
    errors = np.zeros(int(num_landmarks), dtype=np.float32)
    counts = np.zeros(BENCHMARK_COUNT_DIM, dtype=np.float32)
    if benchmark_row is None:
        return errors, counts

    # Compact callback format: [pre_error..., four counters].
    if isinstance(benchmark_row, np.ndarray):
        data = np.asarray(benchmark_row, dtype=np.float32).reshape(-1)
        if data.size >= int(num_landmarks) + BENCHMARK_COUNT_DIM:
            errors[:] = data[:int(num_landmarks)]
            counts[:] = data[int(num_landmarks):int(num_landmarks) + BENCHMARK_COUNT_DIM]
            return errors, counts

    # Legacy callback format:
    # (pre_error, landmarks_real_p, out_of_world, landmark_collision,
    #  agent_collision, obstacle_collision, ...)
    if isinstance(benchmark_row, (tuple, list)):
        if len(benchmark_row) > 0:
            raw_errors = np.asarray(benchmark_row[0], dtype=np.float32).reshape(-1)
            errors[:min(raw_errors.size, errors.size)] = raw_errors[:errors.size]
        if len(benchmark_row) >= 6:
            counts[:] = np.asarray(benchmark_row[2:6], dtype=np.float32)
    return errors, counts


def pack_agent_info(agent, benchmark_row=None, num_landmarks=1):
    """将一个 agent 的诊断状态打包为固定宽度 float32 数组。"""
    num_landmarks = int(num_landmarks)
    row = np.zeros(info_width(num_landmarks), dtype=np.float32)
    # ``None`` 表示字段尚未设置；全零是合法动作，不能再用
    # ``np.any(action)`` 判断字段是否存在，否则零动作会被错误回退。
    row[field_slice('raw_action', num_landmarks)] = _action2(
        getattr(agent, 'action_raw_u', None)
    )
    row[field_slice('constrained_action', num_landmarks)] = _action2(
        getattr(agent, 'action_constrained_u', None)
    )
    row[field_slice('applied_action', num_landmarks)] = _action2(
        getattr(agent, 'action_executed_u', None)
    )
    row[field_slice('filter_active', num_landmarks)] = float(
        bool(getattr(agent, 'action_filter_active', False))
    )
    # ``action_correction_norm`` 保留为旧字段别名，统一表示
    # constrained_action 到 rule_action 的规则修正幅度，不再混入平滑器差异。
    correction = getattr(agent, 'action_rule_correction_norm', None)
    if correction is None:
        correction = getattr(agent, 'action_correction_norm', None)
    if correction is None:
        constrained = row[field_slice('constrained_action', num_landmarks)]
        rule_value = getattr(agent, 'action_rule_u', None)
        rule_action = _action2(rule_value)
        if rule_value is None:
            # 兼容尚未设置新字段的旧实体；当前环境会始终显式设置 rule_action。
            rule_action = row[field_slice('applied_action', num_landmarks)]
        correction = np.linalg.norm(rule_action - constrained)
    row[field_slice('action_correction_norm', num_landmarks)] = float(correction)

    components = getattr(agent, 'reward_last_components', {}) or {}
    row[field_slice('reward_components', num_landmarks)] = np.asarray(
        [float(components.get(key, 0.0)) for key in REWARD_COMPONENT_KEYS],
        dtype=np.float32,
    )
    metrics = getattr(agent, 'reward_last_metrics', {}) or {}
    row[field_slice('reward_metrics', num_landmarks)] = np.asarray(
        [float(metrics.get(key, 0.0)) for key in REWARD_METRIC_KEYS],
        dtype=np.float32,
    )
    row[field_slice('done_reason', num_landmarks)] = float(
        DONE_REASON_CODES.get(getattr(agent, 'reward_done_reason', 'none'), 0)
    )
    errors, counts = _benchmark_values(benchmark_row, num_landmarks)
    row[field_slice('landmark_error', num_landmarks)] = errors
    row[field_slice('benchmark_counts', num_landmarks)] = counts

    # 动作字段扩展：明确区分规则投影和 EMA 平滑后的最终动作。
    rule_value = getattr(agent, 'action_rule_u', None)
    rule_action = _action2(rule_value)
    if rule_value is None:
        decision = getattr(agent, 'action_filter_decision', None)
        if isinstance(decision, dict):
            rule_value = decision.get('rule_action')
        elif decision is not None:
            rule_value = getattr(decision, 'rule_action', None)
        rule_action = _action2(rule_value)
    if rule_value is None:
        rule_action = row[field_slice('constrained_action', num_landmarks)]

    smoothed_value = getattr(agent, 'action_smoothed_u', None)
    smoothed_action = _action2(smoothed_value)
    if smoothed_value is None:
        smoothed_action = row[field_slice('applied_action', num_landmarks)]
    raw_action = row[field_slice('raw_action', num_landmarks)]
    constrained_action = row[field_slice('constrained_action', num_landmarks)]
    action_norms = {
        'raw_to_constrained_norm': getattr(agent, 'action_clip_correction_norm', None),
        'constrained_to_rule_norm': getattr(agent, 'action_rule_correction_norm', None),
        'rule_to_smoothed_norm': getattr(agent, 'action_smooth_correction_norm', None),
        'raw_to_smoothed_norm': getattr(agent, 'action_total_correction_norm', None),
    }
    fallback_norms = {
        'raw_to_constrained_norm': np.linalg.norm(constrained_action - raw_action),
        'constrained_to_rule_norm': np.linalg.norm(rule_action - constrained_action),
        'rule_to_smoothed_norm': np.linalg.norm(smoothed_action - rule_action),
        'raw_to_smoothed_norm': np.linalg.norm(smoothed_action - raw_action),
    }
    row[field_slice('rule_action', num_landmarks)] = rule_action
    row[field_slice('smoothed_action', num_landmarks)] = smoothed_action
    for key, fallback in fallback_norms.items():
        value = action_norms[key]
        row[field_slice(key, num_landmarks)] = float(
            fallback if value is None else value
        )

    decision = getattr(agent, 'action_filter_decision', None)
    if isinstance(decision, dict):
        encounter_type = decision.get('encounter_type')
    else:
        encounter_type = getattr(decision, 'encounter_type', None)
    filter_mode = getattr(agent, 'action_filter_mode', None)
    row[field_slice('encounter_type_code', num_landmarks)] = float(
        encode_encounter_type(encounter_type)
    )
    row[field_slice('filter_mode_code', num_landmarks)] = float(
        encode_filter_mode(filter_mode)
    )
    # DCPA/TCPA 仅作为规则审计和教师样本元数据，不会拼接到观测输入。
    if isinstance(decision, dict):
        dcpa, tcpa = decision.get('dcpa'), decision.get('tcpa')
    else:
        dcpa = getattr(decision, 'dcpa', None)
        tcpa = getattr(decision, 'tcpa', None)
    row[field_slice('dcpa', num_landmarks)] = float(dcpa) if dcpa is not None and np.isfinite(dcpa) else 1.0e9
    row[field_slice('tcpa', num_landmarks)] = float(tcpa) if tcpa is not None and np.isfinite(tcpa) else 1.0e9
    row[field_slice('info_schema_version', num_landmarks)] = float(
        RULE_INFO_SCHEMA_VERSION
    )
    row[field_slice('info_schema_magic', num_landmarks)] = float(
        RULE_INFO_SCHEMA_MAGIC
    )
    return row


def get_agent_field(env_info, agent_i, field, num_landmarks=1):
    """读取固定数组字段；旧格式由调用方自行走兼容路径。"""
    if not is_fixed_info_array(env_info):
        return None
    if agent_i < 0 or agent_i >= env_info.shape[0]:
        return None
    value = env_info[agent_i, field_slice(field, num_landmarks)]
    return np.asarray(value, dtype=np.float32)


def decode_done_reason(code):
    """将固定数组中的 done reason 编码还原为旧字符串。"""
    try:
        return DONE_REASON_NAMES.get(int(round(float(code))), 'none')
    except (TypeError, ValueError):
        return 'none'
