"""A compact, independent replay buffer for COLREGs teacher transitions.

The main replay buffer remains responsible for all SAC transitions.  This
buffer stores only active filter decisions and is sampled exclusively by the
actor imitation term.  Records are stored per agent so a sampled rule batch
can be passed directly to that agent's actor without changing the critic
batch or PER priorities.
"""

from __future__ import annotations

import pickle
import time

import numpy as np

from .rule_metadata import (
    ENCOUNTER_TYPE_CODES,
    rule_type_mapping,
)


# v2 明确保存 raw/constrained/rule/smoothed 四个动作层级。
# v1 只有 raw/applied，无法判断 applied 中混入了多少平滑器行为，不能安全
# 地作为规则教师池继续恢复。
RULE_BUFFER_FORMAT_VERSION = 2


class RuleReplayBuffer:
    """Fixed-capacity NumPy ring buffer for rule-filtered transitions."""

    def __init__(
        self,
        capacity,
        history_length,
        obs_dim,
        action_dim,
        num_agents=1,
        seed=None,
    ):
        self.capacity = int(capacity)
        self.history_length = int(history_length)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.num_agents = int(num_agents)
        if self.capacity <= 0:
            raise ValueError("RuleReplayBuffer capacity must be positive.")
        if self.history_length <= 0 or self.obs_dim <= 0 or self.action_dim <= 0:
            raise ValueError("RuleReplayBuffer dimensions must be positive.")
        if self.num_agents <= 0:
            raise ValueError("RuleReplayBuffer num_agents must be positive.")

        self.rng = np.random.default_rng(seed)
        self.position = 0
        self.size = 0
        self.inserted_total = 0
        self.sampled_total = 0
        self._last_sample_indices = np.empty((0,), dtype=np.int64)
        self._last_sample_stats = {
            "actual_batch_size": 0,
            "unique_sample_ratio": 0.0,
            "reuse_ratio": 0.0,
            "sample_time_ms": 0.0,
            "encounter_type_counts": {},
        }
        self._allocate()

    def _allocate(self):
        shape_his_obs = (self.capacity, self.history_length, self.obs_dim)
        shape_his_action = (self.capacity, self.history_length, self.action_dim)
        shape_action = (self.capacity, self.action_dim)
        self.history_obs = np.zeros(shape_his_obs, dtype=np.float32)
        self.history_action = np.zeros(shape_his_action, dtype=np.float32)
        self.observation = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.raw_action = np.zeros(shape_action, dtype=np.float32)
        self.constrained_action = np.zeros(shape_action, dtype=np.float32)
        self.rule_action = np.zeros(shape_action, dtype=np.float32)
        # applied_action 保留为最终平滑执行动作的兼容字段。
        self.applied_action = np.zeros(shape_action, dtype=np.float32)
        self.correction_norm = np.zeros((self.capacity,), dtype=np.float32)
        self.encounter_type_code = np.zeros((self.capacity,), dtype=np.int16)
        self.filter_mode_code = np.zeros((self.capacity,), dtype=np.int16)
        self.agent_index = np.zeros((self.capacity,), dtype=np.int16)
        self.episode_id = np.zeros((self.capacity,), dtype=np.int64)
        self.environment_step = np.zeros((self.capacity,), dtype=np.int32)
        self.insertion_id = np.zeros((self.capacity,), dtype=np.int64)
        self.sample_count = np.zeros((self.capacity,), dtype=np.int64)
        # 可选 DAgger 教师元数据；不作为 SAC 网络输入。
        self.teacher_valid = np.zeros((self.capacity,), dtype=np.uint8)
        self.teacher_confidence = np.ones((self.capacity,), dtype=np.float32)
        self.dcpa = np.full((self.capacity,), np.nan, dtype=np.float32)
        self.tcpa = np.full((self.capacity,), np.nan, dtype=np.float32)
        self.dagger_iteration = np.zeros((self.capacity,), dtype=np.int32)

    def __len__(self):
        return int(self.size)

    @property
    def current_size(self):
        return int(self.size)

    def _coerce(self, value, shape, name):
        array = np.asarray(value, dtype=np.float32)
        if array.size != int(np.prod(shape)):
            raise ValueError(
                "%s has %d values; expected %d."
                % (name, array.size, int(np.prod(shape)))
            )
        return np.ascontiguousarray(array.reshape(shape), dtype=np.float32)

    def push(
        self,
        history_obs,
        history_action,
        observation,
        raw_action,
        applied_action,
        correction_norm=None,
        encounter_type_code=0,
        filter_mode_code=0,
        agent_index=0,
        episode_id=0,
        environment_step=0,
        constrained_action=None,
        rule_action=None,
        teacher_valid=True,
        teacher_confidence=1.0,
        dcpa=np.nan,
        tcpa=np.nan,
        dagger_iteration=0,
    ):
        """Insert one active rule transition, taking owned copies."""
        agent_index = int(agent_index)
        if not 0 <= agent_index < self.num_agents:
            raise ValueError("agent_index is outside the configured agent range.")
        index = int(self.position)
        self.history_obs[index] = self._coerce(
            history_obs, (self.history_length, self.obs_dim), "history_obs"
        )
        self.history_action[index] = self._coerce(
            history_action, (self.history_length, self.action_dim), "history_action"
        )
        self.observation[index] = self._coerce(
            observation, (self.obs_dim,), "observation"
        )
        self.raw_action[index] = self._coerce(
            raw_action, (self.action_dim,), "raw_action"
        )
        if constrained_action is None:
            # 仅用于旧调用方的内存兼容；新训练路径始终显式传入该字段。
            constrained_action = raw_action
        if rule_action is None:
            # 旧记录没有纯规则动作，只能把 applied 当作临时回退；旧文件
            # 恢复会被 load_state_dict 明确拒绝，不会把它伪装成 v2 数据。
            rule_action = applied_action
        self.constrained_action[index] = self._coerce(
            constrained_action, (self.action_dim,), "constrained_action"
        )
        self.rule_action[index] = self._coerce(
            rule_action, (self.action_dim,), "rule_action"
        )
        self.applied_action[index] = self._coerce(
            applied_action, (self.action_dim,), "applied_action"
        )
        if correction_norm is None:
            correction_norm = np.linalg.norm(
                self.rule_action[index] - self.constrained_action[index]
            )
        self.correction_norm[index] = float(correction_norm)
        self.encounter_type_code[index] = int(encounter_type_code)
        self.filter_mode_code[index] = int(filter_mode_code)
        self.agent_index[index] = agent_index
        self.episode_id[index] = int(episode_id)
        self.environment_step[index] = int(environment_step)
        self.insertion_id[index] = int(self.inserted_total)
        self.sample_count[index] = 0
        self.teacher_valid[index] = int(bool(teacher_valid))
        self.teacher_confidence[index] = float(np.clip(teacher_confidence, 0.0, 1.0))
        self.dcpa[index] = float(dcpa) if np.isfinite(dcpa) else np.nan
        self.tcpa[index] = float(tcpa) if np.isfinite(tcpa) else np.nan
        self.dagger_iteration[index] = int(dagger_iteration)

        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.inserted_total += 1

    def push_batch(self, records):
        """Insert an iterable of dictionaries; return the number accepted."""
        inserted = 0
        if records is None:
            return inserted
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("Rule replay records must be dictionaries.")
            # 普通规则池要求记录明确标记为过滤器介入或 DAgger 访问样本。
            # DAgger 的无风险状态允许 teacher_valid=False，以保留策略访问
            # 分布；其模仿损失由 MASAC 中的有效性掩码归零。
            is_dagger_record = "dagger_iteration" in record
            if (
                not bool(record.get("filter_active", False))
                and not bool(record.get("teacher_valid", False))
                and not is_dagger_record
            ):
                continue
            self.push(
                history_obs=record["history_obs"],
                history_action=record["history_action"],
                observation=record["observation"],
                raw_action=record["raw_action"],
                applied_action=record["applied_action"],
                constrained_action=record.get("constrained_action"),
                rule_action=record.get("rule_action"),
                correction_norm=record.get("correction_norm"),
                encounter_type_code=record.get("encounter_type_code", 0),
                filter_mode_code=record.get("filter_mode_code", 0),
                agent_index=record.get("agent_index", 0),
                episode_id=record.get("episode_id", 0),
                environment_step=record.get("environment_step", 0),
                teacher_valid=record.get("teacher_valid", True),
                teacher_confidence=record.get("teacher_confidence", 1.0),
                dcpa=record.get("dcpa", np.nan),
                tcpa=record.get("tcpa", np.nan),
                dagger_iteration=record.get("dagger_iteration", 0),
            )
            inserted += 1
        return inserted

    def _valid_indices(self, agent_index=None, teacher_valid_only=False):
        if self.size <= 0:
            return np.empty((0,), dtype=np.int64)
        # Ring slots are valid even after wrapping; only the newest ``size``
        # slots contain owned transitions.
        indices = np.arange(self.size, dtype=np.int64)
        if self.size == self.capacity:
            indices = np.arange(self.capacity, dtype=np.int64)
        if agent_index is not None:
            indices = indices[self.agent_index[indices] == int(agent_index)]
        if teacher_valid_only:
            indices = indices[self.teacher_valid[indices] > 0]
        return indices

    def can_sample(self, batch_size, agent_index=None, teacher_valid_only=False):
        return (
            self._valid_indices(agent_index, teacher_valid_only).size > 0
            and int(batch_size) > 0
        )

    def count(self, agent_index=None, teacher_valid_only=False):
        return int(self._valid_indices(agent_index, teacher_valid_only).size)

    def _stratified_indices(self, candidates, batch_size):
        if candidates.size == 0:
            return np.empty((0,), dtype=np.int64)
        batch_size = int(batch_size)
        # Stratification is by encounter class.  Empty classes are skipped;
        # replacement is allowed because active rule events can be sparse.
        groups = []
        for code in sorted(np.unique(self.encounter_type_code[candidates]).tolist()):
            group = candidates[self.encounter_type_code[candidates] == code]
            if group.size:
                groups.append(group)
        if not groups:
            return self.rng.choice(candidates, size=batch_size, replace=True)
        selected = []
        for i in range(batch_size):
            group = groups[i % len(groups)]
            selected.append(int(self.rng.choice(group)))
        self.rng.shuffle(selected)
        return np.asarray(selected, dtype=np.int64)

    def sample(
        self,
        batch_size,
        agent_index=None,
        mode="stratified",
        teacher_valid_only=False,
    ):
        """Sample a rule batch and update reuse statistics.

        The returned dictionary contains NumPy arrays with a stable schema;
        it is intentionally separate from the seven/eight-field SAC batch.
        """
        start = time.perf_counter()
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("Rule sample batch_size must be positive.")
        candidates = self._valid_indices(agent_index, teacher_valid_only)
        if candidates.size == 0:
            raise ValueError("Cannot sample from an empty rule replay buffer.")
        mode = str(mode or "uniform").strip().lower()
        if mode == "stratified":
            indices = self._stratified_indices(candidates, batch_size)
        elif mode == "uniform":
            indices = self.rng.choice(candidates, size=batch_size, replace=True)
        else:
            raise ValueError("Rule sample mode must be 'uniform' or 'stratified'.")

        unique_count = int(np.unique(indices).size)
        old_counts = self.sample_count[indices].copy()
        self.sample_count[indices] += 1
        self.sampled_total += int(indices.size)
        encounter_codes, encounter_counts = np.unique(
            self.encounter_type_code[indices], return_counts=True
        )
        self._last_sample_indices = indices.copy()
        self._last_sample_stats = {
            "actual_batch_size": int(indices.size),
            "unique_sample_ratio": float(unique_count / max(indices.size, 1)),
            "reuse_ratio": float(np.mean(old_counts > 0)) if indices.size else 0.0,
            "sample_time_ms": float((time.perf_counter() - start) * 1000.0),
            "encounter_type_counts": {
                int(code): int(count)
                for code, count in zip(encounter_codes, encounter_counts)
            },
        }
        return {
            "history_obs": self.history_obs[indices].copy(),
            "history_action": self.history_action[indices].copy(),
            "observation": self.observation[indices].copy(),
            "raw_action": self.raw_action[indices].copy(),
            "constrained_action": self.constrained_action[indices].copy(),
            "rule_action": self.rule_action[indices].copy(),
            "applied_action": self.applied_action[indices].copy(),
            "correction_norm": self.correction_norm[indices].copy(),
            "encounter_type_code": self.encounter_type_code[indices].copy(),
            "filter_mode_code": self.filter_mode_code[indices].copy(),
            "agent_index": self.agent_index[indices].copy(),
            "episode_id": self.episode_id[indices].copy(),
            "environment_step": self.environment_step[indices].copy(),
            "sample_indices": indices.copy(),
            "teacher_valid": self.teacher_valid[indices].copy(),
            "teacher_confidence": self.teacher_confidence[indices].copy(),
            "dcpa": self.dcpa[indices].copy(),
            "tcpa": self.tcpa[indices].copy(),
            "dagger_iteration": self.dagger_iteration[indices].copy(),
            # Private transport metadata is consumed by MASAC for logging;
            # it is not a feature or a target used by the actor loss.
            "_sample_stats": self.last_sample_stats(),
        }

    def last_sample_stats(self):
        return dict(self._last_sample_stats)

    def stats(self):
        indices = self._valid_indices()
        if indices.size:
            unknown_rate = float(
                np.mean(self.encounter_type_code[indices] == ENCOUNTER_TYPE_CODES["unknown"])
            )
            teacher_valid_rate = float(np.mean(self.teacher_valid[indices] > 0))
            teacher_confidence_mean = float(np.mean(self.teacher_confidence[indices]))
            per_agent = {
                int(agent): int(np.sum(self.agent_index[indices] == agent))
                for agent in range(self.num_agents)
            }
        else:
            unknown_rate = 0.0
            teacher_valid_rate = 0.0
            teacher_confidence_mean = 0.0
            per_agent = {int(agent): 0 for agent in range(self.num_agents)}
        return {
            "size": int(self.size),
            "capacity": int(self.capacity),
            "fill_ratio": float(self.size / max(self.capacity, 1)),
            "inserted_total": int(self.inserted_total),
            "sampled_total": int(self.sampled_total),
            "unknown_type_rate": unknown_rate,
            "teacher_valid_rate": teacher_valid_rate,
            "teacher_confidence_mean": teacher_confidence_mean,
            "per_agent_size": per_agent,
            "format_version": RULE_BUFFER_FORMAT_VERSION,
        }

    def state_dict(self):
        return {
            "format_version": RULE_BUFFER_FORMAT_VERSION,
            "capacity": self.capacity,
            "history_length": self.history_length,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "num_agents": self.num_agents,
            "position": self.position,
            "size": self.size,
            "inserted_total": self.inserted_total,
            "sampled_total": self.sampled_total,
            "history_obs": self.history_obs,
            "history_action": self.history_action,
            "observation": self.observation,
            "raw_action": self.raw_action,
            "constrained_action": self.constrained_action,
            "rule_action": self.rule_action,
            "applied_action": self.applied_action,
            "correction_norm": self.correction_norm,
            "encounter_type_code": self.encounter_type_code,
            "filter_mode_code": self.filter_mode_code,
            "agent_index": self.agent_index,
            "episode_id": self.episode_id,
            "environment_step": self.environment_step,
            "insertion_id": self.insertion_id,
            "sample_count": self.sample_count,
            "teacher_valid": self.teacher_valid,
            "teacher_confidence": self.teacher_confidence,
            "dcpa": self.dcpa,
            "tcpa": self.tcpa,
            "dagger_iteration": self.dagger_iteration,
            "rng_state": self.rng.bit_generator.state,
            "type_mapping": rule_type_mapping(),
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            raise ValueError("Invalid rule replay state.")
        saved_version = int(state.get("format_version", 0))
        if saved_version != RULE_BUFFER_FORMAT_VERSION:
            if saved_version == 1:
                raise ValueError(
                    "Rule replay format v1 is not safe to resume: it lacks "
                    "constrained_action and rule_action. Start a new rule buffer."
                )
            raise ValueError(
                "Unsupported rule replay format version: %s" % saved_version
            )
        for key, expected in (
            ("capacity", self.capacity),
            ("history_length", self.history_length),
            ("obs_dim", self.obs_dim),
            ("action_dim", self.action_dim),
            ("num_agents", self.num_agents),
        ):
            if key in state and int(state[key]) != int(expected):
                raise ValueError(
                    "Rule replay %s mismatch: saved=%s current=%s"
                    % (key, state[key], expected)
                )
        for key in (
            "history_obs",
            "history_action",
            "observation",
            "raw_action",
            "constrained_action",
            "rule_action",
            "applied_action",
            "correction_norm",
            "encounter_type_code",
            "filter_mode_code",
            "agent_index",
            "episode_id",
            "environment_step",
            "insertion_id",
            "sample_count",
        ):
            if key not in state:
                raise ValueError("Rule replay state is missing '%s'." % key)
            value = np.asarray(state[key])
            target = getattr(self, key)
            if value.shape != target.shape:
                raise ValueError(
                    "Rule replay field %s shape mismatch: saved=%s current=%s"
                    % (key, value.shape, target.shape)
                )
            target[...] = value.astype(target.dtype, copy=False)
        self.position = int(state.get("position", 0)) % self.capacity
        self.size = min(max(int(state.get("size", 0)), 0), self.capacity)
        self.inserted_total = max(int(state.get("inserted_total", self.size)), 0)
        self.sampled_total = max(int(state.get("sampled_total", 0)), 0)
        # 旧 v2 文件没有这些字段，使用兼容默认值继续运行。
        for key, default in (("teacher_valid", 1), ("teacher_confidence", 1.0),
                             ("dcpa", np.nan), ("tcpa", np.nan), ("dagger_iteration", 0)):
            target = getattr(self, key)
            value = state.get(key)
            if value is None:
                target.fill(default)
            else:
                value = np.asarray(value)
                if value.shape != target.shape:
                    raise ValueError("Rule replay field %s shape mismatch." % key)
                target[...] = value.astype(target.dtype, copy=False)
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state

    def save(self, file):
        with open(file, "wb") as handle:
            pickle.dump(self.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def reload(self, file):
        with open(file, "rb") as handle:
            state = pickle.load(handle)
        self.load_state_dict(state)
