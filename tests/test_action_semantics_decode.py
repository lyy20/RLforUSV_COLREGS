"""动作语义固定数组/字典解码的回归测试。

训练、普通测试、自定义测试和策略独立测试都必须坚持同一条动作链：
``raw -> constrained -> rule -> smoothed/applied``。本文件专门防止测试
工具在后续维护时把最终执行动作重新误解为纯规则动作。
"""

import unittest

import numpy as np

from see_trained_custom import _decode_info as decode_custom_info
from see_trained_once import decode_step_info as decode_once_info
from see_trained_policy_only import decode_step_info as decode_policy_only_info
from utilities.info_schema import (
    BENCHMARK_COUNT_DIM,
    INFO_FIXED_PREFIX,
    field_slice,
    pack_agent_info,
)


ACTION_CHAIN_KEYS = (
    "raw_to_constrained_norm",
    "constrained_to_rule_norm",
    "rule_to_smoothed_norm",
    "raw_to_smoothed_norm",
)


class _ActionAgent:
    """构造一个带有完整动作链字段的最小环境实体。"""

    def __init__(self, raw, constrained, rule, smoothed, filter_active=True):
        self.action_raw_u = np.asarray(raw, dtype=np.float32)
        self.action_constrained_u = np.asarray(constrained, dtype=np.float32)
        self.action_rule_u = np.asarray(rule, dtype=np.float32)
        self.action_smoothed_u = np.asarray(smoothed, dtype=np.float32)
        # 当前执行契约中 applied 是 smoothed 的兼容别名，二者必须相同。
        self.action_executed_u = np.asarray(smoothed, dtype=np.float32)
        self.action_clip_correction_norm = float(
            np.linalg.norm(self.action_constrained_u - self.action_raw_u)
        )
        self.action_rule_correction_norm = float(
            np.linalg.norm(self.action_rule_u - self.action_constrained_u)
        )
        self.action_smooth_correction_norm = float(
            np.linalg.norm(self.action_smoothed_u - self.action_rule_u)
        )
        self.action_total_correction_norm = float(
            np.linalg.norm(self.action_smoothed_u - self.action_raw_u)
        )
        self.action_filter_active = bool(filter_active)
        self.action_filter_mode = "colregs_head_on" if filter_active else "none"
        self.action_filter_decision = type(
            "Decision", (), {"encounter_type": "head_on"}
        )()
        self.reward_last_components = {}
        self.reward_last_metrics = {}
        self.reward_done_reason = "none"


class ActionSemanticsDecodeTests(unittest.TestCase):
    """验证三套测试工具对四层动作的解码结果完全一致。"""

    def setUp(self):
        self.num_agents = 1
        self.num_landmarks = 1
        self.raw = np.asarray([[0.25, -0.50]], dtype=np.float32)
        self.constrained = np.asarray([[0.10, -0.25]], dtype=np.float32)
        self.rule = np.asarray([[-0.45, -0.10]], dtype=np.float32)
        self.smoothed = np.asarray([[-0.20, -0.05]], dtype=np.float32)
        self.fallback = np.asarray([[0.88, -0.77]], dtype=np.float32)

    @property
    def decoders(self):
        return (
            ("once", decode_once_info),
            ("custom", decode_custom_info),
            ("policy_only", decode_policy_only_info),
        )

    @staticmethod
    def _norms(raw, constrained, rule, smoothed):
        return {
            "raw_to_constrained_norm": np.linalg.norm(constrained - raw, axis=1),
            "constrained_to_rule_norm": np.linalg.norm(rule - constrained, axis=1),
            "rule_to_smoothed_norm": np.linalg.norm(smoothed - rule, axis=1),
            "raw_to_smoothed_norm": np.linalg.norm(smoothed - raw, axis=1),
        }

    def _assert_decoded_chain(
        self,
        decoded,
        raw,
        constrained,
        rule,
        smoothed,
    ):
        """断言解码出的动作层与由动作本身推导的范数均正确。"""
        np.testing.assert_allclose(decoded["raw_actions"], raw)
        np.testing.assert_allclose(decoded["constrained_actions"], constrained)
        np.testing.assert_allclose(decoded["rule_actions"], rule)
        np.testing.assert_allclose(decoded["smoothed_actions"], smoothed)
        np.testing.assert_allclose(decoded["applied_actions"], smoothed)
        expected_norms = self._norms(raw, constrained, rule, smoothed)
        for key, expected in expected_norms.items():
            np.testing.assert_allclose(decoded["action_chain_norms"][key], expected)

    def test_current_fixed_schema_decodes_all_action_layers(self):
        """当前 schema v2 必须保留四层动作与四段差异范数。"""
        row = pack_agent_info(
            _ActionAgent(
                self.raw[0],
                self.constrained[0],
                self.rule[0],
                self.smoothed[0],
            ),
            num_landmarks=self.num_landmarks,
        ).reshape(self.num_agents, -1)

        for name, decoder in self.decoders:
            with self.subTest(decoder=name):
                decoded = decoder(
                    row,
                    self.num_agents,
                    self.num_landmarks,
                    self.fallback,
                )
                self._assert_decoded_chain(
                    decoded,
                    self.raw,
                    self.constrained,
                    self.rule,
                    self.smoothed,
                )

    def test_zero_raw_action_is_not_treated_as_a_missing_field(self):
        """零动作是合法 SAC 输出，不能被非零 fallback 覆盖。"""
        zero_raw = np.zeros_like(self.raw)
        row = pack_agent_info(
            _ActionAgent(
                zero_raw[0],
                self.constrained[0],
                self.rule[0],
                self.smoothed[0],
            ),
            num_landmarks=self.num_landmarks,
        ).reshape(self.num_agents, -1)

        for name, decoder in self.decoders:
            with self.subTest(decoder=name):
                decoded = decoder(
                    row,
                    self.num_agents,
                    self.num_landmarks,
                    self.fallback,
                )
                np.testing.assert_array_equal(decoded["raw_actions"], zero_raw)
                self._assert_decoded_chain(
                    decoded,
                    zero_raw,
                    self.constrained,
                    self.rule,
                    self.smoothed,
                )

    def test_legacy_fixed_schema_uses_conservative_rule_fallback(self):
        """旧固定数组没有纯规则字段时，rule 只能退化为 constrained。"""
        legacy_width = INFO_FIXED_PREFIX + self.num_landmarks + BENCHMARK_COUNT_DIM
        legacy = np.zeros((self.num_agents, legacy_width), dtype=np.float32)
        legacy[:, field_slice("raw_action", self.num_landmarks)] = self.raw
        legacy[:, field_slice("constrained_action", self.num_landmarks)] = self.constrained
        legacy[:, field_slice("applied_action", self.num_landmarks)] = self.smoothed

        for name, decoder in self.decoders:
            with self.subTest(decoder=name):
                decoded = decoder(
                    legacy,
                    self.num_agents,
                    self.num_landmarks,
                    self.fallback,
                )
                self._assert_decoded_chain(
                    decoded,
                    self.raw,
                    self.constrained,
                    self.constrained,
                    self.smoothed,
                )

    def test_dict_schema_decodes_all_action_layers(self):
        """旧 dict 传输也应完整保留新动作语义字段。"""
        info = {
            "raw_actions": self.raw.copy(),
            "constrained_actions": self.constrained.copy(),
            "rule_actions": self.rule.copy(),
            "smoothed_actions": self.smoothed.copy(),
            "applied_actions": self.smoothed.copy(),
            "filter_active": [True],
            "action_correction_norm": [
                float(np.linalg.norm(self.rule[0] - self.constrained[0]))
            ],
            "done_reason": ["none"],
        }

        for name, decoder in self.decoders:
            with self.subTest(decoder=name):
                decoded = decoder(
                    info,
                    self.num_agents,
                    self.num_landmarks,
                    self.fallback,
                )
                self._assert_decoded_chain(
                    decoded,
                    self.raw,
                    self.constrained,
                    self.rule,
                    self.smoothed,
                )


if __name__ == "__main__":
    unittest.main()
