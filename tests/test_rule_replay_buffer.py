import os
import tempfile
import unittest

import numpy as np

from utilities.info_schema import (
    BENCHMARK_COUNT_DIM,
    INFO_FIXED_PREFIX,
    field_slice,
    has_rule_metadata,
    infer_num_landmarks,
    info_width,
    is_fixed_info_array,
    pack_agent_info,
)
from utilities.rule_metadata import (
    ENCOUNTER_TYPE_CODES,
    decode_encounter_type,
    encode_encounter_type,
    encode_filter_mode,
)
from utilities.rule_replay_buffer import RuleReplayBuffer


class RuleReplayBufferTests(unittest.TestCase):
    def setUp(self):
        self.history_length = 3
        self.obs_dim = 7
        self.action_dim = 2
        self.buffer = RuleReplayBuffer(
            capacity=4,
            history_length=self.history_length,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            num_agents=2,
            seed=11,
        )

    def record(self, value, agent=0, encounter=2):
        return {
            "history_obs": np.full(
                (self.history_length, self.obs_dim), value, dtype=np.float64
            ),
            "history_action": np.full(
                (self.history_length, self.action_dim), value, dtype=np.float64
            ),
            "observation": np.full(self.obs_dim, value, dtype=np.float64),
            "raw_action": np.asarray([value, -value], dtype=np.float64),
            "constrained_action": np.asarray([value * 0.75, -value * 0.75], dtype=np.float64),
            "rule_action": np.asarray([value / 2.0, -value / 2.0], dtype=np.float64),
            "applied_action": np.asarray([value / 3.0, -value / 3.0], dtype=np.float64),
            "correction_norm": float(np.linalg.norm(np.asarray([value / 2.0, -value / 2.0]) - np.asarray([value * 0.75, -value * 0.75]))),
            "encounter_type_code": encounter,
            "filter_mode_code": encode_filter_mode("colregs_head_on"),
            "agent_index": agent,
            "filter_active": True,
        }

    def test_owned_float32_and_ring_overwrite(self):
        records = [self.record(float(i), agent=i % 2, encounter=2 + (i % 2)) for i in range(5)]
        self.buffer.push_batch(records)
        self.assertEqual(len(self.buffer), 4)
        self.assertEqual(self.buffer.history_obs.dtype, np.float32)
        records[4]["observation"][0] = 999.0
        self.assertFalse(np.any(self.buffer.observation == 999.0))
        self.assertEqual(self.buffer.stats()["inserted_total"], 5)
        self.assertEqual(self.buffer.stats()["format_version"], 2)
        np.testing.assert_allclose(
            self.buffer.correction_norm[3],
            np.linalg.norm(self.buffer.rule_action[3] - self.buffer.constrained_action[3]),
        )

    def test_agent_filter_and_stratified_sampling(self):
        self.buffer.push_batch([
            self.record(1.0, agent=0, encounter=2),
            self.record(2.0, agent=0, encounter=3),
            self.record(3.0, agent=1, encounter=5),
        ])
        self.assertEqual(self.buffer.count(agent_index=0), 2)
        sample = self.buffer.sample(8, agent_index=0, mode="stratified")
        self.assertEqual(sample["observation"].shape, (8, self.obs_dim))
        self.assertTrue(np.all(sample["agent_index"] == 0))
        self.assertEqual(sample["observation"].dtype, np.float32)
        self.assertEqual(self.buffer.stats()["sampled_total"], 8)
        self.assertGreaterEqual(sample["_sample_stats"]["unique_sample_ratio"], 0.0)

    def test_dagger_visited_state_can_be_stored_without_teacher_weight(self):
        record = self.record(6.0)
        record["filter_active"] = False
        record["teacher_valid"] = False
        record["teacher_confidence"] = 0.0
        record["dagger_iteration"] = 2
        self.assertEqual(self.buffer.push_batch([record]), 1)
        self.assertEqual(self.buffer.count(agent_index=0), 1)
        self.assertEqual(self.buffer.count(agent_index=0, teacher_valid_only=True), 0)

    def test_save_reload(self):
        self.buffer.push_batch([self.record(4.0), self.record(5.0, encounter=3)])
        self.buffer.sample(3, mode="uniform")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "rule_buffer.file")
            self.buffer.save(path)
            restored = RuleReplayBuffer(
                capacity=4,
                history_length=self.history_length,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                num_agents=2,
                seed=99,
            )
            restored.reload(path)
            self.assertEqual(restored.stats(), self.buffer.stats())
            np.testing.assert_array_equal(restored.observation, self.buffer.observation)
            np.testing.assert_array_equal(restored.constrained_action, self.buffer.constrained_action)
            np.testing.assert_array_equal(restored.rule_action, self.buffer.rule_action)

    def test_v1_state_is_rejected(self):
        state = self.buffer.state_dict()
        state["format_version"] = 1
        state.pop("constrained_action")
        state.pop("rule_action")
        with self.assertRaisesRegex(ValueError, "v1 is not safe"):
            self.buffer.load_state_dict(state)


class RuleMetadataTests(unittest.TestCase):
    def test_stable_codes(self):
        self.assertEqual(encode_encounter_type("head_on"), ENCOUNTER_TYPE_CODES["head_on"])
        self.assertEqual(decode_encounter_type(ENCOUNTER_TYPE_CODES["head_on"]), "head_on")
        self.assertEqual(encode_encounter_type("not_a_rule"), 0)

    def test_info_layout_is_backward_compatible(self):
        class Agent:
            action_raw_u = np.zeros(2)
            action_constrained_u = np.zeros(2)
            action_rule_u = np.zeros(2)
            action_smoothed_u = np.zeros(2)
            action_executed_u = np.zeros(2)
            action_clip_correction_norm = 0.0
            action_rule_correction_norm = 0.0
            action_smooth_correction_norm = 0.0
            action_total_correction_norm = 0.0
            action_filter_active = True
            action_filter_mode = "colregs_head_on"
            action_correction_norm = 0.2
            action_filter_decision = type("Decision", (), {
                "encounter_type": "head_on",
            })()
            reward_last_components = {}
            reward_last_metrics = {}
            reward_done_reason = "none"

        current = pack_agent_info(Agent(), num_landmarks=1)
        self.assertEqual(current.shape[0], info_width(1))
        current_2d = current.reshape(1, -1)
        self.assertTrue(is_fixed_info_array(current_2d))
        self.assertTrue(has_rule_metadata(current_2d))
        self.assertEqual(infer_num_landmarks(current_2d.shape[1], True), 1)
        self.assertEqual(
            int(current[field_slice("encounter_type_code", 1)][0]),
            ENCOUNTER_TYPE_CODES["head_on"],
        )
        np.testing.assert_array_equal(current[field_slice("rule_action", 1)], [0.0, 0.0])
        np.testing.assert_array_equal(current[field_slice("smoothed_action", 1)], [0.0, 0.0])

        legacy_width = INFO_FIXED_PREFIX + 1 + BENCHMARK_COUNT_DIM
        legacy = np.zeros((1, legacy_width), dtype=np.float32)
        self.assertTrue(is_fixed_info_array(legacy, num_landmarks=1))
        self.assertFalse(has_rule_metadata(legacy))
        self.assertEqual(infer_num_landmarks(legacy_width), 1)


if __name__ == "__main__":
    unittest.main()
