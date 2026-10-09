import unittest

import numpy as np

from algorithms.sac.masac import MASAC
from utilities.observation_schema import observation_dim


class NoOpLogger:
    def add_scalars(self, *args, **kwargs):
        return None


def make_main_samples(
    batch_size=4,
    history_length=3,
    obs_dim=observation_dim(num_dynamic_slots=2, num_static_slots=1),
    action_dim=2,
):
    history = [[np.zeros((history_length, obs_dim), dtype=np.float32)] for _ in range(batch_size)]
    history_action = [
        np.zeros((1, history_length, action_dim), dtype=np.float32)
        for _ in range(batch_size)
    ]
    observation = [[np.zeros(obs_dim, dtype=np.float32)] for _ in range(batch_size)]
    raw_action = [
        np.asarray([[0.2, -0.3]], dtype=np.float32) for _ in range(batch_size)
    ]
    reward = [np.asarray([0.1], dtype=np.float32) for _ in range(batch_size)]
    next_observation = [[np.zeros(obs_dim, dtype=np.float32)] for _ in range(batch_size)]
    done = [np.asarray([0.0], dtype=np.float32) for _ in range(batch_size)]
    constrained_action = [
        np.asarray([[0.2, 0.25]], dtype=np.float32) for _ in range(batch_size)
    ]
    rule_action = [
        np.asarray([[-0.4, -0.6]], dtype=np.float32) for _ in range(batch_size)
    ]
    smoothed_action = [
        np.asarray([[-0.1, -0.2]], dtype=np.float32) for _ in range(batch_size)
    ]
    filter_active = [np.asarray([[1.0]], dtype=np.float32) for _ in range(batch_size)]
    return [
        history,
        history_action,
        observation,
        raw_action,
        reward,
        next_observation,
        done,
        raw_action,
        constrained_action,
        rule_action,
        smoothed_action,
        filter_active,
    ]


class MASACRuleBatchTests(unittest.TestCase):
    def make_agent(self, enabled):
        return MASAC(
            num_agents=1,
            num_ob=2,
            num_static_ob_slots=1,
            action_dim=2,
            rnn=False,
            dim_1=16,
            dim_2=16,
            device="cpu",
            rule_buffer_enabled=enabled,
            rule_imitation_weight=0.15,
        )

    def rule_batch(
        self,
        batch_size=3,
        history_length=3,
        obs_dim=observation_dim(num_dynamic_slots=2, num_static_slots=1),
        action_dim=2,
    ):
        return {
            "history_obs": np.zeros((batch_size, history_length, obs_dim), dtype=np.float32),
            "history_action": np.zeros((batch_size, history_length, action_dim), dtype=np.float32),
            "observation": np.zeros((batch_size, obs_dim), dtype=np.float32),
            "raw_action": np.full((batch_size, action_dim), 0.4, dtype=np.float32),
            "constrained_action": np.full((batch_size, action_dim), 0.2, dtype=np.float32),
            "rule_action": np.full((batch_size, action_dim), -0.2, dtype=np.float32),
            # applied_action is intentionally different: it must not be used
            # as the rule teacher after the action-semantics fix.
            "applied_action": np.full((batch_size, action_dim), -0.05, dtype=np.float32),
            "correction_norm": np.full((batch_size,), 0.5, dtype=np.float32),
            "_sample_stats": {
                "unique_sample_ratio": 0.66,
                "reuse_ratio": 0.33,
                "sample_time_ms": 0.1,
            },
        }

    def test_enabled_rule_batch_updates_without_touching_priority_shape(self):
        masac = self.make_agent(enabled=True)
        priorities = masac.update(
            make_main_samples(),
            0,
            NoOpLogger(),
            rule_samples=self.rule_batch(),
        )
        self.assertEqual(np.asarray(priorities).shape[0], 4)

    def test_disabled_path_accepts_legacy_masked_rule_samples(self):
        masac = self.make_agent(enabled=False)
        priorities = masac.update(make_main_samples(), 0, NoOpLogger())
        self.assertEqual(np.asarray(priorities).shape[0], 4)


if __name__ == "__main__":
    unittest.main()
