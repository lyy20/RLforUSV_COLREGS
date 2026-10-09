import unittest

import numpy as np
import torch

from algorithms.sac.masac import MASAC


class MASACFiniteGuardTests(unittest.TestCase):
    def make_model(self):
        return MASAC(
            num_agents=1,
            num_ob=2,
            num_static_ob_slots=1,
            action_dim=2,
            rnn=False,
            dim_1=8,
            dim_2=8,
            device="cpu",
            automatic_entropy_tuning=False,
        )

    def test_nonfinite_tensor_is_rejected_with_context(self):
        model = self.make_model()
        # 非有限值必须暴露为明确错误，不能被 nan_to_num 静默改写。
        with self.assertRaisesRegex(FloatingPointError, r"agent=0.*field=unit_test_tensor"):
            model._assert_finite(
                torch.tensor([0.0, float("nan")]),
                "unit_test_tensor",
                0,
            )

    def test_nonfinite_numpy_array_is_rejected(self):
        model = self.make_model()
        # PER 优先级等 NumPy 数据同样需要经过严格检查。
        with self.assertRaisesRegex(FloatingPointError, r"field=unit_test_priority"):
            model._assert_finite(
                np.asarray([1.0, np.inf], dtype=np.float32),
                "unit_test_priority",
                0,
            )

    def test_finite_mean_is_preserved(self):
        model = self.make_model()
        value = model._finite_mean(torch.tensor([1.0, 3.0]), "unit_test_mean", 0)
        self.assertEqual(value, 2.0)


if __name__ == "__main__":
    unittest.main()
