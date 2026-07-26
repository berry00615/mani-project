import unittest

import numpy as np
import torch

from ppo import ActorCritic, RolloutBuffer


class PPOCoreTests(unittest.TestCase):
    def test_masked_log_prob_ignores_overridden_dimension(self):
        policy = ActorCritic(3, 2, [8], [8], -np.ones(2), np.ones(2))
        obs = torch.zeros(1, 3)
        actions_a = torch.tensor([[0.1, -0.7]])
        actions_b = torch.tensor([[0.1, 0.9]])
        mask = torch.tensor([[1.0, 0.0]])
        logp_a, entropy_a, _ = policy.evaluate_actions_masked(obs, actions_a, mask)
        logp_b, entropy_b, _ = policy.evaluate_actions_masked(obs, actions_b, mask)
        torch.testing.assert_close(logp_a, logp_b)
        torch.testing.assert_close(entropy_a, entropy_b)

    def test_buffer_reset_restores_action_masks(self):
        buffer = RolloutBuffer(2, 3, 2, num_envs=1)
        buffer.add(
            torch.zeros(1, 3), torch.zeros(1, 2), torch.zeros(1),
            torch.zeros(1), torch.zeros(1), torch.zeros(1),
            action_dim_mask=torch.zeros(1, 2),
        )
        buffer.reset()
        self.assertTrue(torch.all(buffer.action_dim_masks == 1))

    def test_partial_rollout_shapes(self):
        buffer = RolloutBuffer(4, 3, 2, num_envs=2)
        buffer.add(
            torch.zeros(2, 3), torch.zeros(2, 2), torch.ones(2),
            torch.zeros(2), torch.zeros(2), torch.zeros(2),
        )
        buffer.compute_advantages(torch.zeros(2), torch.zeros(2))
        data = buffer.get_training_data()
        self.assertEqual(data["observations"].shape, (2, 3))
        self.assertEqual(data["actions"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
