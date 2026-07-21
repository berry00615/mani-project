"""
Rollout buffer for storing trajectories and computing GAE advantages.

Supports per-dimension action masking for on-policy consistency:
when the environment deterministically overrides certain action
dimensions, a per-step ``action_dim_mask`` (shape ``(num_envs, act_dim)``)
is stored alongside the transition so that the PPO update can exclude
those dimensions from the policy gradient.
"""

import torch
import numpy as np


class RolloutBuffer:
    """Fixed-size buffer for on-policy rollouts."""

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        act_dim: int,
        num_envs: int = 1,
        device: torch.device = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_envs = num_envs
        self.device = device or torch.device("cpu")
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.observations = torch.zeros(
            (buffer_size, num_envs, obs_dim), device=self.device
        )
        self.actions = torch.zeros(
            (buffer_size, num_envs, act_dim), device=self.device
        )
        self.rewards = torch.zeros(
            (buffer_size, num_envs), device=self.device
        )
        self.values = torch.zeros(
            (buffer_size, num_envs), device=self.device
        )
        self.log_probs = torch.zeros(
            (buffer_size, num_envs), device=self.device
        )
        self.dones = torch.zeros(
            (buffer_size, num_envs), device=self.device
        )

        # Per-dimension action masks for on-policy correctness.
        # 1.0 = dimension was controllable by the policy at that step,
        # 0.0 = dimension was deterministically overridden by the env.
        # Defaults to all-ones (no masking) for backward compatibility.
        self.action_dim_masks = torch.ones(
            (buffer_size, num_envs, act_dim), device=self.device
        )

        # Time-limit bootstrap values
        self.bootstrap_values = torch.zeros(
            (buffer_size, num_envs), device=self.device
        )

        self.advantages = torch.zeros(
            (buffer_size, num_envs), device=self.device
        )
        self.returns = torch.zeros(
            (buffer_size, num_envs), device=self.device
        )

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: torch.Tensor,
        bootstrap_value: torch.Tensor = None,
        action_dim_mask: torch.Tensor = None,
    ):
        """Add a transition. Shapes should be (num_envs, ...).

        Parameters
        ----------
        bootstrap_value : torch.Tensor or None
            For truncated episodes: V(final_observation).
        action_dim_mask : torch.Tensor or None, shape ``(num_envs, act_dim)``
            Per-dimension mask (1.0 = controllable, 0.0 = overridden).
            Defaults to all-ones when not provided.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)
        if isinstance(done, bool):
            done = torch.tensor([done], device=self.device)
        elif done.dim() == 0:
            done = done.unsqueeze(0)

        if value.ndim == 2 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        if value.shape != (self.num_envs,):
            raise ValueError(
                f"value shape {value.shape} != expected ({self.num_envs},). "
                f"value must be (num_envs,) or (num_envs, 1)."
            )

        if log_prob.ndim == 2 and log_prob.shape[-1] == 1:
            log_prob = log_prob.squeeze(-1)
        if log_prob.shape != (self.num_envs,):
            raise ValueError(
                f"log_prob shape {log_prob.shape} != expected ({self.num_envs},). "
                f"log_prob must be (num_envs,) or (num_envs, 1)."
            )

        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done.float()
        if bootstrap_value is not None:
            self.bootstrap_values[self.pos] = bootstrap_value
        if action_dim_mask is not None:
            self.action_dim_masks[self.pos] = action_dim_mask

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True

    def compute_advantages(self, last_value: torch.Tensor, last_done: torch.Tensor):
        """Compute GAE advantages and returns."""
        n_filled = min(self.pos, self.buffer_size)
        if n_filled == 0:
            return

        advantages = torch.zeros(
            (self.buffer_size, self.num_envs), device=self.device
        )
        returns = torch.zeros(
            (self.buffer_size, self.num_envs), device=self.device
        )

        if last_value.ndim == 2 and last_value.shape[-1] == 1:
            last_value = last_value.squeeze(-1)
        if last_value.dim() == 0:
            last_value = last_value.unsqueeze(0)
        if last_value.shape != (self.num_envs,):
            raise ValueError(
                f"last_value shape {last_value.shape} != expected "
                f"({self.num_envs},)."
            )
        if isinstance(last_done, bool) or last_done.dim() == 0:
            last_done = torch.tensor([last_done], device=self.device).float()

        gae = torch.zeros(self.num_envs, device=self.device)
        next_value = last_value

        for step in reversed(range(n_filled)):
            mask = 1.0 - self.dones[step]

            if self.dones[step].any():
                td_bootstrap = torch.where(
                    self.dones[step].bool(),
                    self.bootstrap_values[step],
                    next_value,
                )
            else:
                td_bootstrap = next_value

            delta = (
                self.rewards[step]
                + self.gamma * td_bootstrap
                - self.values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * mask * gae

            advantages[step] = gae
            returns[step] = advantages[step] + self.values[step]

            gae = gae * mask
            next_value = self.values[step]

        self.advantages = advantages
        self.returns = returns

    def get_training_data(self) -> dict[str, torch.Tensor]:
        """Return flattened training data for PPO update.

        Only returns actually filled entries.
        """
        n = min(self.pos, self.buffer_size)
        if n == 0:
            n = self.buffer_size
        return {
            "observations": self.observations[:n].reshape(-1, self.obs_dim),
            "actions": self.actions[:n].reshape(-1, self.act_dim),
            "rewards": self.rewards[:n].reshape(-1),
            "log_probs": self.log_probs[:n].reshape(-1),
            "advantages": self.advantages[:n].reshape(-1),
            "returns": self.returns[:n].reshape(-1),
            "values": self.values[:n].reshape(-1),
            "action_dim_masks": self.action_dim_masks[:n].reshape(-1, self.act_dim),
        }

    def reset(self):
        """Reset buffer position and restore default action dim masks."""
        self.pos = 0
        self.full = False
        self.action_dim_masks.fill_(1.0)
