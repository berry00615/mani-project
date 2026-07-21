"""
PPO algorithm implementation.

Computes clipped surrogate objective with value loss and entropy bonus.
Supports per-dimension action masking for on-policy correctness.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class PPO:
    """Proximal Policy Optimization (clipped objective)."""

    def __init__(
        self,
        policy: nn.Module,
        learning_rate: float = 3e-4,
        n_epochs: int = 10,
        batch_size: int = 64,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: torch.device = None,
    ):
        self.policy = policy
        self.device = device or torch.device("cpu")
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=learning_rate
        )

    def update(self, buffer) -> dict[str, float]:
        """
        Perform PPO update using data from the rollout buffer.

        When ``action_dim_masks`` are present in the buffer data,
        :meth:`ActorCritic.evaluate_actions_masked` is used so that
        dimensions which were deterministically overridden by the
        environment are excluded from the policy gradient.

        Returns a dictionary of training metrics.
        """
        data = buffer.get_training_data()
        n_samples = data["observations"].shape[0]

        # Normalize advantages
        advantages = data["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Per-dimension action masks for on-policy correctness
        action_dim_masks = data["action_dim_masks"]  # (n_samples, act_dim)

        dataset = TensorDataset(
            data["observations"],
            data["actions"],
            data["log_probs"],
            advantages,
            data["returns"],
            data["values"],
            action_dim_masks,
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            for batch in dataloader:
                (obs_b, act_b, old_logp_b, adv_b, ret_b,
                 old_val_b, mask_b) = batch

                # Evaluate current policy with per-dimension masking
                new_logp, entropy, new_val = self.policy.evaluate_actions_masked(
                    obs_b, act_b, mask_b
                )

                # Flatten value: value_net output is (batch, 1) -> (batch,)
                if new_val.ndim == 2 and new_val.shape[-1] == 1:
                    new_val = new_val.squeeze(-1)

                # Ratio — only reflects controllable dimensions
                ratio = torch.exp(new_logp - old_logp_b)

                # Clipped surrogate objective
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped) — value is never masked
                value_pred_clipped = old_val_b + torch.clamp(
                    new_val - old_val_b, -self.clip_range, self.clip_range
                )
                value_loss_unclipped = (new_val - ret_b) ** 2
                value_loss_clipped = (value_pred_clipped - ret_b) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Entropy bonus — only over controllable dimensions
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

                # Approximate KL divergence
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (new_logp - old_logp_b)).mean().item()
                total_approx_kl += approx_kl

                n_updates += 1

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "approx_kl": total_approx_kl / n_updates,
        }

    def state_dict(self) -> dict:
        """Get training state for checkpointing."""
        return {
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict):
        """Restore training state from checkpoint."""
        self.optimizer.load_state_dict(state["optimizer"])
