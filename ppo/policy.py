"""
Actor-Critic network for PPO.

Architecture:
  obs -> shared_features -> action_mean (actor head)
                         -> value (critic head)

The policy outputs a diagonal Gaussian distribution.
Actions are sampled via reparameterization during training
and deterministically (mean) during evaluation.

Per-dimension action masking (on-policy fix)
--------------------------------------------
When the environment deterministically overrides certain action
dimensions (e.g. forcing gripper open when far from the cube),
those dimensions must be excluded from the PPO objective.
Otherwise the policy gradient wrongly attributes outcomes to
actions the policy never chose.

:meth:`evaluate_actions_masked` accepts an ``action_dim_mask``
tensor of shape ``(batch, act_dim)``.  Masked-out dimensions
contribute zero to both log_prob and entropy, so they produce
no policy gradient.  The value function is never masked — it
always sees the full observation.
"""

import torch
import torch.nn as nn
import numpy as np


def build_mlp(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
    """Build a simple MLP with Tanh activations."""
    layers = []
    prev_dim = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(nn.Tanh())
        prev_dim = h
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Actor-Critic with shared feature extractor and separate heads."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        policy_hidden_sizes: list[int] = None,
        value_hidden_sizes: list[int] = None,
        action_low: np.ndarray = None,
        action_high: np.ndarray = None,
        log_std_init: float = 0.0,
    ):
        super().__init__()
        if policy_hidden_sizes is None:
            policy_hidden_sizes = [256, 256, 256]
        if value_hidden_sizes is None:
            value_hidden_sizes = [256, 256, 256]

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Register action bounds for rescaling
        if action_low is not None:
            self.register_buffer("action_low", torch.from_numpy(action_low).float())
        else:
            self.register_buffer("action_low", -torch.ones(act_dim))
        if action_high is not None:
            self.register_buffer("action_high", torch.from_numpy(action_high).float())
        else:
            self.register_buffer("action_high", torch.ones(act_dim))

        # Shared feature extractor
        self.features = build_mlp(obs_dim, policy_hidden_sizes[:-1], policy_hidden_sizes[-1])

        # Actor head: outputs mean of diagonal Gaussian in [-1, 1]
        self.actor_head = nn.Linear(policy_hidden_sizes[-1], act_dim)

        # Value head
        self.value_net = build_mlp(
            policy_hidden_sizes[-1], value_hidden_sizes, 1
        )

        # Learnable log std
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

        # Cache for architecture info
        self.policy_hidden_sizes = policy_hidden_sizes
        self.value_hidden_sizes = value_hidden_sizes

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (action_mean_rescaled, value).

        action_mean is in the original action space (after tanh rescaling).
        """
        features = self.features(obs)
        action_mean_tanh = torch.tanh(self.actor_head(features))
        action_mean = self._rescale_action(action_mean_tanh)
        value = self.value_net(features)
        return action_mean, value

    def _rescale_action(self, action_tanh: torch.Tensor) -> torch.Tensor:
        """Rescale actions from [-1, 1] to [action_low, action_high]."""
        low = self.action_low
        high = self.action_high
        return low + (high - low) * (action_tanh + 1.0) / 2.0

    def _forward_actor(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared forward pass for actor: returns (action_mean, std, value)."""
        features = self.features(obs)
        action_mean_tanh = torch.tanh(self.actor_head(features))
        action_mean = self._rescale_action(action_mean_tanh)
        value = self.value_net(features)
        std = torch.exp(self.log_std)
        return action_mean, std, value

    # ------------------------------------------------------------------
    # Action sampling (backward-compatible API)
    # ------------------------------------------------------------------

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Returns:
            action: the sampled (or mean) action
            log_prob: log probability of the sampled action (summed over dims)
            value: value estimate
        """
        action_mean, std, value = self._forward_actor(obs)
        dist = torch.distributions.Normal(action_mean, std)

        if deterministic:
            action = action_mean
        else:
            action = dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def get_action_dist(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action and return distribution parameters.

        Returns:
            action:     (batch, act_dim)  sampled (or mean) action
            log_prob:   (batch,)          summed log_prob
            value:      (batch, 1)        value estimate
            action_mean: (batch, act_dim) Gaussian mean
            std:        (act_dim,)         Gaussian std
        """
        action_mean, std, value = self._forward_actor(obs)
        dist = torch.distributions.Normal(action_mean, std)

        if deterministic:
            action = action_mean
        else:
            action = dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value, action_mean, std

    # ------------------------------------------------------------------
    # Per-dimension log_prob (for on-policy masking)
    # ------------------------------------------------------------------

    def get_log_prob(
        self, action_mean: torch.Tensor, std: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Summed log_prob of ``action`` under Normal(action_mean, std).

        No forward pass — reuses pre-computed distribution parameters.
        """
        dist = torch.distributions.Normal(action_mean, std)
        return dist.log_prob(action).sum(dim=-1)

    def get_log_prob_per_dim(
        self, action_mean: torch.Tensor, std: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Per-dimension log_prob, shape ``(batch, act_dim)``.

        No forward pass — reuses pre-computed ``(action_mean, std)``.
        Caller is responsible for masking and summing.
        """
        dist = torch.distributions.Normal(action_mean, std)
        return dist.log_prob(action)  # (batch, act_dim)

    # ------------------------------------------------------------------
    # Policy evaluation (for PPO update)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log_prob and entropy for given actions (all dimensions).

        Used during PPO updates when no action masking is needed.
        Backward-compatible.

        Returns:
            log_prob: (batch,)   summed over all action dims
            entropy:  (batch,)   summed over all action dims
            value:    (batch, 1) value estimate
        """
        features = self.features(obs)
        action_mean_tanh = torch.tanh(self.actor_head(features))
        action_mean = self._rescale_action(action_mean_tanh)
        value = self.value_net(features)

        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, value

    def evaluate_actions_masked(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_dim_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log_prob and entropy with per-dimension masking.

        Masked-out dimensions (mask == 0) contribute zero to both
        log_prob and entropy, so they produce **no policy gradient**
        and do not affect the PPO ratio.

        The value function is **never masked** — it always sees the
        full observation and predicts the full return.

        Parameters
        ----------
        obs : (batch, obs_dim)
        actions : (batch, act_dim)
        action_dim_mask : (batch, act_dim)
            1.0 for controllable dimensions, 0.0 for masked dimensions.

        Returns
        -------
        log_prob : (batch,)   sum over *active* dimensions only
        entropy :  (batch,)   sum over *active* dimensions only
        value :    (batch, 1) value estimate (never masked)
        """
        features = self.features(obs)
        action_mean_tanh = torch.tanh(self.actor_head(features))
        action_mean = self._rescale_action(action_mean_tanh)
        value = self.value_net(features)

        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, std)

        log_prob_per_dim = dist.log_prob(actions)      # (batch, act_dim)
        entropy_per_dim = dist.entropy()                # (batch, act_dim)

        log_prob = (log_prob_per_dim * action_dim_mask).sum(dim=-1)
        entropy = (entropy_per_dim * action_dim_mask).sum(dim=-1)

        return log_prob, entropy, value

    def get_architecture_info(self) -> dict:
        """Return architecture info for checkpointing."""
        return {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "policy_hidden_sizes": self.policy_hidden_sizes,
            "value_hidden_sizes": self.value_hidden_sizes,
            "action_low": self.action_low.cpu().numpy(),
            "action_high": self.action_high.cpu().numpy(),
        }
