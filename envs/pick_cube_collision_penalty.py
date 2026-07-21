"""
Custom PickCube environment with table collision penalty.

Registers as ``PickCubeCollisionPenalty-v1`` via gymnasium.

Detects non-expected contacts between robot links and the table surface
and subtracts a penalty from the original PickCube-v1 dense reward.

Collision Detection Strategy
-----------------------------
Uses ``scene.get_pairwise_contact_forces(link, table)`` which returns
a ``(num_envs, 3)`` tensor of net contact force vectors.  The L2 norm
of the force vector is compared against a configurable threshold.

Exclusions
----------
- ``panda_link0`` — the robot base is mounted and always in contact
  with the mounting surface; penalising it would add constant noise.
- Cube↔table contacts are *never checked* because we only iterate over
  robot links, so normal cube-on-table resting contacts are ignored
  automatically.
- ``panda_hand_tcp`` (link index 10) is a virtual link without collision
  geometry — it never produces contacts, but we skip it explicitly for
  clarity.
"""

from typing import Any

import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env


# ---------- custom env ----------

@register_env("PickCubeCollisionPenalty-v1", max_episode_steps=50)
class PickCubeCollisionPenaltyEnv(PickCubeEnv):
    """PickCube-v1 with an extra penalty for robot-table collisions.

    New keyword arguments (also settable via YAML ``env_kwargs``)
    --------------------------------------------------------------
    - ``table_collision_penalty_coef`` (float, default 0.2):
        Multiplier applied to the contact-force norm.
    - ``table_collision_force_threshold`` (float, default 0.01):
        Forces whose L2 norm is below this value (in Newtons) are
        treated as zero.  Helps filter out tiny residual contacts.
    - ``table_collision_penalty_max`` (float, default 1.0):
        Maximum penalty subtracted **per step** (across all links).
    - ``table_collision_excluded_links`` (list[str], default
        ``["panda_link0", "panda_hand_tcp"]``):
        Link names that are never checked for table collisions.
    """

    # Excluded links — these are either the fixed base or virtual links.
    DEFAULT_EXCLUDED_LINKS = {"panda_link0", "panda_hand_tcp"}

    def __init__(self, *args, **kwargs):
        # Pop our custom kwargs before passing the rest to the parent
        self._table_collision_penalty_coef = float(
            kwargs.pop("table_collision_penalty_coef", 0.2)
        )
        self._table_collision_force_threshold = float(
            kwargs.pop("table_collision_force_threshold", 0.01)
        )
        self._table_collision_penalty_max = float(
            kwargs.pop("table_collision_penalty_max", 1.0)
        )
        excluded = kwargs.pop("table_collision_excluded_links", None)
        if excluded is None:
            self._table_collision_excluded_links = self.DEFAULT_EXCLUDED_LINKS
        else:
            self._table_collision_excluded_links = set(excluded)

        super().__init__(*args, **kwargs)

        # Will be populated on first use (after scene is built)
        self._collision_check_links = None

    # ------------------------------------------------------------------
    # Collision helpers
    # ------------------------------------------------------------------

    def _get_collision_check_links(self):
        """Return the list of robot links that should be checked for table collisions.

        Cached after first call — the link set does not change after ``_load_scene``.
        """
        if self._collision_check_links is not None:
            return self._collision_check_links

        all_links = self.agent.robot.get_links()
        check_links = []
        for link in all_links:
            if link.name in self._table_collision_excluded_links:
                continue
            check_links.append(link)

        self._collision_check_links = check_links
        return check_links

    def _compute_table_collision_penalty(self) -> torch.Tensor:
        """Compute per-environment collision penalty.

        Returns
        -------
        penalty : torch.Tensor, shape ``(num_envs,)``
            Scalar penalty in [0, ``table_collision_penalty_max``].
        collision_mask : torch.Tensor, shape ``(num_envs,)``, bool
            True for environments where at least one robot link is in
            contact with the table above the force threshold.
        """
        device = self.device
        num_envs = self.num_envs
        check_links = self._get_collision_check_links()
        table = self.table_scene.table

        penalty = torch.zeros(num_envs, device=device)
        collision_mask = torch.zeros(num_envs, device=device, dtype=torch.bool)

        thresh = self._table_collision_force_threshold

        for link in check_links:
            forces = self.scene.get_pairwise_contact_forces(link, table)
            # forces: (num_envs, 3)
            force_norm = forces.norm(dim=-1)  # (num_envs,)

            # Only count force above threshold
            above_thresh = force_norm > thresh
            collision_mask = collision_mask | above_thresh

            # Add contribution (clipped per-link to avoid single-link spikes,
            # but the total is capped later)
            penalty = penalty + force_norm * above_thresh.float()

        # Apply coefficient
        penalty = penalty * self._table_collision_penalty_coef

        # Clip total penalty per step
        penalty = torch.clamp(penalty, max=self._table_collision_penalty_max)

        return penalty, collision_mask

    # ------------------------------------------------------------------
    # Reward overrides
    # ------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Original reward from PickCubeEnv
        original_reward = super().compute_dense_reward(obs, action, info)

        # Compute collision penalty
        collision_penalty, collision_mask = self._compute_table_collision_penalty()

        # Store for logging
        self._last_collision_penalty = collision_penalty
        self._last_collision_mask = collision_mask
        self._last_original_reward = original_reward

        # Subtract penalty from reward
        reward = original_reward - collision_penalty

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        # For the normalized reward (used by some algorithms), just divide by 5
        # (the same normalization as the original)
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    # ------------------------------------------------------------------
    # Info extension — expose collision data to the training loop
    # ------------------------------------------------------------------

    def get_collision_info(self):
        """Return the most recent collision penalty and mask.

        Called by the training loop after ``env.step`` to retrieve
        per-step collision statistics.

        Returns
        -------
        penalty : torch.Tensor, shape ``(num_envs,)``
        mask : torch.Tensor, shape ``(num_envs,)``, bool
        """
        return (
            getattr(self, "_last_collision_penalty", None),
            getattr(self, "_last_collision_mask", None),
        )

    def get_reward_components(self):
        """Return the most recent reward components for diagnostics.

        Returns
        -------
        original_reward : torch.Tensor or None, shape ``(num_envs,)``
            The base PickCubeEnv dense reward (approach + grasp + place + static).
        collision_penalty : torch.Tensor or None, shape ``(num_envs,)``
        """
        return (
            getattr(self, "_last_original_reward", None),
            getattr(self, "_last_collision_penalty", None),
        )
