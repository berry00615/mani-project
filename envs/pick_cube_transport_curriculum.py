"""
Custom PickCube environment with transport curriculum for Stage 3.

Registers as ``PickCubeTransportCurriculum-v1`` via gymnasium.

Extends ``PickCubeGripperCurriculum-v1`` with post-grasp transport rewards:

- **Approach reward reduction**: when the cube is grasped, the TCP-to-cube
  approach reward is scaled down (or removed) so the policy focuses on
  transport rather than staying near the cube.
- **Post-grasp hold reward**: constant per-step bonus for maintaining grasp,
  counterbalancing the reduced approach reward.
- **Lift reward**: proportional to the cube's *relative* height above its
  episode-initial z position (not absolute world-z).  Clamped to ≥0 so
  the policy never gets lift reward for resting on the table.
- **Target distance reward**: amplified reward for moving the cube toward
  the green goal position (builds on top of the base ``place_reward``).
- **Post-grasp open penalty**: penalty for opening the gripper while
  holding the cube, discouraging premature release.
- **Success bonus**: additional reward when ``success`` is achieved,
  added on top of the base environment's ``reward[success]=5`` override.

Pre-grasp behavior is identical to Stage 2 — the approach reward,
collision penalty, gripper-timing rewards, and action masking all
remain unchanged.

Reward formula (raw, pre-normalization)
---------------------------------------

Pre-grasp (same as Stage 2)::

    reward = base_reward - collision_penalty + gripper_open_bonus - early_close_penalty

Post-grasp::

    lift_height = max(0, cube_z - episode_initial_cube_z)
    lift_reward = clamp(lift_height * lift_reward_coef, max=lift_reward_max)

    reward = base_reward - collision_penalty + gripper_open_bonus - early_close_penalty
           - approach_reward * (1 - approach_reward_grasped_scale)  # reduce approach
           + post_grasp_hold_reward                                  # maintain grasp
           + lift_reward                                             # lift (relative)
           + (1 - tanh(5 * obj_to_goal_dist)) * target_distance_reward_coef  # approach goal
           - post_grasp_open_penalty * is_gripper_open               # don't release

Success::

    reward += success_reward_bonus  (on top of base 5.0 success override)

Normalization: all components divided by 5.

Relevant YAML configuration keys
---------------------------------
- ``post_grasp_hold_reward`` (float, default 0.5)
- ``post_grasp_open_penalty`` (float, default 1.0)
- ``lift_reward_coef`` (float, default 5.0)
- ``lift_reward_max`` (float, default 2.0)
- ``target_distance_reward_coef`` (float, default 1.0)
- ``success_reward_bonus`` (float, default 3.0)
- ``approach_reward_grasped_scale`` (float, default 0.0)
"""

from typing import Any, Union

import numpy as np
import torch

from mani_skill.utils.registration import register_env

from .pick_cube_gripper_curriculum import PickCubeGripperCurriculumEnv


@register_env("PickCubeTransportCurriculum-v1", max_episode_steps=100)
class PickCubeTransportCurriculumEnv(PickCubeGripperCurriculumEnv):
    """PickCube with transport rewards for Stage 3 curriculum.

    Inherits all logic from ``PickCubeGripperCurriculumEnv`` (action masking,
    collision penalty, gripper timing) and adds post-grasp transport rewards
    on top.

    Lift reward uses *relative* height — the cube's current z minus its
    episode-initial z — so resting on the table always gives zero lift reward,
    regardless of table height or cube half-size.
    """

    def __init__(self, *args, **kwargs):
        # ---- Pop transport kwargs before parent sees them ----
        self._post_grasp_hold_reward = float(
            kwargs.pop("post_grasp_hold_reward", 0.5)
        )
        self._post_grasp_open_penalty = float(
            kwargs.pop("post_grasp_open_penalty", 1.0)
        )
        self._lift_reward_coef = float(
            kwargs.pop("lift_reward_coef", 5.0)
        )
        self._lift_reward_max = float(
            kwargs.pop("lift_reward_max", 2.0)
        )
        self._target_distance_reward_coef = float(
            kwargs.pop("target_distance_reward_coef", 1.0)
        )
        self._success_reward_bonus = float(
            kwargs.pop("success_reward_bonus", 3.0)
        )
        self._approach_reward_grasped_scale = float(
            kwargs.pop("approach_reward_grasped_scale", 0.0)
        )

        # Per-env initial cube z — MUST be set BEFORE super().__init__()
        # because _initialize_episode is called from the parent constructor
        # chain (via reset()), and our override accesses this attribute.
        self._episode_initial_cube_z: torch.Tensor | None = None

        super().__init__(*args, **kwargs)

        # ---- Per-step tracking for transport diagnostics ----
        self._last_transport_reward: torch.Tensor | None = None
        self._last_lift_reward: torch.Tensor | None = None
        self._last_target_distance_reward: torch.Tensor | None = None
        self._last_hold_reward: torch.Tensor | None = None
        self._last_post_grasp_open_penalty: torch.Tensor | None = None
        self._last_approach_reduction: torch.Tensor | None = None
        self._last_success_bonus: torch.Tensor | None = None

        # State for per-episode transport tracking
        self._last_cube_world_z: torch.Tensor | None = None
        self._last_cube_lift_height: torch.Tensor | None = None
        self._last_obj_to_goal_dist: torch.Tensor | None = None
        self._last_is_grasped: torch.Tensor | None = None
        self._last_is_gripper_open: torch.Tensor | None = None
        self._last_success: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Episode initialization — track per-env initial cube z
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Extend parent to capture per-env initial cube z after spawn."""
        super()._initialize_episode(env_idx, options)

        # Lazy-init the tracking tensor on first call
        if self._episode_initial_cube_z is None:
            self._episode_initial_cube_z = torch.zeros(
                self.num_envs, device=self.device
            )

        # Record initial cube z for the newly-initialised environments.
        # This is called by both the initial reset and auto-reset, and
        # env_idx only contains the indices that were just reset, so
        # other environments' initial z values are *not* overwritten.
        self._episode_initial_cube_z[env_idx] = (
            self.cube.pose.p[env_idx, 2].clone()
        )

    # ------------------------------------------------------------------
    # Transport helpers
    # ------------------------------------------------------------------

    def _get_cube_world_z(self) -> torch.Tensor:
        """Return cube absolute z-coordinate (world frame)."""
        return self.cube.pose.p[:, 2]  # (num_envs,)

    def _get_cube_lift_height(self) -> torch.Tensor:
        """Return cube *relative* lift height above episode-initial z.

        Clamped to ≥0 so that dropping below the initial height (e.g.
        pushing the cube down) does not produce negative lift reward.
        """
        cube_z = self._get_cube_world_z()
        if self._episode_initial_cube_z is None:
            # Before first _initialize_episode — use current z as baseline
            return torch.zeros_like(cube_z)
        lift = cube_z - self._episode_initial_cube_z
        return torch.clamp(lift, min=0.0)

    def _get_obj_to_goal_dist(self) -> torch.Tensor:
        """Return Euclidean distance from cube to goal position."""
        return torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )  # (num_envs,)

    # ------------------------------------------------------------------
    # Transport reward
    # ------------------------------------------------------------------

    def _compute_transport_reward(
        self,
        is_grasped: torch.Tensor,
        success: torch.Tensor,
        tcp_to_obj_dist: torch.Tensor,
    ):
        """Compute post-grasp transport rewards.

        Parameters
        ----------
        is_grasped : torch.Tensor, shape ``(num_envs,)``, bool
        success : torch.Tensor, shape ``(num_envs,)``, bool
        tcp_to_obj_dist : torch.Tensor, shape ``(num_envs,)``
            TCP-to-cube Euclidean distance (used for approach reduction).

        Returns
        -------
        transport_reward : torch.Tensor, shape ``(num_envs,)``
            Total transport reward adjustment (raw, pre-normalization).
        """
        num_envs = self.num_envs
        device = self.device
        transport_reward = torch.zeros(num_envs, device=device)

        grasp_f = is_grasped.float()

        # ---- 1. Reduce approach reward when grasping ----
        approach_reward = 1.0 - torch.tanh(5.0 * tcp_to_obj_dist)
        approach_reduction = (
            approach_reward
            * grasp_f
            * (1.0 - self._approach_reward_grasped_scale)
        )
        transport_reward -= approach_reduction

        # ---- 2. Hold reward (constant per-step bonus for maintaining grasp) ----
        transport_reward += self._post_grasp_hold_reward * grasp_f

        # ---- 3. Lift reward (relative height, clamped ≥0) ----
        lift_height = self._get_cube_lift_height()  # already clamp(min=0)
        lift_reward = torch.clamp(
            lift_height * self._lift_reward_coef,
            max=self._lift_reward_max,
        )
        transport_reward += lift_reward * grasp_f

        # ---- 4. Target distance reward (amplified cube-to-goal) ----
        obj_to_goal_dist = self._get_obj_to_goal_dist()
        target_reward = (
            1.0 - torch.tanh(5.0 * obj_to_goal_dist)
        ) * self._target_distance_reward_coef
        transport_reward += target_reward * grasp_f

        # ---- 5. Post-grasp open penalty ----
        gripper_width = self._get_gripper_width()
        is_gripper_open = gripper_width > self._gripper_open_threshold
        open_penalty = is_gripper_open.float() * self._post_grasp_open_penalty
        transport_reward -= open_penalty * grasp_f

        # ---- 6. Success bonus (additive, after base reward[success]=5) ----
        transport_reward += success.float() * self._success_reward_bonus

        # ---- Store for diagnostics ----
        self._last_transport_reward = transport_reward
        self._last_lift_reward = lift_reward * grasp_f
        self._last_target_distance_reward = target_reward * grasp_f
        self._last_hold_reward = self._post_grasp_hold_reward * grasp_f
        self._last_post_grasp_open_penalty = open_penalty * grasp_f
        self._last_approach_reduction = approach_reduction
        self._last_success_bonus = success.float() * self._success_reward_bonus

        # State for logging (world-z and relative lift-height are both stored)
        self._last_cube_world_z = self._get_cube_world_z()
        self._last_cube_lift_height = lift_height
        self._last_obj_to_goal_dist = obj_to_goal_dist
        self._last_is_grasped = is_grasped
        self._last_is_gripper_open = is_gripper_open
        self._last_success = success

        return transport_reward

    # ------------------------------------------------------------------
    # Reward overrides
    # ------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """Compute dense reward with transport shaping.

        Extends the parent (Stage 2) reward by adding transport-specific
        terms.  The base environment's ``reward[success] = 5`` override
        is preserved in the parent reward; our success bonus is added
        on top.
        """
        # Parent reward: base - collision_penalty + gripper_timing
        # (includes base reward[success]=5 for successful envs)
        parent_reward = super().compute_dense_reward(obs, action, info)

        is_grasped = info["is_grasped"]
        success = info.get("success",
                           torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))

        tcp_to_obj_dist = self._compute_tcp_to_cube_dist()

        # Compute transport reward (includes success_bonus added AFTER base override)
        transport_reward = self._compute_transport_reward(
            is_grasped=is_grasped,
            success=success,
            tcp_to_obj_dist=tcp_to_obj_dist,
        )

        reward = parent_reward + transport_reward
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    # ------------------------------------------------------------------
    # Transport info for logging
    # ------------------------------------------------------------------

    def get_transport_info(self):
        """Return the most recent transport state for diagnostics.

        Called by the training loop after ``env.step``.

        Returns
        -------
        is_grasped : torch.Tensor or None, shape ``(num_envs,)``, bool
        is_gripper_open : torch.Tensor or None, shape ``(num_envs,)``, bool
        cube_world_z : torch.Tensor or None, shape ``(num_envs,)``
            Absolute cube z (world frame).
        cube_lift_height : torch.Tensor or None, shape ``(num_envs,)``
            Relative height above episode-initial z (clamped ≥0).
        obj_to_goal_dist : torch.Tensor or None, shape ``(num_envs,)``
        """
        return (
            getattr(self, "_last_is_grasped", None),
            getattr(self, "_last_is_gripper_open", None),
            getattr(self, "_last_cube_world_z", None),
            getattr(self, "_last_cube_lift_height", None),
            getattr(self, "_last_obj_to_goal_dist", None),
        )

    def get_reward_components(self):
        """Return all reward components for diagnostics.

        Extends the parent's (5 components) with transport terms.

        Returns
        -------
        original_reward : torch.Tensor or None    [0] Base PickCubeEnv reward (raw).
        collision_penalty : torch.Tensor or None  [1]
        gripper_reward : torch.Tensor or None     [2]
        early_close_penalty : torch.Tensor or None [3]
        gripper_open_bonus : torch.Tensor or None [4]
        transport_reward : torch.Tensor or None   [5] Total transport adjustment.
        lift_reward : torch.Tensor or None        [6]
        target_distance_reward : torch.Tensor or None [7]
        hold_reward : torch.Tensor or None        [8]
        post_grasp_open_penalty : torch.Tensor or None [9]
        approach_reduction : torch.Tensor or None [10]
        success_bonus : torch.Tensor or None      [11]
        """
        parent_comps = super().get_reward_components()
        return (
            parent_comps[0],
            parent_comps[1],
            parent_comps[2],
            parent_comps[3],
            parent_comps[4],
            getattr(self, "_last_transport_reward", None),
            getattr(self, "_last_lift_reward", None),
            getattr(self, "_last_target_distance_reward", None),
            getattr(self, "_last_hold_reward", None),
            getattr(self, "_last_post_grasp_open_penalty", None),
            getattr(self, "_last_approach_reduction", None),
            getattr(self, "_last_success_bonus", None),
        )
