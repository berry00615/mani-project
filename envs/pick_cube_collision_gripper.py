"""
Custom PickCube environment with table collision penalty + gripper timing rewards.

Registers as ``PickCubeCollisionGripper-v1`` via gymnasium.

Extends ``PickCubeCollisionPenalty-v1`` with gripper-open/close timing rewards:

- **Early close penalty**: when TCP is far from the cube and the gripper is
  already closed, a penalty is subtracted (clip-safe, bounded).
- **Open-near-cube bonus**: when TCP is near the cube and the gripper stays
  open, a small bonus is added.
- **Grasp bonus**: the original ``is_grasped`` reward already gives +1, so
  we do NOT add another grasp bonus here.  We simply suppress the early-close
  penalty once the cube is grasped.

All coefficients are YAML-configurable via ``env_kwargs``.

Relevant YAML configuration keys
---------------------------------
- ``early_gripper_close_penalty_coef`` (float, default 0.2)
- ``gripper_open_near_cube_bonus`` (float, default 0.1)
- ``gripper_near_distance`` (float, default 0.08)
- ``gripper_far_distance`` (float, default 0.15)
- ``gripper_open_threshold`` (float, default 0.03)
- ``gripper_closed_threshold`` (float, default 0.01)
"""

from typing import Any

import torch

from mani_skill.utils.registration import register_env

from .pick_cube_collision_penalty import PickCubeCollisionPenaltyEnv


@register_env("PickCubeCollisionGripper-v1", max_episode_steps=100)
class PickCubeCollisionGripperEnv(PickCubeCollisionPenaltyEnv):
    """PickCube with table-collision penalty *and* gripper-timing rewards.

    Inherits all collision-penalty logic from ``PickCubeCollisionPenaltyEnv``
    and adds gripper-open/close timing on top.
    """

    # Gripper joint indices in the robot qpos (Panda: index 7 = finger1, 8 = finger2)
    GRIPPER_FINGER1_IDX = 7
    GRIPPER_FINGER2_IDX = 8

    def __init__(self, *args, **kwargs):
        # Pop our new kwargs before the parent constructor sees them
        self._early_gripper_close_penalty_coef = float(
            kwargs.pop("early_gripper_close_penalty_coef", 0.2)
        )
        self._gripper_open_near_cube_bonus = float(
            kwargs.pop("gripper_open_near_cube_bonus", 0.1)
        )
        self._gripper_near_distance = float(
            kwargs.pop("gripper_near_distance", 0.08)
        )
        self._gripper_far_distance = float(
            kwargs.pop("gripper_far_distance", 0.15)
        )
        self._gripper_open_threshold = float(
            kwargs.pop("gripper_open_threshold", 0.03)
        )
        self._gripper_closed_threshold = float(
            kwargs.pop("gripper_closed_threshold", 0.01)
        )

        super().__init__(*args, **kwargs)

        # Per-step tracking for logging
        self._last_gripper_width = None
        self._last_early_close_mask = None
        self._last_gripper_open_near_mask = None
        self._last_early_close_penalty = None
        self._last_gripper_open_bonus = None

    # ------------------------------------------------------------------
    # Gripper helpers
    # ------------------------------------------------------------------

    def _get_gripper_width(self) -> torch.Tensor:
        """Return gripper opening width per environment.

        For Panda the two finger joints open symmetrically, so the
        effective opening width is ``finger1_qpos + finger2_qpos``.

        Returns
        -------
        torch.Tensor, shape ``(num_envs,)``
            ~0.08 when fully open, ~0.0 when closed.
        """
        qpos = self.agent.robot.get_qpos()  # (num_envs, 9)
        return qpos[..., self.GRIPPER_FINGER1_IDX] + qpos[..., self.GRIPPER_FINGER2_IDX]

    def _compute_tcp_to_cube_dist(self) -> torch.Tensor:
        """Return Euclidean distance from TCP to cube centre."""
        return torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pose.p, axis=1
        )

    # ------------------------------------------------------------------
    # Gripper-timing reward
    # ------------------------------------------------------------------

    def _compute_gripper_timing_reward(
        self, tcp_to_cube_dist: torch.Tensor, is_grasped: torch.Tensor
    ):
        """Compute gripper-timing penalty / bonus.

        Returns
        -------
        gripper_bonus : torch.Tensor, shape ``(num_envs,)``
            Scalar reward adjustment (negative for penalty, positive for bonus).
        early_close_mask : torch.Tensor, shape ``(num_envs,)``, bool
            True for environments penalised for early gripper close.
        gripper_open_near_mask : torch.Tensor, shape ``(num_envs,)``, bool
            True for environments where gripper is open near cube.
        """
        gripper_width = self._get_gripper_width()
        self._last_gripper_width = gripper_width

        is_far = tcp_to_cube_dist > self._gripper_far_distance
        is_near = tcp_to_cube_dist < self._gripper_near_distance
        gripper_closed = gripper_width < self._gripper_closed_threshold
        gripper_open = gripper_width > self._gripper_open_threshold

        # ---- Early close penalty ----
        # Condition: TCP far from cube, gripper already closed, AND not already
        # grasping (so we don't penalise holding the cube).
        early_close = is_far & gripper_closed & (~is_grasped)
        early_close_penalty = (
            early_close.float() * self._early_gripper_close_penalty_coef
        )

        # ---- Open-near-cube bonus ----
        # Condition: TCP near cube, gripper still open, not yet grasped.
        gripper_open_near = is_near & gripper_open & (~is_grasped)
        gripper_open_bonus = (
            gripper_open_near.float() * self._gripper_open_near_cube_bonus
        )

        # Clip each component to avoid dominating the dense reward
        early_close_penalty = torch.clamp(early_close_penalty, max=1.0)
        gripper_open_bonus = torch.clamp(gripper_open_bonus, max=1.0)

        # Total gripper-timing reward (bonus - penalty)
        gripper_reward = gripper_open_bonus - early_close_penalty

        # Store for logging
        self._last_early_close_mask = early_close
        self._last_gripper_open_near_mask = gripper_open_near
        self._last_early_close_penalty = early_close_penalty
        self._last_gripper_open_bonus = gripper_open_bonus

        return gripper_reward, early_close, gripper_open_near

    # ------------------------------------------------------------------
    # Reward overrides
    # ------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Get the collision-aware reward from the immediate parent
        reward = super().compute_dense_reward(obs, action, info)

        # Compute TCP-to-cube distance
        tcp_to_cube_dist = self._compute_tcp_to_cube_dist()

        # Gripper-timing reward
        gripper_reward, early_close_mask, gripper_open_near_mask = (
            self._compute_gripper_timing_reward(tcp_to_cube_dist, info["is_grasped"])
        )

        # Store the parent's original_reward for reward-component diagnostics
        # (super().compute_dense_reward already set _last_original_reward)
        self._last_gripper_reward = gripper_reward

        reward = reward + gripper_reward

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    # ------------------------------------------------------------------
    # Gripper info for logging
    # ------------------------------------------------------------------

    def get_gripper_info(self):
        """Return the most recent gripper-timing statistics.

        Called by the training loop after ``env.step``.

        Returns
        -------
        gripper_width : torch.Tensor, shape ``(num_envs,)``
        early_close_mask : torch.Tensor, shape ``(num_envs,)``, bool
        gripper_open_near_mask : torch.Tensor, shape ``(num_envs,)``, bool
        early_close_penalty : torch.Tensor, shape ``(num_envs,)``
        gripper_open_bonus : torch.Tensor, shape ``(num_envs,)``
        """
        return (
            getattr(self, "_last_gripper_width", None),
            getattr(self, "_last_early_close_mask", None),
            getattr(self, "_last_gripper_open_near_mask", None),
            getattr(self, "_last_early_close_penalty", None),
            getattr(self, "_last_gripper_open_bonus", None),
        )

    def get_reward_components(self):
        """Return all reward components for diagnostics.

        Extends the parent's components with gripper-timing terms.

        Returns
        -------
        original_reward : torch.Tensor or None
        collision_penalty : torch.Tensor or None
        gripper_reward : torch.Tensor or None
        early_close_penalty : torch.Tensor or None
        gripper_open_bonus : torch.Tensor or None
        """
        orig_rew, col_pen = super().get_reward_components()
        return (
            orig_rew,
            col_pen,
            getattr(self, "_last_gripper_reward", None),
            getattr(self, "_last_early_close_penalty", None),
            getattr(self, "_last_gripper_open_bonus", None),
        )
