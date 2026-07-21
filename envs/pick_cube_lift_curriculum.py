"""
Custom PickCube environment with lift curriculum for Stage 4.

Registers as ``PickCubeLiftCurriculum-v1`` via gymnasium.

Extends ``PickCubeGripperCurriculum-v1`` with lift-focused post-grasp rewards.
Unlike Stage 3, this environment does NOT include cube-to-goal transport rewards.
The sole objective is: grasp → hold closed → lift the cube at least 5 cm.

Reward formula (raw, pre-normalization)
---------------------------------------

Pre-grasp (same as Stage 2)::

    reward = base_reward - collision_penalty + gripper_open_bonus - early_close_penalty

Post-grasp::

    lift_height = max(0, cube_z - episode_initial_cube_z)
    lift_reward = clamp(lift_height * lift_reward_coef, max=lift_reward_max)

    reward = base_reward - collision_penalty + gripper_open_bonus - early_close_penalty
           - approach_reward * (1 - approach_reward_grasped_scale)  # reduce approach
           + min_hold_reward                                         # tiny hold bonus
           + lift_reward                                             # continuous lift
           + milestone_bonus                                         # one-time lift milestones
           - drop_penalty * is_drop_event                            # opening after grasp

Normalization: all components divided by 5.

Milestone bonuses (one-time per episode, cumulative)
----------------------------------------------------
- lift_height > 0.01 m  →  +0.5 raw
- lift_height > 0.03 m  →  +1.5 raw  (cumulative: 2.0)
- lift_height > 0.05 m  →  +3.0 raw  (cumulative: 5.0)

These are tracked per-environment and reset on episode boundaries.

Relevant YAML configuration keys
---------------------------------
- ``min_hold_reward`` (float, default 0.02)
- ``lift_reward_coef`` (float, default 15.0)
- ``lift_reward_max`` (float, default 1.5)
- ``lift_1cm_bonus`` (float, default 0.5)
- ``lift_3cm_bonus`` (float, default 1.5)
- ``lift_5cm_bonus`` (float, default 3.0)
- ``drop_penalty`` (float, default 1.0)
- ``approach_reward_grasped_scale`` (float, default 0.0)
"""

from typing import Any

import torch

from mani_skill.utils.registration import register_env

from .pick_cube_gripper_curriculum import PickCubeGripperCurriculumEnv


@register_env("PickCubeLiftCurriculum-v1", max_episode_steps=100)
class PickCubeLiftCurriculumEnv(PickCubeGripperCurriculumEnv):
    """PickCube with lift curriculum for Stage 4.

    Inherits all logic from ``PickCubeGripperCurriculumEnv`` (action masking,
    collision penalty, gripper timing) and adds lift-focused post-grasp rewards.

    Design principle: hold reward is deliberately tiny (0.02).  Continuous
    lift reward provides the main gradient, and milestone bonuses reward
    specific height achievements.
    """

    def __init__(self, *args, **kwargs):
        # ---- Pop lift kwargs before parent sees them ----
        self._min_hold_reward = float(
            kwargs.pop("min_hold_reward", 0.02)
        )
        self._lift_reward_coef = float(
            kwargs.pop("lift_reward_coef", 15.0)
        )
        self._lift_reward_max = float(
            kwargs.pop("lift_reward_max", 1.5)
        )
        self._lift_1cm_bonus = float(
            kwargs.pop("lift_1cm_bonus", 0.5)
        )
        self._lift_3cm_bonus = float(
            kwargs.pop("lift_3cm_bonus", 1.5)
        )
        self._lift_5cm_bonus = float(
            kwargs.pop("lift_5cm_bonus", 3.0)
        )
        self._drop_penalty = float(
            kwargs.pop("drop_penalty", 1.0)
        )
        self._approach_reward_grasped_scale = float(
            kwargs.pop("approach_reward_grasped_scale", 0.0)
        )

        # Per-env episode state — MUST be set BEFORE super().__init__()
        # because _initialize_episode is called from the parent chain.
        self._episode_initial_cube_z: torch.Tensor | None = None
        self._lift_1cm_reached: torch.Tensor | None = None
        self._lift_3cm_reached: torch.Tensor | None = None
        self._lift_5cm_reached: torch.Tensor | None = None
        self._was_grasped: torch.Tensor | None = None

        super().__init__(*args, **kwargs)

        # ---- Per-step tracking for diagnostics ----
        self._last_lift_bonus_total: torch.Tensor | None = None  # total lift adjustment
        self._last_lift_continuous: torch.Tensor | None = None   # continuous lift only
        self._last_milestone_bonus: torch.Tensor | None = None
        self._last_drop_penalty: torch.Tensor | None = None
        self._last_hold_reward_val: torch.Tensor | None = None
        self._last_approach_reduction: torch.Tensor | None = None

        # State for logging
        self._last_cube_world_z: torch.Tensor | None = None
        self._last_cube_lift_height: torch.Tensor | None = None
        self._last_is_grasped: torch.Tensor | None = None
        self._last_is_drop_event: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Episode initialization — per-env state
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Extend parent to reset per-env lift state for newly-reset envs."""
        super()._initialize_episode(env_idx, options)

        num_envs = self.num_envs
        device = self.device

        # Lazy-init tracking tensors on first call
        if self._episode_initial_cube_z is None:
            self._episode_initial_cube_z = torch.zeros(num_envs, device=device)
            self._lift_1cm_reached = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self._lift_3cm_reached = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self._lift_5cm_reached = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self._was_grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Capture initial cube z for the reset envs
        self._episode_initial_cube_z[env_idx] = (
            self.cube.pose.p[env_idx, 2].clone()
        )

        # Reset milestone flags and grasp state for newly-reset envs
        self._lift_1cm_reached[env_idx] = False
        self._lift_3cm_reached[env_idx] = False
        self._lift_5cm_reached[env_idx] = False
        self._was_grasped[env_idx] = False

    # ------------------------------------------------------------------
    # Lift helpers
    # ------------------------------------------------------------------

    def _get_cube_world_z(self) -> torch.Tensor:
        return self.cube.pose.p[:, 2]

    def _get_cube_lift_height(self) -> torch.Tensor:
        """Relative lift height above episode-initial z, clamped ≥0."""
        cube_z = self._get_cube_world_z()
        if self._episode_initial_cube_z is None:
            return torch.zeros_like(cube_z)
        lift = cube_z - self._episode_initial_cube_z
        return torch.clamp(lift, min=0.0)

    # ------------------------------------------------------------------
    # Lift reward
    # ------------------------------------------------------------------

    def _compute_lift_reward(
        self,
        is_grasped: torch.Tensor,
        tcp_to_obj_dist: torch.Tensor,
    ):
        """Compute post-grasp lift rewards.

        Parameters
        ----------
        is_grasped : torch.Tensor, shape ``(num_envs,)``, bool
        tcp_to_obj_dist : torch.Tensor, shape ``(num_envs,)``

        Returns
        -------
        lift_bonus : torch.Tensor, shape ``(num_envs,)``
            Total lift reward adjustment (raw, pre-normalization).
        """
        num_envs = self.num_envs
        device = self.device
        lift_bonus = torch.zeros(num_envs, device=device)
        grasp_f = is_grasped.float()

        # ---- 1. Reduce approach reward when grasping ----
        approach_reward = 1.0 - torch.tanh(5.0 * tcp_to_obj_dist)
        approach_reduction = (
            approach_reward * grasp_f
            * (1.0 - self._approach_reward_grasped_scale)
        )
        lift_bonus -= approach_reduction

        # ---- 2. Tiny hold reward (just enough to prefer grasp over no-grasp) ----
        hold_rew = self._min_hold_reward * grasp_f
        lift_bonus += hold_rew

        # ---- 3. Continuous lift reward ----
        lift_height = self._get_cube_lift_height()
        lift_continuous = torch.clamp(
            lift_height * self._lift_reward_coef,
            max=self._lift_reward_max,
        )
        lift_bonus += lift_continuous * grasp_f

        # ---- 4. Milestone bonuses (one-time per episode) ----
        milestone = torch.zeros(num_envs, device=device)

        if self._lift_1cm_reached is not None:
            new_1cm = (lift_height > 0.01) & ~self._lift_1cm_reached & is_grasped
            milestone += new_1cm.float() * self._lift_1cm_bonus
            self._lift_1cm_reached |= new_1cm

            new_3cm = (lift_height > 0.03) & ~self._lift_3cm_reached & is_grasped
            milestone += new_3cm.float() * self._lift_3cm_bonus
            self._lift_3cm_reached |= new_3cm

            new_5cm = (lift_height > 0.05) & ~self._lift_5cm_reached & is_grasped
            milestone += new_5cm.float() * self._lift_5cm_bonus
            self._lift_5cm_reached |= new_5cm

        lift_bonus += milestone

        # ---- 5. Drop penalty ----
        # Penalize opening gripper and losing the cube after having grasped.
        gripper_width = self._get_gripper_width()
        is_gripper_open = gripper_width > self._gripper_open_threshold
        is_drop = self._was_grasped & is_gripper_open & ~is_grasped
        drop_pen = is_drop.float() * self._drop_penalty
        lift_bonus -= drop_pen

        # Update was_grasped for next step
        if self._was_grasped is not None:
            # Stay True if still grasping, clear if not grasping
            self._was_grasped = is_grasped.clone()

        # ---- Store for diagnostics ----
        self._last_lift_bonus_total = lift_bonus
        self._last_lift_continuous = lift_continuous * grasp_f
        self._last_milestone_bonus = milestone
        self._last_drop_penalty = drop_pen
        self._last_hold_reward_val = hold_rew
        self._last_approach_reduction = approach_reduction

        self._last_cube_world_z = self._get_cube_world_z()
        self._last_cube_lift_height = lift_height
        self._last_is_grasped = is_grasped
        self._last_is_drop_event = is_drop

        return lift_bonus

    # ------------------------------------------------------------------
    # Reward overrides
    # ------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """Compute dense reward with lift shaping.

        Extends the parent (Stage 2) reward by adding lift-focused
        post-grasp terms.  Target-distance transport rewards are NOT
        included.
        """
        parent_reward = super().compute_dense_reward(obs, action, info)

        is_grasped = info["is_grasped"]
        tcp_to_obj_dist = self._compute_tcp_to_cube_dist()

        lift_bonus = self._compute_lift_reward(
            is_grasped=is_grasped,
            tcp_to_obj_dist=tcp_to_obj_dist,
        )

        return parent_reward + lift_bonus

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    # ------------------------------------------------------------------
    # Lift info for logging
    # ------------------------------------------------------------------

    def get_lift_info(self):
        """Return the most recent lift state for diagnostics.

        Returns
        -------
        is_grasped : torch.Tensor or None, shape ``(num_envs,)``, bool
        is_drop_event : torch.Tensor or None, shape ``(num_envs,)``, bool
        cube_world_z : torch.Tensor or None, shape ``(num_envs,)``
        cube_lift_height : torch.Tensor or None, shape ``(num_envs,)``
        lifted_1cm : torch.Tensor or None, shape ``(num_envs,)``, bool
        lifted_3cm : torch.Tensor or None, shape ``(num_envs,)``, bool
        lifted_5cm : torch.Tensor or None, shape ``(num_envs,)``, bool
        """
        return (
            getattr(self, "_last_is_grasped", None),
            getattr(self, "_last_is_drop_event", None),
            getattr(self, "_last_cube_world_z", None),
            getattr(self, "_last_cube_lift_height", None),
            getattr(self, "_lift_1cm_reached", None),
            getattr(self, "_lift_3cm_reached", None),
            getattr(self, "_lift_5cm_reached", None),
        )

    def get_reward_components(self):
        """Return all reward components for diagnostics.

        Returns (12-element tuple, compatible with Stage 3 index layout):
        [0] original_reward, [1] collision_penalty, [2] gripper_reward,
        [3] early_close_penalty, [4] gripper_open_bonus,
        [5] lift_bonus (total), [6] lift_continuous, [7] milestone_bonus,
        [8] hold_reward, [9] drop_penalty, [10] approach_reduction,
        [11] unused (zeros)
        """
        parent_comps = super().get_reward_components()
        return (
            parent_comps[0],                                    # original_reward
            parent_comps[1],                                    # collision_penalty
            parent_comps[2],                                    # gripper_reward
            parent_comps[3],                                    # early_close_penalty
            parent_comps[4],                                    # gripper_open_bonus
            getattr(self, "_last_lift_bonus_total", None),       # lift_bonus (total) [5]
            getattr(self, "_last_lift_continuous", None),        # lift_continuous [6]
            getattr(self, "_last_milestone_bonus", None),       # milestone_bonus [7]
            getattr(self, "_last_hold_reward_val", None),        # hold_reward [8]
            getattr(self, "_last_drop_penalty", None),           # drop_penalty [9]
            getattr(self, "_last_approach_reduction", None),     # approach_reduction [10]
            torch.zeros(self.num_envs, device=self.device),     # unused [11]
        )
