"""
Custom PickCube environment with stable-grasp lift curriculum for Stage 4.5.

Registers as ``PickCubeStableLiftCurriculum-v1`` via gymnasium.

Extends ``PickCubeLiftCurriculum-v1`` with two key improvements:

1. **Initial state constraint**: cube-goal distance must be ≥ 0.10 m on reset.
   Goal positions are resampled for envs that don't satisfy this.

2. **Stable grasp**: full lift reward (continuous + milestones) only activates
   after ``is_grasped`` has been True for ≥ 3 consecutive steps.  Before that,
   only a tiny hold reward is given and no lift milestones are awarded.
   Early drops (within 1-5 steps after grasp start) incur an extra penalty.

Inherits all other logic from Stage 4 (action masking, approach reduction,
relative lift height, milestone bonuses, drop penalty).

Reward formula (raw, pre-normalization)
---------------------------------------

Pre-grasp: same as Stage 2/4.

Grasp streak < 3 (unstable)::

    reward = parent_reward - approach_reduction + min_hold_reward
           - early_drop_penalty  (if dropped within 5 steps)

Grasp streak ≥ 3 (stable)::

    lift_height = max(0, cube_z - episode_initial_cube_z)
    lift_continuous = clamp(lift_height * lift_reward_coef, max=lift_reward_max)

    reward = parent_reward - approach_reduction + min_hold_reward
           + lift_continuous
           + milestone_bonus  (1cm/3cm/5cm, one-time)
           - early_drop_penalty

Relevant YAML configuration keys
---------------------------------
- ``stable_grasp_steps`` (int, default 3)
- ``early_drop_penalty`` (float, default 0.5)
- ``early_drop_max_steps`` (int, default 5)
- ``min_initial_goal_distance`` (float, default 0.10)
Plus all Stage 4 lift keys.
"""

from typing import Any

import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from .pick_cube_lift_curriculum import PickCubeLiftCurriculumEnv


@register_env("PickCubeStableLiftCurriculum-v1", max_episode_steps=100)
class PickCubeStableLiftCurriculumEnv(PickCubeLiftCurriculumEnv):
    """PickCube with stable-grasp requirement for Stage 4.5.

    Inherits all Stage 4 lift logic and adds:
    - Minimum cube-goal distance on reset
    - Stable grasp (≥3 consecutive steps) before full lift reward
    - Early drop penalty
    """

    def __init__(self, *args, **kwargs):
        # ---- Pop stable-grasp kwargs before parent ----
        self._stable_grasp_steps = int(
            kwargs.pop("stable_grasp_steps", 3)
        )
        self._early_drop_penalty = float(
            kwargs.pop("early_drop_penalty", 0.5)
        )
        self._early_drop_max_steps = int(
            kwargs.pop("early_drop_max_steps", 5)
        )
        self._min_initial_goal_distance = float(
            kwargs.pop("min_initial_goal_distance", 0.10)
        )

        # Per-env state — MUST be set BEFORE super().__init__()
        # because _initialize_episode is called from parent chain.
        self._grasp_streak: torch.Tensor | None = None          # consecutive grasp steps
        self._steps_since_grasp_start: torch.Tensor | None = None  # steps since first grasp
        self._invalid_initial_count: int = 0

        super().__init__(*args, **kwargs)

        # ---- Per-step tracking for stable-grasp diagnostics ----
        self._last_stable_grasp: torch.Tensor | None = None
        self._last_grasp_streak: torch.Tensor | None = None
        self._last_early_drop: torch.Tensor | None = None
        self._last_is_grasped_stable: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Episode initialization — enforce min goal distance
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Extend parent to enforce minimum cube-goal distance on reset."""
        super()._initialize_episode(env_idx, options)

        num_reset = len(env_idx)
        device = self.device

        # Lazy-init stable-grasp tracking
        num_envs = self.num_envs
        if self._grasp_streak is None:
            self._grasp_streak = torch.zeros(num_envs, dtype=torch.int32, device=device)
            self._steps_since_grasp_start = torch.zeros(num_envs, dtype=torch.int32, device=device)

        # Reset grasp tracking for newly-reset envs
        self._grasp_streak[env_idx] = 0
        self._steps_since_grasp_start[env_idx] = 0

        # --- Enforce minimum cube-goal distance ---
        # The parent already set cube and goal poses.  We check each
        # newly-reset env and resample the goal if the distance is
        # too small.  We only modify env_idx envs.
        max_retries = 20
        for _ in range(max_retries):
            cube_pos = self.cube.pose.p[env_idx]       # (n_reset, 3)
            goal_pos = self.goal_site.pose.p[env_idx]   # (n_reset, 3)
            dist = torch.linalg.norm(goal_pos - cube_pos, dim=-1)  # (n_reset,)
            too_close = dist < self._min_initial_goal_distance

            if not too_close.any():
                break

            # Count invalid initial states (only for envs still too close)
            self._invalid_initial_count += int(too_close.sum().item())

            # Resample goal for too-close envs
            fix_idx = env_idx[too_close]  # subset that needs fixing
            b = len(fix_idx)
            if b == 0:
                break

            goal_xyz = torch.zeros((b, 3), device=device)
            goal_xyz[:, :2] = (
                torch.rand((b, 2), device=device)
                * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            goal_xyz[:, 0] += self.cube_spawn_center[0]
            goal_xyz[:, 1] += self.cube_spawn_center[1]
            goal_xyz[:, 2] = (
                torch.rand((b,), device=device) * self.max_goal_height
                + self.cube.pose.p[fix_idx, 2]
            )
            # Update goal position for these specific envs
            current_goal = self.goal_site.pose.raw_pose.clone()
            current_goal[fix_idx, :3] = goal_xyz
            self.goal_site.set_pose(Pose.create_from_pq(
                current_goal[:, :3], current_goal[:, 3:]
            ))

    # ------------------------------------------------------------------
    # Stable grasp logic
    # ------------------------------------------------------------------

    def _update_stable_grasp(self, is_grasped: torch.Tensor):
        """Update grasp streak and stable-grasp state.

        Called once per step, BEFORE computing rewards.

        Returns
        -------
        stable_grasp : torch.Tensor, shape ``(num_envs,)``, bool
            True where grasp streak ≥ stable_grasp_steps.
        early_drop : torch.Tensor, shape ``(num_envs,)``, bool
            True where env dropped within early_drop_max_steps of first grasp.
        """
        was_grasped = self._was_grasped  # from parent (Stage 4 lift env)

        # Update streak
        self._grasp_streak = torch.where(
            is_grasped,
            self._grasp_streak + 1,
            torch.zeros_like(self._grasp_streak),
        )

        # Update steps_since_grasp_start
        # Increment while grasping, reset to 0 when not grasping
        self._steps_since_grasp_start = torch.where(
            is_grasped,
            self._steps_since_grasp_start + 1,
            torch.zeros_like(self._steps_since_grasp_start),
        )

        stable_grasp = self._grasp_streak >= self._stable_grasp_steps

        # Early drop: was grasping, now not, and total grasp duration < max
        early_drop = (
            was_grasped
            & ~is_grasped
            & (self._steps_since_grasp_start < self._early_drop_max_steps)
        )

        # Store for diagnostics
        self._last_stable_grasp = stable_grasp
        self._last_grasp_streak = self._grasp_streak.clone()
        self._last_early_drop = early_drop
        self._last_is_grasped_stable = is_grasped & stable_grasp

        return stable_grasp, early_drop

    # ------------------------------------------------------------------
    # Lift reward — gated by stable grasp
    # ------------------------------------------------------------------

    def _compute_lift_reward(
        self,
        is_grasped: torch.Tensor,
        tcp_to_obj_dist: torch.Tensor,
    ):
        """Compute post-grasp lift rewards with stable-grasp gating.

        Before stable grasp: only tiny hold reward, no lift/milestones.
        After stable grasp: full lift reward from Stage 4.
        """
        num_envs = self.num_envs
        device = self.device

        # --- Update stable grasp state ---
        stable_grasp, early_drop = self._update_stable_grasp(is_grasped)

        grasp_f = is_grasped.float()

        # ---- 1. Reduce approach reward when grasping ----
        approach_reward = 1.0 - torch.tanh(5.0 * tcp_to_obj_dist)
        approach_reduction = (
            approach_reward * grasp_f
            * (1.0 - self._approach_reward_grasped_scale)
        )

        # ---- 2. Tiny hold reward (always, even before stable) ----
        hold_rew = self._min_hold_reward * grasp_f

        # ---- 3. Continuous lift + milestones (ONLY when stable_grasp) ----
        lift_continuous = torch.zeros(num_envs, device=device)
        milestone = torch.zeros(num_envs, device=device)

        if stable_grasp.any():
            stable_f = stable_grasp.float()
            lift_height = self._get_cube_lift_height()

            # Continuous lift (only for stable grasp)
            lift_continuous = torch.clamp(
                lift_height * self._lift_reward_coef,
                max=self._lift_reward_max,
            ) * stable_f

            # Milestones (only for stable grasp, one-time per episode)
            if self._lift_1cm_reached is not None:
                new_1cm = (
                    (lift_height > 0.01)
                    & ~self._lift_1cm_reached
                    & stable_grasp
                )
                milestone += new_1cm.float() * self._lift_1cm_bonus
                self._lift_1cm_reached |= new_1cm

                new_3cm = (
                    (lift_height > 0.03)
                    & ~self._lift_3cm_reached
                    & stable_grasp
                )
                milestone += new_3cm.float() * self._lift_3cm_bonus
                self._lift_3cm_reached |= new_3cm

                new_5cm = (
                    (lift_height > 0.05)
                    & ~self._lift_5cm_reached
                    & stable_grasp
                )
                milestone += new_5cm.float() * self._lift_5cm_bonus
                self._lift_5cm_reached |= new_5cm

        # ---- 4. Early drop penalty ----
        early_drop_pen = early_drop.float() * self._early_drop_penalty

        # ---- 5. Regular drop penalty (from Stage 4 parent logic) ----
        # We still apply the parent's drop penalty logic
        gripper_width = self._get_gripper_width()
        is_gripper_open = gripper_width > self._gripper_open_threshold
        is_drop = self._was_grasped & is_gripper_open & ~is_grasped
        drop_pen = (is_drop.float() * self._drop_penalty
                     + early_drop_pen)  # early drop adds on top

        # Update was_grasped for next step
        if self._was_grasped is not None:
            self._was_grasped = is_grasped.clone()

        # ---- Total lift bonus ----
        lift_bonus = (
            - approach_reduction
            + hold_rew
            + lift_continuous
            + milestone
            - drop_pen
        )

        # ---- Store for diagnostics ----
        self._last_lift_bonus_total = lift_bonus
        self._last_lift_continuous = lift_continuous
        self._last_milestone_bonus = milestone
        self._last_drop_penalty = drop_pen
        self._last_hold_reward_val = hold_rew
        self._last_approach_reduction = approach_reduction

        self._last_cube_world_z = self._get_cube_world_z()
        self._last_cube_lift_height = self._get_cube_lift_height()  # always track for logging
        self._last_is_grasped = is_grasped
        self._last_is_drop_event = is_drop

        return lift_bonus

    # ------------------------------------------------------------------
    # Lift info extension
    # ------------------------------------------------------------------

    def get_lift_info(self):
        """Return lift + stable-grasp state for diagnostics.

        Returns (10-tuple):
        [0] is_grasped, [1] is_drop_event, [2] cube_world_z,
        [3] cube_lift_height, [4] lifted_1cm, [5] lifted_3cm,
        [6] lifted_5cm, [7] stable_grasp, [8] grasp_streak,
        [9] early_drop
        """
        base = super().get_lift_info()
        return (
            base[0],  # is_grasped
            base[1],  # is_drop_event
            base[2],  # cube_world_z
            base[3],  # cube_lift_height
            base[4],  # lifted_1cm
            base[5],  # lifted_3cm
            base[6],  # lifted_5cm
            getattr(self, "_last_stable_grasp", None),      # [7] stable_grasp
            getattr(self, "_last_grasp_streak", None),      # [8] grasp_streak
            getattr(self, "_last_early_drop", None),         # [9] early_drop
        )

    def get_stable_lift_info(self):
        """Return stable-grasp-specific diagnostics.

        Returns
        -------
        stable_grasp : torch.Tensor or None, shape ``(num_envs,)``, bool
        grasp_streak : torch.Tensor or None, shape ``(num_envs,)``, int32
        early_drop : torch.Tensor or None, shape ``(num_envs,)``, bool
        is_grasped_stable : torch.Tensor or None, shape ``(num_envs,)``, bool
            True where both is_grasped AND stable_grasp.
        """
        return (
            getattr(self, "_last_stable_grasp", None),
            getattr(self, "_last_grasp_streak", None),
            getattr(self, "_last_early_drop", None),
            getattr(self, "_last_is_grasped_stable", None),
        )
