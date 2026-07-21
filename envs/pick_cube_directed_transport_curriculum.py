"""
Custom PickCube environment with directed smooth transport curriculum (Stage 5.5).

Registers as ``PickCubeDirectedTransportCurriculum-v1`` via gymnasium.

Extends ``PickCubeTargetTransportCurriculum-v1`` (Stage 5) with:

1. **Strengthened progress reward**: coef 2.0 → 15.0 for clear directional gradient.
2. **Directional velocity reward**: dot(cube_velocity, goal_direction) ≥ 0 rewarded.
3. **Wrong-way penalty**: negative velocity toward goal penalized harder.
4. **Stagnation penalty**: consecutive low-progress steps incur small penalty.
5. **Action smoothness penalty**: penalizes large action deltas (arm only).
6. **Lateral motion penalty**: penalizes velocity perpendicular to goal direction.
7. **Reduced stationary-lift profitability**: safe_height 0.5→0.1, lift_scale 0.3→0.05.
8. **Best distance bonus**: one-time reward when beating episode-best distance.
9. **Near-goal zone**: reduced direction/stagnation to allow fine positioning.

Phase A & B: inherited from Stage 5 (unchanged approach + lift).
Phase C: fully redesigned directed transport.

Relevant YAML keys (new / modified)
------------------------------------
target_progress_reward_coef    (float, default 15.0)   — strengthened from 2.0
target_progress_clip           (float, default 0.05)    — unchanged
direction_reward_coef          (float, default 5.0)    — NEW
wrong_way_penalty_coef         (float, default 8.0)    — NEW
direction_velocity_clip        (float, default 0.5)    — NEW
action_smoothness_coef         (float, default 0.01)   — NEW
lateral_motion_penalty_coef    (float, default 1.0)    — NEW
stagnation_progress_threshold  (float, default 0.001)  — NEW
stagnation_steps_threshold     (int,   default 5)      — NEW
stagnation_penalty_coef        (float, default 0.05)   — NEW
transport_safe_height_reward_coef  (float, default 0.1)    — reduced from 0.5
transport_lift_reward_scale    (float, default 0.05)   — reduced from 0.3
best_distance_bonus_coef       (float, default 2.0)    — NEW
best_distance_improvement_clip (float, default 0.05)   — NEW
near_goal_direction_scale      (float, default 0.3)    — NEW
near_goal_stagnation_scale     (float, default 0.2)    — NEW
near_goal_threshold            (float, default 0.05)   — unchanged
"""

from typing import Any

import torch

from mani_skill.utils.registration import register_env

from .pick_cube_target_transport_curriculum import PickCubeTargetTransportCurriculumEnv


# Panda arm joint action indices (first 7 of 8 action dims)
ARM_INDICES = [0, 1, 2, 3, 4, 5, 6]


@register_env("PickCubeDirectedTransportCurriculum-v1", max_episode_steps=100)
class PickCubeDirectedTransportCurriculumEnv(PickCubeTargetTransportCurriculumEnv):
    """PickCube with directed smooth transport curriculum for Stage 5.5.

    Inherits all Stage 5 logic (stable grasp, transport gate, distance/
    progress/bonus rewards) and adds directional signals, smoothness,
    and anti-stagnation shaping.
    """

    def __init__(self, *args, **kwargs):
        # ---- Pop ONLY Stage-5.5-exclusive kwargs before parent ----
        # (These don't exist in any parent, so we must pop them.)

        # Directional velocity
        self._direction_reward_coef = float(
            kwargs.pop("direction_reward_coef", 5.0))
        self._wrong_way_penalty_coef = float(
            kwargs.pop("wrong_way_penalty_coef", 8.0))
        self._direction_velocity_clip = float(
            kwargs.pop("direction_velocity_clip", 0.5))

        # Action smoothness
        self._action_smoothness_coef = float(
            kwargs.pop("action_smoothness_coef", 0.01))

        # Lateral motion
        self._lateral_motion_penalty_coef = float(
            kwargs.pop("lateral_motion_penalty_coef", 1.0))

        # Stagnation
        self._stagnation_progress_threshold = float(
            kwargs.pop("stagnation_progress_threshold", 0.001))
        self._stagnation_steps_threshold = int(
            kwargs.pop("stagnation_steps_threshold", 5))
        self._stagnation_penalty_coef = float(
            kwargs.pop("stagnation_penalty_coef", 0.05))

        # Best distance bonus
        self._best_distance_bonus_coef = float(
            kwargs.pop("best_distance_bonus_coef", 2.0))
        self._best_distance_improvement_clip = float(
            kwargs.pop("best_distance_improvement_clip", 0.05))

        # Near-goal adjustments
        self._near_goal_direction_scale = float(
            kwargs.pop("near_goal_direction_scale", 0.3))
        self._near_goal_stagnation_scale = float(
            kwargs.pop("near_goal_stagnation_scale", 0.2))

        # ---- Params that ALSO exist in parent but with DIFFERENT defaults ----
        # We read our defaults from kwargs (or use Stage 5.5 defaults), let the
        # parent consume them, then overwrite after super().__init__().
        _my_target_progress_coef = float(
            kwargs.get("target_progress_reward_coef", 15.0))
        _my_safe_height_coef = float(
            kwargs.get("transport_safe_height_reward_coef", 0.1))
        _my_lift_scale = float(
            kwargs.get("transport_lift_reward_scale", 0.05))

        # ---- Per-env state for Stage 5.5 (BEFORE super().__init__()) ----
        self._prev_cube_pos: torch.Tensor | None = None        # (num_envs, 3)
        self._prev_action: torch.Tensor | None = None           # (num_envs, act_dim)
        self._best_goal_dist: torch.Tensor | None = None        # (num_envs,)
        self._stagnation_steps: torch.Tensor | None = None      # (num_envs,)

        super().__init__(*args, **kwargs)

        # ---- Overwrite parent-set values with Stage 5.5 defaults ----
        self._target_progress_reward_coef = _my_target_progress_coef
        self._transport_safe_height_reward_coef = _my_safe_height_coef
        self._transport_lift_reward_scale = _my_lift_scale

        # ---- Per-step tracking for Stage 5.5 diagnostics ----
        self._last_direction_rew: torch.Tensor | None = None
        self._last_wrong_way_pen: torch.Tensor | None = None
        self._last_action_smoothness_pen: torch.Tensor | None = None
        self._last_lateral_motion_pen: torch.Tensor | None = None
        self._last_stagnation_pen: torch.Tensor | None = None
        self._last_best_dist_bonus: torch.Tensor | None = None
        self._last_directional_vel: torch.Tensor | None = None
        self._last_lateral_vel_norm: torch.Tensor | None = None
        self._last_action_delta_l2: torch.Tensor | None = None
        self._last_is_near_goal: torch.Tensor | None = None
        self._last_stagnation_active: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Episode init — extend for Stage 5.5 state
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        num_envs = self.num_envs
        device = self.device

        # Lazy-init Stage 5.5 tensors
        if self._prev_cube_pos is None:
            self._prev_cube_pos = torch.zeros(num_envs, 3, device=device)
            # action_space may not be available during super().__init__()
            # Panda has 8 action dims (7 arm joints + 1 gripper).
            try:
                act_dim = self.action_space.shape[-1]
            except AttributeError:
                act_dim = 8
            self._prev_action = torch.zeros(num_envs, act_dim, device=device)
            self._best_goal_dist = torch.zeros(num_envs, device=device)
            self._stagnation_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)

        # Reset Stage 5.5 state for newly-reset envs
        self._prev_cube_pos[env_idx] = self.cube.pose.p[env_idx].clone()
        self._prev_action[env_idx] = 0.0
        self._best_goal_dist[env_idx] = float('inf')
        self._stagnation_steps[env_idx] = 0

    # ------------------------------------------------------------------
    # Directed transport reward (overrides parent Phase C)
    # ------------------------------------------------------------------

    def _compute_transport_reward(
        self,
        is_grasped: torch.Tensor,
        stable_grasp: torch.Tensor,
        lift_height: torch.Tensor,
        success: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        """Compute Phase C directed transport rewards.

        Overrides the parent's _compute_transport_reward with strengthened
        directional signals, smoothness penalties, and anti-stagnation.
        """
        num_envs = self.num_envs
        device = self.device

        transport_gate = stable_grasp & (lift_height >= self._transport_min_lift_height)
        gate_f = transport_gate.float()

        transport_reward = torch.zeros(num_envs, device=device)

        cube_goal_dist = self._get_cube_goal_dist()

        # --- Handle first-step-of-transport ---
        gate_just_opened = transport_gate & ~self._transport_gate_prev
        self._prev_cube_goal_dist = torch.where(
            gate_just_opened,
            cube_goal_dist,
            self._prev_cube_goal_dist,
        )

        # Mark transport as started
        if self._transport_started is not None:
            self._transport_started |= transport_gate
            self._initial_transport_goal_dist = torch.where(
                gate_just_opened,
                cube_goal_dist,
                self._initial_transport_goal_dist,
            )

        # Initialize best_goal_dist when transport first starts
        if self._best_goal_dist is not None:
            self._best_goal_dist = torch.where(
                gate_just_opened,
                cube_goal_dist,
                self._best_goal_dist,
            )

        # First-step prev_cube_pos reset
        if self._prev_cube_pos is not None:
            self._prev_cube_pos = torch.where(
                gate_just_opened.unsqueeze(-1),
                self.cube.pose.p,
                self._prev_cube_pos,
            )

        # First-step prev_action reset (no penalty on first transport step)
        # We don't flag here; action_smoothness sets prev_action = action when gate_just_opened.

        # Near-goal flag (used for scaling)
        is_near_goal = (cube_goal_dist <= self._near_goal_threshold) & transport_gate
        near_f = is_near_goal.float()

        # ================================================================
        # 1. Target distance reward (unchanged from Stage 5)
        # ================================================================
        target_dist_rew = (
            1.0 - torch.tanh(self._target_distance_scale * cube_goal_dist)
        ) * self._target_distance_reward_coef * gate_f
        transport_reward += target_dist_rew

        # ================================================================
        # 2. Strengthened target progress reward
        # ================================================================
        progress = self._prev_cube_goal_dist - cube_goal_dist
        progress = torch.clamp(progress, -self._target_progress_clip, self._target_progress_clip)
        target_progress_rew = progress * self._target_progress_reward_coef * gate_f
        transport_reward += target_progress_rew

        # Update prev_dist for next step
        self._prev_cube_goal_dist = cube_goal_dist.clone()

        # ================================================================
        # 3. Directional velocity reward (NEW)
        # ================================================================
        cube_pos = self.cube.pose.p  # (num_envs, 3)
        goal_pos = self.goal_site.pose.p  # (num_envs, 3)
        goal_vector = goal_pos - cube_pos  # (num_envs, 3)
        goal_dist_norm = torch.linalg.norm(goal_vector, dim=-1, keepdim=True).clamp(min=1e-6)
        goal_direction = goal_vector / goal_dist_norm  # (num_envs, 3)

        # Use simulator's true linear velocity
        cube_vel = self.cube.linear_velocity  # (num_envs, 3)

        directional_vel = (cube_vel * goal_direction).sum(dim=-1)  # (num_envs,)

        # Direction reward: velocity toward goal, clamped
        direction_rew = (
            torch.clamp(directional_vel, min=0.0, max=self._direction_velocity_clip)
            * self._direction_reward_coef
            * gate_f
        )
        # Scale down near goal to avoid overshooting
        direction_rew = direction_rew * (1.0 - near_f * (1.0 - self._near_goal_direction_scale))
        transport_reward += direction_rew

        # Wrong-way penalty: velocity away from goal
        wrong_way_pen = (
            torch.clamp(-directional_vel, min=0.0, max=self._direction_velocity_clip)
            * self._wrong_way_penalty_coef
            * gate_f
        )
        transport_reward -= wrong_way_pen

        # ================================================================
        # 4. Lateral motion penalty (NEW)
        # ================================================================
        parallel_vel = directional_vel.unsqueeze(-1) * goal_direction  # (num_envs, 3)
        lateral_vel = cube_vel - parallel_vel  # (num_envs, 3)
        lateral_vel_norm = torch.linalg.norm(lateral_vel, dim=-1)  # (num_envs,)
        lateral_motion_pen = (
            lateral_vel_norm
            * self._lateral_motion_penalty_coef
            * gate_f
        )
        transport_reward -= lateral_motion_pen

        # ================================================================
        # 5. Safe-height reward (reduced from Stage 5: 0.5→0.1)
        # ================================================================
        safe_margin = max(lift_height.max().item(), 0.03)
        safe_height_rew = torch.clamp(
            (lift_height - self._transport_min_lift_height) / max(safe_margin, 0.001),
            0.0, 1.0,
        ) * self._transport_safe_height_reward_coef * gate_f
        transport_reward += safe_height_rew

        # ================================================================
        # 6. Height drop below threshold penalty
        # ================================================================
        height_drop = torch.clamp(
            self._transport_min_lift_height - lift_height, min=0.0)
        height_drop_pen = height_drop * self._transport_height_drop_penalty_coef * gate_f
        transport_reward -= height_drop_pen

        # ================================================================
        # 7. Transport drop penalty (unchanged)
        # ================================================================
        gripper_width = self._get_gripper_width()
        is_gripper_open = gripper_width > self._gripper_open_threshold
        transport_drop_event = (
            self._was_grasped & is_gripper_open & ~is_grasped & self._transport_gate_prev
        )
        transport_drop_pen = transport_drop_event.float() * self._transport_drop_penalty
        transport_reward -= transport_drop_pen

        # ================================================================
        # 8. Further-reduced lift during transport (0.3→0.05)
        # ================================================================
        lift_continuous_parent = torch.clamp(
            lift_height * self._lift_reward_coef, max=self._lift_reward_max)
        scaled_lift = lift_continuous_parent * self._transport_lift_reward_scale * gate_f
        lift_adjustment = scaled_lift - lift_continuous_parent * gate_f
        transport_reward += lift_adjustment

        # ================================================================
        # 9. Stagnation penalty (NEW)
        # ================================================================
        if self._stagnation_steps is not None:
            is_stagnating = (
                (progress.abs() < self._stagnation_progress_threshold)
                & transport_gate
            )
            self._stagnation_steps = torch.where(
                is_stagnating,
                self._stagnation_steps + 1,
                torch.zeros_like(self._stagnation_steps),
            )
            stagnation_active = (
                (self._stagnation_steps >= self._stagnation_steps_threshold)
                & transport_gate
            )
            # Reduce stagnation penalty near goal
            stagnation_pen = (
                stagnation_active.float()
                * self._stagnation_penalty_coef
                * (1.0 - near_f * (1.0 - self._near_goal_stagnation_scale))
            )
            transport_reward -= stagnation_pen
        else:
            stagnation_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
            stagnation_pen = torch.zeros(num_envs, device=device)

        # ================================================================
        # 10. Action smoothness penalty (NEW)
        # ================================================================
        if self._prev_action is not None and action is not None:
            act_dim = action.shape[-1]
            # Only penalize arm joint actions (indices 0-6), not gripper
            action_delta = action[:, :7] - self._prev_action[:, :7]
            # On first transport step, zero out delta to avoid spike
            action_delta = torch.where(
                gate_just_opened.unsqueeze(-1),
                torch.zeros_like(action_delta),
                action_delta,
            )
            action_delta_l2 = (action_delta ** 2).mean(dim=-1)  # mean over arm dims
            action_smoothness_pen = (
                action_delta_l2
                * self._action_smoothness_coef
                * gate_f
            )
            transport_reward -= action_smoothness_pen
            # Update prev_action (always, not just during transport)
            self._prev_action = action.clone()
        else:
            action_delta_l2 = torch.zeros(num_envs, device=device)
            action_smoothness_pen = torch.zeros(num_envs, device=device)
            # First call: store current action for next step
            if action is not None and self._prev_action is not None:
                self._prev_action = action.clone()

        # ================================================================
        # 11. Best distance bonus (NEW)
        # ================================================================
        if self._best_goal_dist is not None:
            improved = cube_goal_dist < self._best_goal_dist
            improvement = torch.clamp(
                self._best_goal_dist - cube_goal_dist,
                min=0.0,
                max=self._best_distance_improvement_clip,
            )
            best_dist_bonus = (
                improvement
                * self._best_distance_bonus_coef
                * improved.float()
                * gate_f
            )
            transport_reward += best_dist_bonus
            # Update best
            self._best_goal_dist = torch.where(
                improved & transport_gate,
                cube_goal_dist,
                self._best_goal_dist,
            )
        else:
            best_dist_bonus = torch.zeros(num_envs, device=device)

        # ================================================================
        # 12. One-time bonuses (unchanged from Stage 5)
        # ================================================================
        if self._near_goal_reached is not None:
            new_near = (
                (cube_goal_dist <= self._near_goal_threshold)
                & ~self._near_goal_reached
                & transport_gate
            )
            transport_reward += new_near.float() * self._near_goal_bonus
            self._near_goal_reached |= new_near

        if self._placed_reached is not None:
            is_placed = cube_goal_dist <= self.goal_thresh
            new_placed = is_placed & ~self._placed_reached
            transport_reward += new_placed.float() * self._placed_bonus
            self._placed_reached |= new_placed

        transport_reward += success.float() * self._success_reward_bonus

        # ================================================================
        # Update transport_gate_prev
        # ================================================================
        self._transport_gate_prev = transport_gate.clone()

        # ---- Store for diagnostics ----
        self._last_transport_gate = transport_gate
        self._last_transport_bonus = transport_reward
        self._last_target_distance_rew = target_dist_rew
        self._last_target_progress_rew = target_progress_rew
        self._last_safe_height_rew = safe_height_rew
        self._last_height_drop_pen = height_drop_pen
        self._last_transport_drop_pen = transport_drop_pen
        self._last_near_goal_bonus = new_near.float() * self._near_goal_bonus if self._near_goal_reached is not None else torch.zeros(num_envs, device=device)
        self._last_placed_bonus = new_placed.float() * self._placed_bonus if self._placed_reached is not None else torch.zeros(num_envs, device=device)
        self._last_success_bonus_st5 = success.float() * self._success_reward_bonus
        self._last_cube_goal_dist = cube_goal_dist
        self._last_is_transport_drop = transport_drop_event

        # Stage 5.5 new diagnostics
        self._last_direction_rew = direction_rew
        self._last_wrong_way_pen = wrong_way_pen
        self._last_action_smoothness_pen = action_smoothness_pen
        self._last_lateral_motion_pen = lateral_motion_pen
        self._last_stagnation_pen = stagnation_pen
        self._last_best_dist_bonus = best_dist_bonus
        self._last_directional_vel = directional_vel
        self._last_lateral_vel_norm = lateral_vel_norm
        self._last_action_delta_l2 = action_delta_l2
        self._last_is_near_goal = is_near_goal
        self._last_stagnation_active = stagnation_active

        return transport_reward

    # ------------------------------------------------------------------
    # Main reward (override to pass action to transport)
    # ------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """Three-phase reward with directed transport in Phase C."""
        parent_reward = super(
            PickCubeTargetTransportCurriculumEnv, self
        ).compute_dense_reward(obs, action, info)

        is_grasped = info["is_grasped"]
        success = info.get("success",
                           torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))

        stable_grasp = getattr(self, "_last_stable_grasp",
                               torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        lift_height = getattr(self, "_last_cube_lift_height",
                              torch.zeros(self.num_envs, device=self.device))

        transport_reward = self._compute_transport_reward(
            is_grasped=is_grasped,
            stable_grasp=stable_grasp,
            lift_height=lift_height,
            success=success,
            action=action,
        )

        return parent_reward + transport_reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    # ------------------------------------------------------------------
    # Extended transport info for logging
    # ------------------------------------------------------------------

    def get_transport_info(self):
        """Return extended transport state (14-tuple, compat with Stage 5's 8).

        [0-7]: Same as Stage 5
        [8]:  direction_reward
        [9]:  wrong_way_penalty
        [10]: action_smoothness_penalty
        [11]: stagnation_penalty
        [12]: best_distance_bonus
        [13]: lateral_motion_penalty
        """
        base = super().get_transport_info()
        return (
            base[0], base[1], base[2], base[3], base[4],
            base[5], base[6], base[7],
            getattr(self, "_last_direction_rew", None),          # [8]
            getattr(self, "_last_wrong_way_pen", None),          # [9]
            getattr(self, "_last_action_smoothness_pen", None),  # [10]
            getattr(self, "_last_stagnation_pen", None),          # [11]
            getattr(self, "_last_best_dist_bonus", None),         # [12]
            getattr(self, "_last_lateral_motion_pen", None),      # [13]
        )

    def get_directed_transport_info(self):
        """Return Stage 5.5-specific diagnostics (12-tuple).

        [0]  directional_velocity
        [1]  lateral_velocity_norm
        [2]  action_delta_l2_mean
        [3]  is_near_goal
        [4]  stagnation_active
        [5]  stagnation_steps
        [6]  best_goal_dist
        [7]  direction_reward
        [8]  wrong_way_penalty
        [9]  action_smoothness_penalty
        [10] lateral_motion_penalty
        [11] best_distance_bonus
        """
        return (
            getattr(self, "_last_directional_vel", None),        # [0]
            getattr(self, "_last_lateral_vel_norm", None),       # [1]
            getattr(self, "_last_action_delta_l2", None),        # [2]
            getattr(self, "_last_is_near_goal", None),            # [3]
            getattr(self, "_last_stagnation_active", None),       # [4]
            getattr(self, "_stagnation_steps", None),             # [5]
            getattr(self, "_best_goal_dist", None),               # [6]
            getattr(self, "_last_direction_rew", None),           # [7]
            getattr(self, "_last_wrong_way_pen", None),           # [8]
            getattr(self, "_last_action_smoothness_pen", None),   # [9]
            getattr(self, "_last_lateral_motion_pen", None),      # [10]
            getattr(self, "_last_best_dist_bonus", None),         # [11]
        )

    def get_reward_components(self):
        """Return all reward components (22-tuple, compat with Stage 5's 16).

        [0-15]: Same as Stage 5
        [16]: direction_reward
        [17]: wrong_way_penalty
        [18]: action_smoothness_penalty
        [19]: lateral_motion_penalty
        [20]: stagnation_penalty
        [21]: best_distance_bonus
        """
        parent = super().get_reward_components()
        return (
            parent[0], parent[1], parent[2], parent[3], parent[4],
            parent[5], parent[6], parent[7], parent[8], parent[9],
            parent[10], parent[11], parent[12], parent[13], parent[14],
            parent[15],
            getattr(self, "_last_direction_rew", None),           # [16]
            getattr(self, "_last_wrong_way_pen", None),           # [17]
            getattr(self, "_last_action_smoothness_pen", None),   # [18]
            getattr(self, "_last_lateral_motion_pen", None),      # [19]
            getattr(self, "_last_stagnation_pen", None),          # [20]
            getattr(self, "_last_best_dist_bonus", None),         # [21]
        )
