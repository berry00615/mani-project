"""
Custom PickCube environment with target transport curriculum for Stage 5.

Registers as ``PickCubeTargetTransportCurriculum-v1`` via gymnasium.

Extends ``PickCubeStableLiftCurriculum-v1`` with three-phase transport:

Phase A — No stable grasp (grasp_streak < 3):
    Stage 4.5 approach + gripper mask + tiny hold. No lift/transport.

Phase B — Stable grasp but lift < 2 cm:
    Stage 4.5 lift rewards only. No transport rewards.

Phase C — Transport gate: stable_grasp AND lift_height >= 2 cm:
    Target distance reward + progress reward + safe-height + transport drop penalty
    + near_goal / placed / success one-time bonuses.

Key anti-hacking safeguards:
- Transport rewards gated by stable_grasp AND lift_height >= min_lift.
- Progress uses prev_dist - current_dist, clamped. First transport step has
  prev_dist = current_dist (no spike).
- prev_cube_goal_dist tracked per env; reset on episode boundaries.
- near_goal/placed/success bonuses are one-time per episode.
- Height drop below threshold while transporting incurs per-step penalty.

Relevant YAML keys
-------------------
transport_min_lift_height      (float, default 0.02)
target_distance_scale          (float, default 5.0)
target_distance_reward_coef    (float, default 1.0)
target_progress_reward_coef    (float, default 2.0)
target_progress_clip           (float, default 0.05)
transport_hold_reward          (float, default 0.02)
transport_safe_height_reward_coef  (float, default 0.5)
transport_height_drop_penalty_coef (float, default 5.0)
transport_drop_penalty         (float, default 3.0)
near_goal_threshold            (float, default 0.05)
near_goal_bonus                (float, default 1.0)
placed_bonus                   (float, default 2.0)
success_reward_bonus           (float, default 3.0)
transport_lift_reward_scale    (float, default 0.3)
"""

from typing import Any

import torch

from mani_skill.utils.registration import register_env

from .pick_cube_stable_lift_curriculum import PickCubeStableLiftCurriculumEnv


@register_env("PickCubeTargetTransportCurriculum-v1", max_episode_steps=100)
class PickCubeTargetTransportCurriculumEnv(PickCubeStableLiftCurriculumEnv):
    """PickCube with target transport curriculum for Stage 5.

    Inherits all Stage 4.5 logic (stable grasp, lift milestones, min goal
    distance) and adds three-phase transport rewards on top.
    """

    def __init__(self, *args, **kwargs):
        # ---- Pop transport kwargs before parent ----
        self._transport_min_lift_height = float(
            kwargs.pop("transport_min_lift_height", 0.02))
        self._target_distance_scale = float(
            kwargs.pop("target_distance_scale", 5.0))
        self._target_distance_reward_coef = float(
            kwargs.pop("target_distance_reward_coef", 1.0))
        self._target_progress_reward_coef = float(
            kwargs.pop("target_progress_reward_coef", 2.0))
        self._target_progress_clip = float(
            kwargs.pop("target_progress_clip", 0.05))
        self._transport_hold_reward = float(
            kwargs.pop("transport_hold_reward", 0.02))
        self._transport_safe_height_reward_coef = float(
            kwargs.pop("transport_safe_height_reward_coef", 0.5))
        self._transport_height_drop_penalty_coef = float(
            kwargs.pop("transport_height_drop_penalty_coef", 5.0))
        self._transport_drop_penalty = float(
            kwargs.pop("transport_drop_penalty", 3.0))
        self._near_goal_threshold = float(
            kwargs.pop("near_goal_threshold", 0.05))
        self._near_goal_bonus = float(
            kwargs.pop("near_goal_bonus", 1.0))
        self._placed_bonus = float(
            kwargs.pop("placed_bonus", 2.0))
        self._success_reward_bonus = float(
            kwargs.pop("success_reward_bonus", 3.0))
        self._transport_lift_reward_scale = float(
            kwargs.pop("transport_lift_reward_scale", 0.3))

        # Per-env transport state — BEFORE super().__init__()
        self._prev_cube_goal_dist: torch.Tensor | None = None
        self._transport_started: torch.Tensor | None = None
        self._initial_transport_goal_dist: torch.Tensor | None = None
        self._near_goal_reached: torch.Tensor | None = None
        self._placed_reached: torch.Tensor | None = None
        self._transport_gate_prev: torch.Tensor | None = None

        super().__init__(*args, **kwargs)

        # ---- Per-step tracking ----
        self._last_transport_gate: torch.Tensor | None = None
        self._last_transport_bonus: torch.Tensor | None = None
        self._last_target_distance_rew: torch.Tensor | None = None
        self._last_target_progress_rew: torch.Tensor | None = None
        self._last_safe_height_rew: torch.Tensor | None = None
        self._last_height_drop_pen: torch.Tensor | None = None
        self._last_transport_drop_pen: torch.Tensor | None = None
        self._last_near_goal_bonus: torch.Tensor | None = None
        self._last_placed_bonus: torch.Tensor | None = None
        self._last_success_bonus_st5: torch.Tensor | None = None
        self._last_cube_goal_dist: torch.Tensor | None = None
        self._last_is_transport_drop: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Episode init — extend for transport state
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        num_envs = self.num_envs
        device = self.device

        # Lazy-init transport tensors
        if self._prev_cube_goal_dist is None:
            self._prev_cube_goal_dist = torch.zeros(num_envs, device=device)
            self._transport_started = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self._initial_transport_goal_dist = torch.zeros(num_envs, device=device)
            self._near_goal_reached = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self._placed_reached = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self._transport_gate_prev = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Reset transport state for newly-reset envs
        current_dist = torch.linalg.norm(
            self.goal_site.pose.p[env_idx] - self.cube.pose.p[env_idx], dim=-1)
        self._prev_cube_goal_dist[env_idx] = current_dist
        self._transport_started[env_idx] = False
        self._initial_transport_goal_dist[env_idx] = 0.0
        self._near_goal_reached[env_idx] = False
        self._placed_reached[env_idx] = False
        self._transport_gate_prev[env_idx] = False

    # ------------------------------------------------------------------
    # Transport helpers
    # ------------------------------------------------------------------

    def _get_cube_goal_dist(self) -> torch.Tensor:
        return torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, dim=-1)

    # ------------------------------------------------------------------
    # Transport reward
    # ------------------------------------------------------------------

    def _compute_transport_reward(
        self,
        is_grasped: torch.Tensor,
        stable_grasp: torch.Tensor,
        lift_height: torch.Tensor,
        success: torch.Tensor,
    ):
        """Compute Phase C transport rewards.

        Only active when transport_gate = stable_grasp AND lift >= min.
        """
        num_envs = self.num_envs
        device = self.device

        transport_gate = stable_grasp & (lift_height >= self._transport_min_lift_height)
        gate_f = transport_gate.float()

        transport_reward = torch.zeros(num_envs, device=device)

        cube_goal_dist = self._get_cube_goal_dist()

        # --- Handle first-step-of-transport for progress ---
        # When transport gate first becomes active, set prev_dist = current_dist
        # to avoid a spike from the distance change during approach.
        gate_just_opened = transport_gate & ~self._transport_gate_prev
        self._prev_cube_goal_dist = torch.where(
            gate_just_opened,
            cube_goal_dist,
            self._prev_cube_goal_dist,
        )

        # Mark transport as started (one-time flag for diagnostics)
        if self._transport_started is not None:
            self._transport_started |= transport_gate
            # Record initial distance when transport first activates
            self._initial_transport_goal_dist = torch.where(
                gate_just_opened,
                cube_goal_dist,
                self._initial_transport_goal_dist,
            )

        # ---- 1. Target distance reward ----
        target_dist_rew = (
            1.0 - torch.tanh(self._target_distance_scale * cube_goal_dist)
        ) * self._target_distance_reward_coef * gate_f
        transport_reward += target_dist_rew

        # ---- 2. Target progress reward ----
        progress = self._prev_cube_goal_dist - cube_goal_dist
        progress = torch.clamp(progress, -self._target_progress_clip, self._target_progress_clip)
        target_progress_rew = progress * self._target_progress_reward_coef * gate_f
        transport_reward += target_progress_rew

        # Update prev_dist for next step (always, even without gate)
        self._prev_cube_goal_dist = cube_goal_dist.clone()

        # ---- 3. Safe-height reward ----
        # Bonus for staying above min lift height during transport.
        safe_margin = max(lift_height.max().item(), 0.03)  # for numerical stability
        safe_height_rew = torch.clamp(
            (lift_height - self._transport_min_lift_height) / max(safe_margin, 0.001),
            0.0, 1.0,
        ) * self._transport_safe_height_reward_coef * gate_f
        transport_reward += safe_height_rew

        # ---- 4. Height drop below threshold penalty ----
        height_drop = torch.clamp(
            self._transport_min_lift_height - lift_height, min=0.0)
        height_drop_pen = height_drop * self._transport_height_drop_penalty_coef * gate_f
        transport_reward -= height_drop_pen

        # ---- 5. Transport drop penalty ----
        gripper_width = self._get_gripper_width()
        is_gripper_open = gripper_width > self._gripper_open_threshold
        transport_drop_event = (
            self._was_grasped & is_gripper_open & ~is_grasped & self._transport_gate_prev
        )
        transport_drop_pen = transport_drop_event.float() * self._transport_drop_penalty
        transport_reward -= transport_drop_pen

        # ---- 6. Scale down lift during transport ----
        # The parent lift_continuous is still computed, we subtract the unscaled
        # portion here (handled in compute_dense_reward).
        # We add back the scaled version.
        lift_continuous_parent = torch.clamp(
            lift_height * self._lift_reward_coef, max=self._lift_reward_max)
        scaled_lift = lift_continuous_parent * self._transport_lift_reward_scale * gate_f
        # Subtract full lift (it was added in parent) and add scaled version
        lift_adjustment = scaled_lift - lift_continuous_parent * gate_f
        transport_reward += lift_adjustment

        # ---- 7. One-time bonuses ----
        # near_goal: cube within near_goal_threshold and transport active
        if self._near_goal_reached is not None:
            new_near = (
                (cube_goal_dist <= self._near_goal_threshold)
                & ~self._near_goal_reached
                & transport_gate
            )
            transport_reward += new_near.float() * self._near_goal_bonus
            self._near_goal_reached |= new_near

        # placed: cube within goal_thresh
        if self._placed_reached is not None:
            is_placed = cube_goal_dist <= self.goal_thresh
            new_placed = is_placed & ~self._placed_reached
            transport_reward += new_placed.float() * self._placed_bonus
            self._placed_reached |= new_placed

        # success: placed + robot static (additive, after base reward[success]=5)
        transport_reward += success.float() * self._success_reward_bonus

        # ---- Update transport_gate_prev ----
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

        return transport_reward

    # ------------------------------------------------------------------
    # Main reward
    # ------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """Three-phase reward with transport gating.

        Phase A (no stable grasp): Stage 4.5
        Phase B (stable, lift < 2cm): Stage 4.5 lift
        Phase C (stable, lift >= 2cm): transport rewards
        """
        # Parent reward (Stage 4.5: approach + lift + milestones + drops)
        parent_reward = super().compute_dense_reward(obs, action, info)

        is_grasped = info["is_grasped"]
        success = info.get("success",
                           torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))

        # Get stable_grasp and lift_height from parent state
        stable_grasp = getattr(self, "_last_stable_grasp",
                               torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        lift_height = getattr(self, "_last_cube_lift_height",
                              torch.zeros(self.num_envs, device=self.device))

        transport_reward = self._compute_transport_reward(
            is_grasped=is_grasped,
            stable_grasp=stable_grasp,
            lift_height=lift_height,
            success=success,
        )

        return parent_reward + transport_reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    # ------------------------------------------------------------------
    # Info for logging
    # ------------------------------------------------------------------

    def get_transport_info(self):
        """Return transport state for diagnostics (8-tuple).

        [0] transport_gate, [1] transport_started, [2] cube_goal_dist,
        [3] initial_transport_goal_dist, [4] target_distance_reward,
        [5] target_progress_reward, [6] transport_drop_event,
        [7] near_goal_reached
        """
        return (
            getattr(self, "_last_transport_gate", None),
            getattr(self, "_transport_started", None),
            getattr(self, "_last_cube_goal_dist", None),
            getattr(self, "_initial_transport_goal_dist", None),
            getattr(self, "_last_target_distance_rew", None),
            getattr(self, "_last_target_progress_rew", None),
            getattr(self, "_last_is_transport_drop", None),
            getattr(self, "_near_goal_reached", None),
        )

    def get_reward_components(self):
        """Return all components (16-tuple, compatible with earlier stages).

        [0-11]: Stage 4.5 components
        [12]: transport_bonus (total)
        [13]: target_distance_reward
        [14]: target_progress_reward
        [15]: safe_height_reward
        """
        parent = super().get_reward_components()
        return (
            parent[0], parent[1], parent[2], parent[3], parent[4],
            parent[5], parent[6], parent[7], parent[8], parent[9],
            parent[10], parent[11],
            getattr(self, "_last_transport_bonus", None),          # [12]
            getattr(self, "_last_target_distance_rew", None),      # [13]
            getattr(self, "_last_target_progress_rew", None),      # [14]
            getattr(self, "_last_safe_height_rew", None),          # [15]
        )
