"""
Custom PickCube environment with center-precision carry for Stage 6.7.

Registers as ``PickCubeCenterPrecisionCarry-v1`` via gymnasium.

Extends ``PickCubePostureStableCarry-v1`` (Stage 6.6) with:

1. **Exponential center reward**: replaces flat tanh distance reward with
   ``center_reward_scale * exp(-goal_distance / center_reward_sigma)``,
   creating a sharp gradient that strongly prefers the goal center.

2. **Center progress reward**: replaces the old flat progress reward with
   ``center_progress_reward_scale * clamp(prev_dist - curr_dist, ±0.01)``,
   a high-coefficient potential-based reward whose cumulative sum telescopes
   to ``(init_dist - final_dist) * scale`` — inherently bounded and
   unfarmable.

3. **Near-goal stall penalty**: detects consecutive steps with negligible
   progress at 2.5-5cm and applies a per-step penalty after a grace period.

4. **Success terminal reward**: success triggers
   ``base_success_reward + fast_success_bonus * (1 - elapsed/max_steps)``,
   making quick success strictly better than timing out near the goal.

5. **Relaxed action scaling**: arm minimum scale 0.35 (was 0.25), wrist 0.10
   (was 0.08) — lets the policy complete the final 1-3cm approach.

6. **Fixed action indexing**: all ``action[... i]`` use ellipsis indexing,
   compatible with both single-env (8,) and batch (N,8) action shapes.

All Stage 6.6 features retained: posture regularization, goal dwell, joint
limits, near-goal speed penalties, elbow fold detection, action scaling.

Relevant YAML keys (new / modified for Stage 6.7)
-------------------------------------------------

Center reward:
  center_reward_scale                 (float, default 8.0)
  center_reward_sigma                 (float, default 0.025)

Center progress:
  center_progress_reward_scale        (float, default 300.0)
  center_progress_clip_min            (float, default -0.01)
  center_progress_clip_max            (float, default 0.01)

Near-goal stall:
  near_goal_stall_distance            (float, default 0.05)
  near_goal_stall_progress_threshold  (float, default 0.0005)
  near_goal_stall_steps_threshold     (int,   default 3)
  near_goal_stall_penalty_coef        (float, default 0.15)

Success terminal:
  base_success_reward                 (float, default 50.0)
  fast_success_bonus                  (float, default 30.0)

Zeroed-out old rewards:
  target_distance_reward_coef: 0.0    (replaced by center_reward)
  far_progress_reward_coef: 0.0       (replaced by center_progress_reward)
  success_reward_bonus: 0.0           (replaced by success_terminal_reward)

Relaxed action scaling:
  arm_delta_scale_min: 0.35           (was 0.25)
  wrist_delta_scale_min: 0.10         (was 0.08)
"""

from typing import Any

import torch

from mani_skill.utils.registration import register_env

from .pick_cube_posture_stable_carry import (
    PickCubePostureStableCarryEnv,
    WRIST_JOINT_INDICES,
)

# Panda arm joint indices
ARM_INDICES = [0, 1, 2, 3, 4, 5, 6]
GRIPPER_INDEX = 7


@register_env("PickCubeCenterPrecisionCarry-v1", max_episode_steps=150)
class PickCubeCenterPrecisionCarryEnv(PickCubePostureStableCarryEnv):
    """PickCube with center-precision carry for Stage 6.7.

    Inherits all Stage 6.6 logic (posture, action scaling, goal dwell, joint
    limits, speed penalties, elbow fold) and adds:

    - Exponential center reward (sharp gradient near goal)
    - Center progress reward (potential-based, bounded cumulative)
    - Near-goal stall penalty (punishes stationary farming)
    - Success terminal reward (quick success dominates farming)
    - Relaxed action scaling (lets final approach complete)
    - Fixed action indexing (ellipsis for single/batch compat)
    """

    # ------------------------------------------------------------------
    # Init — pop Stage 6.7 kwargs before parent
    # ------------------------------------------------------------------

    def __init__(self, *args, **kwargs):
        # ---- Center reward ----
        self._center_reward_scale = float(
            kwargs.pop("center_reward_scale", 8.0))
        self._center_reward_sigma = float(
            kwargs.pop("center_reward_sigma", 0.025))

        # ---- Center progress ----
        self._center_progress_reward_scale = float(
            kwargs.pop("center_progress_reward_scale", 300.0))
        self._center_progress_clip_min = float(
            kwargs.pop("center_progress_clip_min", -0.01))
        self._center_progress_clip_max = float(
            kwargs.pop("center_progress_clip_max", 0.01))

        # ---- Near-goal stall ----
        self._near_goal_stall_distance = float(
            kwargs.pop("near_goal_stall_distance", 0.05))
        self._near_goal_stall_progress_threshold = float(
            kwargs.pop("near_goal_stall_progress_threshold", 0.0005))
        self._near_goal_stall_steps_threshold = int(
            kwargs.pop("near_goal_stall_steps_threshold", 3))
        self._near_goal_stall_penalty_coef = float(
            kwargs.pop("near_goal_stall_penalty_coef", 0.15))

        # ---- Success terminal ----
        self._base_success_reward = float(
            kwargs.pop("base_success_reward", 50.0))
        self._fast_success_bonus = float(
            kwargs.pop("fast_success_bonus", 30.0))

        # ---- Per-env state for Stage 6.7 ----
        self._near_goal_stall_steps: torch.Tensor | None = None  # (N,) int32
        self._last_elapsed_steps_s67: torch.Tensor | None = None  # (N,) int32

        super().__init__(*args, **kwargs)

        # ---- Stage 6.7 per-step diagnostic storage ----
        self._last_s67_center_reward: torch.Tensor | None = None
        self._last_s67_center_progress_reward: torch.Tensor | None = None
        self._last_s67_center_progress: torch.Tensor | None = None
        self._last_s67_near_goal_stall_pen: torch.Tensor | None = None
        self._last_s67_near_goal_stall_active: torch.Tensor | None = None
        self._last_s67_success_terminal_reward: torch.Tensor | None = None
        self._last_s67_fast_success_bonus_amt: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Episode init — extend for Stage 6.7 state
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        num_envs = self.num_envs
        device = self.device

        # Lazy-init Stage 6.7 tensors
        if self._near_goal_stall_steps is None:
            self._near_goal_stall_steps = torch.zeros(
                num_envs, dtype=torch.int32, device=device)

        # Reset Stage 6.7 state for newly-reset envs
        self._near_goal_stall_steps[env_idx] = 0

    # ------------------------------------------------------------------
    # Fixed action scaling — use [..., i] for single/batch compat
    # ------------------------------------------------------------------

    def _apply_action_scaling(
        self, action: torch.Tensor,
        arm_scale: torch.Tensor,
        wrist_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Scale arm joint deltas using ellipsis indexing.

        Compatible with both single-env action=(8,) and batch action=(N,8).
        """
        action_scaled = action.clone()
        arm_scale = arm_scale.to(action.device)
        wrist_scale = wrist_scale.to(action.device)

        for i in range(7):
            if i in WRIST_JOINT_INDICES:
                action_scaled[..., i] = action[..., i] * wrist_scale
            else:
                action_scaled[..., i] = action[..., i] * arm_scale

        # Gripper (index 7) — unchanged
        return action_scaled

    # ------------------------------------------------------------------
    # Transport reward override — add Stage 6.7 center-precision rewards
    # ------------------------------------------------------------------

    def _compute_transport_reward(
        self,
        is_grasped: torch.Tensor,
        stable_grasp: torch.Tensor,
        lift_height: torch.Tensor,
        success: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        """Add Stage 6.7 center-precision rewards on top of Stage 6.6.

        Super() call returns Stage 6.6 transport reward which includes:
        - Stage 6.5: all transport components (target_distance=0 from config,
          progress_reward=0 from config, speed_band, premature_stop,
          precision_dist, near_motion, placed zone rewards, pose constraints,
          joint limits, clearance, dragging, one-time bonuses,
          success_bonus=0 from config)
        - Stage 6.6: posture, near-goal speed penalties, goal dwell, goal
          exit, elbow fold

        Stage 6.7 adds:
        - Exponential center reward (replaces zeroed target_distance)
        - Center progress reward (replaces zeroed progress reward)
        - Near-goal stall penalty
        - Success terminal reward (replaces zeroed success_bonus)
        """
        num_envs = self.num_envs
        device = self.device

        transport_gate = stable_grasp & (
            lift_height >= self._transport_min_lift_height)
        gate_f = transport_gate.float()

        cube_goal_dist = self._get_cube_goal_dist()

        # ---- Capture prev distance BEFORE super() updates it ----
        # Stage 6.5's _compute_transport_reward updates _prev_cube_goal_dist
        # to the current value. We must save the old value before that happens
        # so we can compute center_progress correctly.
        _prev_dist_before = getattr(
            self, "_prev_cube_goal_dist",
            cube_goal_dist.clone())

        # Gate-just-opened: first transport step has no prior distance.
        # Force _prev_dist_before = cube_goal_dist so progress=0 for them.
        _transport_gate_prev_before = getattr(
            self, "_transport_gate_prev",
            torch.zeros(num_envs, dtype=torch.bool, device=device))
        gate_just_opened_s67 = transport_gate & ~_transport_gate_prev_before
        _prev_dist_before = torch.where(
            gate_just_opened_s67,
            cube_goal_dist,
            _prev_dist_before,
        )

        # ---- Call Stage 6.6 parent to get ALL existing rewards ----
        transport_reward = super()._compute_transport_reward(
            is_grasped=is_grasped,
            stable_grasp=stable_grasp,
            lift_height=lift_height,
            success=success,
            action=action,
        )

        # ================================================================
        # S7.1  Exponential center reward
        #       Replaces the flat tanh target_distance_reward (set to 0 via
        #       config: target_distance_reward_coef=0.0).
        #
        #       center_reward = center_reward_scale * exp(-dist / sigma)
        #
        #       At 5cm:  8.0 * exp(-2.0) = 1.08
        #       At 2.5cm: 8.0 * exp(-1.0) = 2.94
        #       At 1cm:  8.0 * exp(-0.4) = 5.36
        #
        #       Gated by transport_gate only (no near_goal_gate restriction
        #       — the exponential shape automatically creates gradient).
        #       The reward is bounded above (max = center_reward_scale = 8.0
        #       at dist=0), stable against numerical overflow since exp(-x)
        #       is monotonically decreasing for positive x.
        # ================================================================
        # Clamp dist >= 0 for numerical stability
        safe_dist = torch.clamp(cube_goal_dist, min=0.0)
        center_reward = (
            self._center_reward_scale
            * torch.exp(-safe_dist / self._center_reward_sigma)
            * gate_f
        )
        transport_reward += center_reward

        # ================================================================
        # S7.2  Center progress reward
        #       Replaces the flat progress reward (set to 0 via config:
        #       far_progress_reward_coef=0.0).
        #
        #       center_progress = clamp(prev_dist - curr_dist, ±0.01)
        #       center_progress_reward = center_progress * scale * gate_f
        #
        #       This is a POTENTIAL-BASED reward: the cumulative sum over a
        #       trajectory telescopes to (init_dist - final_dist) * scale.
        #       This is inherently bounded and unfarmable — you cannot earn
        #       more by taking more steps.  If the policy stalls, it earns
        #       zero per step (not positive, not negative).
        #
        #       The coefficient is high (300) so that even 1mm progress
        #       earns a clear signal: 0.001 × 300 = 0.3 per step.
        #       Moving backward (negative progress) is penalised
        #       symmetrically up to the clip bound.
        # ================================================================
        # Use the saved previous distance (captured before super() call).
        # On first transport step (gate just opened), _prev_dist_before equals
        # cube_goal_dist, so progress=0 — correct since there's no prior step.
        progress = _prev_dist_before - cube_goal_dist

        # On first transport step or gate-open, prev_dist == curr_dist
        # so progress=0, harmless.
        center_progress = torch.clamp(
            progress,
            self._center_progress_clip_min,
            self._center_progress_clip_max,
        )
        center_progress_reward = (
            center_progress * self._center_progress_reward_scale * gate_f
        )
        transport_reward += center_progress_reward

        # ================================================================
        # S7.3  Near-goal stall penalty
        #
        #       Detects consecutive steps with negligible progress in the
        #       2.5-5cm zone. After a grace period (3 steps), applies a
        #       per-step penalty.
        #
        #       Conditions:
        #       - near_goal: dist < 0.05m
        #       - not_success: dist > 0.025m (goal_thresh)
        #       - stalled: |progress| < progress_threshold (0.0005m)
        #       - transport_gate active
        #
        #       Consecutive stalled steps counted. After threshold (3),
        #       penalty activates. Counter resets on any movement.
        #
        #       Penalty: near_goal_stall_penalty_coef * gate_f per step.
        # ================================================================
        near_goal = (
            (cube_goal_dist < self._near_goal_stall_distance)
            & transport_gate
        )
        not_success_region = cube_goal_dist > self.goal_thresh
        is_stalled = (
            near_goal
            & not_success_region
            & (progress.abs() < self._near_goal_stall_progress_threshold)
            & transport_gate
        )

        self._near_goal_stall_steps = torch.where(
            is_stalled,
            self._near_goal_stall_steps + 1,
            torch.zeros_like(self._near_goal_stall_steps),
        )

        stall_active = (
            self._near_goal_stall_steps
            >= self._near_goal_stall_steps_threshold
        )
        near_goal_stall_pen = (
            stall_active.float()
            * self._near_goal_stall_penalty_coef
            * gate_f
        )
        # No penalty inside the goal zone (dist <= goal_thresh)
        near_goal_stall_pen = near_goal_stall_pen * not_success_region.float()
        transport_reward -= near_goal_stall_pen

        # ================================================================
        # S7.4  Success terminal reward
        #       Replaces the flat success_bonus (set to 0 via config:
        #       success_reward_bonus=0.0).
        #
        #       success_terminal_reward = success * (
        #           base_success_reward
        #           + fast_success_bonus * (1 - elapsed/max_steps)
        #       )
        #
        #       Quick success (e.g., 50 steps): 50 + 30*0.67 = 70.1
        #       Slow success (e.g., 140 steps): 50 + 30*0.07 = 52.1
        #       No success: 0
        #
        #       This ensures:
        #       - Quick success > slow success > failure (reward-wise)
        #       - The reward for quick success dominates potential farming
        # ================================================================
        elapsed = getattr(
            self, "_last_elapsed_steps_s67",
            torch.zeros(num_envs, dtype=torch.int32, device=device))
        _max_steps = float(getattr(
            self, "_max_episode_steps", 150))
        remaining_frac = torch.clamp(
            1.0 - elapsed.float() / max(_max_steps, 1.0),
            0.0, 1.0,
        )
        fast_bonus_amt = (
            success.float() * self._fast_success_bonus * remaining_frac)
        base_amt = success.float() * self._base_success_reward
        success_terminal_reward = base_amt + fast_bonus_amt
        transport_reward += success_terminal_reward

        # ---- Store Stage 6.7 diagnostics ----
        self._last_s67_center_reward = center_reward
        self._last_s67_center_progress_reward = center_progress_reward
        self._last_s67_center_progress = center_progress
        self._last_s67_near_goal_stall_pen = near_goal_stall_pen
        self._last_s67_near_goal_stall_active = stall_active
        self._last_s67_success_terminal_reward = success_terminal_reward
        self._last_s67_fast_success_bonus_amt = fast_bonus_amt

        return transport_reward

    # ------------------------------------------------------------------
    # Main reward override — store elapsed_steps before MRO chain
    # ------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """Store elapsed_steps then delegate to Stage 6.6 parent chain.

        The stored elapsed_steps is used by _compute_transport_reward to
        compute the fast_success_bonus.
        """
        self._last_elapsed_steps_s67 = info.get(
            "elapsed_steps",
            torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device))
        return super().compute_dense_reward(obs, action, info)

    # ------------------------------------------------------------------
    # Stage 6.7 diagnostic info
    # ------------------------------------------------------------------

    def get_center_precision_info(self):
        """Return Stage 6.7 center-precision diagnostics (14-tuple).

        [0]  center_reward              — per-step exponential center reward
        [1]  center_progress_reward     — per-step progress reward
        [2]  center_progress            — raw (prev_dist - curr_dist)
        [3]  near_goal_stall_penalty    — per-step stall penalty
        [4]  near_goal_stall_active     — bool: stall penalty active
        [5]  near_goal_stall_steps      — consecutive stalled steps
        [6]  success_terminal_reward    — total success reward (base+fast)
        [7]  fast_success_bonus_amt     — fast bonus portion
        [8]  cube_goal_dist             — current distance to goal
        [9]  transport_gate             — bool: transport phase active
        [10] near_goal_gate             — S6.6 near-goal gate [0,1]
        [11] is_obj_placed              — bool: inside goal_thresh
        [12] progress                   — same as [2], for convenience
        [13] near_goal_stall_rate_flag  — bool: in stall zone (near&!success)
        """
        cube_goal_dist = self._get_cube_goal_dist()
        is_obj_placed = cube_goal_dist <= self.goal_thresh

        is_grasped = self.agent.is_grasping(self.cube)
        stable_grasp = getattr(
            self, "_last_stable_grasp",
            torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device))
        lift_height = getattr(
            self, "_last_cube_lift_height",
            torch.zeros(self.num_envs, device=self.device))
        transport_gate = stable_grasp & (
            lift_height >= self._transport_min_lift_height)

        # near_goal_gate from Stage 6.6
        near_goal_gate = getattr(
            self, "_last_s66_near_goal_gate",
            torch.zeros(self.num_envs, device=self.device))

        # Stall zone flag (for computing stall rate)
        near_and_not_success = (
            (cube_goal_dist < self._near_goal_stall_distance)
            & (cube_goal_dist > self.goal_thresh)
            & transport_gate
        )

        return (
            getattr(self, "_last_s67_center_reward", None),           # [0]
            getattr(self, "_last_s67_center_progress_reward", None),  # [1]
            getattr(self, "_last_s67_center_progress", None),         # [2]
            getattr(self, "_last_s67_near_goal_stall_pen", None),     # [3]
            getattr(self, "_last_s67_near_goal_stall_active", None),  # [4]
            getattr(self, "_near_goal_stall_steps", None),            # [5]
            getattr(self, "_last_s67_success_terminal_reward", None), # [6]
            getattr(self, "_last_s67_fast_success_bonus_amt", None),  # [7]
            cube_goal_dist,                                            # [8]
            transport_gate,                                            # [9]
            near_goal_gate,                                            # [10]
            is_obj_placed,                                             # [11]
            getattr(self, "_last_s67_center_progress", None),         # [12]
            near_and_not_success,                                      # [13]
        )

    def get_reward_components(self):
        """Return all reward components including Stage 6.7 additions.

        Extends Stage 6.6's tuple with center_reward, center_progress_reward,
        near_goal_stall_pen, and success_terminal_reward.
        """
        parent = super().get_reward_components()  # tuple from Stage 6.6

        # Sum of S6.7 components (for reward reconstruction)
        s67_sum = torch.zeros(self.num_envs, device=self.device)
        for attr in [
            "_last_s67_center_reward",
            "_last_s67_center_progress_reward",
        ]:
            val = getattr(self, attr, None)
            if val is not None:
                s67_sum = s67_sum + val

        # Subtract stall penalty
        sp = getattr(self, "_last_s67_near_goal_stall_pen", None)
        if sp is not None:
            s67_sum = s67_sum - sp

        # Add success terminal reward
        st = getattr(self, "_last_s67_success_terminal_reward", None)
        if st is not None:
            s67_sum = s67_sum + st

        return parent + (s67_sum,)
