"""
Custom PickCube environment with efficient center carry for Stage 6.8.

Registers as ``PickCubeEfficientCenterCarry-v1`` via gymnasium.

Extends ``PickCubeCenterPrecisionCarry-v1`` (Stage 6.7) with:

1. **Reduced center reward**: ``1.5 * exp(-dist/0.025)`` — small local shaping
   only, no longer the primary reward.  This eliminates the loophole where
   staying at 2.7-3.0 cm could farm ~2.5 reward/step for 100+ steps.

2. **Strengthened progress reward**: ``400 * clamp(prev-curr, ±0.01)`` —
   potential-based, cumulative telescopes to ``(init-final)*400``, unfarmable.

3. **Best-so-far improvement reward**: ``200 * clamp(best_dist - curr_dist, 0, 0.01)``
   — only rewards when the episode-best distance is beaten.  Cannot be farmed
   by staying still.

4. **Tiered near-goal stall penalties**: three concentric zones with
   progressively tighter grace periods and higher penalties:
   - 3.5-5.0 cm: 5-step grace, 0.10/step
   - 3.0-3.5 cm: 3-step grace, 0.25/step
   - 2.5-3.0 cm: 2-step grace, 0.50/step

5. **Away-from-goal penalty**: ``200 * clamp(-progress, 0, 0.005)`` —
   strong penalty for moving AWAY from the goal in the near zone.

6. **Success efficiency reward**: ``success * (80 + 70 * remaining_fraction)``
   — fast success (20 steps) ≈ 140.7, slow success (149 steps) ≈ 80.5.
   Makes success strictly dominate any farming strategy.

7. **Timeout terminal penalty**: ``-60`` one-time penalty when episode
   truncates without success.  Ensures failure total reward < success.

8. **Small per-step time cost**: ``0.02/step`` — adds up to -3 over 150 steps,
   a mild efficiency bias that doesn't disrupt the grasp phase.

All Stage 6.6 and 6.7 features retained: posture regularization, goal dwell,
joint limits, near-goal speed penalties, elbow fold detection, action scaling,
ellipsis indexing.

Relevant YAML keys (new / modified for Stage 6.8)
-------------------------------------------------

Center reward (reduced):
  center_reward_scale                 (float, default 1.5, was 8.0)

Center progress (strengthened):
  center_progress_reward_scale        (float, default 400.0, was 300.0)

Best-so-far improvement:
  best_improvement_scale              (float, default 200.0)
  best_improvement_max                (float, default 0.01)

Tiered stall:
  tiered_stall_35_50_threshold        (int, default 5)
  tiered_stall_35_50_penalty          (float, default 0.10)
  tiered_stall_30_35_threshold        (int, default 3)
  tiered_stall_30_35_penalty          (float, default 0.25)
  tiered_stall_25_30_threshold        (int, default 2)
  tiered_stall_25_30_penalty          (float, default 0.50)

Away-from-goal:
  away_from_goal_penalty_scale        (float, default 200.0)
  away_from_goal_clip                 (float, default 0.005)

Time cost:
  time_penalty_per_step               (float, default 0.02)

Success efficiency:
  base_success_reward                 (float, default 80.0, was 50.0)
  fast_success_bonus                  (float, default 70.0, was 30.0)

Timeout:
  timeout_penalty_value               (float, default 60.0)
"""

from typing import Any

import torch

from mani_skill.utils.registration import register_env

from .pick_cube_center_precision_carry import PickCubeCenterPrecisionCarryEnv


@register_env("PickCubeEfficientCenterCarry-v1", max_episode_steps=150)
class PickCubeEfficientCenterCarryEnv(PickCubeCenterPrecisionCarryEnv):
    """PickCube with efficient center carry for Stage 6.8.

    Inherits all Stage 6.7 logic and overrides the transport reward with
    time-normalized efficiency rewards that eliminate the reward inversion
    between success and failure trajectories.
    """

    # ------------------------------------------------------------------
    # Init — pop Stage 6.8 kwargs before parent
    # ------------------------------------------------------------------

    def __init__(self, *args, **kwargs):
        # ---- Center reward (REDUCED) ----
        self._center_reward_scale_s68 = float(
            kwargs.pop("center_reward_scale", 1.0))
        self._center_reward_time_decay = bool(
            kwargs.pop("center_reward_time_decay", True))

        # ---- Center progress (STRENGTHENED) ----
        self._center_progress_reward_scale_s68 = float(
            kwargs.pop("center_progress_reward_scale", 400.0))

        # ---- Best-so-far improvement ----
        self._best_improvement_scale = float(
            kwargs.pop("best_improvement_scale", 200.0))
        self._best_improvement_max = float(
            kwargs.pop("best_improvement_max", 0.01))

        # ---- Tiered stall: 3.5-5.0 cm ----
        self._tiered_stall_35_50_threshold = int(
            kwargs.pop("tiered_stall_35_50_threshold", 5))
        self._tiered_stall_35_50_penalty = float(
            kwargs.pop("tiered_stall_35_50_penalty", 0.10))

        # ---- Tiered stall: 3.0-3.5 cm ----
        self._tiered_stall_30_35_threshold = int(
            kwargs.pop("tiered_stall_30_35_threshold", 3))
        self._tiered_stall_30_35_penalty = float(
            kwargs.pop("tiered_stall_30_35_penalty", 0.25))

        # ---- Tiered stall: 2.5-3.0 cm ----
        self._tiered_stall_25_30_threshold = int(
            kwargs.pop("tiered_stall_25_30_threshold", 2))
        self._tiered_stall_25_30_penalty = float(
            kwargs.pop("tiered_stall_25_30_penalty", 0.50))

        # ---- Away-from-goal penalty ----
        self._away_from_goal_penalty_scale = float(
            kwargs.pop("away_from_goal_penalty_scale", 200.0))
        self._away_from_goal_clip = float(
            kwargs.pop("away_from_goal_clip", 0.005))

        # ---- Time cost ----
        self._time_penalty_per_step = float(
            kwargs.pop("time_penalty_per_step", 0.02))

        # ---- Success efficiency (STRENGTHENED) ----
        # parent pops base_success_reward and fast_success_bonus, so we
        # intercept them here with our new defaults before passing to parent
        self._base_success_reward_s68 = float(
            kwargs.pop("base_success_reward", 80.0))
        self._fast_success_bonus_s68 = float(
            kwargs.pop("fast_success_bonus", 70.0))

        # ---- Timeout penalty ----
        self._timeout_penalty_value = float(
            kwargs.pop("timeout_penalty_value", 60.0))

        # ---- Per-env state for Stage 6.8 ----
        self._best_goal_dist_s68: torch.Tensor | None = None  # (N,) float
        self._stall_counter_35_50: torch.Tensor | None = None  # (N,) int32
        self._stall_counter_30_35: torch.Tensor | None = None  # (N,) int32
        self._stall_counter_25_30: torch.Tensor | None = None  # (N,) int32

        # Pass remaining kwargs to parent (Stage 6.7).
        # NOTE: parent will pop its own center_reward_scale etc., but we've
        # already popped ours above.  We pass our reduced defaults through
        # so parent doesn't see missing keys.
        super().__init__(*args, **kwargs)

        # ---- Stage 6.8 per-step diagnostic storage ----
        self._last_s68_center_reward_small: torch.Tensor | None = None
        self._last_s68_center_progress_reward: torch.Tensor | None = None
        self._last_s68_center_progress: torch.Tensor | None = None
        self._last_s68_best_improvement_reward: torch.Tensor | None = None
        self._last_s68_best_improvement: torch.Tensor | None = None
        self._last_s68_tiered_stall_penalty: torch.Tensor | None = None
        self._last_s68_away_penalty: torch.Tensor | None = None
        self._last_s68_time_penalty: torch.Tensor | None = None
        self._last_s68_success_efficiency_reward: torch.Tensor | None = None
        self._last_s68_timeout_penalty: torch.Tensor | None = None
        self._last_s68_elapsed_fraction: torch.Tensor | None = None
        self._last_s68_remaining_fraction: torch.Tensor | None = None

        # Tiered stall per-zone active flags for diagnostics
        self._last_s68_stall_35_50_active: torch.Tensor | None = None
        self._last_s68_stall_30_35_active: torch.Tensor | None = None
        self._last_s68_stall_25_30_active: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Episode init — extend for Stage 6.8 state
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        num_envs = self.num_envs
        device = self.device

        # Lazy-init Stage 6.8 tensors
        if self._best_goal_dist_s68 is None:
            self._best_goal_dist_s68 = torch.full(
                (num_envs,), float('inf'), device=device)
        if self._stall_counter_35_50 is None:
            self._stall_counter_35_50 = torch.zeros(
                num_envs, dtype=torch.int32, device=device)
        if self._stall_counter_30_35 is None:
            self._stall_counter_30_35 = torch.zeros(
                num_envs, dtype=torch.int32, device=device)
        if self._stall_counter_25_30 is None:
            self._stall_counter_25_30 = torch.zeros(
                num_envs, dtype=torch.int32, device=device)

        # Reset Stage 6.8 state for newly-reset envs
        self._best_goal_dist_s68[env_idx] = float('inf')
        self._stall_counter_35_50[env_idx] = 0
        self._stall_counter_30_35[env_idx] = 0
        self._stall_counter_25_30[env_idx] = 0

    # ------------------------------------------------------------------
    # Transport reward override — Stage 6.8 efficiency rewards
    # ------------------------------------------------------------------

    def _compute_transport_reward(
        self,
        is_grasped: torch.Tensor,
        stable_grasp: torch.Tensor,
        lift_height: torch.Tensor,
        success: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        """Replace Stage 6.7 transport rewards with Stage 6.8 efficiency rewards.

        Strategy:
        1. Skip Stage 6.7 entirely — call super(PickCubeCenterPrecisionCarryEnv)
           to get the Stage 6.6 base transport reward.
        2. Add all Stage 6.8 reward components from scratch.

        This avoids the complexity of subtracting S6.7 components and adding
        S6.8 ones, at the cost of re-implementing the progress capture logic.
        """
        num_envs = self.num_envs
        device = self.device

        transport_gate = stable_grasp & (
            lift_height >= self._transport_min_lift_height)
        gate_f = transport_gate.float()

        cube_goal_dist = self._get_cube_goal_dist()

        # ---- Capture prev distance BEFORE super() updates it ----
        _prev_dist_before = getattr(
            self, "_prev_cube_goal_dist",
            cube_goal_dist.clone())

        _transport_gate_prev_before = getattr(
            self, "_transport_gate_prev",
            torch.zeros(num_envs, dtype=torch.bool, device=device))
        gate_just_opened_s68 = transport_gate & ~_transport_gate_prev_before
        _prev_dist_before = torch.where(
            gate_just_opened_s68,
            cube_goal_dist,
            _prev_dist_before,
        )

        # ---- Skip Stage 6.7, go directly to Stage 6.6 base ----
        transport_reward = super(
            PickCubeCenterPrecisionCarryEnv, self
        )._compute_transport_reward(
            is_grasped=is_grasped,
            stable_grasp=stable_grasp,
            lift_height=lift_height,
            success=success,
            action=action,
        )

        # ---- Compute time fractions early (used by multiple S8 sections) ----
        _elapsed = getattr(
            self, "_last_elapsed_steps_s67",
            torch.zeros(num_envs, dtype=torch.int32, device=device))
        _max_steps = float(getattr(self, "_max_episode_steps", 150))
        elapsed_fraction = torch.clamp(
            _elapsed.float() / max(_max_steps, 1.0), 0.0, 1.0)
        remaining_fraction = 1.0 - elapsed_fraction

        # ================================================================
        # S8.1  Reduced center reward with time decay
        #
        #       center_reward_small = scale * exp(-dist/sigma) * gate
        #                            * clamp(remaining_fraction, 0.1)
        #
        #       The remaining_fraction factor means: the longer you take
        #       to reach the goal, the less each step's center reward.
        #       At step 20: ×0.87, at step 100: ×0.33, at step 140: ×0.10
        #
        #       At 2.5cm with remaining=0.87 (step 20):
        #         1.0 * exp(-1.0) * 0.87 = 0.320
        #       At 2.5cm with remaining=0.33 (step 100):
        #         1.0 * exp(-1.0) * 0.33 = 0.122
        #
        #       This strongly discourages lingering near the goal.
        #       Farming at 2.7cm for 100 steps:
        #         old (no decay): 100 * 0.51 = 51
        #         new (with decay): ~40 (declining over time)
        # ================================================================
        safe_dist = torch.clamp(cube_goal_dist, min=0.0)
        center_reward_small = (
            self._center_reward_scale_s68
            * torch.exp(-safe_dist / self._center_reward_sigma)
            * gate_f
        )

        # Apply time decay: multiply by remaining_fraction (clamped to 0.1)
        if self._center_reward_time_decay:
            _cr_decay = torch.clamp(remaining_fraction, 0.1, 1.0)
            center_reward_small = center_reward_small * _cr_decay

        transport_reward += center_reward_small

        # ================================================================
        # S8.2  Strengthened center progress reward
        #
        #       progress = prev_dist - curr_dist
        #       center_progress = clamp(progress, ±0.01)
        #       center_progress_reward = 400 * center_progress * gate_f
        #
        #       Cumulative bound: (init - final) * 400 ≤ typically ~40-80
        #       This is potential-based, inherently unfarmable.
        # ================================================================
        progress = _prev_dist_before - cube_goal_dist
        center_progress = torch.clamp(
            progress,
            self._center_progress_clip_min,
            self._center_progress_clip_max,
        )
        center_progress_reward = (
            center_progress * self._center_progress_reward_scale_s68 * gate_f
        )
        transport_reward += center_progress_reward

        # ================================================================
        # S8.3  Best-so-far improvement reward
        #
        #       Only rewards when the episode-best goal distance is beaten.
        #       best_improvement = clamp(old_best - curr_dist, 0, max)
        #       best_improvement_reward = scale * best_improvement * gate_f
        #
        #       This is inherently unfarmable: staying still gives 0,
        #       each mm of improvement can only be rewarded ONCE per episode.
        #       We capture old_best BEFORE updating _best_goal_dist_s68.
        # ================================================================
        _old_best = self._best_goal_dist_s68.clone()
        self._best_goal_dist_s68 = torch.minimum(
            self._best_goal_dist_s68, cube_goal_dist)
        best_improvement_raw = _old_best - cube_goal_dist
        best_improvement = torch.clamp(
            best_improvement_raw, 0.0, self._best_improvement_max)
        best_improvement_reward = (
            best_improvement * self._best_improvement_scale * gate_f
        )
        transport_reward += best_improvement_reward

        # ================================================================
        # S8.4  Tiered near-goal stall penalties
        #
        #       Three concentric zones with different thresholds:
        #
        #       Zone 1 (3.5-5.0 cm):
        #         near_35_50 = dist in [0.035, 0.05) AND transport_gate
        #         stalled = |progress| < 0.0002
        #         After 5 consecutive stalled steps: 0.10/step
        #
        #       Zone 2 (3.0-3.5 cm):
        #         near_30_35 = dist in [0.030, 0.035) AND transport_gate
        #         stalled = |progress| < 0.0002
        #         After 3 consecutive stalled steps: 0.25/step
        #
        #       Zone 3 (2.5-3.0 cm):
        #         near_25_30 = dist in (goal_thresh, 0.030) AND transport_gate
        #         stalled = |progress| < 0.0002
        #         After 2 consecutive stalled steps: 0.50/step
        #
        #       Counter resets on any progress > threshold.
        #       No penalty inside goal zone (dist <= goal_thresh).
        # ================================================================
        no_progress = progress.abs() < 0.0002

        # Zone 1: 3.5-5.0 cm
        zone_35_50 = (
            (cube_goal_dist >= 0.035)
            & (cube_goal_dist < 0.05)
            & transport_gate
        )
        stalled_35_50 = zone_35_50 & no_progress
        self._stall_counter_35_50 = torch.where(
            stalled_35_50,
            self._stall_counter_35_50 + 1,
            torch.zeros_like(self._stall_counter_35_50),
        )
        stall_active_35_50 = (
            self._stall_counter_35_50 >= self._tiered_stall_35_50_threshold
        ) & zone_35_50

        # Zone 2: 3.0-3.5 cm
        zone_30_35 = (
            (cube_goal_dist >= 0.030)
            & (cube_goal_dist < 0.035)
            & transport_gate
        )
        stalled_30_35 = zone_30_35 & no_progress
        self._stall_counter_30_35 = torch.where(
            stalled_30_35,
            self._stall_counter_30_35 + 1,
            torch.zeros_like(self._stall_counter_30_35),
        )
        stall_active_30_35 = (
            self._stall_counter_30_35 >= self._tiered_stall_30_35_threshold
        ) & zone_30_35

        # Zone 3: 2.5-3.0 cm
        zone_25_30 = (
            (cube_goal_dist > self.goal_thresh)
            & (cube_goal_dist < 0.030)
            & transport_gate
        )
        stalled_25_30 = zone_25_30 & no_progress
        self._stall_counter_25_30 = torch.where(
            stalled_25_30,
            self._stall_counter_25_30 + 1,
            torch.zeros_like(self._stall_counter_25_30),
        )
        stall_active_25_30 = (
            self._stall_counter_25_30 >= self._tiered_stall_25_30_threshold
        ) & zone_25_30

        tiered_stall_penalty = (
            stall_active_35_50.float() * self._tiered_stall_35_50_penalty
            + stall_active_30_35.float() * self._tiered_stall_30_35_penalty
            + stall_active_25_30.float() * self._tiered_stall_25_30_penalty
        ) * gate_f
        transport_reward -= tiered_stall_penalty

        # ================================================================
        # S8.5  Away-from-goal penalty
        #
        #       Penalizes moving AWAY from the goal when near it.
        #       away_progress = clamp(-progress, 0, clip)
        #       away_penalty = scale * away_progress * gate_f
        #
        #       Only triggers on backward motion (progress < 0).
        #       Clipped to 0.005m to limit per-step penalty magnitude.
        # ================================================================
        away_progress = torch.clamp(-progress, 0.0, self._away_from_goal_clip)
        away_penalty = (
            away_progress * self._away_from_goal_penalty_scale * gate_f
        )
        transport_reward -= away_penalty

        # ================================================================
        # S8.6  Small per-step time cost
        #
        #       time_penalty = time_penalty_per_step (= 0.02)
        #
        #       Over 150 steps: cumulative -3.0 — a mild efficiency bias.
        #       Not gated (applies to entire episode, all phases).
        # ================================================================
        time_penalty = torch.full(
            (num_envs,), self._time_penalty_per_step, device=device)
        transport_reward -= time_penalty

        # ================================================================
        # S8.7  Success efficiency reward
        #
        #       success_efficiency = success * (
        #           base_success_reward (100)
        #           + fast_success_bonus (120) * remaining_fraction
        #       )
        #
        #       Examples:
        #       - 20 steps: 100 + 120*(1-20/150) ≈ 204
        #       - 50 steps: 100 + 120*(1-50/150) ≈ 180
        #       - 100 steps: 100 + 120*(1-100/150) ≈ 140
        #       - 149 steps: 100 + 120*(1-149/150) ≈ 100.8
        #       - Failure: 0
        #
        #       elapsed_fraction and remaining_fraction are precomputed above.
        # ================================================================
        fast_bonus_amt = (
            success.float()
            * self._fast_success_bonus_s68
            * remaining_fraction
        )
        base_amt = success.float() * self._base_success_reward_s68
        success_efficiency_reward = base_amt + fast_bonus_amt
        transport_reward += success_efficiency_reward

        # ================================================================
        # S8.8  Timeout terminal penalty
        # ================================================================
        is_last_step = _elapsed.float() >= (_max_steps - 1.0)
        timeout_failure = is_last_step & (~success) & transport_gate
        timeout_penalty = (
            timeout_failure.float() * self._timeout_penalty_value
        )
        transport_reward -= timeout_penalty

        # ---- Store Stage 6.8 diagnostics ----
        self._last_s68_center_reward_small = center_reward_small
        self._last_s68_center_progress_reward = center_progress_reward
        self._last_s68_center_progress = center_progress
        self._last_s68_best_improvement_reward = best_improvement_reward
        self._last_s68_best_improvement = best_improvement
        self._last_s68_tiered_stall_penalty = tiered_stall_penalty
        self._last_s68_away_penalty = away_penalty
        self._last_s68_time_penalty = time_penalty
        self._last_s68_success_efficiency_reward = success_efficiency_reward
        self._last_s68_timeout_penalty = timeout_penalty
        self._last_s68_elapsed_fraction = elapsed_fraction
        self._last_s68_remaining_fraction = remaining_fraction
        self._last_s68_stall_35_50_active = stall_active_35_50
        self._last_s68_stall_30_35_active = stall_active_30_35
        self._last_s68_stall_25_30_active = stall_active_25_30

        return transport_reward

    # ------------------------------------------------------------------
    # Main reward override — store elapsed_steps before MRO chain
    # ------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """Store elapsed_steps then delegate to parent chain."""
        self._last_elapsed_steps_s67 = info.get(
            "elapsed_steps",
            torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device))
        return super().compute_dense_reward(obs, action, info)

    # ------------------------------------------------------------------
    # Stage 6.8 diagnostic info
    # ------------------------------------------------------------------

    def get_efficient_center_info(self):
        """Return Stage 6.8 efficiency diagnostics (25-tuple).

        [0]  center_reward_small         — per-step reduced center reward
        [1]  center_progress_reward      — per-step progress reward
        [2]  center_progress             — raw (prev_dist - curr_dist)
        [3]  best_improvement_reward     — per-step best-improvement reward
        [4]  best_improvement            — raw improvement amount
        [5]  tiered_stall_penalty        — total tiered stall penalty this step
        [6]  away_penalty                — away-from-goal penalty this step
        [7]  time_penalty                — per-step time cost
        [8]  success_efficiency_reward   — terminal success reward
        [9]  timeout_penalty             — terminal timeout penalty
        [10] elapsed_fraction            — elapsed / max_steps
        [11] remaining_fraction          — 1 - elapsed_fraction
        [12] cube_goal_dist              — current distance to goal
        [13] transport_gate              — bool: transport phase active
        [14] is_obj_placed               — bool: inside goal_thresh
        [15] best_goal_dist_s68          — episode-best goal distance
        [16] success                     — bool: success this step
        [17] progress                    — same as [2], for convenience
        [18] stall_35_50_active          — bool: zone 1 stall active
        [19] stall_30_35_active          — bool: zone 2 stall active
        [20] stall_25_30_active          — bool: zone 3 stall active
        [21] near_zone_flag              — bool: in any near zone (2.5-5cm)
        [22] away_flag                   — bool: moving away from goal
        [23] is_last_step                — bool: last step of episode
        [24] timeout_failure             — bool: timeout without success
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

        elapsed = getattr(
            self, "_last_elapsed_steps_s67",
            torch.zeros(self.num_envs, dtype=torch.int32, device=self.device))
        _max_steps = float(getattr(self, "_max_episode_steps", 150))
        is_last_step = elapsed.float() >= (_max_steps - 1.0)

        # Success from the stored diagnostic or compute from dist
        success = is_obj_placed  # simplified; actual success may need dwell

        near_zone = (
            (cube_goal_dist > self.goal_thresh)
            & (cube_goal_dist < 0.05)
            & transport_gate
        )

        # Away flag: progress < -0.0001
        progress = getattr(self, "_last_s68_center_progress", None)
        if progress is None:
            progress = torch.zeros(self.num_envs, device=self.device)
        away_flag = (progress < -0.0001) & transport_gate

        timeout_failure = is_last_step & (~success) & transport_gate

        return (
            getattr(self, "_last_s68_center_reward_small", None),     # [0]
            getattr(self, "_last_s68_center_progress_reward", None),  # [1]
            getattr(self, "_last_s68_center_progress", None),         # [2]
            getattr(self, "_last_s68_best_improvement_reward", None), # [3]
            getattr(self, "_last_s68_best_improvement", None),        # [4]
            getattr(self, "_last_s68_tiered_stall_penalty", None),    # [5]
            getattr(self, "_last_s68_away_penalty", None),            # [6]
            getattr(self, "_last_s68_time_penalty", None),            # [7]
            getattr(self, "_last_s68_success_efficiency_reward", None), # [8]
            getattr(self, "_last_s68_timeout_penalty", None),         # [9]
            getattr(self, "_last_s68_elapsed_fraction", None),        # [10]
            getattr(self, "_last_s68_remaining_fraction", None),      # [11]
            cube_goal_dist,                                            # [12]
            transport_gate,                                            # [13]
            is_obj_placed,                                             # [14]
            getattr(self, "_best_goal_dist_s68", None),               # [15]
            success,                                                    # [16]
            progress,                                                   # [17]
            getattr(self, "_last_s68_stall_35_50_active", None),      # [18]
            getattr(self, "_last_s68_stall_30_35_active", None),      # [19]
            getattr(self, "_last_s68_stall_25_30_active", None),      # [20]
            near_zone,                                                  # [21]
            away_flag,                                                  # [22]
            is_last_step,                                               # [23]
            timeout_failure,                                            # [24]
        )

    def get_reward_components(self):
        """Return all reward components including Stage 6.8 additions.

        Extends parent's tuple with the S6.8 reward sum for reconstruction.
        """
        parent = super().get_reward_components()

        # Sum of S6.8 components (for reward reconstruction)
        s68_sum = torch.zeros(self.num_envs, device=self.device)
        for attr in [
            "_last_s68_center_reward_small",
            "_last_s68_center_progress_reward",
            "_last_s68_best_improvement_reward",
            "_last_s68_success_efficiency_reward",
        ]:
            val = getattr(self, attr, None)
            if val is not None:
                s68_sum = s68_sum + val

        # Subtract penalties
        for attr in [
            "_last_s68_tiered_stall_penalty",
            "_last_s68_away_penalty",
            "_last_s68_time_penalty",
            "_last_s68_timeout_penalty",
        ]:
            val = getattr(self, attr, None)
            if val is not None:
                s68_sum = s68_sum - val

        return parent + (s68_sum,)
