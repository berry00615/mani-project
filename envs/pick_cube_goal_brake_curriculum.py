"""
Custom PickCube environment with goal-zone brake curriculum for Stage 6.

Registers as ``PickCubeGoalBrakeCurriculum-v1`` via gymnasium.

Extends ``PickCubeDirectedTransportCurriculum-v1`` (Stage 5.5) with:

1. **Four-zone state machine**: pre-grasp (A), far transport (B, >8cm),
   brake approach (C, 2.5–8cm), placed hold (D, ≤2.5cm).

2. **Brake scale**: continuous interpolation from far-transport behavior
   at 8cm to full-brake behavior at 2.5cm (goal_thresh).

3. **Placed-zone reward shutdown**: progress, direction, best-distance,
   and stagnation rewards are zeroed inside the goal zone.

4. **Cube speed penalty**: penalizes high cube velocity, ramping up
   in brake zone and maxing in placed zone.

5. **Robot static shaping**: continuous reward for low joint velocity,
   ramping up in placed zone.

6. **Placed hold / center / dwell rewards**: multi-component incentives
   to stay in the goal zone with low speed and become static.

7. **Placed exit penalty**: one-time penalty when leaving the goal zone
   after having entered.

8. **Hysteresis**: placed-control state uses 0.025m enter / 0.032m exit
   to prevent boundary oscillation. Base ``is_obj_placed`` and ``success``
   semantics are NOT modified.

Phase A & B: inherited from Stage 5.5 (unchanged approach + lift).
Phase C (transport gate): fully redesigned with brake zones.

Relevant YAML keys (new / modified)
------------------------------------
brake_zone_outer                    (float, default 0.08)
placed_control_enter_threshold      (float, default 0.025)
placed_control_exit_threshold       (float, default 0.032)

# Far zone (B) — inherited from Stage 5.5
far_progress_reward_coef            (float, default 15.0)     — same as Stage 5.5
far_direction_reward_coef           (float, default 5.0)      — same as Stage 5.5
far_wrong_way_penalty_coef          (float, default 8.0)      — same as Stage 5.5

# Brake zone (C) — progressive
approach_cube_speed_penalty_coef    (float, default 2.0)
near_robot_motion_penalty_coef      (float, default 0.5)
near_action_smoothness_coef         (float, default 0.02)

# Placed zone (D)
placed_cube_speed_penalty_coef      (float, default 8.0)
placed_static_reward_coef           (float, default 2.0)
static_shaping_scale                (float, default 5.0)      — inside tanh
placed_hold_reward_coef             (float, default 0.5)
placed_center_reward_coef           (float, default 1.0)
placed_exit_penalty_coef            (float, default 5.0)
placed_action_magnitude_coef        (float, default 0.05)

# Dwell bonuses
placed_dwell_3_bonus                (float, default 0.5)
placed_dwell_5_bonus                (float, default 1.0)
placed_dwell_10_bonus               (float, default 2.0)
placed_dwell_cube_speed_threshold   (float, default 0.05)
placed_dwell_robot_qvel_threshold   (float, default 0.4)     — relaxed vs success

# One-time bonuses (unchanged from Stage 5.5)
near_goal_bonus, placed_bonus, success_reward_bonus
"""

from typing import Any

import torch

from mani_skill.utils.registration import register_env

from .pick_cube_directed_transport_curriculum import PickCubeDirectedTransportCurriculumEnv
from .pick_cube_target_transport_curriculum import PickCubeTargetTransportCurriculumEnv

# Panda arm joint indices (first 7 of 8)
ARM_INDICES = [0, 1, 2, 3, 4, 5, 6]


@register_env("PickCubeGoalBrakeCurriculum-v1", max_episode_steps=100)
class PickCubeGoalBrakeCurriculumEnv(PickCubeDirectedTransportCurriculumEnv):
    """PickCube with goal-zone brake curriculum for Stage 6.

    Inherits all Stage 5.5 logic (directed transport, smoothness,
    stagnation) and adds four-zone brake/shutdown/hold shaping.
    """

    def __init__(self, *args, **kwargs):
        # ---- Pop Stage-6-exclusive kwargs before parent ----

        # Zone thresholds
        self._brake_zone_outer = float(
            kwargs.pop("brake_zone_outer", 0.08))
        self._placed_control_enter_threshold = float(
            kwargs.pop("placed_control_enter_threshold", 0.025))
        self._placed_control_exit_threshold = float(
            kwargs.pop("placed_control_exit_threshold", 0.032))

        # Far zone coefficients (same defaults as Stage 5.5)
        self._far_progress_reward_coef = float(
            kwargs.pop("far_progress_reward_coef", 15.0))
        self._far_direction_reward_coef = float(
            kwargs.pop("far_direction_reward_coef", 5.0))
        self._far_wrong_way_penalty_coef = float(
            kwargs.pop("far_wrong_way_penalty_coef", 8.0))

        # Brake zone (C) — progressive
        self._approach_cube_speed_penalty_coef = float(
            kwargs.pop("approach_cube_speed_penalty_coef", 2.0))
        self._near_robot_motion_penalty_coef = float(
            kwargs.pop("near_robot_motion_penalty_coef", 0.5))
        self._near_action_smoothness_coef = float(
            kwargs.pop("near_action_smoothness_coef", 0.02))

        # Placed zone (D)
        self._placed_cube_speed_penalty_coef = float(
            kwargs.pop("placed_cube_speed_penalty_coef", 8.0))
        self._placed_static_reward_coef = float(
            kwargs.pop("placed_static_reward_coef", 2.0))
        self._static_shaping_scale = float(
            kwargs.pop("static_shaping_scale", 5.0))
        self._placed_hold_reward_coef = float(
            kwargs.pop("placed_hold_reward_coef", 0.5))
        self._placed_center_reward_coef = float(
            kwargs.pop("placed_center_reward_coef", 1.0))
        self._placed_exit_penalty_coef = float(
            kwargs.pop("placed_exit_penalty_coef", 5.0))
        self._placed_action_magnitude_coef = float(
            kwargs.pop("placed_action_magnitude_coef", 0.05))

        # Dwell bonuses
        self._placed_dwell_3_bonus = float(
            kwargs.pop("placed_dwell_3_bonus", 0.5))
        self._placed_dwell_5_bonus = float(
            kwargs.pop("placed_dwell_5_bonus", 1.0))
        self._placed_dwell_10_bonus = float(
            kwargs.pop("placed_dwell_10_bonus", 2.0))
        self._placed_dwell_cube_speed_threshold = float(
            kwargs.pop("placed_dwell_cube_speed_threshold", 0.05))
        self._placed_dwell_robot_qvel_threshold = float(
            kwargs.pop("placed_dwell_robot_qvel_threshold", 0.4))

        # Cube speed clip
        self._cube_speed_clip = float(
            kwargs.pop("cube_speed_clip", 0.5))
        self._robot_qvel_clip = float(
            kwargs.pop("robot_qvel_clip", 1.0))

        # ---- Far action smoothness (inherited default, stored locally) ----
        self._far_action_smoothness_coef = float(
            kwargs.pop("far_action_smoothness_coef", 0.01))

        # ---- Override params shared with parent ----
        _my_target_progress_coef = float(
            kwargs.get("target_progress_reward_coef", 15.0))
        _my_safe_height_coef = float(
            kwargs.get("transport_safe_height_reward_coef", 0.1))
        _my_lift_scale = float(
            kwargs.get("transport_lift_reward_scale", 0.05))

        # ---- Per-env state for Stage 6 (BEFORE super().__init__()) ----
        self._placed_control_active: torch.Tensor | None = None
        self._was_obj_placed_for_exit: torch.Tensor | None = None
        self._placed_dwell_steps: torch.Tensor | None = None
        self._stable_placed_dwell_steps: torch.Tensor | None = None
        self._placed_3_bonus_given: torch.Tensor | None = None
        self._placed_5_bonus_given: torch.Tensor | None = None
        self._placed_10_bonus_given: torch.Tensor | None = None
        self._prev_ep_success: torch.Tensor | None = None

        super().__init__(*args, **kwargs)

        # ---- Overwrite parent-set values ----
        self._target_progress_reward_coef = _my_target_progress_coef
        self._transport_safe_height_reward_coef = _my_safe_height_coef
        self._transport_lift_reward_scale = _my_lift_scale

        # ---- Per-step tracking for Stage 6 diagnostics ----
        self._last_brake_scale: torch.Tensor | None = None
        self._last_is_brake_zone: torch.Tensor | None = None
        self._last_is_placed_control: torch.Tensor | None = None
        self._last_cube_speed: torch.Tensor | None = None
        self._last_cube_speed_pen: torch.Tensor | None = None
        self._last_robot_qvel_norm: torch.Tensor | None = None
        self._last_static_shaping_rew: torch.Tensor | None = None
        self._last_placed_hold_rew: torch.Tensor | None = None
        self._last_center_rew: torch.Tensor | None = None
        self._last_placed_exit_pen: torch.Tensor | None = None
        self._last_placed_dwell_bonus: torch.Tensor | None = None
        self._last_placed_action_mag_pen: torch.Tensor | None = None
        self._last_is_obj_placed: torch.Tensor | None = None
        self._last_is_robot_static: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Episode init — extend for Stage 6 state
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        num_envs = self.num_envs
        device = self.device

        # Lazy-init Stage 6 tensors
        if self._placed_control_active is None:
            self._placed_control_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self._was_obj_placed_for_exit = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self._placed_dwell_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
            self._stable_placed_dwell_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
            self._placed_3_bonus_given = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self._placed_5_bonus_given = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self._placed_10_bonus_given = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self._prev_ep_success = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Reset Stage 6 state for newly-reset envs
        self._placed_control_active[env_idx] = False
        self._was_obj_placed_for_exit[env_idx] = False
        self._placed_dwell_steps[env_idx] = 0
        self._stable_placed_dwell_steps[env_idx] = 0
        self._placed_3_bonus_given[env_idx] = False
        self._placed_5_bonus_given[env_idx] = False
        self._placed_10_bonus_given[env_idx] = False
        self._prev_ep_success[env_idx] = False

    # ------------------------------------------------------------------
    # Helper: get robot qvel norm (arm joints only, matches base env)
    # ------------------------------------------------------------------

    def _get_robot_qvel_norm(self) -> torch.Tensor:
        """Return ||qvel_arm|| for each env, shape (num_envs,).

        Uses the same logic as base PickCubeEnv.compute_dense_reward:
        panda qvel excludes last 2 dims (gripper joints).
        """
        qvel = self.agent.robot.get_qvel()
        if self.robot_uids in ["panda", "widowxai"]:
            qvel = qvel[..., :-2]
        elif self.robot_uids == "so100":
            qvel = qvel[..., :-1]
        return torch.linalg.norm(qvel, dim=-1)

    # ------------------------------------------------------------------
    # Helper: cube speed (real simulator velocity)
    # ------------------------------------------------------------------

    def _get_cube_speed(self) -> torch.Tensor:
        """Return ||cube_linear_velocity|| for each env, shape (num_envs,)."""
        return torch.linalg.norm(self.cube.linear_velocity, dim=-1)

    # ------------------------------------------------------------------
    # Helper: compute brake scale
    # ------------------------------------------------------------------

    def _compute_brake_scale(self, cube_goal_dist: torch.Tensor) -> torch.Tensor:
        """Continuous brake scale: 1 at brake_zone_outer, 0 at goal_thresh.

        brake_scale = clamp((dist - goal_thresh) / (brake_zone_outer - goal_thresh), 0, 1)

        - dist >= brake_zone_outer  → 1.0 (full far-transport)
        - dist = goal_thresh        → 0.0 (full placed behavior)
        - dist between              → linear interpolation
        """
        margin = self._brake_zone_outer - self.goal_thresh
        return torch.clamp(
            (cube_goal_dist - self.goal_thresh) / max(margin, 0.001),
            0.0, 1.0,
        )

    # ------------------------------------------------------------------
    # Hysteresis: update placed-control state
    # ------------------------------------------------------------------

    def _update_placed_control(self, cube_goal_dist: torch.Tensor):
        """Update placed-control active state with hysteresis.

        Enter: dist <= placed_control_enter_threshold (0.025)
        Exit:  dist >  placed_control_exit_threshold  (0.032)

        This is separate from is_obj_placed / success.
        """
        newly_entered = (
            (cube_goal_dist <= self._placed_control_enter_threshold)
            & ~self._placed_control_active
        )
        newly_exited = (
            (cube_goal_dist > self._placed_control_exit_threshold)
            & self._placed_control_active
        )

        self._placed_control_active = torch.where(
            newly_entered,
            torch.ones_like(self._placed_control_active, dtype=torch.bool),
            self._placed_control_active,
        )
        self._placed_control_active = torch.where(
            newly_exited,
            torch.zeros_like(self._placed_control_active, dtype=torch.bool),
            self._placed_control_active,
        )

        return self._placed_control_active

    # ------------------------------------------------------------------
    # Directed transport reward (overrides Stage 5.5 Phase C)
    # ------------------------------------------------------------------

    def _compute_transport_reward(
        self,
        is_grasped: torch.Tensor,
        stable_grasp: torch.Tensor,
        lift_height: torch.Tensor,
        success: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        """Compute Phase C transport rewards with brake zones.

        Four-zone design:
        - Far (dist > brake_zone_outer): Stage 5.5 behavior
        - Brake (goal_thresh < dist <= brake_zone_outer): progressive braking
        - Placed-control (hysteresis): full brake + hold
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

        # --- Compute zones ---
        brake_scale = self._compute_brake_scale(cube_goal_dist)  # 1=far, 0=placed
        self._update_placed_control(cube_goal_dist)
        placed_control = self._placed_control_active
        placed_f = placed_control.float()

        # is_obj_placed (base semantics, unchanged)
        is_obj_placed = cube_goal_dist <= self.goal_thresh
        is_robot_static = self.agent.is_static(0.2)

        # Brake-zone flag
        is_brake_zone = (cube_goal_dist <= self._brake_zone_outer) & transport_gate & ~placed_control
        is_far_zone = (cube_goal_dist > self._brake_zone_outer) & transport_gate

        # ---- Prevent gate-first-step spikes for placed control ----
        # On gate-just-opened, reset placed control to avoid stale state
        self._placed_control_active = torch.where(
            gate_just_opened,
            torch.zeros_like(self._placed_control_active, dtype=torch.bool),
            self._placed_control_active,
        )
        self._was_obj_placed_for_exit = torch.where(
            gate_just_opened,
            torch.zeros_like(self._was_obj_placed_for_exit, dtype=torch.bool),
            self._was_obj_placed_for_exit,
        )

        # ================================================================
        # 1. Target distance reward (unchanged)
        # ================================================================
        target_dist_rew = (
            1.0 - torch.tanh(self._target_distance_scale * cube_goal_dist)
        ) * self._target_distance_reward_coef * gate_f
        transport_reward += target_dist_rew

        # ================================================================
        # 2. Progress reward — zero in placed-control
        # ================================================================
        progress = self._prev_cube_goal_dist - cube_goal_dist
        progress = torch.clamp(progress, -self._target_progress_clip, self._target_progress_clip)
        # In placed-control: progress reward = 0
        # In brake zone: scaled by brake_scale
        # In far zone: full
        progress_scale = torch.where(
            placed_control,
            torch.zeros_like(brake_scale),
            brake_scale,
        )
        target_progress_rew = (
            progress * self._far_progress_reward_coef * gate_f * progress_scale
        )
        transport_reward += target_progress_rew

        self._prev_cube_goal_dist = cube_goal_dist.clone()

        # ================================================================
        # 3. Directional velocity reward — zero in placed-control
        # ================================================================
        cube_pos = self.cube.pose.p
        goal_pos = self.goal_site.pose.p
        goal_vector = goal_pos - cube_pos
        goal_dist_norm = torch.linalg.norm(goal_vector, dim=-1, keepdim=True).clamp(min=1e-6)
        goal_direction = goal_vector / goal_dist_norm

        cube_vel = self.cube.linear_velocity
        directional_vel = (cube_vel * goal_direction).sum(dim=-1)

        # Direction scale: brake_scale in brake zone, 0 in placed-control
        dir_scale = torch.where(
            placed_control,
            torch.zeros_like(brake_scale),
            brake_scale,
        )
        # Near-goal scaling (inherited from Stage 5.5, only in far/brake)
        is_near_goal = (cube_goal_dist <= self._near_goal_threshold) & transport_gate
        near_f = is_near_goal.float()

        direction_rew = (
            torch.clamp(directional_vel, min=0.0, max=self._direction_velocity_clip)
            * self._far_direction_reward_coef
            * gate_f
            * dir_scale
        )
        direction_rew = direction_rew * (1.0 - near_f * (1.0 - self._near_goal_direction_scale))
        transport_reward += direction_rew

        # Wrong-way penalty — zero in placed-control
        wrong_way_pen = (
            torch.clamp(-directional_vel, min=0.0, max=self._direction_velocity_clip)
            * self._far_wrong_way_penalty_coef
            * gate_f
            * dir_scale
        )
        transport_reward -= wrong_way_pen

        # ================================================================
        # 4. Lateral motion penalty (unchanged, inactive in placed-control)
        # ================================================================
        parallel_vel = directional_vel.unsqueeze(-1) * goal_direction
        lateral_vel = cube_vel - parallel_vel
        lateral_vel_norm = torch.linalg.norm(lateral_vel, dim=-1)
        # Scale by brake_scale (zero in placed-control)
        lateral_motion_pen = (
            lateral_vel_norm
            * self._lateral_motion_penalty_coef
            * gate_f
            * brake_scale
        )
        transport_reward -= lateral_motion_pen

        # ================================================================
        # 5. Safe-height reward (reduced, unchanged from Stage 5.5)
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
        # 8. Lift during transport (reduced, unchanged from Stage 5.5)
        # ================================================================
        lift_continuous_parent = torch.clamp(
            lift_height * self._lift_reward_coef, max=self._lift_reward_max)
        scaled_lift = lift_continuous_parent * self._transport_lift_reward_scale * gate_f
        lift_adjustment = scaled_lift - lift_continuous_parent * gate_f
        transport_reward += lift_adjustment

        # ================================================================
        # 9. Stagnation penalty — zero in placed-control
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
            # Zero stagnation penalty in placed-control
            stag_scale = torch.where(
                placed_control,
                torch.zeros_like(brake_scale),
                torch.ones_like(brake_scale),
            )
            stagnation_pen = (
                stagnation_active.float()
                * self._stagnation_penalty_coef
                * stag_scale
                * (1.0 - near_f * (1.0 - self._near_goal_stagnation_scale))
            )
            transport_reward -= stagnation_pen
        else:
            stagnation_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
            stagnation_pen = torch.zeros(num_envs, device=device)

        # ================================================================
        # 10. Action smoothness — enhanced in brake/placed zones
        # ================================================================
        if self._prev_action is not None and action is not None:
            action_delta = action[:, :7] - self._prev_action[:, :7]
            action_delta = torch.where(
                gate_just_opened.unsqueeze(-1),
                torch.zeros_like(action_delta),
                action_delta,
            )
            action_delta_l2 = (action_delta ** 2).mean(dim=-1)

            # Smoothness coef: far_coef + (near_coef - far_coef) * (1 - brake_scale)
            effective_smoothness = (
                self._far_action_smoothness_coef
                + (self._near_action_smoothness_coef - self._far_action_smoothness_coef)
                * (1.0 - brake_scale)
            )
            action_smoothness_pen = (
                action_delta_l2 * effective_smoothness * gate_f
            )
            transport_reward -= action_smoothness_pen
            self._prev_action = action.clone()
        else:
            action_delta_l2 = torch.zeros(num_envs, device=device)
            action_smoothness_pen = torch.zeros(num_envs, device=device)
            if action is not None and self._prev_action is not None:
                self._prev_action = action.clone()

        # ================================================================
        # 11. Placed action magnitude penalty (NEW)
        # ================================================================
        if self._prev_action is not None and action is not None:
            action_magnitude_arm = (action[:, :7] ** 2).mean(dim=-1)
            placed_action_mag_pen = (
                action_magnitude_arm
                * self._placed_action_magnitude_coef
                * placed_f
            )
            transport_reward -= placed_action_mag_pen
        else:
            placed_action_mag_pen = torch.zeros(num_envs, device=device)

        # ================================================================
        # 12. Best distance bonus — zero in placed-control
        # ================================================================
        if self._best_goal_dist is not None:
            improved = cube_goal_dist < self._best_goal_dist
            improvement = torch.clamp(
                self._best_goal_dist - cube_goal_dist,
                min=0.0,
                max=self._best_distance_improvement_clip,
            )
            # Zero in placed-control
            best_scale = torch.where(
                placed_control,
                torch.zeros_like(brake_scale),
                torch.ones_like(brake_scale),
            )
            best_dist_bonus = (
                improvement
                * self._best_distance_bonus_coef
                * improved.float()
                * gate_f
                * best_scale
            )
            transport_reward += best_dist_bonus
            self._best_goal_dist = torch.where(
                improved & transport_gate & ~placed_control,
                cube_goal_dist,
                self._best_goal_dist,
            )
        else:
            best_dist_bonus = torch.zeros(num_envs, device=device)

        # ================================================================
        # 13. Cube speed penalty (NEW)
        #   In brake zone: progressively stronger as dist decreases
        #   In placed-control: full coef
        # ================================================================
        cube_speed = self._get_cube_speed()
        cube_speed_clipped = torch.clamp(cube_speed, max=self._cube_speed_clip)

        # In far zone: no speed penalty (brake_scale ≈ 1, 1-brake_scale ≈ 0)
        # In brake zone: progressive
        # In placed: full (1-brake_scale = 1.0)
        approach_speed_pen = (
            cube_speed_clipped
            * self._approach_cube_speed_penalty_coef
            * (1.0 - brake_scale)
            * gate_f
        )
        # Additional penalty in placed-control
        placed_speed_pen = (
            cube_speed_clipped
            * self._placed_cube_speed_penalty_coef
            * placed_f
        )
        transport_reward -= approach_speed_pen
        transport_reward -= placed_speed_pen

        # Combined for diagnostics
        cube_speed_pen_total = approach_speed_pen + placed_speed_pen

        # ================================================================
        # 14. Robot static shaping (NEW)
        # ================================================================
        robot_qvel_norm = self._get_robot_qvel_norm()
        robot_qvel_clipped = torch.clamp(robot_qvel_norm, max=self._robot_qvel_clip)

        # Base static shaping: 1 - tanh(scale * qvel), replicates base env formula
        static_shaping = (
            1.0 - torch.tanh(self._static_shaping_scale * robot_qvel_clipped)
        )

        # In brake zone: progressive activation
        # In placed-control: full activation
        static_shaping_rew = (
            static_shaping
            * self._placed_static_reward_coef
            * (1.0 - brake_scale)
            * gate_f
        )
        # Additional boost in placed-control
        static_shaping_rew = static_shaping_rew + (
            static_shaping
            * self._placed_static_reward_coef
            * 0.5  # extra boost when actually placed
            * placed_f
        )
        transport_reward += static_shaping_rew

        # ================================================================
        # 15. Placed hold reward (NEW)
        #   Only in placed-control, scaled by low-speed factor
        # ================================================================
        low_speed_factor = 1.0 - torch.tanh(5.0 * cube_speed_clipped)
        # Also factor in robot static-ness
        robot_static_factor = 1.0 - torch.tanh(5.0 * robot_qvel_clipped)
        placed_stable_factor = low_speed_factor * robot_static_factor

        placed_hold_rew = (
            self._placed_hold_reward_coef
            * placed_f
            * placed_stable_factor
        )
        transport_reward += placed_hold_rew

        # ================================================================
        # 16. Center reward (NEW)
        #   Higher reward closer to center of goal zone
        # ================================================================
        center_rew = (
            (1.0 - cube_goal_dist / max(self.goal_thresh, 0.001))
            * self._placed_center_reward_coef
            * placed_f
        )
        # Clamp to non-negative
        center_rew = torch.clamp(center_rew, min=0.0)
        transport_reward += center_rew

        # ================================================================
        # 17. Placed exit penalty (NEW)
        # ================================================================
        # Track whether we were in placed-control last step
        was_placed_prev = self._was_obj_placed_for_exit.clone()

        # Update tracking: mark envs that are now in placed-control
        self._was_obj_placed_for_exit = torch.where(
            placed_control,
            torch.ones_like(self._was_obj_placed_for_exit, dtype=torch.bool),
            self._was_obj_placed_for_exit,
        )
        # But clear for envs that exited (hysteresis exit)
        self._was_obj_placed_for_exit = torch.where(
            (~placed_control) & (cube_goal_dist > self._placed_control_exit_threshold),
            torch.zeros_like(self._was_obj_placed_for_exit, dtype=torch.bool),
            self._was_obj_placed_for_exit,
        )

        # Exit event: was placed last step, not placed now, NOT due to success/reset
        prev_success = self._prev_ep_success
        placed_exit_event = (
            was_placed_prev
            & ~placed_control
            & ~success  # don't penalize success-terminated episodes
            & ~prev_success  # don't penalize if just reset from success
        )
        placed_exit_pen = (
            placed_exit_event.float()
            * self._placed_exit_penalty_coef
        )
        transport_reward -= placed_exit_pen

        self._prev_ep_success = success.clone()

        # ================================================================
        # 18. Placed dwell bonuses (NEW)
        # ================================================================
        # Increment dwell counters
        self._placed_dwell_steps = torch.where(
            placed_control,
            self._placed_dwell_steps + 1,
            torch.zeros_like(self._placed_dwell_steps),
        )

        # Stable dwell: placed AND low speed AND robot fairly static
        placed_stable = (
            placed_control
            & (cube_speed < self._placed_dwell_cube_speed_threshold)
            & (robot_qvel_norm < self._placed_dwell_robot_qvel_threshold)
        )
        self._stable_placed_dwell_steps = torch.where(
            placed_stable,
            self._stable_placed_dwell_steps + 1,
            torch.zeros_like(self._stable_placed_dwell_steps),
        )

        dwell_bonus = torch.zeros(num_envs, device=device)

        # 3-step bonus
        new_3 = (
            (self._placed_dwell_steps >= 3)
            & ~self._placed_3_bonus_given
            & placed_control
        )
        dwell_bonus += new_3.float() * self._placed_dwell_3_bonus
        self._placed_3_bonus_given |= new_3

        # 5-step bonus
        new_5 = (
            (self._placed_dwell_steps >= 5)
            & ~self._placed_5_bonus_given
            & placed_control
        )
        dwell_bonus += new_5.float() * self._placed_dwell_5_bonus
        self._placed_5_bonus_given |= new_5

        # 10-step bonus
        new_10 = (
            (self._placed_dwell_steps >= 10)
            & ~self._placed_10_bonus_given
            & placed_control
        )
        dwell_bonus += new_10.float() * self._placed_dwell_10_bonus
        self._placed_10_bonus_given |= new_10

        transport_reward += dwell_bonus

        # ================================================================
        # 19. One-time bonuses (unchanged from Stage 5.5)
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
            new_placed = is_obj_placed & ~self._placed_reached
            transport_reward += new_placed.float() * self._placed_bonus
            self._placed_reached |= new_placed

        transport_reward += success.float() * self._success_reward_bonus

        # ================================================================
        # Update transport_gate_prev
        # ================================================================
        self._transport_gate_prev = transport_gate.clone()

        # ---- Store for diagnostics (Stage 5 inherited) ----
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

        # Stage 5.5 diagnostics
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

        # Stage 6 diagnostics
        self._last_brake_scale = brake_scale
        self._last_is_brake_zone = is_brake_zone
        self._last_is_placed_control = placed_control
        self._last_cube_speed = cube_speed
        self._last_cube_speed_pen = cube_speed_pen_total
        self._last_robot_qvel_norm = robot_qvel_norm
        self._last_static_shaping_rew = static_shaping_rew
        self._last_placed_hold_rew = placed_hold_rew
        self._last_center_rew = center_rew
        self._last_placed_exit_pen = placed_exit_pen
        self._last_placed_dwell_bonus = dwell_bonus
        self._last_placed_action_mag_pen = placed_action_mag_pen
        self._last_is_obj_placed = is_obj_placed
        self._last_is_robot_static = is_robot_static

        return transport_reward

    # ------------------------------------------------------------------
    # Main reward (override to pass action to transport)
    # ------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """Three-phase reward with brake-zone transport in Phase C.

        Uses super(PickCubeTargetTransportCurriculumEnv, self) to skip
        BOTH Stage 5 and Stage 5.5 transport computations, getting only
        the Stage 4.5 parent reward. Then adds Stage 6 transport.
        """
        # Skip BOTH Stage 5 (TargetTransport) and Stage 5.5 (DirectedTransport)
        # transport — get only Stage 4.5 parent reward
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
    # Stage 6 diagnostic info
    # ------------------------------------------------------------------

    def get_brake_info(self):
        """Return Stage 6 brake diagnostics (16-tuple).

        [0]  brake_scale
        [1]  is_brake_zone
        [2]  is_placed_control
        [3]  cube_speed
        [4]  cube_speed_penalty
        [5]  robot_qvel_norm
        [6]  static_shaping_reward
        [7]  placed_hold_reward
        [8]  center_reward
        [9]  placed_exit_penalty
        [10] placed_dwell_bonus
        [11] placed_dwell_steps
        [12] stable_placed_dwell_steps
        [13] placed_action_mag_penalty
        [14] is_obj_placed
        [15] is_robot_static
        """
        return (
            getattr(self, "_last_brake_scale", None),             # [0]
            getattr(self, "_last_is_brake_zone", None),           # [1]
            getattr(self, "_last_is_placed_control", None),       # [2]
            getattr(self, "_last_cube_speed", None),              # [3]
            getattr(self, "_last_cube_speed_pen", None),          # [4]
            getattr(self, "_last_robot_qvel_norm", None),         # [5]
            getattr(self, "_last_static_shaping_rew", None),      # [6]
            getattr(self, "_last_placed_hold_rew", None),         # [7]
            getattr(self, "_last_center_rew", None),              # [8]
            getattr(self, "_last_placed_exit_pen", None),         # [9]
            getattr(self, "_last_placed_dwell_bonus", None),      # [10]
            getattr(self, "_placed_dwell_steps", None),           # [11]
            getattr(self, "_stable_placed_dwell_steps", None),    # [12]
            getattr(self, "_last_placed_action_mag_pen", None),   # [13]
            getattr(self, "_last_is_obj_placed", None),           # [14]
            getattr(self, "_last_is_robot_static", None),         # [15]
        )

    def get_reward_components(self):
        """Return all reward components (28-tuple, compat with Stage 5.5's 22).

        [0-21]: Same as Stage 5.5
        [22]: cube_speed_penalty (total)
        [23]: static_shaping_reward
        [24]: placed_hold_reward
        [25]: center_reward
        [26]: placed_exit_penalty
        [27]: placed_dwell_bonus
        """
        parent = super().get_reward_components()
        return (
            parent[0], parent[1], parent[2], parent[3], parent[4],
            parent[5], parent[6], parent[7], parent[8], parent[9],
            parent[10], parent[11], parent[12], parent[13], parent[14],
            parent[15], parent[16], parent[17], parent[18], parent[19],
            parent[20], parent[21],
            getattr(self, "_last_cube_speed_pen", None),          # [22]
            getattr(self, "_last_static_shaping_rew", None),      # [23]
            getattr(self, "_last_placed_hold_rew", None),         # [24]
            getattr(self, "_last_center_rew", None),              # [25]
            getattr(self, "_last_placed_exit_pen", None),         # [26]
            getattr(self, "_last_placed_dwell_bonus", None),      # [27]
        )
