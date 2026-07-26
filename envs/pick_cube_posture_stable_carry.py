"""
Custom PickCube environment with posture-stable carry curriculum for Stage 6.6.

Registers as ``PickCubePostureStableCarry-v1`` via gymnasium.

Extends ``PickCubePrecisionCarryCurriculumEnv`` (Stage 6.5) with:

1. **Near-goal joint-delta action scaling**: progressively reduces arm joint delta
   magnitude as the cube approaches the goal. Wrist joints (j5, j6) are scaled
   more aggressively. Gripper action is unchanged.

2. **Joint limit soft penalty** (strengthened vs Stage 6.5): uses robot's real
   qlimits, lower margin, smooth squared penalty.

3. **Posture regularization**: soft penalty pulling arm toward a natural
   reference qpos. Gated: OFF during approach/grasp, weak during far transport,
   strong near goal. Uses normalised qpos error.

4. **Near-goal speed penalties**: separate from brake-zone speed band.
   Penalises cube speed, TCP speed, and arm qvel in the near-goal zone.

5. **Goal-dwell tracking**: counts consecutive steps inside the goal zone
   with low cube speed AND low robot qvel ("stable dwell").

6. **Goal exit penalty** (strengthened): stronger penalty for leaving goal.

7. **Episode length**: 150 steps (vs 100 in Stage 6.5).

8. **Reward rebalancing**: reduced continuous lift/transport scaling once
   skills are learned; increased success bonus.

9. **Self-collision monitoring**: approximates link proximity via min
   non-adjacent link distance (diagnostic-only for now).

All Stage 6.5 features retained: speed band, premature stop, min progress,
grasp reference, gripper alignment, clearance, dragging.

Relevant YAML keys (new / modified for Stage 6.6)
-------------------------------------------------

Near-goal action scaling:
  action_scale_far_dist               (float, default 0.12)
  action_scale_near_dist              (float, default 0.04)
  arm_delta_scale_min                 (float, default 0.25)
  wrist_delta_scale_min               (float, default 0.08)
  near_goal_action_scale_coef         (float, default 0.0  - diagnostic only)

Joint limit penalty (strengthened):
  joint_limit_margin                  (float, default 0.70  — lower than 0.80)
  joint_limit_penalty_coef            (float, default 0.3   — higher than 0.1)
  wrist_joint_weight                  (float, default 2.0   — higher than 1.5)

Posture regularization:
  posture_reg_coef                    (float, default 0.05)
  posture_near_goal_boost             (float, default 3.0)
  posture_far_scale                   (float, default 0.2)

Near-goal speed penalties:
  near_goal_cube_speed_coef           (float, default 0.5)
  near_goal_tcp_speed_coef            (float, default 0.3)
  near_goal_qvel_coef                 (float, default 0.2)

Goal dwell:
  goal_dwell_cube_speed_threshold     (float, default 0.03)
  goal_dwell_robot_qvel_threshold     (float, default 0.3)
  goal_dwell_3_bonus                  (float, default 1.0)
  goal_dwell_5_bonus                  (float, default 2.0)
  goal_dwell_10_bonus                 (float, default 5.0)

Goal exit:
  goal_exit_penalty_coef              (float, default 8.0)

Reward rebalancing:
  transport_lift_reward_scale         (float, default 0.02  — reduced from 0.05)
  success_reward_bonus                (float, default 10.0  — increased from 3.0)
"""

from typing import Any

import torch

from mani_skill.utils.registration import register_env

from .pick_cube_precision_carry_curriculum import (
    PickCubePrecisionCarryCurriculumEnv,
    PANDA_JOINT_LOWER,
    PANDA_JOINT_UPPER,
    WRIST_JOINT_INDICES,
)

# Panda arm joint indices (first 7 of 8 action dims)
ARM_INDICES = [0, 1, 2, 3, 4, 5, 6]
GRIPPER_INDEX = 7

# Natural reference qpos: Panda neutral pose (from ManiSkill default reset)
# This is a relaxed mid-range configuration, not at any joint limit.
PANDA_REFERENCE_QPOS = torch.tensor([
    0.0,      # j1 — centred
    0.0,      # j2 — centred
    0.0,      # j3 — centred
    -1.5708,  # j4 — mid-range forearm rotation
    0.0,      # j5 — centred wrist pitch
    1.5708,   # j6 — mid-range wrist roll
    0.0,      # j7 — centred
])


@register_env("PickCubePostureStableCarry-v1", max_episode_steps=150)
class PickCubePostureStableCarryEnv(PickCubePrecisionCarryCurriculumEnv):
    """PickCube with posture-stable carry curriculum for Stage 6.6.

    Inherits all Stage 6.5 logic (speed band, premature stop, precision
    distance, grasp reference, joint limits, clearance, dragging) and adds:

    - Near-goal joint-delta action scaling (wrist scaled more aggressively)
    - Posture regularization toward natural reference qpos
    - Strengthened joint limit penalty (lower margin, higher weight)
    - Goal-dwell tracking with bonuses
    - Goal exit penalty (strengthened)
    - Near-goal speed penalties
    - Reward rebalancing (reduced continuous lift/transport, increased success)
    - 150-step episodes
    """

    # ------------------------------------------------------------------
    # Init — pop Stage 6.6 kwargs before parent
    # ------------------------------------------------------------------

    def __init__(self, *args, **kwargs):
        # Stage 6.6 per-env state for TCP velocity tracking
        self._prev_tcp_pos_s66: torch.Tensor | None = None  # (N, 3)
        self._tcp_speed_s66: torch.Tensor | None = None      # (N,)
        # ---- Action scaling ----
        self._action_scale_far_dist = float(
            kwargs.pop("action_scale_far_dist", 0.12))
        self._action_scale_near_dist = float(
            kwargs.pop("action_scale_near_dist", 0.04))
        self._arm_delta_scale_min = float(
            kwargs.pop("arm_delta_scale_min", 0.25))
        self._wrist_delta_scale_min = float(
            kwargs.pop("wrist_delta_scale_min", 0.08))

        # ---- Joint limit penalty (strengthened vs Stage 6.5) ----
        # Override parent defaults with stronger values
        # Note: Stage 6.5 pops these too, so we pop them BEFORE super().__init__()
        # But Stage 6.5 already pops them... We need to handle this carefully.
        # Stage 6.6 pops its OWN copies with different defaults.
        # Actually the parent (Stage 6.5) pops these with its own defaults.
        # To override, we pop BEFORE super().__init__(), so our values
        # take precedence. The parent will NOT re-pop because kwargs keys
        # are already consumed.
        self._joint_limit_margin = float(
            kwargs.pop("joint_limit_margin", 0.70))
        self._joint_limit_penalty_coef = float(
            kwargs.pop("joint_limit_penalty_coef", 0.3))
        self._wrist_joint_weight = float(
            kwargs.pop("wrist_joint_weight", 2.0))

        # ---- Posture regularization ----
        self._posture_reg_coef = float(
            kwargs.pop("posture_reg_coef", 0.05))
        self._posture_near_goal_boost = float(
            kwargs.pop("posture_near_goal_boost", 3.0))
        self._posture_far_scale = float(
            kwargs.pop("posture_far_scale", 0.2))

        # ---- Near-goal speed penalties ----
        self._near_goal_cube_speed_coef = float(
            kwargs.pop("near_goal_cube_speed_coef", 0.5))
        self._near_goal_tcp_speed_coef = float(
            kwargs.pop("near_goal_tcp_speed_coef", 0.3))
        self._near_goal_qvel_coef = float(
            kwargs.pop("near_goal_qvel_coef", 0.2))

        # ---- Goal dwell ----
        self._goal_dwell_cube_speed_threshold = float(
            kwargs.pop("goal_dwell_cube_speed_threshold", 0.03))
        self._goal_dwell_robot_qvel_threshold = float(
            kwargs.pop("goal_dwell_robot_qvel_threshold", 0.3))
        self._goal_dwell_3_bonus = float(
            kwargs.pop("goal_dwell_3_bonus", 1.0))
        self._goal_dwell_5_bonus = float(
            kwargs.pop("goal_dwell_5_bonus", 2.0))
        self._goal_dwell_10_bonus = float(
            kwargs.pop("goal_dwell_10_bonus", 5.0))

        # ---- Goal exit ----
        self._goal_exit_penalty_coef = float(
            kwargs.pop("goal_exit_penalty_coef", 8.0))

        # ---- Reward rebalancing ----
        # Override parent's transport_lift_reward_scale and success_reward_bonus
        self._transport_lift_reward_scale = float(
            kwargs.pop("transport_lift_reward_scale", 0.02))
        self._success_reward_bonus = float(
            kwargs.pop("success_reward_bonus", 10.0))

        # ---- Per-env state for Stage 6.6 ----
        self._goal_entry_ever: torch.Tensor | None = None        # (N,) bool
        self._goal_dwell_steps: torch.Tensor | None = None       # (N,) int32
        self._stable_goal_dwell_steps: torch.Tensor | None = None  # (N,) int32
        self._goal_dwell_3_bonus_given: torch.Tensor | None = None
        self._goal_dwell_5_bonus_given: torch.Tensor | None = None
        self._goal_dwell_10_bonus_given: torch.Tensor | None = None
        self._prev_goal_entry_s66: torch.Tensor | None = None     # (N,) bool

        # Reference qpos on device (set in super().__init__)
        self._ref_qpos: torch.Tensor | None = None

        super().__init__(*args, **kwargs)

        # ---- Move reference qpos to device ----
        self._ref_qpos = PANDA_REFERENCE_QPOS.to(self.device)

        # ---- Per-step diagnostic storage (Stage 6.6) ----
        self._last_s66_arm_action_scale: torch.Tensor | None = None
        self._last_s66_wrist_action_scale: torch.Tensor | None = None
        self._last_s66_raw_action_norm: torch.Tensor | None = None
        self._last_s66_exec_action_norm: torch.Tensor | None = None
        self._last_s66_arm_action_norm: torch.Tensor | None = None
        self._last_s66_wrist_action_norm: torch.Tensor | None = None
        self._last_s66_gripper_action: torch.Tensor | None = None
        self._last_s66_posture_error: torch.Tensor | None = None
        self._last_s66_posture_pen: torch.Tensor | None = None
        self._last_s66_posture_gate: torch.Tensor | None = None
        self._last_s66_near_goal_gate: torch.Tensor | None = None
        self._last_s66_cube_speed_near: torch.Tensor | None = None
        self._last_s66_tcp_speed_near: torch.Tensor | None = None
        self._last_s66_qvel_near: torch.Tensor | None = None
        self._last_s66_near_cube_speed_pen: torch.Tensor | None = None
        self._last_s66_near_tcp_speed_pen: torch.Tensor | None = None
        self._last_s66_near_qvel_pen: torch.Tensor | None = None
        self._last_s66_is_goal_entry: torch.Tensor | None = None
        self._last_s66_goal_exit_pen: torch.Tensor | None = None
        self._last_s66_goal_dwell_bonus: torch.Tensor | None = None
        self._last_s66_is_stable_goal_dwell: torch.Tensor | None = None
        self._last_s66_elbow_fold_ratio: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Episode init — extend for Stage 6.6 state
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        num_envs = self.num_envs
        device = self.device

        # Lazy-init Stage 6.6 tensors
        if self._goal_entry_ever is None:
            self._goal_entry_ever = torch.zeros(
                num_envs, dtype=torch.bool, device=device)
            self._goal_dwell_steps = torch.zeros(
                num_envs, dtype=torch.int32, device=device)
            self._stable_goal_dwell_steps = torch.zeros(
                num_envs, dtype=torch.int32, device=device)
            self._goal_dwell_3_bonus_given = torch.zeros(
                num_envs, dtype=torch.bool, device=device)
            self._goal_dwell_5_bonus_given = torch.zeros(
                num_envs, dtype=torch.bool, device=device)
            self._goal_dwell_10_bonus_given = torch.zeros(
                num_envs, dtype=torch.bool, device=device)
            self._prev_goal_entry_s66 = torch.zeros(
                num_envs, dtype=torch.bool, device=device)

        # Reset Stage 6.6 state for newly-reset envs
        self._goal_entry_ever[env_idx] = False
        self._goal_dwell_steps[env_idx] = 0
        self._stable_goal_dwell_steps[env_idx] = 0
        self._goal_dwell_3_bonus_given[env_idx] = False
        self._goal_dwell_5_bonus_given[env_idx] = False
        self._goal_dwell_10_bonus_given[env_idx] = False
        self._prev_goal_entry_s66[env_idx] = False

    # ------------------------------------------------------------------
    # Posture regularization
    # ------------------------------------------------------------------

    def _compute_posture_penalty(
        self, transport_gate: torch.Tensor, near_goal_gate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute posture regularization penalty vs natural reference qpos.

        Only for arm joints (0:7). Gated:
        - OFF when transport_gate=False (approach/grasp phase)
        - Weak when far from goal (posture_far_scale)
        - Strong near goal (posture_far_scale + (1-posture_far_scale)*near_goal_gate)

        Returns: (posture_penalty, posture_error, posture_gate_strength)
        """
        num_envs = self.num_envs
        device = self.device

        qpos = self.agent.robot.get_qpos()[..., :7]  # (N, 7)
        ref = self._ref_qpos  # (7,)

        # Normalised error using joint ranges
        lower = PANDA_JOINT_LOWER.to(device)
        upper = PANDA_JOINT_UPPER.to(device)
        half_range = (upper - lower) / 2.0  # (7,)
        half_range = torch.clamp(half_range, min=0.05)  # avoid div by zero

        # L2 of normalised error per env
        norm_error = ((qpos - ref) / half_range) ** 2  # (N, 7)
        posture_error = norm_error.mean(dim=-1)  # (N,) — mean over 7 joints

        # Gate strength
        ng = near_goal_gate.float()
        gate_strength = (
            self._posture_far_scale
            + (1.0 - self._posture_far_scale) * ng
        )
        gate_f = transport_gate.float() * gate_strength

        posture_pen = posture_error * self._posture_reg_coef * gate_f

        return posture_pen, posture_error, gate_strength

    # ------------------------------------------------------------------
    # Elbow fold detection (simplified link proximity proxy)
    # ------------------------------------------------------------------

    def _compute_elbow_fold_ratio(self) -> torch.Tensor:
        """Detect mechanically folded configurations.

        Uses joint 3 (elbow) and joint 2 (shoulder lift) — when both are
        near their extremes in the same direction, the arm is folded.

        Returns: (N,) float ratio in [0, 1] where >0.5 indicates folding.
        """
        device = self.device
        num_envs = self.num_envs

        qpos = self.agent.robot.get_qpos()[..., :7]  # (N, 7)
        lower = PANDA_JOINT_LOWER.to(device)
        upper = PANDA_JOINT_UPPER.to(device)

        # Normalise to [-1, 1]
        q_norm = 2.0 * (qpos - lower) / (upper - lower) - 1.0  # (N, 7)

        # j2 (index 1): shoulder lift, j3 (index 2): elbow, j4 (index 3): forearm roll
        # Extreme configurations: j2 near -1 and j3 near -1 (arm folded under)
        # OR j2 near +1 and j3 near -1 (arm folded over)
        j2 = q_norm[:, 1]  # shoulder lift
        j3 = q_norm[:, 2]  # elbow
        j4 = q_norm[:, 3]  # forearm roll

        # Fold signal: |j2| near 1 AND |j3| near 1 in opposite or same direction
        # Simple composite: max(|j2|, |j3|, |j4|) > 0.85
        extreme = torch.max(torch.abs(torch.stack([j2, j3, j4], dim=-1)), dim=-1).values
        elbow_fold = torch.clamp((extreme - 0.7) / 0.3, 0.0, 1.0)  # (N,)

        return elbow_fold

    # ------------------------------------------------------------------
    # Near-goal gate
    # ------------------------------------------------------------------

    def _compute_near_goal_gate(
        self, cube_goal_dist: torch.Tensor, transport_gate: torch.Tensor
    ) -> torch.Tensor:
        """Smooth near-goal gate based on distance to goal.

        Returns: (N,) float gate in [0, 1].
        """
        far = self._action_scale_far_dist  # 0.12
        near = self._action_scale_near_dist  # 0.04

        # Clamp progress to [0, 1]
        progress = (far - cube_goal_dist) / max(far - near, 0.001)
        progress = torch.clamp(progress, 0.0, 1.0)

        gate = progress * transport_gate.float()
        return gate

    # ------------------------------------------------------------------
    # Action scaling — apply in step()
    # ------------------------------------------------------------------

    def _compute_action_scale(
        self, near_goal_gate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-env scaling factors for arm and wrist joint deltas.

        Returns: (arm_scale, wrist_scale), both (N,) in [min, 1.0].
        """
        ng = near_goal_gate  # (N,) in [0, 1]

        arm_scale = 1.0 - ng * (1.0 - self._arm_delta_scale_min)
        wrist_scale = 1.0 - ng * (1.0 - self._wrist_delta_scale_min)

        return arm_scale, wrist_scale

    def _apply_action_scaling(
        self, action: torch.Tensor,
        arm_scale: torch.Tensor,
        wrist_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Scale arm joint deltas (indices 0:7) based on near-goal gate.

        - All arm joints (0:7) get arm_scale
        - Wrist joints (4, 5) get additionally scaled by wrist_scale/arm_scale
          ratio so wrist = action * wrist_scale, non-wrist arm = action * arm_scale
        - Gripper (index 7) is UNCHANGED
        """
        action_scaled = action.clone()
        # Ensure scales are on the same device as action
        arm_scale = arm_scale.to(action.device)
        wrist_scale = wrist_scale.to(action.device)

        # Apply arm_scale to all 7 arm joints
        for i in range(7):
            if i in WRIST_JOINT_INDICES:
                action_scaled[..., i] = action[..., i] * wrist_scale
            else:
                action_scaled[..., i] = action[..., i] * arm_scale

        # Gripper (index 7) — unchanged
        return action_scaled

    # ------------------------------------------------------------------
    # Step override — apply action scaling, then call parent step
    # ------------------------------------------------------------------

    def step(self, action):
        """Apply near-goal action scaling, then delegate to parent step().

        The action scaling is a deterministic transform applied BEFORE
        the gripper mask (which runs inside the parent step chain).
        PPO logprob is computed against the ORIGINAL policy action;
        the scaling is treated as part of the environment dynamics.
        """
        # ---- Compute action scaling ----
        cube_goal_dist = self._get_cube_goal_dist()

        # Transport gate requires stable_grasp + lift
        # Guard against None on first step before parent reward has run
        is_grasped = self.agent.is_grasping(self.cube)
        _sg = getattr(self, "_last_stable_grasp", None)
        _lh = getattr(self, "_last_cube_lift_height", None)
        stable_grasp = (
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            if _sg is None else _sg)
        lift_height = (
            torch.zeros(self.num_envs, device=self.device)
            if _lh is None else _lh)
        transport_gate = stable_grasp & (
            lift_height >= self._transport_min_lift_height)

        near_goal_gate = self._compute_near_goal_gate(
            cube_goal_dist, transport_gate)
        arm_scale, wrist_scale = self._compute_action_scale(near_goal_gate)

        # ---- Ensure action is on the correct device ----
        if not isinstance(action, torch.Tensor):
            action = torch.from_numpy(action).float()
        if action.device != self.device:
            action = action.to(self.device)

        # ---- Apply scaling ----
        action_scaled = self._apply_action_scaling(
            action, arm_scale, wrist_scale)

        # ---- Store diagnostics ----
        raw_arm_norm = torch.linalg.norm(action[..., :7], dim=-1)  # (N,)
        raw_wrist_norm = torch.linalg.norm(
            action[..., WRIST_JOINT_INDICES], dim=-1)  # (N,)
        scaled_arm_norm = torch.linalg.norm(action_scaled[..., :7], dim=-1)
        scaled_wrist_norm = torch.linalg.norm(
            action_scaled[..., WRIST_JOINT_INDICES], dim=-1)

        self._last_s66_raw_action_norm = torch.linalg.norm(action, dim=-1)
        self._last_s66_exec_action_norm = torch.linalg.norm(
            action_scaled, dim=-1)
        self._last_s66_arm_action_norm = scaled_arm_norm
        self._last_s66_wrist_action_norm = scaled_wrist_norm
        self._last_s66_gripper_action = action[..., GRIPPER_INDEX].clone()
        self._last_s66_arm_action_scale = arm_scale
        self._last_s66_wrist_action_scale = wrist_scale
        self._last_s66_near_goal_gate = near_goal_gate

        return super().step(action_scaled)

    # ------------------------------------------------------------------
    # Transport reward override
    # ------------------------------------------------------------------

    def _compute_transport_reward(
        self,
        is_grasped: torch.Tensor,
        stable_grasp: torch.Tensor,
        lift_height: torch.Tensor,
        success: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        """Add Stage 6.6 reward components on top of Stage 6.5.

        New components:
        - Posture regularization (gated)
        - Near-goal speed penalties
        - Goal-dwell tracking + bonuses
        - Goal exit penalty (strengthened)
        - Elbow fold penalty
        """
        # ---- Call Stage 6.5 parent to get ALL existing rewards ----
        transport_reward = super()._compute_transport_reward(
            is_grasped=is_grasped,
            stable_grasp=stable_grasp,
            lift_height=lift_height,
            success=success,
            action=action,
        )

        num_envs = self.num_envs
        device = self.device

        transport_gate = stable_grasp & (
            lift_height >= self._transport_min_lift_height)
        gate_f = transport_gate.float()

        cube_goal_dist = self._get_cube_goal_dist()

        # ---- Compute gates ----
        near_goal_gate = self._compute_near_goal_gate(
            cube_goal_dist, transport_gate)
        near_goal_f = near_goal_gate  # already [0,1] float

        is_obj_placed = cube_goal_dist <= self.goal_thresh

        # ================================================================
        # 28. Posture regularization (NEW)
        # ================================================================
        posture_pen, posture_error, posture_gate = (
            self._compute_posture_penalty(transport_gate, near_goal_gate))
        transport_reward -= posture_pen

        # ================================================================
        # 29. Near-goal speed penalties (NEW)
        #     Penalise cube speed, TCP speed, arm qvel near goal
        # ================================================================
        cube_speed = self._get_cube_speed()
        cube_speed_clipped = torch.clamp(cube_speed, max=1.0)
        near_cube_speed_pen = (
            cube_speed_clipped
            * self._near_goal_cube_speed_coef
            * near_goal_f
            * gate_f
        )
        transport_reward -= near_cube_speed_pen

        # TCP speed (from position difference between steps)
        tcp_pos = self.agent.tcp_pose.p  # (N, 3)
        if self._prev_tcp_pos_s66 is None:
            self._prev_tcp_pos_s66 = tcp_pos.clone()
            self._tcp_speed_s66 = torch.zeros(num_envs, device=device)
        tcp_vel_est = (tcp_pos - self._prev_tcp_pos_s66) / max(
            self.physx_dt if hasattr(self, 'physx_dt') else 0.04, 0.001)
        tcp_speed = torch.linalg.norm(tcp_vel_est, dim=-1)  # (N,)
        tcp_speed_clipped = torch.clamp(tcp_speed, max=1.0)
        # Store for next step
        self._prev_tcp_pos_s66 = tcp_pos.clone()
        self._tcp_speed_s66 = tcp_speed
        near_tcp_speed_pen = (
            tcp_speed_clipped
            * self._near_goal_tcp_speed_coef
            * near_goal_f
            * gate_f
        )
        transport_reward -= near_tcp_speed_pen

        # Robot arm qvel norm
        robot_qvel_norm = self._get_robot_qvel_norm()
        robot_qvel_clipped = torch.clamp(robot_qvel_norm, max=1.0)
        near_qvel_pen = (
            robot_qvel_clipped
            * self._near_goal_qvel_coef
            * near_goal_f
            * gate_f
        )
        transport_reward -= near_qvel_pen

        # ================================================================
        # 30. Goal-dwell tracking (NEW)
        # ================================================================
        # Dwell: inside goal zone AND cube speed low AND robot qvel low
        is_goal_dwell = (
            is_obj_placed
            & (cube_speed < self._goal_dwell_cube_speed_threshold)
            & (robot_qvel_norm < self._goal_dwell_robot_qvel_threshold)
        )
        self._goal_dwell_steps = torch.where(
            is_goal_dwell,
            self._goal_dwell_steps + 1,
            torch.zeros_like(self._goal_dwell_steps),
        )
        # Stable dwell: goal_dwell AND also TCP speed low
        is_stable_goal_dwell = (
            is_goal_dwell
            & (tcp_speed < self._goal_dwell_cube_speed_threshold)
        )
        self._stable_goal_dwell_steps = torch.where(
            is_stable_goal_dwell,
            self._stable_goal_dwell_steps + 1,
            torch.zeros_like(self._stable_goal_dwell_steps),
        )

        # Dwell bonuses (one-time)
        dwell_bonus_s66 = torch.zeros(num_envs, device=device)
        new_3 = (
            (self._goal_dwell_steps >= 3)
            & ~self._goal_dwell_3_bonus_given
            & is_obj_placed
        )
        dwell_bonus_s66 += new_3.float() * self._goal_dwell_3_bonus
        self._goal_dwell_3_bonus_given |= new_3
        new_5 = (
            (self._goal_dwell_steps >= 5)
            & ~self._goal_dwell_5_bonus_given
            & is_obj_placed
        )
        dwell_bonus_s66 += new_5.float() * self._goal_dwell_5_bonus
        self._goal_dwell_5_bonus_given |= new_5
        new_10 = (
            (self._goal_dwell_steps >= 10)
            & ~self._goal_dwell_10_bonus_given
            & is_obj_placed
        )
        dwell_bonus_s66 += new_10.float() * self._goal_dwell_10_bonus
        self._goal_dwell_10_bonus_given |= new_10
        transport_reward += dwell_bonus_s66

        # ================================================================
        # 31. Goal exit penalty (STRENGTHENED vs Stage 6)
        # ================================================================
        # Track whether env ever entered goal zone
        self._goal_entry_ever = self._goal_entry_ever | is_obj_placed

        # Detect exit: was in goal zone, now not
        goal_exit_event = (
            self._prev_goal_entry_s66
            & ~is_obj_placed
            & self._goal_entry_ever
            & ~success
            & transport_gate
        )
        goal_exit_pen = (
            goal_exit_event.float() * self._goal_exit_penalty_coef)
        transport_reward -= goal_exit_pen

        self._prev_goal_entry_s66 = is_obj_placed.clone()

        # ================================================================
        # 32. Elbow fold penalty (NEW — diagnostic + penalty)
        # ================================================================
        elbow_fold = self._compute_elbow_fold_ratio()
        # Small penalty for extreme folding — stronger near goal
        elbow_pen = (
            elbow_fold * 0.1 * gate_f * (1.0 + near_goal_f * 2.0))
        transport_reward -= elbow_pen

        # ---- Store Stage 6.6 diagnostics ----
        self._last_s66_posture_error = posture_error
        self._last_s66_posture_pen = posture_pen
        self._last_s66_posture_gate = posture_gate
        self._last_s66_near_goal_gate = near_goal_gate
        self._last_s66_cube_speed_near = (
            cube_speed_clipped * near_goal_f)
        self._last_s66_tcp_speed_near = (
            tcp_speed_clipped * near_goal_f)
        self._last_s66_qvel_near = (
            robot_qvel_clipped * near_goal_f)
        self._last_s66_near_cube_speed_pen = near_cube_speed_pen
        self._last_s66_near_tcp_speed_pen = near_tcp_speed_pen
        self._last_s66_near_qvel_pen = near_qvel_pen
        self._last_s66_is_goal_entry = is_obj_placed
        self._last_s66_goal_exit_pen = goal_exit_pen
        self._last_s66_goal_dwell_bonus = dwell_bonus_s66
        self._last_s66_is_stable_goal_dwell = is_stable_goal_dwell
        self._last_s66_elbow_fold_ratio = elbow_fold

        return transport_reward

    # ------------------------------------------------------------------
    # Main reward override (reuse Stage 6.5 MRO logic)
    # ------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """Compute dense reward with all Stage 6.5 + 6.6 components.

        Uses same MRO skip as Stage 6.5 — skip ALL transport layers.
        """
        from .pick_cube_target_transport_curriculum import (
            PickCubeTargetTransportCurriculumEnv)

        # Skip ALL transport layers — get only Stage 4.5 parent
        parent_reward = super(
            PickCubeTargetTransportCurriculumEnv, self
        ).compute_dense_reward(obs, action, info)

        is_grasped = info["is_grasped"]
        success = info.get(
            "success",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))

        stable_grasp = getattr(
            self, "_last_stable_grasp",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        lift_height = getattr(
            self, "_last_cube_lift_height",
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
    # Stage 6.6 diagnostic info (extended)
    # ------------------------------------------------------------------

    def get_posture_stable_info(self):
        """Return Stage 6.6 posture-stable diagnostics (36-tuple).

        [0]  arm_action_scale
        [1]  wrist_action_scale
        [2]  raw_action_norm
        [3]  executed_action_norm
        [4]  arm_action_norm
        [5]  wrist_action_norm
        [6]  gripper_action
        [7]  near_goal_gate
        [8]  posture_error
        [9]  posture_penalty
        [10] posture_gate_strength
        [11] near_cube_speed_pen
        [12] near_tcp_speed_pen
        [13] near_qvel_pen
        [14] cube_speed_near_goal
        [15] tcp_speed_near_goal
        [16] arm_qvel_near_goal
        [17] is_goal_entry
        [18] goal_entry_ever
        [19] goal_exit_penalty
        [20] goal_dwell_steps
        [21] stable_goal_dwell_steps
        [22] is_stable_goal_dwell
        [23] goal_dwell_bonus
        [24] elbow_fold_ratio
        [25] robot_qvel_norm
        [26] cube_speed
        [27] tcp_speed
        [28] transport_gate
        [29] is_obj_placed
        [30] cube_goal_dist
        [31] joint_limit_pen (from S6.5)
        [32] max_norm_qpos (from S6.5)
        [33] wrist_near_limit (from S6.5)
        [34] dragging (from S6.5)
        [35] grasp_reference_valid (from S6.5)
        """
        # Compute some on-the-fly
        cube_speed = self._get_cube_speed()
        robot_qvel_norm = self._get_robot_qvel_norm()
        tcp_speed = getattr(self, "_tcp_speed_s66",
            torch.zeros(self.num_envs, device=self.device))

        cube_goal_dist = self._get_cube_goal_dist()
        is_obj_placed = cube_goal_dist <= self.goal_thresh

        is_grasped = self.agent.is_grasping(self.cube)
        stable_grasp = getattr(
            self, "_last_stable_grasp",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        lift_height = getattr(
            self, "_last_cube_lift_height",
            torch.zeros(self.num_envs, device=self.device))
        transport_gate = stable_grasp & (
            lift_height >= self._transport_min_lift_height)

        return (
            getattr(self, "_last_s66_arm_action_scale", None),     # [0]
            getattr(self, "_last_s66_wrist_action_scale", None),   # [1]
            getattr(self, "_last_s66_raw_action_norm", None),      # [2]
            getattr(self, "_last_s66_exec_action_norm", None),     # [3]
            getattr(self, "_last_s66_arm_action_norm", None),      # [4]
            getattr(self, "_last_s66_wrist_action_norm", None),    # [5]
            getattr(self, "_last_s66_gripper_action", None),       # [6]
            getattr(self, "_last_s66_near_goal_gate", None),       # [7]
            getattr(self, "_last_s66_posture_error", None),        # [8]
            getattr(self, "_last_s66_posture_pen", None),          # [9]
            getattr(self, "_last_s66_posture_gate", None),         # [10]
            getattr(self, "_last_s66_near_cube_speed_pen", None),  # [11]
            getattr(self, "_last_s66_near_tcp_speed_pen", None),   # [12]
            getattr(self, "_last_s66_near_qvel_pen", None),        # [13]
            getattr(self, "_last_s66_cube_speed_near", None),      # [14]
            getattr(self, "_last_s66_tcp_speed_near", None),       # [15]
            getattr(self, "_last_s66_qvel_near", None),            # [16]
            is_obj_placed,                                          # [17]
            getattr(self, "_goal_entry_ever", None),                # [18]
            getattr(self, "_last_s66_goal_exit_pen", None),        # [19]
            getattr(self, "_goal_dwell_steps", None),               # [20]
            getattr(self, "_stable_goal_dwell_steps", None),        # [21]
            getattr(self, "_last_s66_is_stable_goal_dwell", None),  # [22]
            getattr(self, "_last_s66_goal_dwell_bonus", None),      # [23]
            getattr(self, "_last_s66_elbow_fold_ratio", None),      # [24]
            robot_qvel_norm,                                        # [25]
            cube_speed,                                              # [26]
            tcp_speed,                                               # [27]
            transport_gate,                                          # [28]
            is_obj_placed,                                           # [29]
            cube_goal_dist,                                          # [30]
            getattr(self, "_last_s65_joint_limit_pen", None),       # [31]
            getattr(self, "_last_s65_max_norm_qpos", None),         # [32]
            getattr(self, "_last_s65_wrist_near_limit", None),      # [33]
            getattr(self, "_last_s65_dragging", None),              # [34]
            getattr(self, "_grasp_reference_valid", None),          # [35]
        )

    def get_reward_components(self):
        """Return all reward components including Stage 6.6 additions."""
        parent = super().get_reward_components()  # 35-tuple from Stage 6.5

        s66_sum = torch.zeros(self.num_envs, device=self.device)
        for attr in [
            "_last_s66_posture_pen",
            "_last_s66_near_cube_speed_pen",
            "_last_s66_near_tcp_speed_pen",
            "_last_s66_near_qvel_pen",
            "_last_s66_goal_exit_pen",
        ]:
            val = getattr(self, attr, None)
            if val is not None:
                s66_sum = s66_sum + val

        # Add goal_dwell_bonus
        dw = getattr(self, "_last_s66_goal_dwell_bonus", None)
        if dw is not None:
            s66_sum = s66_sum + dw

        # Elbow fold penalty
        ef = getattr(self, "_last_s66_elbow_fold_ratio", None)
        if ef is not None:
            # penalty ≈ elbow_fold * 0.1 * gate * (1 + near*2)
            ng = getattr(self, "_last_s66_near_goal_gate", None)
            tg = getattr(self, "_last_transport_gate", None)
            if ng is not None and tg is not None:
                gf = tg.float()
                s66_sum = s66_sum - ef * 0.1 * gf * (1.0 + ng * 2.0)

        return parent + (s66_sum,)
