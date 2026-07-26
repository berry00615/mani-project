"""
Custom PickCube environment with precision carry curriculum for Stage 6.5.

Registers as ``PickCubePrecisionCarryCurriculum-v1`` via gymnasium.

Extends ``PickCubeGoalBrakeCurriculum-v1`` (Stage 6) with:

1. **Target speed band**: replaces simple cube-speed-penalty with a distance-dependent
   desired forward speed that penalises BOTH too-fast AND too-slow (stationary)
   in the brake zone.

2. **Premature stop penalty**: detects stationary behaviour outside the goal zone
   and applies a small penalty scaled by distance.

3. **Min progress scale**: guarantees progress reward never fully vanishes
   outside the goal zone, providing a tiny persistent incentive to keep advancing.

4. **Precision distance reward**: small continuous ``(1 - dist/brake_zone_outer)``
   signal inside the brake zone.

5. **Near-goal motion shaping**: weaker version of static shaping in the brake
   zone — encourages lower speed but does NOT pay for complete stillness.

6. **Placed zone**: full static shaping, hold, centre, dwell, exit penalty —
   same as Stage 6 but ONLY active once the cube actually enters the placed zone.

7. **Grasp reference relative pose**: records TCP↔cube relative transform at
   first stable grasp and penalises drift during transport.

8. **Gripper down alignment**: weakly penalises sideways wrist orientations.

9. **Joint limit margin penalty**: penalises joint positions near hardware limits,
   with higher weight on wrist joints prone to twisting (j5, j6).

10. **Cube clearance + TCP height**: penalises dragging the cube along the table
    and TCP dropping below the cube centre.

11. **Dragging detection**: penalises low-clearance + high-lateral-speed events.

Phase A & B: inherited from Stage 5.5 (unchanged approach + lift).
Phase C: fully redesigned precision carry.

Relevant YAML keys (new / modified)
------------------------------------

Zone thresholds (inherited from Stage 6):
  brake_zone_outer                   (float, default 0.08)
  placed_control_enter_threshold     (float, default 0.025)
  placed_control_exit_threshold      (float, default 0.032)

Target speed band:
  speed_band_max                     (float, default 0.15)
  speed_band_sigma                   (float, default 0.05)
  speed_band_reward_coef             (float, default 1.0)

Premature stop:
  stop_velocity_threshold            (float, default 0.005)
  premature_stop_steps_threshold     (int,   default 3)
  premature_stop_coef                (float, default 0.3)

Min progress:
  min_progress_scale                 (float, default 0.05)

Precision distance:
  precision_dist_coef                (float, default 0.3)

Near-goal motion shaping:
  near_motion_shaping_coef           (float, default 0.3)
  near_motion_tanh_scale             (float, default 3.0)

Pose constraints:
  rel_pos_error_coef                 (float, default 0.3)
  rel_rot_error_coef                 (float, default 0.5)
  upright_penalty_coef               (float, default 0.1)

Joint limits:
  joint_limit_margin                 (float, default 0.8)
  joint_limit_penalty_coef           (float, default 0.1)
  wrist_joint_weight                 (float, default 1.5)

Clearance:
  min_cube_clearance                 (float, default 0.01)
  cube_clearance_coef                (float, default 2.0)
  min_tcp_cube_offset                (float, default 0.02)
  tcp_height_penalty_coef            (float, default 1.0)

Dragging:
  drag_clearance_threshold           (float, default 0.005)
  drag_speed_threshold               (float, default 0.05)
  drag_penalty_coef                  (float, default 2.0)

All Stage 6 placed-zone and bonus params are inherited unchanged.
"""

from typing import Any

import torch

from mani_skill.utils.registration import register_env

from .pick_cube_goal_brake_curriculum import PickCubeGoalBrakeCurriculumEnv
from .pick_cube_target_transport_curriculum import PickCubeTargetTransportCurriculumEnv

# Panda arm joint indices (first 7 of 8)
ARM_INDICES = [0, 1, 2, 3, 4, 5, 6]

# Panda joint limits from URDF
PANDA_JOINT_LOWER = torch.tensor([
    -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
])
PANDA_JOINT_UPPER = torch.tensor([
    2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973,
])

# Wrist joints (most prone to twisting): j5=index 4, j6=index 5
WRIST_JOINT_INDICES = [4, 5]


@register_env("PickCubePrecisionCarryCurriculum-v1", max_episode_steps=100)
class PickCubePrecisionCarryCurriculumEnv(PickCubeGoalBrakeCurriculumEnv):
    """PickCube with precision carry curriculum for Stage 6.5.

    Inherits all Stage 6 logic (brake zones, placed-control hysteresis,
    hold/centre/dwell/exit rewards) and adds:

    - Target speed band instead of simple speed penalty
    - Premature stop penalty outside goal zone
    - Min progress scale guarantee
    - Precision distance reward
    - Grasp reference pose tracking with relative pose penalties
    - Gripper down alignment, joint limit margin, cube clearance, dragging
    """

    def __init__(self, *args, **kwargs):
        # ---- Pop Stage-6.5-exclusive kwargs before parent ----

        # Target speed band
        self._speed_band_max = float(
            kwargs.pop("speed_band_max", 0.15))
        self._speed_band_sigma = float(
            kwargs.pop("speed_band_sigma", 0.05))
        self._speed_band_reward_coef = float(
            kwargs.pop("speed_band_reward_coef", 1.0))

        # Premature stop
        self._stop_velocity_threshold = float(
            kwargs.pop("stop_velocity_threshold", 0.005))
        self._premature_stop_steps_threshold = int(
            kwargs.pop("premature_stop_steps_threshold", 3))
        self._premature_stop_coef = float(
            kwargs.pop("premature_stop_coef", 0.3))

        # Min progress
        self._min_progress_scale = float(
            kwargs.pop("min_progress_scale", 0.05))

        # Precision distance
        self._precision_dist_coef = float(
            kwargs.pop("precision_dist_coef", 0.3))

        # Near-goal motion shaping
        self._near_motion_shaping_coef = float(
            kwargs.pop("near_motion_shaping_coef", 0.3))
        self._near_motion_tanh_scale = float(
            kwargs.pop("near_motion_tanh_scale", 3.0))

        # Pose constraints
        self._rel_pos_error_coef = float(
            kwargs.pop("rel_pos_error_coef", 0.3))
        self._rel_rot_error_coef = float(
            kwargs.pop("rel_rot_error_coef", 0.5))
        self._upright_penalty_coef = float(
            kwargs.pop("upright_penalty_coef", 0.1))

        # Joint limits
        self._joint_limit_margin = float(
            kwargs.pop("joint_limit_margin", 0.8))
        self._joint_limit_penalty_coef = float(
            kwargs.pop("joint_limit_penalty_coef", 0.1))
        self._wrist_joint_weight = float(
            kwargs.pop("wrist_joint_weight", 1.5))

        # Clearance
        self._min_cube_clearance = float(
            kwargs.pop("min_cube_clearance", 0.01))
        self._cube_clearance_coef = float(
            kwargs.pop("cube_clearance_coef", 2.0))
        self._min_tcp_cube_offset = float(
            kwargs.pop("min_tcp_cube_offset", 0.02))
        self._tcp_height_penalty_coef = float(
            kwargs.pop("tcp_height_penalty_coef", 1.0))

        # Dragging
        self._drag_clearance_threshold = float(
            kwargs.pop("drag_clearance_threshold", 0.005))
        self._drag_speed_threshold = float(
            kwargs.pop("drag_speed_threshold", 0.05))
        self._drag_penalty_coef = float(
            kwargs.pop("drag_penalty_coef", 2.0))

        # ---- Per-env state for Stage 6.5 (BEFORE super().__init__()) ----
        self._grasp_reference_valid: torch.Tensor | None = None
        self._grasp_reference_rel_pos: torch.Tensor | None = None   # (N, 3)
        self._grasp_reference_rel_quat: torch.Tensor | None = None  # (N, 4)
        self._premature_stop_steps: torch.Tensor | None = None      # (N,) int32
        self._prev_stable_grasp_s65: torch.Tensor | None = None     # (N,) bool

        super().__init__(*args, **kwargs)

        # ---- Per-step diagnostic storage ----
        self._last_s65_desired_speed: torch.Tensor | None = None
        self._last_s65_speed_band_rew: torch.Tensor | None = None
        self._last_s65_premature_stop_pen: torch.Tensor | None = None
        self._last_s65_precision_dist_rew: torch.Tensor | None = None
        self._last_s65_near_motion_rew: torch.Tensor | None = None
        self._last_s65_rel_pos_err: torch.Tensor | None = None
        self._last_s65_rel_rot_err: torch.Tensor | None = None
        self._last_s65_rel_pos_pen: torch.Tensor | None = None
        self._last_s65_rel_rot_pen: torch.Tensor | None = None
        self._last_s65_upright_pen: torch.Tensor | None = None
        self._last_s65_down_align: torch.Tensor | None = None
        self._last_s65_joint_limit_pen: torch.Tensor | None = None
        self._last_s65_max_norm_qpos: torch.Tensor | None = None
        self._last_s65_wrist_near_limit: torch.Tensor | None = None
        self._last_s65_cube_clearance: torch.Tensor | None = None
        self._last_s65_tcp_cube_offset: torch.Tensor | None = None
        self._last_s65_clear_pen: torch.Tensor | None = None
        self._last_s65_tcp_height_pen: torch.Tensor | None = None
        self._last_s65_dragging: torch.Tensor | None = None
        self._last_s65_drag_pen: torch.Tensor | None = None
        self._last_s65_parallel_vel: torch.Tensor | None = None
        self._last_s65_forward_speed_err: torch.Tensor | None = None
        self._last_s65_is_premature_stop: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Episode init — extend for Stage 6.5 state
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        num_envs = self.num_envs
        device = self.device

        # Lazy-init Stage 6.5 tensors
        if self._grasp_reference_valid is None:
            self._grasp_reference_valid = torch.zeros(
                num_envs, dtype=torch.bool, device=device)
            self._grasp_reference_rel_pos = torch.zeros(
                num_envs, 3, device=device)
            self._grasp_reference_rel_quat = torch.zeros(
                num_envs, 4, device=device)
            self._premature_stop_steps = torch.zeros(
                num_envs, dtype=torch.int32, device=device)
            self._prev_stable_grasp_s65 = torch.zeros(
                num_envs, dtype=torch.bool, device=device)

        # Reset Stage 6.5 state for newly-reset envs
        self._grasp_reference_valid[env_idx] = False
        self._grasp_reference_rel_pos[env_idx] = 0.0
        self._grasp_reference_rel_quat[env_idx] = 0.0
        self._grasp_reference_rel_quat[env_idx, 0] = 1.0  # identity quat
        self._premature_stop_steps[env_idx] = 0
        self._prev_stable_grasp_s65[env_idx] = False

    # ------------------------------------------------------------------
    # Quaternion utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions. q = (w, x, y, z)."""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=-1)

    @staticmethod
    def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
        """Conjugate of quaternion q = (w, x, y, z)."""
        result = q.clone()
        result[..., 1:] = -result[..., 1:]
        return result

    @staticmethod
    def _quat_geodesic_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Geodesic angle (radians) between two quaternions.

        Handles q ≡ -q ambiguity.
        q1, q2: (..., 4) with format (w, x, y, z).
        Returns: (...,) radians in [0, π].
        """
        dot = torch.abs((q1 * q2).sum(dim=-1))
        dot = torch.clamp(dot, 0.0, 1.0)
        return 2.0 * torch.acos(dot)

    # ------------------------------------------------------------------
    # Pose constraint helpers
    # ------------------------------------------------------------------

    def _get_tcp_rotation_matrix(self) -> torch.Tensor:
        """Return TCP rotation matrix (N, 3, 3) in world frame.

        Columns: [ortho(x), closing(y), approaching(z)].
        """
        return self.agent.tcp_pose.to_transformation_matrix()[..., :3, :3]

    def _get_tcp_approach_axis(self) -> torch.Tensor:
        """Return TCP approach axis (z) in world frame, shape (N, 3)."""
        R = self._get_tcp_rotation_matrix()
        return R[..., :, 2]  # column 2 = approaching = local z

    def _capture_grasp_reference(self, stable_grasp: torch.Tensor):
        """Record TCP-cube relative pose on first stable grasp.

        Only captures for envs where stable_grasp just became True
        and reference is not yet valid.
        """
        newly_stable = stable_grasp & ~self._prev_stable_grasp_s65
        should_capture = newly_stable & ~self._grasp_reference_valid

        if should_capture.any():
            cube_pos = self.cube.pose.p[should_capture]       # (k, 3)
            tcp_pos = self.agent.tcp_pose.p[should_capture]   # (k, 3)
            cube_quat = self.cube.pose.q[should_capture]      # (k, 4)
            tcp_quat = self.agent.tcp_pose.q[should_capture]  # (k, 4)

            # Relative position: cube - tcp (world frame)
            self._grasp_reference_rel_pos[should_capture] = cube_pos - tcp_pos

            # Relative rotation: tcp_quat^-1 * cube_quat
            tcp_quat_inv = self._quat_conjugate(tcp_quat)
            self._grasp_reference_rel_quat[should_capture] = self._quat_multiply(
                tcp_quat_inv, cube_quat)

            self._grasp_reference_valid[should_capture] = True

    def _clear_grasp_reference(self, stable_grasp: torch.Tensor):
        """Clear reference when stable grasp is lost."""
        lost_stable = self._prev_stable_grasp_s65 & ~stable_grasp
        if lost_stable.any():
            self._grasp_reference_valid[lost_stable] = False
            self._grasp_reference_rel_pos[lost_stable] = 0.0
            self._grasp_reference_rel_quat[lost_stable] = 0.0
            self._grasp_reference_rel_quat[lost_stable, 0] = 1.0

    def _compute_relative_pose_error(
        self, stable_grasp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute relative position and rotation errors w.r.t. grasp reference.

        Returns: (rel_pos_error, rel_rot_error_rad), both (N,).
        """
        num_envs = self.num_envs
        device = self.device
        zero = torch.zeros(num_envs, device=device)

        valid = self._grasp_reference_valid & stable_grasp
        if not valid.any():
            return zero, zero

        v = valid

        # Relative position error
        current_rel_pos = self.cube.pose.p[v] - self.agent.tcp_pose.p[v]
        ref_pos = self._grasp_reference_rel_pos[v]
        pos_err = torch.linalg.norm(current_rel_pos - ref_pos, dim=-1)

        # Relative rotation error
        cube_quat = self.cube.pose.q[v]
        tcp_quat = self.agent.tcp_pose.q[v]
        tcp_quat_inv = self._quat_conjugate(tcp_quat)
        current_rel_quat = self._quat_multiply(tcp_quat_inv, cube_quat)
        ref_quat = self._grasp_reference_rel_quat[v]
        rot_err = self._quat_geodesic_angle(current_rel_quat, ref_quat)

        pos_err_full = torch.zeros(num_envs, device=device)
        rot_err_full = torch.zeros(num_envs, device=device)
        pos_err_full[v] = pos_err
        rot_err_full[v] = rot_err
        return pos_err_full, rot_err_full

    # ------------------------------------------------------------------
    # Joint limit helper
    # ------------------------------------------------------------------

    def _compute_joint_limit_penalty(self) -> tuple[torch.Tensor, torch.Tensor,
                                                      torch.Tensor, torch.Tensor]:
        """Compute joint limit margin penalty.

        Returns: (penalty, max_abs_norm_q, wrist_near_limit, norm_qpos_full)
        all shape (N,).
        """
        num_envs = self.num_envs
        device = self.device

        qpos = self.agent.robot.get_qpos()[..., :7]  # (N, 7) arm only
        lower = PANDA_JOINT_LOWER.to(device)
        upper = PANDA_JOINT_UPPER.to(device)

        # Normalize to [-1, +1]
        q_norm = 2.0 * (qpos - lower) / (upper - lower) - 1.0  # (N, 7)

        # Margin penalty: relu(|q_norm| - margin)²
        margin = self._joint_limit_margin
        excess = torch.relu(torch.abs(q_norm) - margin)  # (N, 7)
        excess_sq = excess ** 2

        # Weights: higher for wrist joints
        weights = torch.ones(7, device=device)
        for idx in WRIST_JOINT_INDICES:
            weights[idx] = self._wrist_joint_weight

        weighted = excess_sq * weights  # (N, 7)
        penalty = weighted.sum(dim=-1)  # (N,)

        max_abs_norm = torch.max(torch.abs(q_norm), dim=-1).values  # (N,)

        # Wrist near-limit: any wrist joint |q_norm| > margin
        wrist_norm = torch.abs(q_norm[:, WRIST_JOINT_INDICES])  # (N, 2)
        wrist_near_limit = (wrist_norm > margin).any(dim=-1)  # (N,)

        return penalty, max_abs_norm, wrist_near_limit, q_norm

    # ------------------------------------------------------------------
    # Clearance helpers
    # ------------------------------------------------------------------

    def _compute_clearance_metrics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cube clearance above table and TCP height above cube.

        Table is at z=0 (TableSceneBuilder convention).
        cube_half_size = 0.02 for Panda.

        Returns: (cube_clearance, tcp_cube_offset), both (N,).
        """
        cube_z = self.cube.pose.p[..., 2]           # (N,)
        tcp_z = self.agent.tcp_pose.p[..., 2]       # (N,)
        cube_bottom_z = cube_z - 0.02               # half_size
        cube_clearance = cube_bottom_z              # table at z=0
        tcp_cube_offset = tcp_z - cube_z
        return cube_clearance, tcp_cube_offset

    # ------------------------------------------------------------------
    # Directed transport reward (overrides Stage 6)
    # ------------------------------------------------------------------

    def _compute_transport_reward(
        self,
        is_grasped: torch.Tensor,
        stable_grasp: torch.Tensor,
        lift_height: torch.Tensor,
        success: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        """Compute Phase C transport rewards with precision carry.

        Key changes from Stage 6:
        - Target speed band replaces simple speed penalty
        - Premature stop penalty outside goal zone
        - Min progress scale guarantees non-zero progress incentive
        - Precision distance reward in brake zone
        - Near-goal motion shaping (weaker than placed static shaping)
        - Full static shaping ONLY in placed-control
        - Grasp reference pose constraints
        - Joint limit, clearance, dragging penalties
        """
        num_envs = self.num_envs
        device = self.device

        transport_gate = stable_grasp & (
            lift_height >= self._transport_min_lift_height)
        gate_f = transport_gate.float()

        transport_reward = torch.zeros(num_envs, device=device)

        cube_goal_dist = self._get_cube_goal_dist()

        # --- Handle first-step-of-transport ---
        gate_just_opened = transport_gate & ~self._transport_gate_prev
        self._prev_cube_goal_dist = torch.where(
            gate_just_opened, cube_goal_dist, self._prev_cube_goal_dist)

        if self._transport_started is not None:
            self._transport_started |= transport_gate
            self._initial_transport_goal_dist = torch.where(
                gate_just_opened, cube_goal_dist,
                self._initial_transport_goal_dist)

        if self._best_goal_dist is not None:
            self._best_goal_dist = torch.where(
                gate_just_opened, cube_goal_dist, self._best_goal_dist)

        if self._prev_cube_pos is not None:
            self._prev_cube_pos = torch.where(
                gate_just_opened.unsqueeze(-1),
                self.cube.pose.p, self._prev_cube_pos)

        # --- Compute zones ---
        brake_scale = self._compute_brake_scale(cube_goal_dist)
        self._update_placed_control(cube_goal_dist)
        placed_control = self._placed_control_active
        placed_f = placed_control.float()

        is_obj_placed = cube_goal_dist <= self.goal_thresh
        is_robot_static = self.agent.is_static(0.2)

        is_brake_zone = (
            (cube_goal_dist <= self._brake_zone_outer)
            & transport_gate
            & ~placed_control
        )
        brake_zone_f = is_brake_zone.float()

        # ---- Clear placed state on gate-open ----
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
        # 1. Grasp reference capture / clear (BEFORE reward computation)
        # ================================================================
        self._capture_grasp_reference(stable_grasp)
        self._clear_grasp_reference(stable_grasp)

        # ================================================================
        # 2. Target distance reward (unchanged from Stage 6)
        # ================================================================
        target_dist_rew = (
            1.0 - torch.tanh(self._target_distance_scale * cube_goal_dist)
        ) * self._target_distance_reward_coef * gate_f
        transport_reward += target_dist_rew

        # ================================================================
        # 3. Progress reward — with min_progress_scale
        # ================================================================
        progress = self._prev_cube_goal_dist - cube_goal_dist
        progress = torch.clamp(
            progress, -self._target_progress_clip, self._target_progress_clip)

        # progress_scale = min_progress_scale + (1-min) * brake_scale
        # Guarantees non-zero progress in brake zone
        progress_scale = (
            self._min_progress_scale
            + (1.0 - self._min_progress_scale) * brake_scale
        )
        # Still fully zero in placed-control
        progress_scale = torch.where(
            placed_control, torch.zeros_like(progress_scale), progress_scale)

        target_progress_rew = (
            progress * self._far_progress_reward_coef * gate_f * progress_scale
        )
        transport_reward += target_progress_rew
        self._prev_cube_goal_dist = cube_goal_dist.clone()

        # ================================================================
        # 4. Directional velocity reward — same brake-scale shutdown
        # ================================================================
        cube_pos = self.cube.pose.p
        goal_pos = self.goal_site.pose.p
        goal_vector = goal_pos - cube_pos
        goal_dist_norm = torch.linalg.norm(
            goal_vector, dim=-1, keepdim=True).clamp(min=1e-6)
        goal_direction = goal_vector / goal_dist_norm

        cube_vel = self.cube.linear_velocity
        directional_vel = (cube_vel * goal_direction).sum(dim=-1)
        parallel_vel_val = directional_vel  # scalar

        dir_scale = torch.where(
            placed_control, torch.zeros_like(brake_scale), brake_scale)

        is_near_goal = (
            (cube_goal_dist <= self._near_goal_threshold) & transport_gate)
        near_f = is_near_goal.float()

        direction_rew = (
            torch.clamp(directional_vel, min=0.0, max=self._direction_velocity_clip)
            * self._far_direction_reward_coef
            * gate_f
            * dir_scale
        )
        direction_rew = direction_rew * (
            1.0 - near_f * (1.0 - self._near_goal_direction_scale))
        transport_reward += direction_rew

        wrong_way_pen = (
            torch.clamp(-directional_vel, min=0.0, max=self._direction_velocity_clip)
            * self._far_wrong_way_penalty_coef
            * gate_f
            * dir_scale
        )
        transport_reward -= wrong_way_pen

        # ================================================================
        # 5. Lateral motion penalty (unchanged)
        # ================================================================
        parallel_vel_vec = directional_vel.unsqueeze(-1) * goal_direction
        lateral_vel = cube_vel - parallel_vel_vec
        lateral_vel_norm = torch.linalg.norm(lateral_vel, dim=-1)
        lateral_motion_pen = (
            lateral_vel_norm
            * self._lateral_motion_penalty_coef
            * gate_f
            * brake_scale
        )
        transport_reward -= lateral_motion_pen

        # ================================================================
        # 6. Safe-height + height drop (unchanged)
        # ================================================================
        safe_margin = max(lift_height.max().item(), 0.03)
        safe_height_rew = torch.clamp(
            (lift_height - self._transport_min_lift_height)
            / max(safe_margin, 0.001),
            0.0, 1.0,
        ) * self._transport_safe_height_reward_coef * gate_f
        transport_reward += safe_height_rew

        height_drop = torch.clamp(
            self._transport_min_lift_height - lift_height, min=0.0)
        height_drop_pen = (
            height_drop * self._transport_height_drop_penalty_coef * gate_f)
        transport_reward -= height_drop_pen

        # ================================================================
        # 7. Transport drop penalty (unchanged)
        # ================================================================
        gripper_width = self._get_gripper_width()
        is_gripper_open = gripper_width > self._gripper_open_threshold
        transport_drop_event = (
            self._was_grasped & is_gripper_open & ~is_grasped
            & self._transport_gate_prev
        )
        transport_drop_pen = (
            transport_drop_event.float() * self._transport_drop_penalty)
        transport_reward -= transport_drop_pen

        # ================================================================
        # 8. Lift during transport (unchanged)
        # ================================================================
        lift_continuous_parent = torch.clamp(
            lift_height * self._lift_reward_coef, max=self._lift_reward_max)
        scaled_lift = (
            lift_continuous_parent * self._transport_lift_reward_scale * gate_f)
        lift_adjustment = scaled_lift - lift_continuous_parent * gate_f
        transport_reward += lift_adjustment

        # ================================================================
        # 9. Stagnation penalty — same as Stage 6
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
            stagnation_active = torch.zeros(
                num_envs, dtype=torch.bool, device=device)
            stagnation_pen = torch.zeros(num_envs, device=device)

        # ================================================================
        # 10. Action smoothness — same as Stage 6
        # ================================================================
        if self._prev_action is not None and action is not None:
            action_delta = action[:, :7] - self._prev_action[:, :7]
            action_delta = torch.where(
                gate_just_opened.unsqueeze(-1),
                torch.zeros_like(action_delta),
                action_delta,
            )
            action_delta_l2 = (action_delta ** 2).mean(dim=-1)
            effective_smoothness = (
                self._far_action_smoothness_coef
                + (self._near_action_smoothness_coef
                   - self._far_action_smoothness_coef)
                * (1.0 - brake_scale)
            )
            action_smoothness_pen = (
                action_delta_l2 * effective_smoothness * gate_f)
            transport_reward -= action_smoothness_pen
            self._prev_action = action.clone()
        else:
            action_delta_l2 = torch.zeros(num_envs, device=device)
            action_smoothness_pen = torch.zeros(num_envs, device=device)
            if action is not None and self._prev_action is not None:
                self._prev_action = action.clone()

        # ================================================================
        # 11. Target speed band (NEW — replaces approach speed penalty)
        # ================================================================
        desired_speed = self._speed_band_max * brake_scale  # (N,)
        forward_speed_error = torch.abs(
            parallel_vel_val - desired_speed)  # (N,)

        # Gaussian reward: exp(-error² / (2 * sigma²))
        # Only active in brake zone (not far zone, not placed zone)
        sigma_sq = self._speed_band_sigma ** 2
        speed_band_rew = (
            torch.exp(-forward_speed_error ** 2 / (2.0 * sigma_sq))
            * self._speed_band_reward_coef
            * gate_f
            * brake_zone_f  # only in brake zone
        )
        transport_reward += speed_band_rew

        # ================================================================
        # 12. Premature stop penalty (NEW)
        # ================================================================
        is_stopped = (
            (torch.abs(parallel_vel_val) < self._stop_velocity_threshold)
            & transport_gate
            & ~is_obj_placed
            & (cube_goal_dist <= self._brake_zone_outer)
        )
        self._premature_stop_steps = torch.where(
            is_stopped,
            self._premature_stop_steps + 1,
            torch.zeros_like(self._premature_stop_steps),
        )
        premature_stop_active = (
            self._premature_stop_steps >= self._premature_stop_steps_threshold
        )
        premature_stop_pen = (
            premature_stop_active.float()
            * self._premature_stop_coef
            * brake_scale  # stronger further from goal
            * gate_f
        )
        # Zero when placed
        premature_stop_pen = premature_stop_pen * (~placed_control).float()
        transport_reward -= premature_stop_pen

        # ================================================================
        # 13. Precision distance reward (NEW)
        # ================================================================
        precision_dist_rew = (
            (1.0 - cube_goal_dist / max(self._brake_zone_outer, 0.001))
            * self._precision_dist_coef
            * brake_zone_f
        )
        precision_dist_rew = torch.clamp(precision_dist_rew, min=0.0)
        transport_reward += precision_dist_rew

        # ================================================================
        # 14. Near-goal motion shaping (NEW — weaker than placed static)
        #     Active in brake zone: encourages lower speed, not zero
        # ================================================================
        robot_qvel_norm = self._get_robot_qvel_norm()
        robot_qvel_clipped = torch.clamp(robot_qvel_norm, max=self._robot_qvel_clip)

        near_motion_shaping = (
            1.0 - torch.tanh(
                self._near_motion_tanh_scale * robot_qvel_clipped)
        )
        near_motion_rew = (
            near_motion_shaping
            * self._near_motion_shaping_coef
            * (1.0 - brake_scale)   # stronger near goal
            * brake_zone_f           # ONLY in brake zone
        )
        transport_reward += near_motion_rew

        # ================================================================
        # 15. Placed static shaping (ONLY in placed-control, same as Stg 6)
        # ================================================================
        static_shaping = (
            1.0 - torch.tanh(self._static_shaping_scale * robot_qvel_clipped)
        )
        placed_static_rew = (
            static_shaping
            * self._placed_static_reward_coef
            * placed_f
            * (1.0 + 0.5)  # base + extra boost in placed
        )
        transport_reward += placed_static_rew

        # ================================================================
        # 16. Placed hold reward (unchanged from Stage 6)
        # ================================================================
        cube_speed = self._get_cube_speed()
        cube_speed_clipped = torch.clamp(cube_speed, max=self._cube_speed_clip)
        low_speed_factor = 1.0 - torch.tanh(5.0 * cube_speed_clipped)
        robot_static_factor = 1.0 - torch.tanh(5.0 * robot_qvel_clipped)
        placed_stable_factor = low_speed_factor * robot_static_factor
        placed_hold_rew = (
            self._placed_hold_reward_coef * placed_f * placed_stable_factor)
        transport_reward += placed_hold_rew

        # ================================================================
        # 17. Center reward (unchanged)
        # ================================================================
        center_rew = (
            (1.0 - cube_goal_dist / max(self.goal_thresh, 0.001))
            * self._placed_center_reward_coef
            * placed_f
        )
        center_rew = torch.clamp(center_rew, min=0.0)
        transport_reward += center_rew

        # ================================================================
        # 18. Placed exit penalty (unchanged)
        # ================================================================
        was_placed_prev = self._was_obj_placed_for_exit.clone()
        self._was_obj_placed_for_exit = torch.where(
            placed_control,
            torch.ones_like(self._was_obj_placed_for_exit, dtype=torch.bool),
            self._was_obj_placed_for_exit,
        )
        self._was_obj_placed_for_exit = torch.where(
            (~placed_control)
            & (cube_goal_dist > self._placed_control_exit_threshold),
            torch.zeros_like(self._was_obj_placed_for_exit, dtype=torch.bool),
            self._was_obj_placed_for_exit,
        )
        prev_success = self._prev_ep_success
        placed_exit_event = (
            was_placed_prev
            & ~placed_control
            & ~success
            & ~prev_success
        )
        placed_exit_pen = (
            placed_exit_event.float() * self._placed_exit_penalty_coef)
        transport_reward -= placed_exit_pen

        self._prev_ep_success = success.clone()

        # ================================================================
        # 19. Placed dwell bonuses (unchanged)
        # ================================================================
        self._placed_dwell_steps = torch.where(
            placed_control,
            self._placed_dwell_steps + 1,
            torch.zeros_like(self._placed_dwell_steps),
        )
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
        new_3 = (
            (self._placed_dwell_steps >= 3)
            & ~self._placed_3_bonus_given
            & placed_control
        )
        dwell_bonus += new_3.float() * self._placed_dwell_3_bonus
        self._placed_3_bonus_given |= new_3
        new_5 = (
            (self._placed_dwell_steps >= 5)
            & ~self._placed_5_bonus_given
            & placed_control
        )
        dwell_bonus += new_5.float() * self._placed_dwell_5_bonus
        self._placed_5_bonus_given |= new_5
        new_10 = (
            (self._placed_dwell_steps >= 10)
            & ~self._placed_10_bonus_given
            & placed_control
        )
        dwell_bonus += new_10.float() * self._placed_dwell_10_bonus
        self._placed_10_bonus_given |= new_10
        transport_reward += dwell_bonus

        # ================================================================
        # 20. Placed action magnitude penalty (unchanged)
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
        # 21. Best distance bonus (unchanged shutdown)
        # ================================================================
        if self._best_goal_dist is not None:
            improved = cube_goal_dist < self._best_goal_dist
            improvement = torch.clamp(
                self._best_goal_dist - cube_goal_dist,
                min=0.0, max=self._best_distance_improvement_clip)
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
                cube_goal_dist, self._best_goal_dist)
        else:
            best_dist_bonus = torch.zeros(num_envs, device=device)

        # ================================================================
        # 22. Relative pose penalties (NEW — grasp reference)
        # ================================================================
        rel_pos_err, rel_rot_err = self._compute_relative_pose_error(
            stable_grasp)
        rel_pos_pen = rel_pos_err * self._rel_pos_error_coef * gate_f
        rel_rot_pen = rel_rot_err * self._rel_rot_error_coef * gate_f
        transport_reward -= rel_pos_pen
        transport_reward -= rel_rot_pen

        # ================================================================
        # 23. Gripper down alignment (NEW)
        # ================================================================
        approach_axis = self._get_tcp_approach_axis()  # (N, 3)
        world_down = torch.tensor(
            [0.0, 0.0, -1.0], device=device).expand(num_envs, -1)
        down_alignment = (approach_axis * world_down).sum(dim=-1)  # (N,)
        upright_pen = (
            (1.0 - down_alignment)
            * self._upright_penalty_coef
            * gate_f
        )
        transport_reward -= upright_pen

        # ================================================================
        # 24. Joint limit margin penalty (NEW)
        # ================================================================
        joint_limit_pen, max_norm_q, wrist_near, _ = \
            self._compute_joint_limit_penalty()
        joint_limit_penalty = (
            joint_limit_pen * self._joint_limit_penalty_coef * gate_f)
        transport_reward -= joint_limit_penalty

        # ================================================================
        # 25. Cube clearance + TCP height (NEW)
        # ================================================================
        cube_clearance, tcp_cube_offset = self._compute_clearance_metrics()
        clearance_pen = (
            torch.relu(self._min_cube_clearance - cube_clearance)
            * self._cube_clearance_coef
            * gate_f
        )
        tcp_height_pen = (
            torch.relu(self._min_tcp_cube_offset - tcp_cube_offset)
            * self._tcp_height_penalty_coef
            * gate_f
        )
        transport_reward -= clearance_pen
        transport_reward -= tcp_height_pen

        # ================================================================
        # 26. Dragging penalty (NEW)
        # ================================================================
        horizontal_speed = torch.linalg.norm(cube_vel[..., :2], dim=-1)
        dragging = (
            transport_gate
            & (cube_clearance < self._drag_clearance_threshold)
            & (horizontal_speed > self._drag_speed_threshold)
        )
        drag_pen = dragging.float() * self._drag_penalty_coef
        transport_reward -= drag_pen

        # ================================================================
        # 27. One-time bonuses (unchanged)
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
        # Update transport_gate_prev and grasp tracking
        # ================================================================
        self._transport_gate_prev = transport_gate.clone()
        self._prev_stable_grasp_s65 = stable_grasp.clone()

        # ---- Store for diagnostics (parent fields) ----
        self._last_transport_gate = transport_gate
        self._last_transport_bonus = transport_reward
        self._last_target_distance_rew = target_dist_rew
        self._last_target_progress_rew = target_progress_rew
        self._last_safe_height_rew = safe_height_rew
        self._last_height_drop_pen = height_drop_pen
        self._last_transport_drop_pen = transport_drop_pen
        self._last_near_goal_bonus = (
            new_near.float() * self._near_goal_bonus
            if self._near_goal_reached is not None
            else torch.zeros(num_envs, device=device))
        self._last_placed_bonus = (
            new_placed.float() * self._placed_bonus
            if self._placed_reached is not None
            else torch.zeros(num_envs, device=device))
        self._last_success_bonus_st5 = (
            success.float() * self._success_reward_bonus)
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
        self._last_cube_speed_pen = (
            torch.zeros(num_envs, device=device))  # replaced by speed band
        self._last_robot_qvel_norm = robot_qvel_norm
        self._last_static_shaping_rew = placed_static_rew + near_motion_rew
        self._last_placed_hold_rew = placed_hold_rew
        self._last_center_rew = center_rew
        self._last_placed_exit_pen = placed_exit_pen
        self._last_placed_dwell_bonus = dwell_bonus
        self._last_placed_action_mag_pen = placed_action_mag_pen
        self._last_is_obj_placed = is_obj_placed
        self._last_is_robot_static = is_robot_static

        # Stage 6.5 diagnostics
        self._last_s65_desired_speed = desired_speed
        self._last_s65_speed_band_rew = speed_band_rew
        self._last_s65_premature_stop_pen = premature_stop_pen
        self._last_s65_precision_dist_rew = precision_dist_rew
        self._last_s65_near_motion_rew = near_motion_rew
        self._last_s65_rel_pos_err = rel_pos_err
        self._last_s65_rel_rot_err = rel_rot_err
        self._last_s65_rel_pos_pen = rel_pos_pen
        self._last_s65_rel_rot_pen = rel_rot_pen
        self._last_s65_upright_pen = upright_pen
        self._last_s65_down_align = down_alignment
        self._last_s65_joint_limit_pen = joint_limit_penalty
        self._last_s65_max_norm_qpos = max_norm_q
        self._last_s65_wrist_near_limit = wrist_near
        self._last_s65_cube_clearance = cube_clearance
        self._last_s65_tcp_cube_offset = tcp_cube_offset
        self._last_s65_clear_pen = clearance_pen
        self._last_s65_tcp_height_pen = tcp_height_pen
        self._last_s65_dragging = dragging
        self._last_s65_drag_pen = drag_pen
        self._last_s65_parallel_vel = parallel_vel_val
        self._last_s65_forward_speed_err = forward_speed_error
        self._last_s65_is_premature_stop = premature_stop_active

        return transport_reward

    # ------------------------------------------------------------------
    # Main reward (override to skip Stage 6 transport)
    # ------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """Three-phase reward with precision carry transport in Phase C.

        Uses super(PickCubeTargetTransportCurriculumEnv, self) to skip ALL
        transport layers (Stage 5, 5.5, 6), getting only the Stage 4.5
        parent reward (base + collision + gripper + lift).
        Then adds Stage 6.5 transport.

        This is the SAME skip target as Stage 6 — because Stage 6's
        compute_dense_reward also skips to Stage 4.5 lift.  Both stages
        skip past all transport layers to avoid double-counting through
        polymorphic self._compute_transport_reward() calls.
        """
        # Skip ALL transport layers (Stage 5/5.5/6) — get only Stage 4.5 parent
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
    # Stage 6.5 diagnostic info
    # ------------------------------------------------------------------

    def get_precision_carry_info(self):
        """Return Stage 6.5 precision carry diagnostics (29-tuple).

        [0]  desired_forward_speed
        [1]  parallel_velocity
        [2]  forward_speed_error
        [3]  speed_band_reward
        [4]  premature_stop_active
        [5]  premature_stop_penalty
        [6]  premature_stop_steps
        [7]  precision_distance_reward
        [8]  near_motion_shaping_reward
        [9]  grasp_reference_valid
        [10] rel_pos_error
        [11] rel_rot_error_rad
        [12] rel_pos_penalty
        [13] rel_rot_penalty
        [14] gripper_down_alignment
        [15] upright_penalty
        [16] joint_limit_penalty
        [17] max_normalized_joint_position
        [18] wrist_near_limit
        [19] cube_clearance
        [20] tcp_cube_vertical_offset
        [21] clearance_penalty
        [22] tcp_height_penalty
        [23] dragging
        [24] dragging_penalty
        [25] is_brake_zone
        [26] is_placed_control
        [27] cube_goal_dist
        [28] robot_qvel_norm
        """
        return (
            getattr(self, "_last_s65_desired_speed", None),          # [0]
            getattr(self, "_last_s65_parallel_vel", None),           # [1]
            getattr(self, "_last_s65_forward_speed_err", None),      # [2]
            getattr(self, "_last_s65_speed_band_rew", None),         # [3]
            getattr(self, "_last_s65_is_premature_stop", None),      # [4]
            getattr(self, "_last_s65_premature_stop_pen", None),     # [5]
            getattr(self, "_premature_stop_steps", None),            # [6]
            getattr(self, "_last_s65_precision_dist_rew", None),     # [7]
            getattr(self, "_last_s65_near_motion_rew", None),        # [8]
            getattr(self, "_grasp_reference_valid", None),           # [9]
            getattr(self, "_last_s65_rel_pos_err", None),            # [10]
            getattr(self, "_last_s65_rel_rot_err", None),            # [11]
            getattr(self, "_last_s65_rel_pos_pen", None),            # [12]
            getattr(self, "_last_s65_rel_rot_pen", None),            # [13]
            getattr(self, "_last_s65_down_align", None),             # [14]
            getattr(self, "_last_s65_upright_pen", None),            # [15]
            getattr(self, "_last_s65_joint_limit_pen", None),        # [16]
            getattr(self, "_last_s65_max_norm_qpos", None),          # [17]
            getattr(self, "_last_s65_wrist_near_limit", None),       # [18]
            getattr(self, "_last_s65_cube_clearance", None),         # [19]
            getattr(self, "_last_s65_tcp_cube_offset", None),        # [20]
            getattr(self, "_last_s65_clear_pen", None),              # [21]
            getattr(self, "_last_s65_tcp_height_pen", None),         # [22]
            getattr(self, "_last_s65_dragging", None),               # [23]
            getattr(self, "_last_s65_drag_pen", None),               # [24]
            getattr(self, "_last_is_brake_zone", None),              # [25]
            getattr(self, "_last_is_placed_control", None),          # [26]
            getattr(self, "_last_cube_goal_dist", None),             # [27]
            getattr(self, "_last_robot_qvel_norm", None),            # [28]
        )

    def get_reward_components(self):
        """Return all reward components (35-tuple).

        [0-27]: Same as Stage 6
        [28]: speed_band_reward
        [29]: premature_stop_penalty
        [30]: near_motion_shaping_reward
        [31]: rel_pos_penalty
        [32]: rel_rot_penalty
        [33]: upright_penalty
        [34]: joint_limit + clearance + tcp_height + dragging (summed)
        """
        parent = super().get_reward_components()
        s65_sum = torch.zeros(self.num_envs, device=self.device)
        for attr in [
            "_last_s65_clear_pen", "_last_s65_tcp_height_pen",
            "_last_s65_drag_pen",
        ]:
            val = getattr(self, attr, None)
            if val is not None:
                s65_sum = s65_sum + val
        # Add joint limit penalty separately
        jl = getattr(self, "_last_s65_joint_limit_pen", None)
        if jl is not None:
            s65_sum = s65_sum + jl

        return (
            parent[0], parent[1], parent[2], parent[3], parent[4],
            parent[5], parent[6], parent[7], parent[8], parent[9],
            parent[10], parent[11], parent[12], parent[13], parent[14],
            parent[15], parent[16], parent[17], parent[18], parent[19],
            parent[20], parent[21], parent[22], parent[23], parent[24],
            parent[25], parent[26], parent[27],
            getattr(self, "_last_s65_speed_band_rew", None),         # [28]
            getattr(self, "_last_s65_premature_stop_pen", None),     # [29]
            getattr(self, "_last_s65_near_motion_rew", None),        # [30]
            getattr(self, "_last_s65_rel_pos_pen", None),            # [31]
            getattr(self, "_last_s65_rel_rot_pen", None),            # [32]
            getattr(self, "_last_s65_upright_pen", None),            # [33]
            s65_sum,                                                  # [34]
        )
