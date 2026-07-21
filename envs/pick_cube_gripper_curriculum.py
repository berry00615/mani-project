"""
Custom PickCube environment with gripper action masking curriculum.

Registers as ``PickCubeGripperCurriculum-v1`` via gymnasium.

Extends ``PickCubeCollisionGripper-v1`` with action-level gripper masking
that **forces the gripper to stay open** when the TCP is far from the cube.
This prevents the policy from learning the bad habit of closing the gripper
before reaching the cube.

Masking rule
------------
When ``force_gripper_open_enabled=True`` and the TCP-to-cube distance
exceeds ``force_gripper_open_until_distance`` **and** the cube is NOT
currently grasped:

    action[..., -1] = +1.0   (force gripper open)

Otherwise the policy's original gripper action is used as-is.

Design
------
The masking is applied in ``step()`` **before** calling ``super().step()``,
so the controller never sees the original gripper action.  The policy
network and checkpoint format are unchanged — only the environment
intercepts the action.

Relevant YAML configuration keys
---------------------------------
- ``force_gripper_open_enabled`` (bool, default True)
- ``force_gripper_open_until_distance`` (float, default 0.10)

Logging (exposed via ``get_gripper_mask_info()``)
--------------------------------------------------
- ``forced_open_mask``: bool tensor, True where gripper action was overridden
- ``policy_gripper_action``: the original action[..., -1] from the policy
- ``executed_gripper_action``: the actual action[..., -1] sent to the controller
"""

from typing import Any, Union

import numpy as np
import torch

from mani_skill.utils.registration import register_env

from .pick_cube_collision_gripper import PickCubeCollisionGripperEnv


@register_env("PickCubeGripperCurriculum-v1", max_episode_steps=100)
class PickCubeGripperCurriculumEnv(PickCubeCollisionGripperEnv):
    """PickCube with gripper action masking + collision penalty + timing rewards.

    Inherits all logic from ``PickCubeCollisionGripperEnv`` and adds
    action-level gripper masking on top.
    """

    def __init__(self, *args, **kwargs):
        # Pop our new kwargs before the parent constructor sees them
        self._force_gripper_open_enabled = bool(
            kwargs.pop("force_gripper_open_enabled", True)
        )
        self._force_gripper_open_until_distance = float(
            kwargs.pop("force_gripper_open_until_distance", 0.10)
        )

        super().__init__(*args, **kwargs)

        # Per-step tracking for logging
        self._last_mask_active: torch.Tensor | None = None
        self._last_action_overridden: torch.Tensor | None = None
        self._last_policy_requested_close: torch.Tensor | None = None
        self._last_policy_gripper_action: torch.Tensor | None = None
        self._last_executed_gripper_action: torch.Tensor | None = None
        self._last_tcp_cube_distance: torch.Tensor | None = None
        self._last_near_cube: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Action masking
    # ------------------------------------------------------------------

    def _apply_gripper_mask(
        self, action: torch.Tensor
    ) -> torch.Tensor:
        """Modify ``action[..., -1]`` to force gripper open when appropriate.

        Parameters
        ----------
        action : torch.Tensor, shape ``(num_envs, act_dim)``
            Original action from the policy, already on the correct device.

        Returns
        -------
        action : torch.Tensor, same shape
            Action with gripper dimension possibly overridden.

        Metric semantics
        ----------------
        * ``mask_active`` — the masking rule is *eligible* to fire (far AND
          not grasping).  This does NOT mean the action was changed; the
          policy may have already requested an open gripper.
        * ``action_overridden`` — mask active **and** the policy requested
          close (action[-1] < 0).  This is the subset where the environment
          truly changed the policy's intent.
        * ``policy_requested_close`` — policy's original action[-1] < 0.
        * ``near_cube`` — TCP-to-cube distance <= threshold.
        * ``tcp_cube_distance`` — raw Euclidean distance.
        """
        # Current state
        tcp_to_cube_dist = self._compute_tcp_to_cube_dist()
        is_grasped = self.agent.is_grasping(self.cube)
        threshold = self._force_gripper_open_until_distance

        # Save policy's original gripper action for logging
        policy_gripper = action[..., -1].clone()

        # ---- Masking rule ----
        # The mask is *active* when the TCP is too far and we haven't
        # grasped the cube yet.  Active does NOT imply the action was
        # changed — the policy may already want the gripper open.
        mask_active = (tcp_to_cube_dist > threshold) & (~is_grasped)

        # Force gripper open where the mask is active
        action[..., -1] = torch.where(
            mask_active,
            torch.tensor(1.0, device=self.device, dtype=action.dtype),
            action[..., -1],
        )

        # ---- Derived metrics ----
        policy_requested_close = policy_gripper < 0.0
        # Action is truly overridden only when the mask was active AND
        # the policy actually wanted to close the gripper.
        action_overridden = mask_active & policy_requested_close
        near_cube = tcp_to_cube_dist <= threshold

        # Store for logging
        self._last_mask_active = mask_active
        self._last_action_overridden = action_overridden
        self._last_policy_requested_close = policy_requested_close
        self._last_policy_gripper_action = policy_gripper
        self._last_executed_gripper_action = action[..., -1].clone()
        self._last_tcp_cube_distance = tcp_to_cube_dist
        self._last_near_cube = near_cube

        return action

    # ------------------------------------------------------------------
    # Step override — intercept action before controller
    # ------------------------------------------------------------------

    def step(
        self, action: Union[None, np.ndarray, torch.Tensor, dict]
    ):
        """Take one environment step with gripper action masking.

        Masking is applied **before** ``super().step()`` so the controller
        never sees the original gripper action.
        """
        if self._force_gripper_open_enabled and action is not None:
            # Normalise to torch.Tensor on the correct device
            if isinstance(action, np.ndarray):
                action_t = torch.from_numpy(action).to(
                    device=self.device, dtype=torch.float32
                )
                was_numpy = True
            elif isinstance(action, torch.Tensor):
                action_t = action.to(device=self.device, dtype=torch.float32)
                was_numpy = False
            elif isinstance(action, dict):
                # Dict actions (e.g. multi-agent or control_mode switch) —
                # pass through unchanged.
                was_numpy = False
                action_t = action
            else:
                raise TypeError(
                    f"action must be np.ndarray, torch.Tensor, or dict, "
                    f"got {type(action)}"
                )

            # Apply masking only for tensor actions (single-agent case)
            if isinstance(action_t, torch.Tensor):
                action_t = self._apply_gripper_mask(action_t)

            # Convert back if the original was numpy
            if was_numpy:
                action = action_t.cpu().numpy()
            else:
                action = action_t

        return super().step(action)

    # ------------------------------------------------------------------
    # Info extension — expose masking data to the training loop
    # ------------------------------------------------------------------

    def get_gripper_mask_info(self):
        """Return the most recent gripper-masking statistics.

        Called by the training loop after ``env.step``.

        Returns
        -------
        mask_active : torch.Tensor or None, shape ``(num_envs,)``, bool
            True where the masking rule was active (far & not grasped).
            Does NOT mean the action was changed.
        action_overridden : torch.Tensor or None, shape ``(num_envs,)``, bool
            True where mask was active AND policy requested close.
            This is where the env truly overrode the policy's intent.
        policy_requested_close : torch.Tensor or None, shape ``(num_envs,)``, bool
            True where policy's original action[-1] < 0.
        policy_gripper_action : torch.Tensor or None, shape ``(num_envs,)``
            The original action[..., -1] requested by the policy.
        executed_gripper_action : torch.Tensor or None, shape ``(num_envs,)``
            The actual action[..., -1] that was executed.
        tcp_cube_distance : torch.Tensor or None, shape ``(num_envs,)``
            Euclidean distance from TCP to cube centre.
        near_cube : torch.Tensor or None, shape ``(num_envs,)``, bool
            True where TCP-to-cube distance <= threshold.
        """
        return (
            getattr(self, "_last_mask_active", None),
            getattr(self, "_last_action_overridden", None),
            getattr(self, "_last_policy_requested_close", None),
            getattr(self, "_last_policy_gripper_action", None),
            getattr(self, "_last_executed_gripper_action", None),
            getattr(self, "_last_tcp_cube_distance", None),
            getattr(self, "_last_near_cube", None),
        )

    def get_action_dim_mask(self, act_dim: int) -> torch.Tensor | None:
        """Return per-dimension action mask for on-policy PPO.

        When the gripper masking rule is active (far & not grasped),
        the gripper dimension (last dim) was deterministically
        overridden by the environment.  These dimensions should be
        excluded from the PPO policy gradient.

        Parameters
        ----------
        act_dim : int
            Total number of action dimensions.

        Returns
        -------
        mask : torch.Tensor or None, shape ``(num_envs, act_dim)``
            1.0 = dimension was controllable by the policy.
            0.0 = dimension was deterministically overridden.
            Returns None when masking is disabled.
        """
        mask_active = getattr(self, "_last_mask_active", None)
        if mask_active is None:
            return None

        num_envs = mask_active.shape[0]
        mask = torch.ones(num_envs, act_dim, device=self.device)
        mask[mask_active, -1] = 0.0  # mask gripper dimension
        return mask
