"""StackCube placement curriculum with on-policy-safe automatic release."""

from typing import Union

import numpy as np
import torch
from mani_skill.utils.registration import register_env

from .stack_cube_local_transport_curriculum import StackCubeLocalTransportCurriculumEnv


@register_env("StackCubeAutoReleaseCurriculum-v1", max_episode_steps=100)
class StackCubeAutoReleaseCurriculumEnv(StackCubeLocalTransportCurriculumEnv):
    """Open the gripper only after upstream's exact on-target predicate holds.

    The overridden gripper dimension is excluded from PPO log-probability, so
    placement receives valid policy gradients while release is scaffolded.
    """

    def __init__(self, *args, **kwargs):
        self.auto_release_enabled = bool(kwargs.pop("auto_release_enabled", True))
        self.auto_release_require_static = bool(
            kwargs.pop("auto_release_require_static", False))
        self._last_auto_release_mask = None
        super().__init__(*args, **kwargs)

    def step(self, action: Union[np.ndarray, torch.Tensor, dict, None]):
        if self.auto_release_enabled and action is not None and not isinstance(action, dict):
            eval_info = self.evaluate()
            on_target = eval_info["is_cubeA_on_cubeB"]
            release_mask = on_target
            if self.auto_release_require_static:
                release_mask = release_mask & eval_info["is_cubeA_static"]
            self._last_auto_release_mask = release_mask.detach().clone()
            if isinstance(action, torch.Tensor):
                # Deliberately mutate the policy action in place; the trainer
                # then stores the actually executed action in its rollout.
                action[..., -1][release_mask] = 1.0
            elif isinstance(action, np.ndarray):
                action[..., -1][release_mask.detach().cpu().numpy()] = 1.0
        else:
            self._last_auto_release_mask = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device)
        return super().step(action)

    def get_action_dim_mask(self, act_dim: int):
        mask = torch.ones(self.num_envs, act_dim, device=self.device)
        if self._last_auto_release_mask is not None:
            mask[self._last_auto_release_mask, -1] = 0.0
        return mask
