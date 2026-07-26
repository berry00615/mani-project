"""StackCube curriculum focused on grasped transport and target alignment."""

from typing import Any

import torch
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.registration import register_env


@register_env("StackCubeTransportCurriculum-v1", max_episode_steps=100)
class StackCubeTransportCurriculumEnv(StackCubeEnv):
    """Preserve upstream success semantics while fixing its transport plateau."""

    def __init__(self, *args, **kwargs):
        self.progress_scale = float(kwargs.pop("transport_progress_scale", 120.0))
        self.progress_clip = float(kwargs.pop("transport_progress_clip", 0.02))
        self.goal_reward_scale = float(kwargs.pop("goal_reward_scale", 4.0))
        self.drop_penalty = float(kwargs.pop("drop_penalty", 4.0))
        self.success_bonus = float(kwargs.pop("success_bonus", 50.0))
        self.time_penalty = float(kwargs.pop("time_penalty", 0.01))
        self._prev_goal_dist = None
        self._was_grasped = None
        super().__init__(*args, **kwargs)

    def _goal_distance(self):
        goal = torch.cat(
            [self.cubeB.pose.p[:, :2],
             (self.cubeB.pose.p[:, 2] + self.cube_half_size[2] * 2)[:, None]],
            dim=1,
        )
        return torch.linalg.norm(self.cubeA.pose.p - goal, dim=1)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        if self._prev_goal_dist is None:
            self._prev_goal_dist = torch.zeros(self.num_envs, device=self.device)
            self._was_grasped = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_goal_dist[env_idx] = self._goal_distance()[env_idx]
        self._was_grasped[env_idx] = False

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        upstream = super().compute_dense_reward(obs, action, info)
        grasped = info["is_cubeA_grasped"]
        goal_dist = self._goal_distance()
        progress = torch.clamp(
            self._prev_goal_dist - goal_dist,
            -self.progress_clip,
            self.progress_clip,
        )

        # Potential-like directional signal applies only after grasping. It
        # cannot be farmed by holding still and strongly rewards motion toward B.
        transport_progress = self.progress_scale * progress * grasped.float()
        goal_shaping = (
            self.goal_reward_scale * (1 - torch.tanh(10 * goal_dist))
            * grasped.float()
        )
        dropped = self._was_grasped & ~grasped & ~info["is_cubeA_on_cubeB"]
        reward = (
            upstream + transport_progress + goal_shaping
            - self.drop_penalty * dropped.float()
            - self.time_penalty
            + self.success_bonus * info["success"].float()
        )
        self._prev_goal_dist = goal_dist.detach().clone()
        self._was_grasped = self._was_grasped | grasped
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Keep reward magnitudes explicit for PPO rather than dividing the
        # success bonus by the upstream hard-coded normalization factor.
        return self.compute_dense_reward(obs, action, info)
