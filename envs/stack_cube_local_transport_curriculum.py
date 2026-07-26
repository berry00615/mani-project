"""Local-target StackCube curriculum with non-farmable transport rewards."""

from typing import Any

import torch
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("StackCubeLocalTransportCurriculum-v1", max_episode_steps=100)
class StackCubeLocalTransportCurriculumEnv(StackCubeEnv):
    def __init__(self, *args, **kwargs):
        self.target_distance_min = float(kwargs.pop("target_distance_min", 0.07))
        self.target_distance_max = float(kwargs.pop("target_distance_max", 0.11))
        self.progress_scale = float(kwargs.pop("transport_progress_scale", 300.0))
        self.best_scale = float(kwargs.pop("best_improvement_scale", 150.0))
        self.progress_clip = float(kwargs.pop("transport_progress_clip", 0.02))
        self.drop_penalty = float(kwargs.pop("drop_penalty", 8.0))
        self.release_penalty = float(kwargs.pop("premature_release_penalty", 5.0))
        self.on_target_bonus = float(kwargs.pop("on_target_bonus", 10.0))
        self.success_bonus = float(kwargs.pop("success_bonus", 100.0))
        self.time_penalty = float(kwargs.pop("time_penalty", 0.01))
        self._prev_goal_dist = None
        self._best_goal_dist = None
        self._prev_grasped = None
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
        # Put B in a collision-free local ring around A. This shortens only
        # the transport phase; grasping, stacking geometry and success remain
        # identical to upstream StackCube.
        b = len(env_idx)
        angle = torch.rand(b, device=self.device) * (2 * torch.pi)
        radius = self.target_distance_min + torch.rand(b, device=self.device) * (
            self.target_distance_max - self.target_distance_min)
        offset = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1) * radius[:, None]
        reset_p = self.cubeB.pose.p[env_idx].clone()
        reset_p[:, :2] = self.cubeA.pose.p[env_idx, :2] + offset
        reset_p[:, 0].clamp_(-0.18, 0.18)
        reset_p[:, 1].clamp_(-0.28, 0.28)
        # Actor.set_pose follows ManiSkill's active reset mask, so partial
        # resets must pass exactly b poses rather than all num_envs poses.
        self.cubeB.set_pose(Pose.create_from_pq(
            p=reset_p, q=self.cubeB.pose.q[env_idx].clone()))

        if self._prev_goal_dist is None:
            self._prev_goal_dist = torch.zeros(self.num_envs, device=self.device)
            self._best_goal_dist = torch.zeros(self.num_envs, device=self.device)
            self._prev_grasped = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device)
        dist = self._goal_distance()
        self._prev_goal_dist[env_idx] = dist[env_idx]
        self._best_goal_dist[env_idx] = dist[env_idx]
        self._prev_grasped[env_idx] = False

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        upstream = super().compute_dense_reward(obs, action, info)
        grasped = info["is_cubeA_grasped"]
        on_target = info["is_cubeA_on_cubeB"]
        goal_dist = self._goal_distance()
        progress = torch.clamp(
            self._prev_goal_dist - goal_dist, -self.progress_clip, self.progress_clip)
        improvement = torch.clamp(
            self._best_goal_dist - goal_dist, 0.0, self.progress_clip)
        dropped = self._prev_grasped & ~grasped & ~on_target

        reward = (
            upstream
            + self.progress_scale * progress * grasped.float()
            + self.best_scale * improvement * grasped.float()
            + self.on_target_bonus * on_target.float()
            - self.drop_penalty * dropped.float()
            - self.release_penalty * (dropped & (goal_dist > 0.05)).float()
            + self.success_bonus * info["success"].float()
            - self.time_penalty
        )
        self._prev_goal_dist = goal_dist.detach().clone()
        self._best_goal_dist = torch.minimum(self._best_goal_dist, goal_dist.detach())
        self._prev_grasped = grasped.detach().clone()
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info)
