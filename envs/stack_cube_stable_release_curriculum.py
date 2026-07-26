"""StackCube target braking and stable auto-release curriculum."""
from typing import Any
import torch
from mani_skill.utils.registration import register_env
from .stack_cube_auto_release_curriculum import StackCubeAutoReleaseCurriculumEnv

@register_env("StackCubeStableReleaseCurriculum-v1", max_episode_steps=100)
class StackCubeStableReleaseCurriculumEnv(StackCubeAutoReleaseCurriculumEnv):
    def __init__(self, *args, **kwargs):
        self.static_shaping_scale = float(kwargs.pop("static_shaping_scale", 10.0))
        self.on_target_static_bonus = float(kwargs.pop("on_target_static_bonus", 30.0))
        self.on_target_speed_penalty = float(kwargs.pop("on_target_speed_penalty", 5.0))
        super().__init__(*args, **kwargs)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        reward = super().compute_dense_reward(obs, action, info)
        on_target = info["is_cubeA_on_cubeB"]
        linear_speed = torch.linalg.norm(self.cubeA.linear_velocity, dim=1)
        angular_speed = torch.linalg.norm(self.cubeA.angular_velocity, dim=1)
        motion = linear_speed * 20.0 + angular_speed * 2.0
        static_shaping = 1.0 - torch.tanh(motion)
        return (reward
                + self.static_shaping_scale * static_shaping * on_target.float()
                + self.on_target_static_bonus * (on_target & info["is_cubeA_static"]).float()
                - self.on_target_speed_penalty
                * torch.clamp(linear_speed + 0.1 * angular_speed, 0.0, 1.0)
                * on_target.float())
