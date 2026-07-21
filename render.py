"""Render and record a trained PPO policy for ManiSkill PickCube-v1."""

import argparse
from pathlib import Path
from typing import Any
import envs  # noqa: F401 - 注册项目自定义环境
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from mani_skill.utils.wrappers import RecordEpisode

class PPOPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        policy_hidden_sizes: list[int],
        value_hidden_sizes: list[int],
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> None:
        super().__init__()

        policy_layers: list[nn.Module] = []
        input_dim = obs_dim

        for hidden_dim in policy_hidden_sizes:
            policy_layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.Tanh(),
                ]
            )
            input_dim = hidden_dim

        self.features = nn.Sequential(*policy_layers)
        self.actor_head = nn.Linear(input_dim, act_dim)

        value_layers: list[nn.Module] = []
        value_input_dim = input_dim

        for hidden_dim in value_hidden_sizes:
            value_layers.extend(
                [
                    nn.Linear(value_input_dim, hidden_dim),
                    nn.Tanh(),
                ]
            )
            value_input_dim = hidden_dim

        value_layers.append(nn.Linear(value_input_dim, 1))
        self.value_net = nn.Sequential(*value_layers)

        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.register_buffer(
            "action_low",
            torch.as_tensor(action_low, dtype=torch.float32),
        )
        self.register_buffer(
            "action_high",
            torch.as_tensor(action_high, dtype=torch.float32),
        )

    @torch.no_grad()
    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.features(obs)
        action_mean = self.actor_head(features)

        return torch.clamp(
            action_mean,
            self.action_low,
            self.action_high,
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="渲染已训练的 ManiSkill PPO 模型")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/ppo_pick_cube/final.pt",
        help="checkpoint 路径",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="录制 episode 数量",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="每个 episode 最大步数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子",
    )
    parser.add_argument(
        "--output-dir",
        default="videos/ppo_pick_cube",
        help="视频输出目录",
    )
    return parser.parse_args()


def is_done(terminated: Any, truncated: Any) -> bool:
    return bool(
        np.asarray(terminated).any()
        or np.asarray(truncated).any()
    )


def extract_rms(rms: Any) -> tuple[np.ndarray, np.ndarray, float] | None:
    if rms is None:
        return None

    if isinstance(rms, dict):
        mean = rms.get("mean")
        var = rms.get("var")
        count = rms.get("count", 1.0)
    else:
        mean = getattr(rms, "mean", None)
        var = getattr(rms, "var", None)
        count = getattr(rms, "count", 1.0)

    if mean is None or var is None:
        return None

    mean_array = np.asarray(mean, dtype=np.float32)
    var_array = np.asarray(var, dtype=np.float32)
    return mean_array, var_array, float(count)


def normalize_observation(
    obs: torch.Tensor,
    obs_rms: tuple[np.ndarray, np.ndarray, float] | None,
    epsilon: float = 1e-8,
    clip_value: float = 10.0,
) -> torch.Tensor:
    if obs_rms is None:
        return obs

    mean, var, _ = obs_rms
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=obs.device)
    var_tensor = torch.as_tensor(var, dtype=torch.float32, device=obs.device)

    normalized = (obs - mean_tensor) / torch.sqrt(var_tensor + epsilon)
    return torch.clamp(normalized, -clip_value, clip_value)


def main() -> None:
    args = parse_args()

    if args.episodes < 1:
        raise ValueError("--episodes 必须大于或等于 1")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")

    print(f"加载 checkpoint：{checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    architecture = checkpoint["architecture"]

    policy = PPOPolicy(
        obs_dim=int(architecture["obs_dim"]),
        act_dim=int(architecture["act_dim"]),
        policy_hidden_sizes=list(architecture["policy_hidden_sizes"]),
        value_hidden_sizes=list(architecture["value_hidden_sizes"]),
        action_low=np.asarray(architecture["action_low"], dtype=np.float32),
        action_high=np.asarray(architecture["action_high"], dtype=np.float32),
    )

    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    policy.eval()

    obs_rms = extract_rms(checkpoint.get("obs_rms"))

    if obs_rms is None:
        print("未检测到可用的 obs_rms，将使用原始观测。")
    else:
        print("已加载观测归一化统计量。")

    env_id = checkpoint.get("env_id", "PickCube-v1")
    obs_mode = checkpoint.get("obs_mode", "state")
    control_mode = checkpoint.get("control_mode", "pd_joint_delta_pos")

    print(f"环境：{env_id}")
    print(f"观测模式：{obs_mode}")
    print(f"控制模式：{control_mode}")
    print(f"训练步数：{checkpoint.get('timestep')}")
    print(f"视频目录：{output_dir}")

    base_env = gym.make(
        env_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        max_episode_steps=args.max_steps,
    )

    env = RecordEpisode(
        base_env,
        output_dir=str(output_dir),
        save_trajectory=False,
        save_video=True,
        save_on_reset=True,
        max_steps_per_video=args.max_steps,
        clean_on_close=True,
        avoid_overwriting_video=True,
    )

    successes = 0

    try:
        env.action_space.seed(args.seed)

        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)

            episode_reward = 0.0
            success = False
            steps = 0

            while steps < args.max_steps:
                if not isinstance(obs, torch.Tensor):
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                else:
                    obs_tensor = obs.detach().to(
                        device="cpu",
                        dtype=torch.float32,
                    )

                if obs_tensor.ndim == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)

                normalized_obs = normalize_observation(obs_tensor, obs_rms)

                action_tensor = policy.deterministic_action(normalized_obs)
                action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)

                obs, reward, terminated, truncated, info = env.step(action)

                # --- Stage 5.5 目标区域诊断 ---
                unwrapped = env.unwrapped

                cube_pos = unwrapped.cube.pose.p
                goal_pos = unwrapped.goal_site.pose.p

                # 单环境渲染，统一取第一个环境
                cube_pos_np = np.asarray(
                    cube_pos.detach().cpu() if hasattr(cube_pos, "detach") else cube_pos
                ).reshape(-1, 3)[0]
                goal_pos_np = np.asarray(
                    goal_pos.detach().cpu() if hasattr(goal_pos, "detach") else goal_pos
                ).reshape(-1, 3)[0]

                goal_dist = float(np.linalg.norm(cube_pos_np - goal_pos_np))
                goal_thresh = float(getattr(unwrapped, "goal_thresh", 0.025))

                placed_geom = goal_dist <= goal_thresh
                near_goal_geom = goal_dist <= 0.05

                info_placed = info.get("is_obj_placed", None)
                info_static = info.get("is_robot_static", None)
                info_success = info.get("success", None)

                def _as_bool(value):
                    if value is None:
                        return None
                    if hasattr(value, "detach"):
                        value = value.detach().cpu().numpy()
                    return bool(np.asarray(value).reshape(-1)[0])

                placed_info = _as_bool(info_placed)
                static_info = _as_bool(info_static)
                success_info = _as_bool(info_success)

                if (
                    near_goal_geom
                    or placed_geom
                    or placed_info
                    or success_info
                    or steps % 10 == 0
                ):
                    print(
                        f"[目标诊断] step={steps:03d} "
                        f"dist={goal_dist:.4f}m "
                        f"threshold={goal_thresh:.4f}m "
                        f"near5cm={near_goal_geom} "
                        f"placed_geom={placed_geom} "
                        f"placed_info={placed_info} "
                        f"static={static_info} "
                        f"success={success_info}"
                    )

                episode_reward += float(np.asarray(reward).mean())
                steps += 1

                if isinstance(info, dict):
                    success_value = info.get("success", False)
                    success = success or bool(np.asarray(success_value).any())

                if is_done(terminated, truncated):
                    break

            successes += int(success)

            print(
                f"episode {episode + 1}/{args.episodes} | "
                f"steps={steps} | "
                f"reward={episode_reward:.3f} | "
                f"success={success}"
            )

    finally:
        env.close()

    print()
    print(f"渲染完成，成功率：{successes}/{args.episodes}")
    print(f"视频已保存到：{output_dir}")


if __name__ == "__main__":
    main()
