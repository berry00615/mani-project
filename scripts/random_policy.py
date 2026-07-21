"""Run a configurable random policy in a ManiSkill GUI environment."""

import argparse
from typing import Any

import gymnasium as gym
import numpy as np
import mani_skill.envs  # noqa: F401 - registers ManiSkill environments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 ManiSkill 随机策略")
    parser.add_argument("--env-id", default="PickCube-v1", help="Gymnasium 环境 ID")
    parser.add_argument("--steps", type=int, default=1000, help="总运行步数")
    parser.add_argument("--seed", type=int, default=0, help="初始随机种子")
    return parser.parse_args()


def scalar(value: Any) -> float:
    return float(np.asarray(value).sum())


def is_done(terminated: Any, truncated: Any) -> bool:
    return bool(np.asarray(terminated).any() or np.asarray(truncated).any())


def main() -> None:
    args = parse_args()
    if args.steps < 0:
        raise ValueError("--steps 必须大于或等于 0")

    env = gym.make(
        args.env_id,
        obs_mode="state",
        render_mode="human",
        sim_backend="physx_cpu",
    )
    completed_episodes = 0
    total_reward = 0.0
    completed_episode_rewards: list[float] = []
    current_episode_reward = 0.0
    try:
        env.action_space.seed(args.seed)
        env.reset(seed=args.seed)
        print(f"环境 {args.env_id} 已创建，将运行 {args.steps} 步。")
        for _ in range(args.steps):
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            env.render()
            step_reward = scalar(reward)
            total_reward += step_reward
            current_episode_reward += step_reward
            if is_done(terminated, truncated):
                completed_episodes += 1
                completed_episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                env.reset()
    finally:
        env.close()

    average_reward = (
        sum(completed_episode_rewards) / completed_episodes
        if completed_episodes
        else 0.0
    )
    print("随机策略运行结束")
    print(f"总步数: {args.steps}")
    print(f"episode 数量: {completed_episodes}")
    print(f"累计 reward: {total_reward:.6f}")
    print(f"平均 episode reward: {average_reward:.6f}")


if __name__ == "__main__":
    main()
