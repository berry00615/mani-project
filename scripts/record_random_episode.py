"""Record random PickCube episodes as RGB MP4 files."""

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import mani_skill.envs  # noqa: F401 - registers ManiSkill environments
from mani_skill.utils.wrappers import RecordEpisode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="录制 ManiSkill 随机 episode")
    parser.add_argument("--env-id", default="PickCube-v1", help="Gymnasium 环境 ID")
    parser.add_argument("--episodes", type=int, default=3, help="录制 episode 数量")
    parser.add_argument("--seed", type=int, default=0, help="初始随机种子")
    parser.add_argument(
        "--output-dir", default="videos/random", help="视频输出目录"
    )
    return parser.parse_args()


def is_done(terminated: Any, truncated: Any) -> bool:
    return bool(np.asarray(terminated).any() or np.asarray(truncated).any())


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes 必须大于或等于 1")

    output_dir = Path(args.output_dir).expanduser().resolve()
    base_env = gym.make(
        args.env_id,
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
    )
    env = RecordEpisode(
        base_env,
        output_dir=str(output_dir),
        save_trajectory=False,
        save_video=True,
        save_on_reset=True,
        max_steps_per_video=200,
        clean_on_close=True,
        avoid_overwriting_video=True,
    )
    try:
        env.action_space.seed(args.seed)
        for episode in range(args.episodes):
            env.reset(seed=args.seed + episode)
            steps = 0
            while steps < 200:
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                steps += 1
                if is_done(terminated, truncated):
                    break
            print(f"episode {episode + 1}/{args.episodes}: {steps} 步")
        # RecordEpisode.close() flushes the final buffered video.
    finally:
        env.close()

    print(f"视频输出目录: {output_dir}")


if __name__ == "__main__":
    main()
