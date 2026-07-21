"""Inspect PickCube-v1 and run a short random-policy GUI session."""

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
import mani_skill.envs  # noqa: F401 - registers ManiSkill environments


def print_observation(value: Any, path: str = "observation", indent: int = 0) -> None:
    """Recursively print mappings and array/tensor-like observation leaves."""
    prefix = "  " * indent
    if isinstance(value, Mapping):
        print(f"{prefix}{path}: {type(value).__name__} ({len(value)} keys)")
        for key, child in value.items():
            print_observation(child, f"{path}.{key}", indent + 1)
        return

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    print(
        f"{prefix}{path}: type={type(value).__name__}, "
        f"shape={shape}, dtype={dtype}"
    )


def is_done(terminated: Any, truncated: Any) -> bool:
    return bool(np.asarray(terminated).any() or np.asarray(truncated).any())


def main() -> None:
    print("[1/3] 创建 PickCube-v1（CPU simulation, human rendering）...")
    env = gym.make(
        "PickCube-v1",
        obs_mode="state",
        render_mode="human",
        sim_backend="physx_cpu",
    )
    try:
        observation, info = env.reset(seed=0)
        print("[2/3] 环境创建成功，observation 结构：")
        print_observation(observation)
        print(f"reset info keys: {list(info.keys())}")
        print(f"action_space: {env.action_space}")
        print(f"action_space.low: {env.action_space.low}")
        print(f"action_space.high: {env.action_space.high}")

        print("[3/3] 随机运行 300 步；关闭 GUI 或按 Ctrl+C 可退出。")
        episodes = 0
        for step in range(1, 301):
            observation, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            env.render()
            if is_done(terminated, truncated):
                episodes += 1
                print(f"  step={step}: episode {episodes} 结束，正在 reset")
                observation, info = env.reset()
        print(f"运行完成：总步数=300，已结束 episodes={episodes}")
    finally:
        env.close()
        print("环境已关闭。")


if __name__ == "__main__":
    main()
