#!/usr/bin/env python3
"""
Verify: does info["elapsed_steps"] reset after auto-reset?
Does terminated properly clear?
What is the correct episode boundary signal?
"""

import sys
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mani_skill.envs  # noqa: F401
import envs  # noqa: F401

NUM_ENVS = 4
device = torch.device("cuda")

env = gym.make(
    "PickCubeCollisionGripper-v1",
    num_envs=NUM_ENVS,
    obs_mode="state",
    render_mode=None,
    sim_backend="auto",
    render_backend="none",
    enable_shadow=False,
    table_collision_penalty_coef=0.01,
    table_collision_force_threshold=1.0,
    table_collision_penalty_max=0.5,
    early_gripper_close_penalty_coef=0.2,
    gripper_open_near_cube_bonus=0.1,
    gripper_near_distance=0.08,
    gripper_far_distance=0.15,
    gripper_open_threshold=0.03,
    gripper_closed_threshold=0.01,
)

act_dim = env.action_space.shape[-1]
obs, info = env.reset(seed=42)

print(f"{'step':>5s} {'elapsed':>8s} {'term':>5s} {'trunc':>5s} {'info_trunc':>10s}")
print("-" * 50)

for step in range(105):
    action = torch.randn(NUM_ENVS, act_dim, device=device) * 0.5
    obs, reward, terminated, truncated, info = env.step(action)

    elapsed = info["elapsed_steps"]

    # Check if info has a TimeLimit key
    tl_truncated = info.get("TimeLimit.truncated", None)
    tl_str = str(tl_truncated.tolist()) if tl_truncated is not None else "None"

    if step >= 95 and step <= 104:
        print(f"{step:5d} {elapsed.tolist()}   {int(terminated.sum().item()):5d} "
              f"{int(truncated.sum().item()):5d}   {tl_str}")

# Key question: does elapsed_steps reset to 1 after truncation?
print(f"\nKey observations:")
print(f"  At step 99 (100th step): truncated should become True")
print(f"  At step 100: does elapsed_steps reset to 1?")
print(f"  Does truncated stay True or become False?")

env.close()
