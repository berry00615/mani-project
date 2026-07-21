#!/usr/bin/env python3
"""
Minimal diagnostic to trace terminated/truncated/done behavior
and episode boundaries in GPU-parallel PickCubeCollisionGripper-v1.

Runs 16 envs for 220+ vector steps and reports per-step done statistics.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mani_skill.envs  # noqa: F401
import envs  # noqa: F401
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=256)
args = parser.parse_args()

NUM_ENVS = args.num_envs
NUM_STEPS = args.steps

print("=" * 70)
print(f"Diagnostic: terminated/truncated/done trace (num_envs={NUM_ENVS})")
print(f"Wrapper: ManiSkillVectorEnv(auto_reset=True)")
print("=" * 70)

base_env = gym.make(
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

env = ManiSkillVectorEnv(base_env, auto_reset=True, ignore_terminations=False)

print(f"\nenv type: {type(env).__name__}")
print(f"auto_reset: {env.auto_reset}")
print(f"sim_backend: {base_env.unwrapped.backend.sim_backend}")
print(f"gpu_sim_enabled: {base_env.unwrapped.gpu_sim_enabled}")
print(f"obs_space: {env.observation_space.shape}")
print(f"act_space: {env.action_space.shape}")

# Check max_episode_steps
if hasattr(env, 'spec') and env.spec is not None:
    print(f"spec.max_episode_steps: {env.spec.max_episode_steps}")
if hasattr(env.unwrapped, '_max_episode_steps'):
    print(f"env._max_episode_steps: {env.unwrapped._max_episode_steps}")

act_dim = env.action_space.shape[-1]
device = torch.device("cuda")

obs, info = env.reset(seed=42)

# Per-env tracking
ep_rewards = torch.zeros(NUM_ENVS, device=device)
ep_lengths = torch.zeros(NUM_ENVS, dtype=torch.int32, device=device)
completed_lengths = []
completed_rewards = []
total_done_count = 0

# Count terminated vs truncated separately
total_term = 0
total_trunc = 0

print(f"\n{'step':>5s} {'term':>6s} {'trunc':>6s} {'done':>6s} "
      f"{'done_idx':>30s} {'ep_lens[0:4]':>20s} "
      f"{'cum_eps':>8s}")

for step in range(NUM_STEPS):
    action = torch.randn(NUM_ENVS, act_dim, device=device) * 0.5
    next_obs, reward, terminated, truncated, info = env.step(action)

    # Device handling
    if reward.device != device:
        reward = reward.to(device)
    if terminated.device != device:
        terminated = terminated.to(device)
    if truncated.device != device:
        truncated = truncated.to(device)

    done = terminated | truncated
    n_term = int(terminated.sum().item())
    n_trunc = int(truncated.sum().item())
    n_done = int(done.sum().item())
    total_term += n_term
    total_trunc += n_trunc
    total_done_count += n_done

    # Per-env tracking (same logic as train_ppo_gpu.py)
    ep_rewards += reward
    ep_lengths += 1

    if done.any():
        done_indices = done.nonzero(as_tuple=True)[0]
        for idx in done_indices:
            i = idx.item()
            completed_lengths.append(int(ep_lengths[i].item()))
            completed_rewards.append(float(ep_rewards[i].item()))
            ep_rewards[i] = 0.0
            ep_lengths[i] = 0

    # Print every step for first 10, then every 10 steps, then at boundaries
    should_print = (
        step < 10
        or step % 25 == 0
        or n_done > 0
    )
    if should_print:
        done_idx_str = str(done.nonzero(as_tuple=True)[0].tolist()) if n_done > 0 else "[]"
        ep_lens_str = str([ep_lengths[i].item() for i in range(min(4, NUM_ENVS))])
        print(f"{step:5d} {n_term:6d} {n_trunc:6d} {n_done:6d} "
              f"{done_idx_str:>30s} {ep_lens_str:>20s} "
              f"{len(completed_lengths):8d}")

    obs = next_obs
    if obs.device != device:
        obs = obs.to(device)

# ---- Summary ----
print(f"\n{'=' * 70}")
print(f"Summary after {NUM_STEPS} vector steps ({NUM_STEPS * NUM_ENVS} env-steps)")
print(f"{'=' * 70}")
print(f"  Total terminated events: {total_term}")
print(f"  Total truncated events:  {total_trunc}")
print(f"  Total done events:       {total_done_count}")
print(f"  Completed episodes:      {len(completed_lengths)}")
print(f"  Episodes per env:        {len(completed_lengths) / NUM_ENVS:.1f}")
print(f"  Steps per episode:       {NUM_STEPS * NUM_ENVS / max(len(completed_lengths), 1):.1f}")

if completed_lengths:
    comp_lens = np.array(completed_lengths)
    print(f"\n  Episode length distribution:")
    print(f"    min:    {comp_lens.min()}")
    print(f"    mean:   {comp_lens.mean():.1f}")
    print(f"    max:    {comp_lens.max()}")
    print(f"    median: {np.median(comp_lens):.1f}")
    print(f"    std:    {comp_lens.std():.1f}")

    comp_rews = np.array(completed_rewards)
    print(f"\n  Episode reward distribution:")
    print(f"    min:    {comp_rews.min():.4f}")
    print(f"    mean:   {comp_rews.mean():.4f}")
    print(f"    max:    {comp_rews.max():.4f}")

    # Conservation check
    sum_completed = comp_lens.sum()
    sum_current = int(ep_lengths.sum().item())
    total_accounted = sum_completed + sum_current
    print(f"\n  Conservation check:")
    print(f"    sum(completed_lengths):     {sum_completed}")
    print(f"    sum(current_ep_lengths):    {sum_current}")
    print(f"    total accounted:            {total_accounted}")
    print(f"    total env-steps:            {NUM_STEPS * NUM_ENVS}")
    print(f"    expected (each env-step increments ep_length by 1): {NUM_STEPS * NUM_ENVS}")
    print(f"    MATCH: {total_accounted == NUM_STEPS * NUM_ENVS}")

# Check what happens at a specific truncation boundary
print(f"\n  Current ep_lengths (all envs):")
for i in range(NUM_ENVS):
    print(f"    env[{i:3d}]: ep_len={ep_lengths[i].item()}", end="")
    if (i + 1) % 4 == 0:
        print()

env.close()
print("\nDone.")
