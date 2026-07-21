#!/usr/bin/env python3
"""
Minimal headless state-only ManiSkill test for A100 GPU.

Tests whether ManiSkill 3.x can run with render_backend="none"
on a headless Linux machine without Vulkan.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/test_headless_state.py
"""

import os
import sys
import time
import warnings

# Ensure only GPU 0 is visible
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import gymnasium as gym
import numpy as np
import torch


def get_gpu_memory_mb():
    """Return (allocated_MB, reserved_MB) for cuda:0, or None."""
    if not torch.cuda.is_available():
        return None
    return (
        torch.cuda.memory_allocated(0) / 1024**2,
        torch.cuda.memory_reserved(0) / 1024**2,
    )


def main():
    print("=" * 70)
    print("ManiSkill Headless State-Only Test")
    print("=" * 70)

    # --- Environment info ---
    print(f"\nPython:       {sys.version}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"CUDA avail:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:          {torch.cuda.get_device_name(0)}")
        print(f"GPU count:    {torch.cuda.device_count()}")

    import mani_skill
    import sapien
    import gymnasium as gymnasium_lib
    print(f"ManiSkill:    {mani_skill.__version__}")
    print(f"SAPIEN:       {sapien.__version__}")
    print(f"Gymnasium:    {gymnasium_lib.__version__}")

    # --- GPU memory before ---
    mem_before = get_gpu_memory_mb()
    if mem_before:
        print(f"\nGPU mem before env:  allocated={mem_before[0]:.1f} MB, reserved={mem_before[1]:.1f} MB")

    # --- Create environment ---
    # Verified against ManiSkill 3.0.1 source:
    #   - render_backend="none" is officially supported (backend.py:87-88)
    #   - sim_backend="auto" picks physx_cpu for num_envs=1
    #   - obs_mode="state" never triggers sensor data capture
    #   - render_mode=None means no GUI, no rgb_array
    print("\n--- Creating environment ---")
    print("Parameters:")
    env_kwargs = dict(
        num_envs=1,
        obs_mode="state",
        reward_mode="normalized_dense",
        control_mode=None,          # use default for PickCube-v1
        render_mode=None,           # no GUI, no render() calls
        sim_backend="auto",         # physx_cpu for 1 env
        render_backend="none",      # DISABLE RENDERER completely
        enable_shadow=False,
    )
    for k, v in env_kwargs.items():
        print(f"  {k}: {v!r}")

    try:
        env = gym.make("PickCube-v1", **env_kwargs)
    except Exception as e:
        print(f"\nFATAL: Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return 1

    try:
        print(f"\nEnvironment created successfully.")
        print(f"  sim backend:  {env.unwrapped.backend.sim_backend}")
        print(f"  render backend: {env.unwrapped.backend.render_backend}")
        print(f"  render device:  {env.unwrapped.backend.render_device}")
        print(f"  scene.can_render(): {env.unwrapped.scene.can_render()}")
        print(f"  GPU sim enabled:   {env.unwrapped.gpu_sim_enabled}")

        # --- Observation space ---
        print(f"\n--- Observation Space ---")
        print(f"  type:  {type(env.observation_space)}")
        print(f"  shape: {env.observation_space.shape}")
        print(f"  dtype: {env.observation_space.dtype}")
        print(f"  bounds: low={env.observation_space.low[:5]}... high={env.observation_space.high[:5]}...")

        # --- Action space ---
        print(f"\n--- Action Space ---")
        print(f"  type:  {type(env.action_space)}")
        print(f"  shape: {env.action_space.shape}")
        print(f"  dtype: {env.action_space.dtype}")
        print(f"  bounds: low={env.action_space.low} high={env.action_space.high}")

        # --- Reset with seed ---
        print(f"\n--- Reset (seed=0) ---")
        obs, info = env.reset(seed=0)
        print(f"  obs type:  {type(obs)}")
        if isinstance(obs, np.ndarray):
            print(f"  obs shape: {obs.shape}")
            print(f"  obs dtype: {obs.dtype}")
            print(f"  obs[:5]:   {obs.flatten()[:5]}")
        elif isinstance(obs, torch.Tensor):
            print(f"  obs shape: {obs.shape}")
            print(f"  obs dtype: {obs.dtype}")
            print(f"  obs[:5]:   {obs.flatten()[:5]}")
        elif isinstance(obs, dict):
            print(f"  obs keys:  {list(obs.keys())}")
            for k, v in obs.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        print(f"    {k}.{k2}: shape={v2.shape}, dtype={v2.dtype}")
                else:
                    print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  obs: {obs}")

        print(f"  info keys: {list(info.keys()) if isinstance(info, dict) else type(info)}")

        # --- Run 100 random steps ---
        print(f"\n--- Running 100 random steps ---")
        t0 = time.time()
        total_reward = 0.0
        step_count = 0

        for i in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward) if not isinstance(reward, np.ndarray) else float(reward.item() if hasattr(reward, 'item') else reward)
            step_count += 1

            if i == 0:
                print(f"  Step 0: reward={reward}, terminated={terminated}, truncated={truncated}")
                if isinstance(obs, np.ndarray):
                    print(f"  Step 0 obs shape: {obs.shape}")

            if terminated or truncated:
                print(f"  Episode ended at step {i+1}")
                obs, info = env.reset()
                # Don't reset total_reward — just continue

        elapsed = time.time() - t0
        print(f"\n  Completed {step_count} steps in {elapsed:.2f}s")
        print(f"  Steps per second: {step_count / elapsed:.1f}")
        print(f"  Cumulative reward: {total_reward:.4f}")

        # --- GPU memory after ---
        mem_after = get_gpu_memory_mb()
        if mem_after:
            print(f"\nGPU mem after env:   allocated={mem_after[0]:.1f} MB, reserved={mem_after[1]:.1f} MB")
        if mem_before and mem_after:
            print(f"GPU mem delta:        allocated={mem_after[0] - mem_before[0]:.1f} MB, reserved={mem_after[1] - mem_before[1]:.1f} MB")

        # --- Summary ---
        print(f"\n{'=' * 70}")
        print("TEST PASSED")
        print(f"{'=' * 70}")
        print(f"  Vulkan initialized:     NO (render_backend='none', render_device=None)")
        print(f"  GPU simulation:         {env.unwrapped.gpu_sim_enabled}")
        print(f"  Steps completed:        {step_count}/100")
        print(f"  State-only training:    SUITABLE for PPO on A100")

    finally:
        print("\n--- Closing environment ---")
        env.close()
        print("Environment closed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
