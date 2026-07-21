#!/usr/bin/env python3
"""
Evaluate a trained PPO policy on PickCube-v1 in headless mode.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_ppo_headless.py \
        --checkpoint checkpoints/ppo_pick_cube/final.pt \
        --episodes 10
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ppo import ActorCritic, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO policy headless")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Evaluation seed (different from training)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu). Default: auto-detect.")
    return parser.parse_args()


def evaluate():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Headless PPO Evaluation")
    print("=" * 70)
    print(f"  Checkpoint:    {ckpt_path}")
    print(f"  Episodes:      {args.episodes}")
    print(f"  Seed:          {args.seed}")
    print(f"  Device:        {device}")

    # --- Load checkpoint ---
    print("\nLoading checkpoint...")
    map_loc = "cuda:0" if device.type == "cuda" else "cpu"
    ckpt = load_checkpoint(str(ckpt_path), map_location=map_loc, device=device)

    env_id = ckpt["env_id"]
    obs_mode = ckpt["obs_mode"]
    control_mode = ckpt["control_mode"]
    arch = ckpt["architecture"]
    config = ckpt.get("config", {})
    timestep = ckpt["timestep"]
    versions = ckpt.get("versions", {})

    print(f"  Env ID:        {env_id}")
    print(f"  Obs mode:       {obs_mode}")
    print(f"  Control mode:   {control_mode}")
    print(f"  Timestep:       {timestep}")
    print(f"  ManiSkill ver:  {versions.get('mani_skill', 'unknown')}")

    # --- Create environment (headless, no Vulkan) ---
    print("\nCreating environment...")
    import mani_skill.envs  # registers ManiSkill environments
    import envs  # noqa: F401 — registers custom project environments
    make_kwargs = dict(
        num_envs=1,
        obs_mode=obs_mode,
        render_mode=None,
        sim_backend="auto",
        render_backend="none",       # CRITICAL: no Vulkan
        enable_shadow=False,
    )
    # Pass env_kwargs from training config if available
    env_kwargs = config.get("env_kwargs", None)
    if env_kwargs:
        make_kwargs.update(env_kwargs)
    env = gym.make(env_id, **make_kwargs)
    print(f"  Obs space:      {env.observation_space.shape}")
    print(f"  Act space:      {env.action_space.shape}")
    print(f"  Control mode:   {env.unwrapped.control_mode}")
    print(f"  Sim backend:    {env.unwrapped.backend.sim_backend}")
    print(f"  Can render:     {env.unwrapped.scene.can_render()}")

    # --- Build policy ---
    action_low = arch.get("action_low", None)
    action_high = arch.get("action_high", None)
    if action_low is not None and not isinstance(action_low, np.ndarray):
        action_low = np.array(action_low)
    if action_high is not None and not isinstance(action_high, np.ndarray):
        action_high = np.array(action_high)

    policy = ActorCritic(
        obs_dim=arch["obs_dim"],
        act_dim=arch["act_dim"],
        policy_hidden_sizes=arch["policy_hidden_sizes"],
        value_hidden_sizes=arch["value_hidden_sizes"],
        action_low=action_low,
        action_high=action_high,
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    print(f"  Policy loaded successfully.")

    # --- Evaluate ---
    print(f"\n{'=' * 70}")
    print(f"Running {args.episodes} episodes...")
    print(f"{'=' * 70}")

    episode_rewards = []
    episode_lengths = []
    successes = []
    # Gripper stats accumulated across all episodes
    eval_early_close_rate = []
    eval_mean_gripper_width = []
    eval_grasp_rate = []

    initial_success_skipped = 0
    valid_episodes = 0
    true_successes = 0
    MAX_RESET_RETRIES = 100

    t_start = time.time()

    for ep in range(args.episodes):
        seed = args.seed + ep

        # --- Reset with initial-success guard ---
        reset_attempts = 0
        while True:
            obs, info = env.reset(seed=seed + reset_attempts * 1000)

            init_success = info.get("success", False)
            if isinstance(init_success, (np.ndarray, torch.Tensor)):
                init_success = bool(init_success.item() if hasattr(init_success, 'item') else init_success)
            init_success = bool(init_success)

            reset_attempts += 1

            if not init_success:
                break  # valid starting state

            initial_success_skipped += 1
            if reset_attempts >= MAX_RESET_RETRIES:
                print(f"  WARNING: {MAX_RESET_RETRIES} resets all started as success — "
                      f"accepting current state for episode {ep+1}")
                break

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(device)
        elif obs.device != device:
            obs = obs.to(device)

        ep_reward = 0.0
        ep_length = 0
        done = False
        ep_gripper_widths = []
        ep_early_close = 0
        ep_grasped = 0
        ep_grasp_steps = 0

        while not done:
            with torch.no_grad():
                action, _, _ = policy.get_action(obs, deterministic=True)

            action_np = action.cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)

            done = bool(terminated or truncated)
            r = float(reward.item() if hasattr(reward, 'item') else reward)
            ep_reward += r
            ep_length += 1

            # Gripper tracking
            if hasattr(env.unwrapped, "get_gripper_info"):
                g_width, g_early_close, g_open_near, _, _ = env.unwrapped.get_gripper_info()
                if g_width is not None:
                    ep_gripper_widths.append(float(g_width.item()))
                if g_early_close is not None:
                    ep_early_close += int(g_early_close.item())
            is_grasped = info.get("is_grasped", None)
            if is_grasped is not None:
                ep_grasp_steps += 1
                if isinstance(is_grasped, (np.ndarray, torch.Tensor)):
                    if bool(is_grasped.item() if hasattr(is_grasped, 'item') else is_grasped):
                        ep_grasped += 1
                elif bool(is_grasped):
                    ep_grasped += 1

            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).float().to(device)
            elif obs.device != device:
                obs = obs.to(device)

            if done:
                # Check success from info
                success = bool(info.get("success", False))
                if isinstance(success, (np.ndarray, torch.Tensor)):
                    success = bool(success.item() if hasattr(success, 'item') else success)
                successes.append(success)
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_length)
                valid_episodes += 1
                if success:
                    true_successes += 1

                # Per-episode gripper stats
                ep_ec_rate = ep_early_close / max(ep_length, 1)
                ep_grip_w = np.mean(ep_gripper_widths) if ep_gripper_widths else 0.0
                ep_g_rate = ep_grasped / max(ep_grasp_steps, 1) if ep_grasp_steps > 0 else 0.0
                eval_early_close_rate.append(ep_ec_rate)
                eval_mean_gripper_width.append(ep_grip_w)
                eval_grasp_rate.append(ep_g_rate)

                print(f"  Ep {ep+1:3d}: reward={ep_reward:>7.3f}, length={ep_length:>4d}, "
                      f"success={success}, ec_rate={ep_ec_rate:.3f}, "
                      f"grip_w={ep_grip_w:.4f}, grasp_rate={ep_g_rate:.3f}")
                break

    total_time = time.time() - t_start

    # --- Summary ---
    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)
    success_rate = np.mean(successes) * 100 if successes else 0.0

    print(f"\n{'=' * 70}")
    print(f"Evaluation Results")
    print(f"{'=' * 70}")
    print(f"  Episodes requested: {args.episodes}")
    print(f"  initial_success_skipped={initial_success_skipped}")
    print(f"  valid_episodes={valid_episodes}")
    print(f"  true_successes={true_successes}")
    if valid_episodes > 0:
        print(f"  True success rate:   {true_successes / valid_episodes * 100:.1f}% ({true_successes}/{valid_episodes})")
    print(f"  ----")
    print(f"  Mean reward:        {rewards.mean():.3f}")
    print(f"  Reward std:         {rewards.std():.3f}")
    print(f"  Mean episode length:{lengths.mean():.1f}")
    # Raw success rate (including any initial-success episodes) for comparison
    raw_success_rate = np.mean(successes) * 100 if successes else 0.0
    print(f"  Raw success rate:   {raw_success_rate:.1f}% ({sum(successes)}/{len(successes)})")
    print(f"  Total time:         {total_time:.1f}s")
    if len(rewards) > 0:
        print(f"  Time per episode:   {total_time / len(rewards):.1f}s")

    # Gripper summary
    if eval_mean_gripper_width:
        print(f"  ----")
        print(f"  Mean early_close_rate:  {np.mean(eval_early_close_rate):.4f}")
        print(f"  Mean gripper_width:     {np.mean(eval_mean_gripper_width):.4f}")
        print(f"  Mean grasp_rate:        {np.mean(eval_grasp_rate):.4f}")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    evaluate()
