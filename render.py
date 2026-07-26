#!/usr/bin/env python3
"""
Render a trained PPO policy on PickCube environments with the ManiSkill GUI viewer.

Supports:
  - Interactive rendering (default)
  - Fixed-seed batch evaluation (--seeds)
  - Deterministic policy (actor mean, no sampling)
  - Video recording (--record)
  - Step-by-step overlay diagnostics

Usage:
    # Interactive single episode
    python render.py --checkpoint checkpoints/.../final.pt

    # Fixed seeds, no rendering
    python render.py --checkpoint checkpoints/.../final.pt \
        --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
        --no-render

    # Video recording
    python render.py --checkpoint checkpoints/.../final.pt --record
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ppo import ActorCritic, load_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scalar(v):
    """Extract Python scalar from tensor/array/primitive."""
    if v is None:
        return 0.0
    if isinstance(v, torch.Tensor):
        return float(v.item()) if v.numel() == 1 else float(v.flatten()[0].item())
    if isinstance(v, np.ndarray):
        return float(v.item()) if v.size == 1 else float(v.flatten()[0])
    if isinstance(v, bool):
        return float(v)
    return float(v)


def _bool(v):
    """Extract Python bool from tensor/array/primitive."""
    if v is None:
        return False
    if isinstance(v, torch.Tensor):
        return bool(v.item()) if v.numel() == 1 else bool(v.flatten()[0].item())
    if isinstance(v, np.ndarray):
        return bool(v.item()) if v.size == 1 else bool(v.flatten()[0])
    return bool(v)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_policy(ckpt_path: str, device: torch.device):
    """Load checkpoint and build ActorCritic policy."""
    map_loc = "cuda:0" if device.type == "cuda" else "cpu"
    ckpt = load_checkpoint(str(ckpt_path), map_location=map_loc, device=device)

    arch = ckpt["architecture"]
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

    # Check for obs_rms
    obs_rms = ckpt.get("obs_rms", None)
    if obs_rms is not None:
        print("检测到 obs_rms，将使用归一化观测")
    else:
        print("未检测到可用的 obs_rms，将使用原始观测")

    return policy, ckpt


def create_env(ckpt: dict, render: bool = True, record: bool = False):
    """Create environment from checkpoint config."""
    import mani_skill.envs  # registers ManiSkill environments
    import envs  # noqa: F401 — registers custom project environments

    env_id = ckpt["env_id"]
    obs_mode = ckpt["obs_mode"]
    config = ckpt.get("config", {})

    render_mode = "rgb_array" if record else ("human" if render else None)
    sim_backend = "auto"

    make_kwargs = dict(
        num_envs=1,
        obs_mode=obs_mode,
        render_mode=render_mode,
        sim_backend=sim_backend,
        # Recording also needs an active renderer even when no GUI is shown.
        render_backend="vulkan" if (render or record) else "none",
        enable_shadow=False,
    )
    env_kwargs = config.get("env_kwargs", None)
    if env_kwargs:
        make_kwargs.update(env_kwargs)

    env = gym.make(env_id, **make_kwargs)
    return env


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

def _frame_to_uint8(frame):
    """Normalize env.render() output to an HxWx3 uint8 numpy array."""
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    while frame.ndim > 3 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating) and frame.max(initial=0) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError(f"Unexpected rendered frame shape: {frame.shape}")
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame


def run_episode(env, policy, device, seed: int, render: bool = True, record_path=None):
    """Run one episode and return results dict."""
    MAX_RESET_RETRIES = 100

    # Reset with initial-success guard
    reset_attempts = 0
    while True:
        obs, info = env.reset(seed=seed + reset_attempts * 1000)
        init_success = _bool(info.get("success", False))
        reset_attempts += 1
        if not init_success:
            break
        if reset_attempts >= MAX_RESET_RETRIES:
            break

    # Move obs to device
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).float().to(device)
    elif obs.device != device:
        obs = obs.to(device)

    frames = []
    if record_path is not None:
        frames.append(_frame_to_uint8(env.render()))

    ep_reward = 0.0
    ep_length = 0
    done = False
    terminated = False
    truncated = False
    last_info = {}

    while not done:
        with torch.no_grad():
            action, _, _ = policy.get_action(obs, deterministic=True)

        # Ensure batch dim for env.step
        action_np = action.cpu().numpy()
        if action_np.ndim == 1:
            action_np = action_np.reshape(1, -1)

        obs, reward, term, trunc, info = env.step(action_np)
        last_info = info
        done = bool(_bool(term) or _bool(trunc))
        terminated = _bool(term)
        truncated = _bool(trunc)
        ep_reward += float(_scalar(reward))
        ep_length += 1

        if record_path is not None:
            frames.append(_frame_to_uint8(env.render()))

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(device)
        elif obs.device != device:
            obs = obs.to(device)

    if record_path is not None:
        import imageio.v2 as imageio
        record_path = Path(record_path)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(record_path, frames, fps=20, macro_block_size=None)

    success = _bool(last_info.get("success", False))
    is_grasped = _bool(last_info.get("is_grasped", False))
    is_robot_static = _bool(last_info.get("is_robot_static", False))

    return {
        "seed": seed,
        "success": success,
        "steps": ep_length,
        "terminated": terminated,
        "truncated": truncated,
        "episode_return": ep_reward,
        "grasped_at_end": is_grasped,
        "static_at_end": is_robot_static,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Render/evaluate a PPO policy on PickCube")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu/mps). Default: auto-detect.")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes (used when --seeds is not provided)")
    parser.add_argument("--base-seed", type=int, default=0,
                        help="Base seed for episodes (used when --seeds is not provided)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of fixed seeds, "
                             "e.g. 0,1,2,...,19. Each episode calls "
                             "env.reset(seed=seed).")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable GUI rendering (headless eval)")
    parser.add_argument("--record", action="store_true",
                        help="Record video instead of interactive render")
    parser.add_argument("--record-dir", type=str, default="videos",
                        help="Directory for recorded videos")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- Checkpoint ---
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    policy, ckpt = load_policy(str(ckpt_path), device)
    print(f"  Env ID:      {ckpt['env_id']}")
    print(f"  Obs mode:    {ckpt['obs_mode']}")
    print(f"  Control:     {ckpt['control_mode']}")
    print(f"  Timestep:    {ckpt['timestep']}")
    print(f"  Obs dim:     {policy.obs_dim}")
    print(f"  Act dim:     {policy.act_dim}")

    # --- Seeds ---
    # --record is an off-screen operation; do not open an interactive viewer.
    render = not args.no_render and not args.record
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.base_seed + i for i in range(args.episodes)]
    print(f"Seeds ({len(seeds)}): {seeds[:5]}...{seeds[-3:]}"
          if len(seeds) > 8 else f"Seeds: {seeds}")

    # --- Environment ---
    print(f"Creating environment (render={render})...")
    env = create_env(ckpt, render=render, record=args.record)

    if args.record:
        from pathlib import Path as P
        record_dir = P(args.record_dir)
        record_dir.mkdir(parents=True, exist_ok=True)

    # --- Run episodes ---
    results = []
    t_start = time.time()

    for ep_idx, seed in enumerate(seeds):
        print(f"\n{'=' * 60}")
        print(f"Episode {ep_idx + 1}/{len(seeds)} — seed={seed}")

        record_path = None
        if args.record:
            record_path = record_dir / f"seed_{seed}.mp4"

        result = run_episode(
            env, policy, device, seed, render=render, record_path=record_path
        )
        results.append(result)

        status = "✓ SUCCESS" if result["success"] else (
            "✗ TIMEOUT" if result["truncated"] else "✗ FAILED")
        print(f"  {status}  steps={result['steps']}  "
              f"return={result['episode_return']:.3f}  "
              f"grasped={result['grasped_at_end']}  "
              f"static={result['static_at_end']}")
        if record_path is not None:
            print(f"  Video: {record_path}")

    total_time = time.time() - t_start

    # --- Summary ---
    n = len(results)
    n_success = sum(1 for r in results if r["success"])
    n_timeout = sum(1 for r in results if r["truncated"])
    returns = np.array([r["episode_return"] for r in results])
    steps = np.array([r["steps"] for r in results])
    success_steps = np.array([r["steps"] for r in results if r["success"]])

    print(f"\n{'=' * 60}")
    print(f"Summary ({n} episodes, {total_time:.1f}s)")
    print(f"{'=' * 60}")
    print(f"  Success:  {n_success}/{n} ({n_success/n*100:.1f}%)")
    print(f"  Timeout:  {n_timeout}/{n} ({n_timeout/n*100:.1f}%)")
    print(f"  Mean return:        {returns.mean():.3f}")
    print(f"  Median return:      {np.median(returns):.3f}")
    print(f"  Mean steps:         {steps.mean():.1f}")
    print(f"  Median steps:       {np.median(steps):.1f}")
    if len(success_steps) > 0:
        print(f"  Mean success steps: {success_steps.mean():.1f}")
        print(f"  Median succ steps:  {np.median(success_steps):.1f}")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
