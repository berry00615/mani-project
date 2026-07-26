#!/usr/bin/env python3
"""Deterministic, GPU-parallel phase diagnostics for StackCube checkpoints."""

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ppo import ActorCritic, load_checkpoint


def as_bool(info, key, n, device):
    value = info.get(key)
    if value is None:
        return torch.zeros(n, dtype=torch.bool, device=device)
    return torch.as_tensor(value, device=device, dtype=torch.bool).reshape(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    device = torch.device("cuda")
    ckpt = load_checkpoint(args.checkpoint, map_location="cuda:0", device=device)
    arch = ckpt["architecture"]
    low = np.asarray(arch["action_low"]) if arch.get("action_low") is not None else None
    high = np.asarray(arch["action_high"]) if arch.get("action_high") is not None else None
    policy = ActorCritic(
        obs_dim=arch["obs_dim"], act_dim=arch["act_dim"],
        policy_hidden_sizes=arch["policy_hidden_sizes"],
        value_hidden_sizes=arch["value_hidden_sizes"],
        action_low=low, action_high=high,
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    import mani_skill.envs  # noqa: F401
    import envs  # noqa: F401

    n = args.episodes
    config = ckpt.get("config", {})
    make_kwargs = dict(
        num_envs=n, obs_mode=ckpt["obs_mode"], render_mode=None,
        sim_backend="auto", render_backend="none", enable_shadow=False,
    )
    make_kwargs.update(config.get("env_kwargs") or {})
    env = gym.make(ckpt["env_id"], **make_kwargs)
    obs, info = env.reset(seed=args.seed)

    ever_grasped = torch.zeros(n, dtype=torch.bool, device=device)
    ever_on = torch.zeros_like(ever_grasped)
    ever_on_static = torch.zeros_like(ever_grasped)
    ever_on_released = torch.zeros_like(ever_grasped)
    ever_success = torch.zeros_like(ever_grasped)
    min_xy_error = torch.full((n,), float("inf"), device=device)
    min_z_error = torch.full((n,), float("inf"), device=device)
    returns = torch.zeros(n, device=device)

    max_steps = env.spec.max_episode_steps or 50
    for _ in range(max_steps):
        with torch.no_grad():
            action, _, _ = policy.get_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        returns += torch.as_tensor(reward, device=device).reshape(n)

        grasped = as_bool(info, "is_cubeA_grasped", n, device)
        on = as_bool(info, "is_cubeA_on_cubeB", n, device)
        static = as_bool(info, "is_cubeA_static", n, device)
        success = as_bool(info, "success", n, device)
        ever_grasped |= grasped
        ever_on |= on
        ever_on_static |= on & static
        ever_on_released |= on & ~grasped
        ever_success |= success

        offset = env.unwrapped.cubeA.pose.p - env.unwrapped.cubeB.pose.p
        min_xy_error = torch.minimum(min_xy_error, torch.linalg.norm(offset[:, :2], dim=1))
        target_z = float(env.unwrapped.cube_half_size[2].item() * 2)
        min_z_error = torch.minimum(min_z_error, torch.abs(offset[:, 2] - target_z))

    final_grasped = as_bool(info, "is_cubeA_grasped", n, device)
    final_on = as_bool(info, "is_cubeA_on_cubeB", n, device)
    final_static = as_bool(info, "is_cubeA_static", n, device)

    def rate(x):
        return float(x.float().mean().item())

    result = {
        "checkpoint": args.checkpoint,
        "timestep": int(ckpt["timestep"]),
        "episodes": n,
        "seed_start": args.seed,
        "strict_success_rate": rate(ever_success),
        "ever_grasped_rate": rate(ever_grasped),
        "ever_on_target_rate": rate(ever_on),
        "ever_on_and_static_rate": rate(ever_on_static),
        "ever_on_and_released_rate": rate(ever_on_released),
        "final_grasped_rate": rate(final_grasped),
        "final_on_target_rate": rate(final_on),
        "final_static_rate": rate(final_static),
        "mean_return": float(returns.mean().item()),
        "mean_min_xy_error_m": float(min_xy_error.mean().item()),
        "median_min_xy_error_m": float(min_xy_error.median().item()),
        "within_3cm_xy_rate": rate(min_xy_error <= 0.03),
        "mean_min_z_error_m": float(min_z_error.mean().item()),
        "within_5mm_z_rate": rate(min_z_error <= 0.005),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    env.close()


if __name__ == "__main__":
    main()
