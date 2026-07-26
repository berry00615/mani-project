#!/usr/bin/env python3
"""Fixed-seed GPU-parallel sweep for ManiSkill's official PPO checkpoints."""

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn


class OfficialAgent(nn.Module):
    """Architecture used by ManiSkill v3.0.1 examples/baselines/ppo/ppo_fast.py."""

    def __init__(self, n_obs: int, n_act: int, device: torch.device):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(n_obs, 256, device=device), nn.Tanh(),
            nn.Linear(256, 256, device=device), nn.Tanh(),
            nn.Linear(256, 256, device=device), nn.Tanh(),
            nn.Linear(256, 1, device=device),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(n_obs, 256, device=device), nn.Tanh(),
            nn.Linear(256, 256, device=device), nn.Tanh(),
            nn.Linear(256, 256, device=device), nn.Tanh(),
            nn.Linear(256, n_act, device=device),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))


def checkpoint_key(path: str):
    name = Path(path).name
    match = re.search(r"ckpt_(\d+)\.pt$", name)
    return (int(match.group(1)) if match else 10**9, name)


def bool_info(info, key, n, device):
    value = info.get(key)
    if value is None:
        return torch.zeros(n, dtype=torch.bool, device=device)
    return torch.as_tensor(value, dtype=torch.bool, device=device).reshape(n)


def evaluate(env, agent, n: int, seed: int, device: torch.device):
    obs, _ = env.reset(seed=seed)
    returns = torch.zeros(n, device=device)
    ever_success = torch.zeros(n, dtype=torch.bool, device=device)
    ever_grasped = torch.zeros_like(ever_success)
    ever_on = torch.zeros_like(ever_success)
    ever_on_static = torch.zeros_like(ever_success)
    ever_on_released = torch.zeros_like(ever_success)
    min_xy = torch.full((n,), float("inf"), device=device)
    min_z = torch.full((n,), float("inf"), device=device)

    with torch.no_grad():
        for _ in range(50):
            action = agent.actor_mean(obs)
            obs, reward, _, _, info = env.step(action)
            returns += reward.reshape(n)
            grasped = bool_info(info, "is_cubeA_grasped", n, device)
            on = bool_info(info, "is_cubeA_on_cubeB", n, device)
            static = bool_info(info, "is_cubeA_static", n, device)
            success = bool_info(info, "success", n, device)
            ever_success |= success
            ever_grasped |= grasped
            ever_on |= on
            ever_on_static |= on & static
            ever_on_released |= on & ~grasped
            offset = env.unwrapped.cubeA.pose.p - env.unwrapped.cubeB.pose.p
            min_xy = torch.minimum(min_xy, torch.linalg.norm(offset[:, :2], dim=1))
            min_z = torch.minimum(min_z, torch.abs(offset[:, 2] - 0.04))

    def rate(x):
        return float(x.float().mean().item())

    failed = ~ever_success
    failure_no_grasp = failed & ~ever_grasped
    failure_never_on = failed & ever_grasped & ~ever_on
    failure_on_never_release = failed & ever_on & ~ever_on_released
    failure_released_not_static = (
        failed & ever_on_released & ~ever_on_static)
    classified = (failure_no_grasp | failure_never_on
                  | failure_on_never_release | failure_released_not_static)
    failure_other = failed & ~classified

    return {
        "success_rate": rate(ever_success),
        "successes": int(ever_success.sum().item()),
        "ever_grasped_rate": rate(ever_grasped),
        "ever_on_target_rate": rate(ever_on),
        "ever_on_static_rate": rate(ever_on_static),
        "ever_on_released_rate": rate(ever_on_released),
        "within_3cm_xy_rate": rate(min_xy <= 0.03),
        "mean_min_xy_error_m": float(min_xy.mean().item()),
        "within_5mm_z_rate": rate(min_z <= 0.005),
        "mean_min_z_error_m": float(min_z.mean().item()),
        "mean_return": float(returns.mean().item()),
        "failure_no_grasp": int(failure_no_grasp.sum().item()),
        "failure_never_on_target": int(failure_never_on.sum().item()),
        "failure_on_target_never_released": int(
            failure_on_never_release.sum().item()),
        "failure_released_not_static": int(
            failure_released_not_static.sum().item()),
        "failure_other_timing": int(failure_other.sum().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-glob", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    checkpoints = sorted(glob.glob(args.checkpoint_glob), key=checkpoint_key)
    if not checkpoints:
        raise FileNotFoundError(args.checkpoint_glob)
    device = torch.device("cuda")
    import mani_skill.envs  # noqa: F401
    env = gym.make(
        "StackCube-v1", num_envs=args.episodes, obs_mode="state",
        render_mode=None, sim_backend="physx_cuda", render_backend="none",
        control_mode="pd_joint_delta_pos", reconfiguration_freq=1,
    )
    agent = OfficialAgent(48, 8, device).eval()
    results = []
    for checkpoint in checkpoints:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        agent.load_state_dict(state)
        metrics = evaluate(env, agent, args.episodes, args.seed, device)
        metrics["checkpoint"] = checkpoint
        metrics["checkpoint_name"] = Path(checkpoint).name
        metrics["episodes"] = args.episodes
        metrics["seed"] = args.seed
        results.append(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)
    env.close()

    results.sort(key=lambda x: (
        x["success_rate"], x["ever_on_static_rate"],
        x["ever_on_released_rate"], x["ever_on_target_rate"],
        x["mean_return"]), reverse=True)
    json_path = Path(args.output_json)
    csv_path = Path(args.output_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("BEST", json.dumps(results[0], sort_keys=True))


if __name__ == "__main__":
    main()
