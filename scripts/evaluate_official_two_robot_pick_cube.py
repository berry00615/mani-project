#!/usr/bin/env python3
"""Fixed-seed evaluation and phase diagnostics for TwoRobotPickCube PPO."""

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class OfficialAgent(nn.Module):
    """Network structure used by ManiSkill 3.0.1 ppo_fast.py."""

    def __init__(self, n_obs, n_act, device):
        super().__init__()

        def network(out, final_std=np.sqrt(2)):
            return nn.Sequential(
                layer_init(nn.Linear(n_obs, 256, device=device)), nn.Tanh(),
                layer_init(nn.Linear(256, 256, device=device)), nn.Tanh(),
                layer_init(nn.Linear(256, 256, device=device)), nn.Tanh(),
                layer_init(
                    nn.Linear(256, out, device=device),
                    std=final_std,
                ),
            )

        self.critic = network(1)
        self.actor_mean = network(n_act, final_std=0.01 * np.sqrt(2))
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))


def checkpoint_key(path):
    match = re.search(r"ckpt_(\d+)\.pt$", Path(path).name)
    return (int(match.group(1)) if match else 10**9, Path(path).name)


def rate(mask):
    return float(mask.float().mean().item())


def evaluate(env, agent, n, seed, device, policy, horizon, action_dim):
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    returns = torch.zeros(n, device=device)
    ever_success = torch.zeros(n, dtype=torch.bool, device=device)
    ever_crossed = torch.zeros_like(ever_success)
    ever_right_grasped = torch.zeros_like(ever_success)
    ever_placed = torch.zeros_like(ever_success)
    ever_placed_static = torch.zeros_like(ever_success)
    ever_fell = torch.zeros_like(ever_success)
    min_goal_dist = torch.full((n,), float("inf"), device=device)
    max_cube_y = torch.full((n,), -float("inf"), device=device)

    with torch.no_grad():
        for _ in range(horizon):
            if policy == "random":
                action = torch.rand((n, action_dim), device=device) * 2 - 1
            else:
                action = agent.actor_mean(obs)
            obs, reward, _, _, info = env.step(action)
            returns += reward.reshape(n)
            base = env.unwrapped
            cube_p = base.cube.pose.p
            goal_dist = torch.linalg.norm(base.goal_site.pose.p - cube_p, dim=1)
            crossed = cube_p[:, 1] >= 0
            right_grasped = base.right_agent.is_grasping(base.cube)
            placed = torch.as_tensor(info["is_obj_placed"], device=device).bool()
            right_static = torch.as_tensor(
                info["is_right_arm_static"], device=device
            ).bool()
            success = torch.as_tensor(info["success"], device=device).bool()
            fell = cube_p[:, 2] < -0.01

            ever_success |= success
            ever_crossed |= crossed
            ever_right_grasped |= right_grasped
            ever_placed |= placed
            ever_placed_static |= placed & right_static
            ever_fell |= fell
            min_goal_dist = torch.minimum(min_goal_dist, goal_dist)
            max_cube_y = torch.maximum(max_cube_y, cube_p[:, 1])

    failed = ~ever_success
    fell = failed & ever_fell
    no_handoff = failed & ~ever_fell & ~ever_crossed
    no_right_grasp = failed & ~ever_fell & ever_crossed & ~ever_right_grasped
    no_goal = (
        failed & ~ever_fell & ever_crossed & ever_right_grasped & ~ever_placed
    )
    no_static = (
        failed & ~ever_fell & ever_crossed & ever_right_grasped
        & ever_placed & ~ever_placed_static
    )
    classified = fell | no_handoff | no_right_grasp | no_goal | no_static
    other = failed & ~classified
    return {
        "successes": int(ever_success.sum()),
        "success_rate": rate(ever_success),
        "ever_crossed_handoff_rate": rate(ever_crossed),
        "ever_right_grasped_rate": rate(ever_right_grasped),
        "ever_placed_rate": rate(ever_placed),
        "ever_placed_static_rate": rate(ever_placed_static),
        "ever_fell_rate": rate(ever_fell),
        "mean_min_goal_distance_m": float(min_goal_dist.mean()),
        "mean_max_cube_y_m": float(max_cube_y.mean()),
        "mean_return": float(returns.mean()),
        "failure_fell": int(fell.sum()),
        "failure_no_handoff": int(no_handoff.sum()),
        "failure_no_right_grasp": int(no_right_grasp.sum()),
        "failure_never_placed": int(no_goal.sum()),
        "failure_placed_not_static": int(no_static.sum()),
        "failure_other": int(other.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy", choices=("random", "untrained", "checkpoint"),
        default="checkpoint",
    )
    parser.add_argument("--checkpoint-glob")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    if args.policy == "checkpoint" and not args.checkpoint_glob:
        parser.error("--checkpoint-glob is required for checkpoint policy")

    device = torch.device("cuda")
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

    env = gym.make(
        "TwoRobotPickCube-v1", num_envs=args.episodes, obs_mode="state",
        reward_mode="normalized_dense", render_mode=None,
        sim_backend="physx_cuda", render_backend="none",
        control_mode="pd_joint_delta_pos", reconfiguration_freq=1,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    obs, _ = env.reset(seed=args.seed)
    if obs.ndim != 2:
        raise RuntimeError(f"expected flat vector observation, got {obs.shape}")
    action_dim = env.action_space.shape[-1]
    torch.manual_seed(args.seed)
    agent = OfficialAgent(obs.shape[-1], action_dim, device).eval()
    if args.policy == "checkpoint":
        candidates = sorted(glob.glob(args.checkpoint_glob), key=checkpoint_key)
        if not candidates:
            raise FileNotFoundError(args.checkpoint_glob)
    else:
        candidates = [args.policy]

    results = []
    for candidate in candidates:
        if args.policy == "checkpoint":
            state = torch.load(candidate, map_location=device, weights_only=True)
            agent.load_state_dict(state)
        metrics = evaluate(
            env, agent, args.episodes, args.seed, device, args.policy,
            args.horizon, action_dim,
        )
        metrics.update(
            candidate=str(candidate), policy=args.policy,
            episodes=args.episodes, seed=args.seed, horizon=args.horizon,
            observation_dim=int(obs.shape[-1]), action_dim=int(action_dim),
        )
        results.append(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)
    env.close()
    results.sort(
        key=lambda x: (
            x["success_rate"], x["ever_placed_rate"],
            x["ever_right_grasped_rate"], -x["mean_min_goal_distance_m"],
            x["mean_return"],
        ),
        reverse=True,
    )
    json_path, csv_path = Path(args.output_json), Path(args.output_csv)
    if json_path.exists() or csv_path.exists():
        raise FileExistsError("refusing to overwrite an evaluation artifact")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print("BEST", json.dumps(results[0], sort_keys=True))


if __name__ == "__main__":
    main()
