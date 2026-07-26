#!/usr/bin/env python3
"""Fixed-seed evaluation and diagnostics for official PegInsertionSide PPO."""

import argparse
import csv
import glob
import json
import math
import re
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn


class OfficialAgent(nn.Module):
    def __init__(self, n_obs, n_act, device):
        super().__init__()
        def network(out):
            return nn.Sequential(
                nn.Linear(n_obs, 256, device=device), nn.Tanh(),
                nn.Linear(256, 256, device=device), nn.Tanh(),
                nn.Linear(256, 256, device=device), nn.Tanh(),
                nn.Linear(256, out, device=device),
            )
        self.critic = network(1)
        self.actor_mean = network(n_act)
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))


def checkpoint_key(path):
    match = re.search(r"ckpt_(\d+)\.pt$", Path(path).name)
    return (int(match.group(1)) if match else 10**9, Path(path).name)


def rate(value):
    return float(value.float().mean().item())


def evaluate(env, agent, n, seed, device, policy, horizon):
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    returns = torch.zeros(n, device=device)
    ever_success = torch.zeros(n, dtype=torch.bool, device=device)
    ever_grasped = torch.zeros_like(ever_success)
    ever_oriented = torch.zeros_like(ever_success)
    ever_preinsert = torch.zeros_like(ever_success)
    ever_entry_aligned = torch.zeros_like(ever_success)
    min_axis_angle = torch.full((n,), math.pi, device=device)
    min_entry_lateral = torch.full((n,), float("inf"), device=device)
    max_insertion_x = torch.full((n,), -float("inf"), device=device)

    with torch.no_grad():
        for _ in range(horizon):
            if policy == "random":
                action = torch.rand((n, 8), device=device) * 2 - 1
            else:
                action = agent.actor_mean(obs)
            obs, reward, _, _, info = env.step(action)
            returns += reward.reshape(n)
            base = env.unwrapped
            grasped = base.agent.is_grasping(base.peg, max_angle=20)
            goal_relative_head = base.goal_pose.inv() * base.peg_head_pose
            goal_relative_center = base.goal_pose.inv() * base.peg.pose
            head_yz = torch.linalg.norm(goal_relative_head.p[:, 1:], dim=1)
            center_yz = torch.linalg.norm(goal_relative_center.p[:, 1:], dim=1)
            preinsert = (head_yz < 0.01) & (center_yz < 0.01)
            hole_relative = base.box_hole_pose.inv() * base.peg_head_pose
            lateral = torch.linalg.norm(hole_relative.p[:, 1:], dim=1)
            entry_aligned = (
                hole_relative.p[:, 1].abs() <= base.box_hole_radii
            ) & (hole_relative.p[:, 2].abs() <= base.box_hole_radii)
            peg_axis = base.peg.pose.to_transformation_matrix()[:, :3, 0]
            goal_axis = base.goal_pose.to_transformation_matrix()[:, :3, 0]
            angle = torch.acos(torch.sum(peg_axis * goal_axis, dim=1).clamp(-1, 1))
            angle = torch.minimum(angle, math.pi - angle)
            success = torch.as_tensor(info["success"], device=device).bool()

            ever_success |= success
            ever_grasped |= grasped
            ever_oriented |= angle <= math.radians(10)
            ever_preinsert |= preinsert
            ever_entry_aligned |= entry_aligned
            min_axis_angle = torch.minimum(min_axis_angle, angle)
            min_entry_lateral = torch.minimum(min_entry_lateral, lateral)
            max_insertion_x = torch.maximum(max_insertion_x, hole_relative.p[:, 0])

    failed = ~ever_success
    no_grasp = failed & ~ever_grasped
    no_orientation = failed & ever_grasped & ~ever_oriented
    no_preinsert = failed & ever_grasped & ever_oriented & ~ever_preinsert
    no_entry = (
        failed & ever_grasped & ever_oriented
        & ever_preinsert & ~ever_entry_aligned
    )
    no_depth = (
        failed & ever_grasped & ever_oriented
        & ever_preinsert & ever_entry_aligned
    )
    classified = no_grasp | no_orientation | no_preinsert | no_entry | no_depth
    other = failed & ~classified
    return {
        "successes": int(ever_success.sum()),
        "success_rate": rate(ever_success),
        "ever_grasped_rate": rate(ever_grasped),
        "ever_oriented_10deg_rate": rate(ever_oriented),
        "ever_preinsert_aligned_rate": rate(ever_preinsert),
        "ever_entry_aligned_rate": rate(ever_entry_aligned),
        "mean_min_axis_angle_deg": float(torch.rad2deg(min_axis_angle).mean()),
        "mean_min_entry_lateral_m": float(min_entry_lateral.mean()),
        "mean_max_insertion_x_m": float(max_insertion_x.mean()),
        "mean_return": float(returns.mean()),
        "failure_no_grasp": int(no_grasp.sum()),
        "failure_no_orientation": int(no_orientation.sum()),
        "failure_no_preinsert_alignment": int(no_preinsert.sum()),
        "failure_no_entry_alignment": int(no_entry.sum()),
        "failure_insufficient_depth": int(no_depth.sum()),
        "failure_other": int(other.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("random", "untrained", "checkpoint"),
                        default="checkpoint")
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
    env = gym.make(
        "PegInsertionSide-v1", num_envs=args.episodes, obs_mode="state",
        reward_mode="normalized_dense", render_mode=None,
        sim_backend="physx_cuda", render_backend="none",
        control_mode="pd_joint_delta_pos", reconfiguration_freq=1,
    )
    torch.manual_seed(args.seed)
    agent = OfficialAgent(43, 8, device).eval()
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
        metrics = evaluate(env, agent, args.episodes, args.seed, device,
                           args.policy, args.horizon)
        metrics.update(candidate=str(candidate), policy=args.policy,
                       episodes=args.episodes, seed=args.seed,
                       horizon=args.horizon)
        results.append(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)
    env.close()
    results.sort(key=lambda x: (x["success_rate"],
                                x["ever_preinsert_aligned_rate"],
                                x["mean_max_insertion_x_m"],
                                x["mean_return"]), reverse=True)
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
