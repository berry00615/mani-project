#!/usr/bin/env python3
"""Record deterministic TwoRobotPickCube episodes from an official PPO model."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
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


def frame_array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    frame = np.asarray(value)
    if frame.ndim == 4:
        frame = frame[0]
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=100)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output = Path(args.output_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

    device = torch.device("cuda")
    env = gym.make(
        "TwoRobotPickCube-v1", num_envs=1, obs_mode="state",
        reward_mode="normalized_dense", render_mode="rgb_array",
        sim_backend="physx_cuda", control_mode="pd_joint_delta_pos",
        reconfiguration_freq=1,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    obs, _ = env.reset(seed=args.start_seed)
    agent = OfficialAgent(
        obs.shape[-1], env.action_space.shape[-1], device
    ).eval()
    agent.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )

    results = []
    for seed in range(args.start_seed, args.start_seed + args.episodes):
        obs, _ = env.reset(seed=seed)
        frames = [frame_array(env.render())]
        ever_success = False
        episode_return = 0.0
        min_goal_dist = float("inf")
        with torch.no_grad():
            for _ in range(args.horizon):
                action = agent.actor_mean(obs)
                obs, reward, _, _, info = env.step(action)
                frames.append(frame_array(env.render()))
                episode_return += float(torch.as_tensor(reward).item())
                ever_success |= bool(torch.as_tensor(info["success"]).item())
                base = env.unwrapped
                distance = torch.linalg.norm(
                    base.goal_site.pose.p - base.cube.pose.p, dim=1
                )
                min_goal_dist = min(min_goal_dist, float(distance.item()))
        status = "success" if ever_success else "failure"
        video = output / f"seed_{seed:04d}_{status}.mp4"
        iio.imwrite(video, np.stack(frames), fps=30)
        result = {
            "seed": seed,
            "strict_success": ever_success,
            "return": episode_return,
            "min_goal_distance_m": min_goal_dist,
            "video": video.name,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    env.close()
    (output / "manifest.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    failures = sum(not x["strict_success"] for x in results)
    print(f"recorded={len(results)} failures={failures}", flush=True)


if __name__ == "__main__":
    main()
