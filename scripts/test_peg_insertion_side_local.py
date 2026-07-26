#!/usr/bin/env python3
"""Run the promoted PegInsertionSide policy locally on CPU PhysX."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch
from torch import nn


class OfficialAgent(nn.Module):
    """ManiSkill v3.0.1 PPO-fast actor/critic checkpoint structure."""

    def __init__(self, n_obs: int, n_act: int, device: torch.device):
        super().__init__()

        def network(out_features: int):
            return nn.Sequential(
                nn.Linear(n_obs, 256, device=device),
                nn.Tanh(),
                nn.Linear(256, 256, device=device),
                nn.Tanh(),
                nn.Linear(256, 256, device=device),
                nn.Tanh(),
                nn.Linear(256, out_features, device=device),
            )

        self.critic = network(1)
        self.actor_mean = network(n_act)
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))


def to_frame(rendered):
    if isinstance(rendered, torch.Tensor):
        rendered = rendered.detach().cpu().numpy()
    frame = np.asarray(rendered)
    if frame.ndim == 4:
        frame = frame[0]
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/ppo_peg_insertion_side_official_ckpt2151_lr1e4_25m/"
            "best_o2_ckpt551_991of1000_matchedseed.pt"
        ),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--video", type=Path)
    parser.add_argument(
        "--camera-view",
        choices=("hero", "front", "side", "top"),
        default="hero",
    )
    args = parser.parse_args()
    if args.video and args.episodes != 1:
        parser.error("--video requires --episodes 1")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; use --device cpu")

    import mani_skill
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils import sapien_utils

    policy_device = torch.device(args.device)
    render_mode = "rgb_array" if args.video else None
    camera_eyes = {
        "hero": [0.5, -0.5, 0.8],
        "front": [0.0, 0.8, 0.32],
        "side": [0.62, 0.05, 0.34],
        "top": [0.001, -0.05, 1.25],
    }
    camera_targets = {
        "hero": [0.05, -0.1, 0.4],
        "front": [0.0, 0.05, 0.13],
        "side": [0.0, 0.05, 0.13],
        "top": [0.0, 0.0, 0.1],
    }
    camera_pose = sapien_utils.look_at(
        camera_eyes[args.camera_view], camera_targets[args.camera_view]
    )
    env = gym.make(
        "PegInsertionSide-v1",
        num_envs=args.episodes,
        obs_mode="state",
        reward_mode="normalized_dense",
        render_mode=render_mode,
        sim_backend="physx_cpu",
        control_mode="pd_joint_delta_pos",
        reconfiguration_freq=1,
        human_render_camera_configs={"pose": camera_pose},
    )
    obs, _ = env.reset(seed=args.seed)
    agent = OfficialAgent(43, 8, policy_device).eval()
    state = torch.load(checkpoint, map_location=policy_device, weights_only=True)
    agent.load_state_dict(state)

    returns = torch.zeros(args.episodes)
    ever_success = torch.zeros(args.episodes, dtype=torch.bool)
    frames = [to_frame(env.render())] if args.video else []
    with torch.no_grad():
        for _ in range(args.horizon):
            policy_obs = obs.to(policy_device)
            action = agent.actor_mean(policy_obs).to("cpu")
            obs, reward, _, _, info = env.step(action)
            returns += torch.as_tensor(reward).reshape(args.episodes).cpu()
            ever_success |= (
                torch.as_tensor(info["success"]).reshape(args.episodes).cpu()
            )
            if args.video:
                frames.append(to_frame(env.render()))
    env.close()

    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(args.video, np.stack(frames), fps=30)

    result = {
        "checkpoint": str(checkpoint),
        "mani_skill_version": mani_skill.__version__,
        "sim_backend": "physx_cpu",
        "policy_device": args.device,
        "seed": args.seed,
        "camera_view": args.camera_view,
        "episodes": args.episodes,
        "successes": int(ever_success.sum()),
        "success_rate": float(ever_success.float().mean()),
        "mean_return": float(returns.mean()),
        "video": str(args.video) if args.video else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
