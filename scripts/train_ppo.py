#!/usr/bin/env python3
"""
Train PPO on PickCube-v1 with state observations.

Server usage (A100 headless, no Vulkan):
    CUDA_VISIBLE_DEVICES=0 python scripts/train_ppo.py \
        --config configs/ppo_pick_cube.yaml \
        --total-timesteps 100000

Smoke test:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_ppo.py \
        --config configs/ppo_pick_cube.yaml \
        --total-timesteps 4096 \
        --run-name smoke_test

Resume:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_ppo.py \
        --config configs/ppo_pick_cube.yaml \
        --resume checkpoints/ppo_pick_cube/checkpoint_10000.pt
"""

import argparse
import os
import sys
import time
import traceback
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
import gymnasium as gym

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ppo import ActorCritic, RolloutBuffer, PPO, save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on ManiSkill")
    parser.add_argument("--config", type=str, default="configs/ppo_pick_cube.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Override total_timesteps from config")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed from config")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cuda/cpu/auto)")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Override num_envs from config")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name for logging (overrides config-derived name)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate setup only, do not train")
    return parser.parse_args()


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Gymnasium seeding is done via env.reset(seed=seed)


def make_env(env_id: str, obs_mode: str, sim_backend: str, render_backend: str,
             render_mode, num_envs: int, enable_shadow: bool = False,
             env_kwargs: dict | None = None) -> gym.Env:
    """Create a headless ManiSkill environment without Vulkan."""
    import mani_skill.envs  # registers ManiSkill environments with gymnasium
    import envs  # noqa: F401 — registers custom project environments  # noqa: F811
    make_kwargs = dict(
        num_envs=num_envs,
        obs_mode=obs_mode,
        render_mode=render_mode,
        sim_backend=sim_backend,
        render_backend=render_backend,
        enable_shadow=enable_shadow,
    )
    if env_kwargs:
        make_kwargs.update(env_kwargs)
    return gym.make(env_id, **make_kwargs)


def compute_explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute explained variance for value function."""
    var_y = torch.var(y_true)
    if var_y < 1e-8:
        return float("nan")
    return float(1.0 - torch.var(y_true - y_pred) / var_y)


def save_csv_log(log_path: str, metrics: dict, header: bool = False):
    """Append a row of metrics to a CSV log file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    mode = "w" if header else "a"
    with open(log_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if header:
            writer.writeheader()
        writer.writerow(metrics)


def train(args):
    # --- Load config ---
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # --- Override config with CLI args ---
    if args.total_timesteps is not None:
        config["total_timesteps"] = args.total_timesteps
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device is not None:
        config["device"] = args.device
    if args.num_envs is not None:
        config["num_envs"] = args.num_envs

    total_timesteps = config["total_timesteps"]
    seed = config["seed"]
    num_envs = config["num_envs"]
    run_name = args.run_name or f"seed{seed}"

    # --- Device setup ---
    device_str = config.get("device", "cuda")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    print("=" * 70)
    print("PPO Training Setup")
    print("=" * 70)
    print(f"  Config:       {config_path}")
    print(f"  Run name:     {run_name}")
    print(f"  Device:       {device}")
    print(f"  CUDA avail:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
    print(f"  Seed:         {seed}")
    print(f"  Num envs:     {num_envs}")
    print(f"  Total steps:  {total_timesteps}")

    # --- Set seeds ---
    set_seed(seed)

    # --- Create environment ---
    env_kwargs = dict(
        env_id=config["env_id"],
        obs_mode=config["obs_mode"],
        sim_backend=config.get("sim_backend", "auto"),
        render_backend=config.get("render_backend", "none"),
        render_mode=config.get("render_mode", None),
        num_envs=num_envs,
        enable_shadow=config.get("enable_shadow", False),
        env_kwargs=config.get("env_kwargs", None),
    )

    print(f"\n  Environment:")
    for k, v in env_kwargs.items():
        print(f"    {k}: {v!r}")

    env = make_env(**env_kwargs)
    print(f"  Created successfully.")
    print(f"  Sim backend:     {env.unwrapped.backend.sim_backend}")
    print(f"  Render backend:  {env.unwrapped.backend.render_backend}")
    print(f"  Render device:   {env.unwrapped.backend.render_device}")
    print(f"  GPU sim enabled: {env.unwrapped.gpu_sim_enabled}")
    print(f"  Obs space:       {env.observation_space.shape}")
    print(f"  Act space:       {env.action_space.shape}")
    print(f"  Control mode:    {env.unwrapped.control_mode}")

    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    # With vectorized envs, action space bounds have shape (num_envs, act_dim).
    # The policy expects per-env bounds: shape (act_dim,).
    if hasattr(env, "single_action_space") and env.single_action_space is not None:
        action_low = env.single_action_space.low
        action_high = env.single_action_space.high
    elif env.action_space.low.ndim >= 2:
        action_low = env.action_space.low[0]
        action_high = env.action_space.high[0]
    else:
        action_low = env.action_space.low
        action_high = env.action_space.high

    # --- Create policy ---
    policy = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        policy_hidden_sizes=config.get("policy_hidden_sizes", [256, 256, 256]),
        value_hidden_sizes=config.get("value_hidden_sizes", [256, 256, 256]),
        action_low=action_low,
        action_high=action_high,
    ).to(device)

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"\n  Policy params:  {total_params:,} total, {trainable_params:,} trainable")

    # --- Create PPO algorithm ---
    ppo_algo = PPO(
        policy=policy,
        learning_rate=config.get("learning_rate", 3e-4),
        n_epochs=config.get("n_epochs", 10),
        batch_size=config.get("batch_size", 64),
        clip_range=config.get("clip_range", 0.2),
        ent_coef=config.get("ent_coef", 0.0),
        vf_coef=config.get("vf_coef", 0.5),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        device=device,
    )

    # --- Create rollout buffer ---
    buffer = RolloutBuffer(
        buffer_size=config.get("n_steps", 2048),
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_envs=num_envs,
        device=device,
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
    )

    # --- Resume from checkpoint if specified ---
    start_timestep = 0
    if args.resume:
        print(f"\n  Resuming from: {args.resume}")
        ckpt = load_checkpoint(args.resume, map_location=device_str, device=device)
        policy.load_state_dict(ckpt["policy_state_dict"])
        ppo_algo.load_state_dict({
            "optimizer": ckpt["optimizer_state_dict"]
        })
        start_timestep = ckpt.get("timestep", 0)
        # Restore RNG state if available
        rng_state = ckpt.get("rng_state", {})
        if rng_state.get("torch_rng") is not None:
            torch.set_rng_state(rng_state["torch_rng"].to("cpu") if hasattr(rng_state["torch_rng"], "to") else rng_state["torch_rng"])
        if rng_state.get("numpy_rng") is not None:
            np.random.set_state(rng_state["numpy_rng"])
        print(f"  Resumed at timestep {start_timestep}")

    # --- Dry run check ---
    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN — setup OK. Not starting training.")
        print("=" * 70)
        env.close()
        return

    # --- Training loop ---
    output_dir = PROJECT_ROOT / config.get("output_dir", "checkpoints/ppo_pick_cube")
    log_dir = PROJECT_ROOT / config.get("log_dir", "logs/ppo_pick_cube")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_log_path = log_dir / f"training_{run_name}.csv"
    checkpoint_interval = config.get("checkpoint_interval", 10000)
    eval_interval = config.get("eval_interval", 10000)
    eval_episodes = config.get("eval_episodes", 10)
    n_steps_per_rollout = config.get("n_steps", 2048)

    print(f"\n{'=' * 70}")
    print(f"Starting Training (run: {run_name})")
    print(f"{'=' * 70}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Log dir:       {log_dir}")
    print(f"  Checkpoint:    every {checkpoint_interval} steps")
    print(f"  Rollout size:  {n_steps_per_rollout} steps")
    print()

    # Initialize CSV log
    csv_fields = [
        "timestep", "episode_reward", "episode_length",
        "policy_loss", "value_loss", "entropy", "approx_kl",
        "explained_variance", "sps", "wall_time_s",
        "table_collision_rate", "table_collision_penalty",
        "success_rate", "invalid_initial_success_count",
        "early_close_rate", "mean_gripper_width", "grasp_rate",
        "gripper_mask_active_rate", "policy_requested_close_rate",
        "gripper_action_overridden_rate", "allowed_gripper_control_rate",
        "mean_tcp_cube_distance", "near_cube_rate",
    ]
    save_csv_log(str(csv_log_path), {f: 0 for f in csv_fields}, header=True)

    # Per-env episode tracking (vectorized — one accumulator per env)
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_lengths_vec = np.zeros(num_envs, dtype=np.int64)

    # Collision / success tracking
    collision_steps_total = 0       # steps where robot-table collision detected
    collision_penalties = []        # per-step collision penalty values
    total_steps = 0
    episode_successes = []          # True/False per completed episode
    invalid_initial_success_count = 0

    # Gripper-timing tracking (for PickCubeCollisionGripper-v1)
    gripper_early_close_steps_total = 0  # steps where early close penalty applied
    gripper_open_near_steps_total = 0    # steps where open-near bonus applied
    gripper_width_sum = 0.0              # cumulative gripper width for averaging
    gripper_width_count = 0
    grasp_steps_total = 0               # steps where cube was grasped
    grasp_check_count = 0

    # Gripper-mask tracking (for PickCubeGripperCurriculum-v1)
    mask_active_steps_total = 0          # steps where mask rule was active
    action_overridden_steps_total = 0    # steps where mask active AND policy wanted close
    policy_close_requested_total = 0     # steps where policy asked to close gripper
    policy_gripper_control_count = 0     # denominator for per-step rates
    tcp_distance_sum = 0.0               # cumulative TCP-to-cube distance
    near_cube_steps_total = 0            # steps where TCP <= threshold

    timestep = start_timestep
    obs, info = env.reset(seed=seed)
    obs_tensor = obs  # ManiSkill returns torch.Tensor for state mode

    if obs_tensor.device != device:
        obs_tensor = obs_tensor.to(device)

    # Check invalid initial success after first reset (vectorized)
    init_success = info.get("success", None)
    if init_success is not None:
        if isinstance(init_success, torch.Tensor):
            invalid_initial_success_count += int(
                init_success.detach().cpu().bool().sum().item()
            )
        elif isinstance(init_success, np.ndarray):
            invalid_initial_success_count += int(
                init_success.astype(bool).sum()
            )
        else:
            if bool(init_success):
                invalid_initial_success_count += num_envs

    t_start = time.time()
    last_log_time = t_start
    last_log_step = timestep

    try:
        while timestep < total_timesteps:
            # --- Collect rollout ---
            buffer.reset()

            for _ in range(n_steps_per_rollout):
                with torch.no_grad():
                    action, log_prob, value = policy.get_action(obs_tensor)

                action_np = action.cpu().numpy()
                next_obs, reward, terminated, truncated, info = env.step(action_np)

                # ── Normalize done ────────────────────────────────────
                done = terminated | truncated
                # done_bool: numpy bool array, shape (num_envs,)
                if isinstance(done, torch.Tensor):
                    done_bool = done.detach().bool().cpu().numpy().reshape(-1)
                else:
                    done_bool = np.asarray(done, dtype=bool).reshape(-1)

                # done_tensor: float32 tensor for buffer, shape (num_envs,)
                if isinstance(done, torch.Tensor):
                    done_tensor = done.float().to(device).reshape(-1)
                elif isinstance(done, np.ndarray):
                    done_tensor = torch.from_numpy(
                        done.astype(np.float32)
                    ).to(device).reshape(-1)
                else:
                    done_tensor = torch.tensor([float(done)], device=device)

                # ── Normalize reward ──────────────────────────────────
                # reward_np: float32 array for per-env tracking, shape (num_envs,)
                if isinstance(reward, torch.Tensor):
                    reward_np = (
                        reward.detach().cpu().numpy().reshape(-1).astype(np.float32)
                    )
                else:
                    reward_np = np.asarray(reward, dtype=np.float32).reshape(-1)

                # reward_tensor: float32 tensor for buffer, shape (num_envs,)
                if isinstance(reward, torch.Tensor):
                    reward_tensor = reward.float().to(device).reshape(-1)
                elif isinstance(reward, np.ndarray):
                    reward_tensor = torch.from_numpy(
                        reward.astype(np.float32)
                    ).to(device).reshape(-1)
                else:
                    reward_tensor = torch.tensor([float(reward)], device=device)

                # ── Per-env episode tracking ──────────────────────────
                episode_returns += reward_np
                episode_lengths_vec += 1

                finished_indices = np.flatnonzero(done_bool)
                for idx in finished_indices:
                    episode_rewards.append(float(episode_returns[idx]))
                    episode_lengths.append(int(episode_lengths_vec[idx]))
                    episode_returns[idx] = 0.0
                    episode_lengths_vec[idx] = 0

                # ── Collision tracking ────────────────────────────────
                total_steps += num_envs
                if hasattr(env.unwrapped, "get_collision_info"):
                    c_pen, c_mask = env.unwrapped.get_collision_info()
                    if c_pen is not None:
                        collision_penalties.append(float(c_pen.mean().item()))
                    if c_mask is not None:
                        collision_steps_total += int(c_mask.sum().item())

                # ── Gripper-timing tracking ───────────────────────────
                if hasattr(env.unwrapped, "get_gripper_info"):
                    g_width, g_early_close, g_open_near, g_ec_pen, g_on_bonus = \
                        env.unwrapped.get_gripper_info()
                    if g_early_close is not None:
                        gripper_early_close_steps_total += int(g_early_close.sum().item())
                    if g_open_near is not None:
                        gripper_open_near_steps_total += int(g_open_near.sum().item())
                    if g_width is not None:
                        gripper_width_sum += float(g_width.sum().item())
                        gripper_width_count += g_width.shape[0]
                    # Track grasp events from info
                    is_grasped = info.get("is_grasped", None)
                    if is_grasped is not None:
                        grasp_check_count += num_envs
                        if isinstance(is_grasped, torch.Tensor):
                            grasp_steps_total += int(is_grasped.sum().item())
                        elif isinstance(is_grasped, np.ndarray):
                            grasp_steps_total += int(is_grasped.sum().item())

                # ── Gripper-mask tracking ─────────────────────────────
                if hasattr(env.unwrapped, "get_gripper_mask_info"):
                    (mask_active, action_overridden, policy_requested_close,
                     policy_gripper_act, exec_gripper_act,
                     tcp_dist, near_cube) = env.unwrapped.get_gripper_mask_info()

                    if mask_active is not None:
                        mask_active_steps_total += int(mask_active.sum().item())
                    if action_overridden is not None:
                        action_overridden_steps_total += int(
                            action_overridden.sum().item()
                        )
                    if policy_requested_close is not None:
                        policy_close_requested_total += int(
                            policy_requested_close.sum().item()
                        )
                        policy_gripper_control_count += policy_requested_close.shape[0]
                    if tcp_dist is not None:
                        tcp_distance_sum += float(tcp_dist.sum().item())
                    if near_cube is not None:
                        near_cube_steps_total += int(near_cube.sum().item())

                # ── Success tracking (per finished env) ───────────────
                if len(finished_indices) > 0:
                    success_val = info.get("success", None)
                    if success_val is None:
                        episode_successes.extend([False] * len(finished_indices))
                    else:
                        if isinstance(success_val, torch.Tensor):
                            success_array = (
                                success_val.detach().cpu().numpy()
                                .astype(bool).reshape(-1)
                            )
                        else:
                            success_array = np.asarray(success_val, dtype=bool).reshape(-1)

                        if success_array.size == num_envs:
                            episode_successes.extend(
                                success_array[finished_indices].tolist()
                            )
                        elif success_array.size == 1:
                            episode_successes.extend(
                                [bool(success_array.item())] * len(finished_indices)
                            )
                        else:
                            raise RuntimeError(
                                "Unexpected success shape: "
                                f"{success_array.shape}, expected scalar or ({num_envs},)"
                            )

                # ── Store in buffer ───────────────────────────────────
                buffer.add(
                    obs=obs_tensor,
                    action=action,
                    reward=reward_tensor,
                    value=value,
                    log_prob=log_prob,
                    done=done_tensor,
                )

                # ── Transition to next observation ────────────────────
                # ManiSkill vectorized env auto-resets done sub-environments:
                # next_obs already carries the reset observation for each
                # finished env — no explicit env.reset() needed mid-rollout.
                obs_tensor = next_obs
                if isinstance(obs_tensor, np.ndarray):
                    obs_tensor = torch.from_numpy(obs_tensor).float().to(device)
                elif obs_tensor.device != device:
                    obs_tensor = obs_tensor.to(device)

                timestep += num_envs
                if timestep >= total_timesteps:
                    break

            # --- Compute advantages ---
            with torch.no_grad():
                _, _, last_value = policy.get_action(obs_tensor)

            # Normalize last_done for GAE: shape (num_envs,) float tensor
            if isinstance(done, torch.Tensor):
                last_done = done.float().to(device).reshape(-1)
            elif isinstance(done, np.ndarray):
                last_done = torch.from_numpy(
                    done.astype(np.float32)
                ).to(device).reshape(-1)
            else:
                last_done = torch.tensor([float(done)], device=device)

            buffer.compute_advantages(last_value, last_done)

            # --- PPO update ---
            metrics = ppo_algo.update(buffer)

            # --- Logging ---
            now = time.time()
            wall_time = now - t_start
            steps_since_log = timestep - last_log_step
            sps = steps_since_log / (now - last_log_time) if (now - last_log_time) > 0 else 0
            last_log_time = now
            last_log_step = timestep

            # Compute explained variance
            data = buffer.get_training_data()
            explained_var = compute_explained_variance(data["values"], data["returns"])

            # Episode stats
            avg_ep_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            avg_ep_length = np.mean(episode_lengths[-10:]) if episode_lengths else 0

            # Collision stats over the rollout window
            table_collision_rate = collision_steps_total / max(total_steps, 1)
            avg_table_collision_penalty = (
                np.mean(collision_penalties[-100:]) if collision_penalties else 0.0
            )
            # Success rate over completed episodes
            success_rate = (
                np.mean(episode_successes[-10:]) if episode_successes else 0.0
            )

            # Gripper-timing stats
            early_close_rate = gripper_early_close_steps_total / max(total_steps, 1)
            mean_gripper_width = (
                gripper_width_sum / max(gripper_width_count, 1)
                if gripper_width_count > 0 else 0.0
            )
            grasp_rate = (
                grasp_steps_total / max(grasp_check_count, 1)
                if grasp_check_count > 0 else 0.0
            )

            # Gripper-mask stats
            gripper_mask_active_rate = mask_active_steps_total / max(total_steps, 1)
            policy_requested_close_rate = (
                policy_close_requested_total / max(policy_gripper_control_count, 1)
                if policy_gripper_control_count > 0 else 0.0
            )
            gripper_action_overridden_rate = (
                action_overridden_steps_total / max(policy_gripper_control_count, 1)
                if policy_gripper_control_count > 0 else 0.0
            )
            allowed_gripper_control_rate = 1.0 - gripper_mask_active_rate
            mean_tcp_cube_distance = (
                tcp_distance_sum / max(policy_gripper_control_count, 1)
                if policy_gripper_control_count > 0 else 0.0
            )
            near_cube_rate = (
                near_cube_steps_total / max(policy_gripper_control_count, 1)
                if policy_gripper_control_count > 0 else 0.0
            )

            log_entry = {
                "timestep": timestep,
                "episode_reward": round(avg_ep_reward, 4),
                "episode_length": round(avg_ep_length, 1),
                "policy_loss": round(metrics["policy_loss"], 6),
                "value_loss": round(metrics["value_loss"], 6),
                "entropy": round(metrics["entropy"], 6),
                "approx_kl": round(metrics["approx_kl"], 6),
                "explained_variance": round(explained_var, 4) if not np.isnan(explained_var) else "nan",
                "sps": round(sps, 1),
                "wall_time_s": round(wall_time, 1),
                "table_collision_rate": round(table_collision_rate, 6),
                "table_collision_penalty": round(avg_table_collision_penalty, 6),
                "success_rate": round(success_rate, 6),
                "invalid_initial_success_count": invalid_initial_success_count,
                "early_close_rate": round(early_close_rate, 6),
                "mean_gripper_width": round(mean_gripper_width, 6),
                "grasp_rate": round(grasp_rate, 6),
                "gripper_mask_active_rate": round(gripper_mask_active_rate, 6),
                "policy_requested_close_rate": round(policy_requested_close_rate, 6),
                "gripper_action_overridden_rate": round(gripper_action_overridden_rate, 6),
                "allowed_gripper_control_rate": round(allowed_gripper_control_rate, 6),
                "mean_tcp_cube_distance": round(mean_tcp_cube_distance, 6),
                "near_cube_rate": round(near_cube_rate, 6),
            }
            save_csv_log(str(csv_log_path), log_entry)

            print(f"  Step {timestep:>8,}/{total_timesteps:,} | "
                  f"ep_rew={avg_ep_reward:>7.3f} | "
                  f"pol_loss={metrics['policy_loss']:.4f} | "
                  f"val_loss={metrics['value_loss']:.4f} | "
                  f"ent={metrics['entropy']:.4f} | "
                  f"kl={metrics['approx_kl']:.4f} | "
                  f"col_rate={table_collision_rate:.3f} | "
                  f"ec_rate={early_close_rate:.3f} | "
                  f"grp_w={mean_gripper_width:.3f} | "
                  f"grasp={grasp_rate:.3f} | "
                  f"mask={gripper_mask_active_rate:.3f} | "
                  f"overridden={gripper_action_overridden_rate:.3f} | "
                  f"pol_close={policy_requested_close_rate:.3f} | "
                  f"near_cube={near_cube_rate:.3f} | "
                  f"succ={success_rate:.2f} | "
                  f"SPS={sps:.0f}")

            # --- Checkpoint ---
            if timestep % checkpoint_interval < n_steps_per_rollout or timestep >= total_timesteps:
                ckpt_path = output_dir / f"checkpoint_{timestep}.pt"

                # Collect RNG state for resumability
                rng_state = {
                    "torch_rng": torch.get_rng_state(),
                    "numpy_rng": np.random.get_state(),
                }
                if torch.cuda.is_available():
                    rng_state["torch_cuda_rng"] = torch.cuda.get_rng_state()

                save_checkpoint(
                    path=str(ckpt_path),
                    policy=policy,
                    ppo=ppo_algo,
                    timestep=timestep,
                    env_id=config["env_id"],
                    obs_mode=config["obs_mode"],
                    control_mode=env.unwrapped.control_mode,
                    config=config,
                    rng_state=rng_state,
                )
                print(f"  -> Saved checkpoint: {ckpt_path}")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving checkpoint before exit...")
        ckpt_path = output_dir / f"interrupted_{timestep}.pt"

        rng_state = {
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda_rng"] = torch.cuda.get_rng_state()

        save_checkpoint(
            path=str(ckpt_path),
            policy=policy,
            ppo=ppo_algo,
            timestep=timestep,
            env_id=config["env_id"],
            obs_mode=config["obs_mode"],
            control_mode=env.unwrapped.control_mode,
            config=config,
            rng_state=rng_state,
        )
        print(f"  Saved interrupted checkpoint: {ckpt_path}")
        env.close()
        sys.exit(0)

    except Exception as e:
        print(f"\nERROR during training: {e}")
        traceback.print_exc()
        env.close()
        sys.exit(1)

    # --- Final checkpoint ---
    final_path = output_dir / "final.pt"
    rng_state = {
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda_rng"] = torch.cuda.get_rng_state()

    save_checkpoint(
        path=str(final_path),
        policy=policy,
        ppo=ppo_algo,
        timestep=timestep,
        env_id=config["env_id"],
        obs_mode=config["obs_mode"],
        control_mode=env.unwrapped.control_mode,
        config=config,
        rng_state=rng_state,
    )
    print(f"\nFinal checkpoint saved: {final_path}")

    total_wall_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"Training Complete")
    print(f"{'=' * 70}")
    print(f"  Total steps:     {timestep:,}")
    print(f"  Wall time:       {total_wall_time:.1f}s")
    print(f"  Average SPS:     {timestep / total_wall_time:.0f}")
    if episode_rewards:
        print(f"  Final avg reward (last 10 eps): {np.mean(episode_rewards[-10:]):.3f}")
        print(f"  Best episode reward:           {np.max(episode_rewards):.3f}")

    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
