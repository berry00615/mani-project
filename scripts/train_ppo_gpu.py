#!/usr/bin/env python3
"""
GPU-parallel PPO training script for ManiSkill with proper multi-env support.

Key differences from ``train_ppo.py``:

- Per-environment episode tracking (reward, length, success per sub-env).
- NO ``env.reset()`` during rollout — ManiSkill GPU envs auto-reset
  terminated sub-environments, and the returned ``obs`` already contains
  the reset observation for those sub-envs.
- Tensor actions passed directly to ``env.step()`` (no CPU round-trip).
- Sub-env termination detected per-index and only finished sub-envs
  have their accumulators reset.
- GPU memory and utilisation reported each log interval.
- Shape verification at startup for all critical tensors.

Usage::

    # Quick validation with 16 envs, 4096 total steps:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_ppo_gpu.py \\
        --config configs/ppo_pick_cube_collision_gripper_gpu.yaml \\
        --num-envs 16 --total-timesteps 4096

    # Benchmark with 64 envs, 8192 total steps:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_ppo_gpu.py \\
        --config configs/ppo_pick_cube_collision_gripper_gpu.yaml \\
        --num-envs 64 --total-timesteps 8192

    # Stress test with 256 envs:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_ppo_gpu.py \\
        --config configs/ppo_pick_cube_collision_gripper_gpu.yaml \\
        --num-envs 256 --total-timesteps 8192
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

from ppo import ActorCritic, RolloutBuffer, PPO, save_checkpoint
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


# ---------------------------------------------------------------------------
# GPU utilities
# ---------------------------------------------------------------------------

def get_gpu_memory_mb():
    """Return (allocated_MiB, reserved_MiB) for cuda:0, or None."""
    if not torch.cuda.is_available():
        return None
    return (
        torch.cuda.memory_allocated(0) / 1024**2,
        torch.cuda.memory_reserved(0) / 1024**2,
    )


def get_gpu_utilization():
    """Query GPU utilisation via nvidia-smi. Returns dict or empty."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            return {
                "gpu_util_pct": float(parts[0].strip()),
                "mem_util_pct": float(parts[1].strip()),
                "gpu_temp_c": float(parts[2].strip()),
            }
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# CLI and config
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU-Parallel PPO Training on ManiSkill"
    )
    parser.add_argument("--config", type=str,
                        default="configs/ppo_pick_cube_collision_gripper_gpu.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None,
                        help="Rollout steps per cycle (overrides YAML config)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark mode: validate n_steps*num_envs == total_timesteps")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate setup only, do not train")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file to resume from. "
                             "Loads policy weights, optimizer state, and timestep. "
                             "Config file is still required for env/hyperparams.")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    var_y = torch.var(y_true)
    if var_y < 1e-8:
        return float("nan")
    return float(1.0 - torch.var(y_true - y_pred) / var_y)


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------

def make_env(env_id: str, obs_mode: str, sim_backend: str, render_backend: str,
             render_mode, num_envs: int, enable_shadow: bool = False,
             env_kwargs: dict | None = None) -> gym.Env:
    """Create a headless ManiSkill environment.

    When ``num_envs > 1`` and ``sim_backend="auto"``, ManiSkill automatically
    picks ``physx_cuda`` for GPU-parallel simulation.
    """
    import mani_skill.envs   # registers built-in envs
    import envs               # registers custom project envs  # noqa: F811

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


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "timestep", "total_env_steps",
    "episode_reward_mean", "episode_reward_std",
    "episode_length_mean", "episode_length_std",
    "success_rate", "n_episodes_completed",
    "table_collision_rate", "table_collision_penalty_mean",
    "early_close_rate", "mean_gripper_width", "grasp_rate",
    "policy_loss", "value_loss", "entropy", "approx_kl",
    "explained_variance", "sps",
    "gpu_util_pct", "gpu_mem_allocated_mb", "gpu_mem_reserved_mb",
    "wall_time_s",
    "obs_mean_abs", "reward_mean_abs",
    # On-policy masking diagnostics
    "action_mismatch_rate", "executed_gripper_action_mean",
    "arm_action_magnitude", "mean_tcp_cube_distance",
    "approach_reward_mean", "collision_penalty_rcomp_mean",
    "gripper_reward_mean", "early_close_penalty_mean",
    "gripper_open_bonus_mean",
    "mask_active_rate", "action_overridden_rate",
    # Reward integrity
    "reward_reconstruction_error",
    # Masked-dimension entropy diagnostics
    "masked_entropy_mean", "active_dim_count_mean",
]


def save_csv_log(log_path: str, metrics: dict, header: bool = False):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    mode = "w" if header else "a"
    with open(log_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if header:
            writer.writeheader()
        writer.writerow(metrics)


# ---------------------------------------------------------------------------
# Shape verification
# ---------------------------------------------------------------------------

def verify_shapes(env, num_envs: int, device: torch.device):
    """Verify all tensor shapes at startup. Raises AssertionError on mismatch."""
    print("\n--- Shape Verification ---")
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    obs, info = env.reset(seed=0)

    # obs
    assert obs.shape == (num_envs, obs_dim), \
        f"obs shape {obs.shape} != ({num_envs}, {obs_dim})"
    # For GPU sim, obs should be on CUDA; for CPU sim, on CPU
    if env.unwrapped.gpu_sim_enabled:
        assert obs.device.type == "cuda", \
            f"GPU sim enabled but obs device={obs.device} != cuda"
    assert obs.dtype == torch.float32, f"obs dtype {obs.dtype} != float32"
    print(f"  [OK] obs:              {obs.shape}  device={obs.device}")

    # step
    action = torch.randn(num_envs, act_dim, device=device)
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == (num_envs, obs_dim), \
        f"next_obs shape {next_obs.shape} != ({num_envs}, {obs_dim})"
    print(f"  [OK] next_obs:          {next_obs.shape}")

    assert reward.shape == (num_envs,), \
        f"reward shape {reward.shape} != ({num_envs},)"
    print(f"  [OK] reward:            {reward.shape}")

    assert terminated.shape == (num_envs,), \
        f"terminated shape {terminated.shape} != ({num_envs},)"
    assert terminated.dtype == torch.bool, \
        f"terminated dtype {terminated.dtype} != bool"
    print(f"  [OK] terminated:        {terminated.shape}  dtype={terminated.dtype}")

    assert truncated.shape == (num_envs,), \
        f"truncated shape {truncated.shape} != ({num_envs},)"
    print(f"  [OK] truncated:         {truncated.shape}  dtype={truncated.dtype}")

    # info fields
    for key in ["success", "is_grasped", "elapsed_steps"]:
        if key in info:
            v = info[key]
            if isinstance(v, torch.Tensor):
                assert v.shape == (num_envs,), \
                    f"info['{key}'] shape {v.shape} != ({num_envs},)"
                print(f"  [OK] info['{key}']:  {v.shape}")

    # collision info
    if hasattr(env.unwrapped, "get_collision_info"):
        penalty, mask = env.unwrapped.get_collision_info()
        assert penalty.shape == (num_envs,), \
            f"collision_penalty shape {penalty.shape} != ({num_envs},)"
        assert mask.shape == (num_envs,), \
            f"collision_mask shape {mask.shape} != ({num_envs},)"
        print(f"  [OK] collision_penalty: {penalty.shape}")
        print(f"  [OK] collision_mask:    {mask.shape}")

    # gripper info
    if hasattr(env.unwrapped, "get_gripper_info"):
        g_width, g_ec, g_on, g_ec_pen, g_on_bonus = env.unwrapped.get_gripper_info()
        assert g_width.shape == (num_envs,), \
            f"gripper_width shape {g_width.shape} != ({num_envs},)"
        assert g_ec.shape == (num_envs,), \
            f"early_close shape {g_ec.shape} != ({num_envs},)"
        print(f"  [OK] gripper_width:     {g_width.shape}")
        print(f"  [OK] early_close_mask:  {g_ec.shape}")

    # numerical sanity
    assert torch.isfinite(obs).all(), "obs contains NaN/Inf!"
    assert torch.isfinite(next_obs).all(), "next_obs contains NaN/Inf!"
    assert torch.isfinite(reward).all(), "reward contains NaN/Inf!"
    print(f"  [OK] all tensors finite")

    print("  Shape verification PASSED\n")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    # ---- Load config ----
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ---- Overrides ----
    if args.total_timesteps is not None:
        config["total_timesteps"] = args.total_timesteps
    if args.n_steps is not None:
        config["n_steps"] = args.n_steps
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device is not None:
        config["device"] = args.device
    if args.num_envs is not None:
        config["num_envs"] = args.num_envs

    total_timesteps = config["total_timesteps"]
    seed = config["seed"]
    num_envs = config["num_envs"]
    run_name = args.run_name or f"seed{seed}_nenvs{num_envs}"

    # ---- Device ----
    device_str = config.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    # ---- Header ----
    print("=" * 70)
    print("PPO GPU-Parallel Training Setup")
    print("=" * 70)
    print(f"  Config:        {config_path}")
    print(f"  Run name:      {run_name}")
    print(f"  Device:        {device}")
    print(f"  CUDA avail:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:           {torch.cuda.get_device_name(0)}")
        mem = get_gpu_memory_mb()
        if mem:
            print(f"  GPU mem init:  {mem[0]:.0f} MiB allocated, {mem[1]:.0f} MiB reserved")
    print(f"  Seed:          {seed}")
    print(f"  Num envs:      {num_envs}")
    print(f"  Total steps:   {total_timesteps}")

    set_seed(seed)

    # ---- Create environment ----
    print(f"\n  Environment:")
    env_kwargs = config.get("env_kwargs", None)
    print(f"    env_id:         {config['env_id']}")
    print(f"    obs_mode:       {config['obs_mode']}")
    print(f"    sim_backend:    {config.get('sim_backend', 'auto')}")
    print(f"    render_backend: {config.get('render_backend', 'none')}")
    print(f"    num_envs:       {num_envs}")
    if env_kwargs:
        for k, v in env_kwargs.items():
            print(f"    env_kwargs.{k}: {v}")

    base_env = make_env(
        env_id=config["env_id"],
        obs_mode=config["obs_mode"],
        sim_backend=config.get("sim_backend", "auto"),
        render_backend=config.get("render_backend", "none"),
        render_mode=config.get("render_mode", None),
        num_envs=num_envs,
        enable_shadow=config.get("enable_shadow", False),
        env_kwargs=env_kwargs,
    )
    env = ManiSkillVectorEnv(
        base_env,
        auto_reset=True,
        ignore_terminations=False,
    )
    print(f"\n  Env wrapper:      ManiSkillVectorEnv(auto_reset=True)")

    sim_backend = env.unwrapped.backend.sim_backend
    gpu_sim = env.unwrapped.gpu_sim_enabled

    print(f"\n  Sim backend:     {sim_backend}")
    print(f"  GPU sim enabled: {gpu_sim}")
    print(f"  Render backend:  {env.unwrapped.backend.render_backend}")
    print(f"  Obs space:       {env.observation_space.shape}")
    print(f"  Act space:       {env.action_space.shape}")
    print(f"  Control mode:    {env.unwrapped.control_mode}")

    # ---- Critical check: GPU backend ----
    if num_envs > 1 and sim_backend != "physx_cuda":
        print(f"\n  *** WARNING: num_envs={num_envs} > 1 but sim_backend={sim_backend} != physx_cuda!")
        print(f"  *** GPU parallel simulation may not be active!")
    elif num_envs > 1:
        print(f"\n  [OK] physx_cuda confirmed for num_envs={num_envs}")

    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    action_low = env.action_space.low
    action_high = env.action_space.high
    # For batched GPU envs, action bounds include the num_envs dimension.
    # Extract the per-env bounds so policy rescaling works with arbitrary
    # batch sizes during PPO updates.
    if isinstance(action_low, np.ndarray) and action_low.ndim > 1:
        action_low = action_low[0]
    if isinstance(action_high, np.ndarray) and action_high.ndim > 1:
        action_high = action_high[0]

    # ---- Shape verification ----
    verify_shapes(env, num_envs, device)

    # ---- Create policy ----
    policy = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        policy_hidden_sizes=config.get("policy_hidden_sizes", [256, 256, 256]),
        value_hidden_sizes=config.get("value_hidden_sizes", [256, 256, 256]),
        action_low=action_low,
        action_high=action_high,
    ).to(device)

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"\n  Policy params: {total_params:,}")

    # ---- Resume from checkpoint (if specified) ----
    resume_timestep = 0
    if args.resume is not None:
        resume_path = PROJECT_ROOT / args.resume
        if not resume_path.exists():
            print(f"ERROR: Resume checkpoint not found: {resume_path}")
            sys.exit(1)
        print(f"\n  Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        # Load policy weights
        policy.load_state_dict(ckpt["policy_state_dict"])
        # Load optimizer state
        # (PPO must be created first, handled below)
        ckpt_optimizer_state = ckpt.get("optimizer_state_dict", None)
        resume_timestep = ckpt.get("timestep", 0)
        ckpt_env_id = ckpt.get("env_id", "unknown")
        ckpt_config = ckpt.get("config", {})
        ckpt_env_kwargs = ckpt_config.get("env_kwargs", {})
        ckpt_gripper_dist = ckpt_env_kwargs.get("force_gripper_open_until_distance", "N/A")
        print(f"    Checkpoint timestep:             {resume_timestep:,}")
        print(f"    Checkpoint env_id:               {ckpt_env_id}")
        print(f"    Checkpoint force_gripper_open_until_distance: {ckpt_gripper_dist}")
        print(f"    Policy weights loaded:           {len(ckpt['policy_state_dict'])} tensors")
        if args.total_timesteps is None:
            total_timesteps = resume_timestep + config["total_timesteps"]
            print(f"    New total_timesteps (resume + config): {total_timesteps:,}")
        else:
            print(f"    Total timesteps (CLI override):  {total_timesteps:,}")
        print(f"    New force_gripper_open_until_distance: "
              f"{config.get('env_kwargs', {}).get('force_gripper_open_until_distance', 'N/A')}")

    # ---- Create PPO ----
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

    # Restore optimizer state after PPO creation
    if args.resume is not None and ckpt_optimizer_state is not None:
        ppo_algo.optimizer.load_state_dict(ckpt_optimizer_state)
        print(f"    Optimizer state restored")

    # ---- Create buffer ----
    n_steps = config.get("n_steps", 2048)
    n_steps_per_rollout = n_steps
    actual_rollout_env_steps = n_steps_per_rollout * num_envs
    # NOTE: rollout_per_cycle should be selected such that
    # n_steps_per_rollout * num_envs does not exceed available memory
    # for the buffer, and is large enough to give a good GAE horizon.

    buffer = RolloutBuffer(
        buffer_size=n_steps,
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_envs=num_envs,
        device=device,
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
    )

    # ---- Logging setup ----
    output_dir = PROJECT_ROOT / config.get("output_dir", "checkpoints/ppo_pick_cube_collision_gripper_gpu")
    log_dir = PROJECT_ROOT / config.get("log_dir", "logs/ppo_pick_cube_collision_gripper_gpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_log_path = log_dir / f"training_{run_name}.csv"
    save_csv_log(str(csv_log_path), {}, header=True)

    checkpoint_interval = config.get("checkpoint_interval", 10000)

    # ---- Benchmark consistency check ----
    if args.benchmark:
        if actual_rollout_env_steps != total_timesteps:
            print(f"ERROR: Benchmark consistency check failed!")
            print(f"  n_steps ({n_steps}) * num_envs ({num_envs}) = "
                  f"{actual_rollout_env_steps} != total_timesteps ({total_timesteps})")
            print(f"  Each benchmark tier must satisfy: n_steps * num_envs == total_timesteps")
            sys.exit(1)
        print(f"  [benchmark] Consistency OK: {n_steps} steps × {num_envs} envs "
              f"= {actual_rollout_env_steps} env-steps == total_timesteps")

    # ---- Dry run ----
    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN — setup OK.")
        print("=" * 70)
        env.close()
        return

    # =====================================================================
    # Training loop (GPU-parallel, per-env tracking)
    # =====================================================================

    print(f"\n{'=' * 70}")
    print(f"Starting Training: {run_name}")
    print(f"{'=' * 70}")
    print(f"  Output dir:     {output_dir}")
    print(f"  Log dir:        {log_dir}")
    print(f"  Rollout size:   {n_steps_per_rollout} steps × {num_envs} envs "
          f"= {actual_rollout_env_steps} env-steps")
    print(f"  Checkpoint:     every {checkpoint_interval} env-steps")
    print()

    # ---- Per-environment episode accumulators ----
    # These are (num_envs,) tensors that track the current episode
    # for each sub-environment independently.
    ep_rewards = torch.zeros(num_envs, device=device)
    ep_lengths = torch.zeros(num_envs, dtype=torch.int32, device=device)

    # Completed episode logs (Python lists, no size limit needed for short tests)
    completed_rewards: list[float] = []
    completed_lengths: list[int] = []
    completed_successes: list[bool] = []

    # Per-step stat accumulators (for collision, gripper, grasp rates)
    total_env_steps = 0
    collision_steps_total = 0
    collision_penalty_sum = 0.0
    collision_penalty_count = 0
    early_close_steps_total = 0
    gripper_width_sum = 0.0
    gripper_width_count = 0
    grasp_steps_total = 0
    grasp_check_count = 0

    # New on-policy-fix accumulators
    action_mismatch_sum = 0.0       # sum of mismatch rate per step
    action_mismatch_count = 0
    executed_gripper_sum = 0.0
    arm_action_magnitude_sum = 0.0
    tcp_cube_distance_sum = 0.0
    tcp_cube_distance_count = 0
    approach_reward_sum = 0.0
    approach_reward_count = 0
    collision_rcomp_sum = 0.0
    collision_rcomp_count = 0
    gripper_reward_sum = 0.0
    gripper_reward_count = 0
    early_close_pen_sum = 0.0
    early_close_pen_count = 0
    gripper_open_bonus_sum = 0.0
    gripper_open_bonus_count = 0
    mask_active_sum = 0.0
    mask_active_count = 0
    action_overridden_sum = 0.0
    action_overridden_count = 0

    # Reward reconstruction error
    reward_recon_error_sum = 0.0
    reward_recon_error_count = 0

    # Active-dimension tracking for masked entropy
    active_dim_sum = 0.0
    active_dim_count = 0

    # ---- Initial reset ----
    obs, info = env.reset(seed=seed)
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).float().to(device)
    elif obs.device != device:
        obs = obs.to(device)

    t_start = time.time()
    last_log_time = t_start
    timestep = resume_timestep  # total env-interactions (env-steps)
    last_log_timestep = timestep

    try:
        while timestep < total_timesteps:
            buffer.reset()

            # ---- Collect rollout ----
            for step_in_rollout in range(n_steps_per_rollout):
                # --- Policy forward ---
                with torch.no_grad():
                    (action_raw, log_prob_raw, value,
                     action_mean, std) = policy.get_action_dist(obs)

                # --- Save original action BEFORE env modifies it ---
                action_original = action_raw.clone()

                # --- Step environment ---
                next_obs, reward, terminated, truncated, info = env.step(action_raw)

                if isinstance(reward, torch.Tensor) and reward.device != device:
                    reward = reward.to(device)
                if isinstance(terminated, torch.Tensor) and terminated.device != device:
                    terminated = terminated.to(device)
                if isinstance(truncated, torch.Tensor) and truncated.device != device:
                    truncated = truncated.to(device)

                done = terminated | truncated

                # --- ON-POLICY FIX: per-dimension masked log_prob ---
                # After env.step(), action_raw has been modified in-place
                # by the gripper mask.  So action_raw IS action_exec.
                action_exec = action_raw

                # Build per-dimension mask: which dims did the policy control?
                action_dim_mask = torch.ones(num_envs, act_dim, device=device)
                if hasattr(env.unwrapped, "get_action_dim_mask"):
                    env_mask = env.unwrapped.get_action_dim_mask(act_dim)
                    if env_mask is not None:
                        action_dim_mask = env_mask

                # Masked log_prob: only controllable dims contribute
                log_prob_per_dim = policy.get_log_prob_per_dim(
                    action_mean, std, action_exec
                )  # (num_envs, act_dim)
                log_prob_exec = (log_prob_per_dim * action_dim_mask).sum(dim=-1)

                # --- Per-step tracking ---
                total_env_steps += num_envs

                # Collision tracking
                if hasattr(env.unwrapped, "get_collision_info"):
                    c_pen, c_mask = env.unwrapped.get_collision_info()
                    if c_mask is not None:
                        collision_steps_total += int(c_mask.sum().item())
                    if c_pen is not None:
                        collision_penalty_sum += float(c_pen.sum().item())
                        collision_penalty_count += num_envs

                # Gripper tracking
                if hasattr(env.unwrapped, "get_gripper_info"):
                    g_width, g_ec, g_on, g_ec_pen, g_on_bonus = \
                        env.unwrapped.get_gripper_info()
                    if g_ec is not None:
                        early_close_steps_total += int(g_ec.sum().item())
                    if g_width is not None:
                        gripper_width_sum += float(g_width.sum().item())
                        gripper_width_count += num_envs

                # Grasp tracking (from info dict)
                is_grasped = info.get("is_grasped", None)
                if is_grasped is not None:
                    grasp_check_count += num_envs
                    grasp_steps_total += int(is_grasped.sum().item())

                # --- New diagnostics: action mismatch, reward components ---
                if hasattr(env.unwrapped, "get_gripper_mask_info"):
                    mask_info = env.unwrapped.get_gripper_mask_info()
                    if mask_info[0] is not None:  # mask_active
                        n = num_envs
                        # Action mismatch: compare executed vs original
                        mismatch = (action_exec[..., -1] != action_original[..., -1]).float()
                        action_mismatch_sum += float(mismatch.sum().item())
                        action_mismatch_count += n
                        # Executed gripper action
                        exec_grip = mask_info[4]  # executed_gripper_action
                        if exec_grip is not None:
                            executed_gripper_sum += float(exec_grip.sum().item())
                        # Arm action magnitude
                        arm_action_magnitude_sum += float(
                            action_exec[..., :-1].norm(dim=-1).sum().item()
                        )
                        # TCP distance
                        tcp_dist = mask_info[5]  # tcp_cube_distance
                        if tcp_dist is not None:
                            tcp_cube_distance_sum += float(tcp_dist.sum().item())
                            tcp_cube_distance_count += n
                        # Mask metrics
                        mask_active_sum += float(mask_info[0].float().sum().item())
                        mask_active_count += n
                        overridden = mask_info[1]  # action_overridden
                        if overridden is not None:
                            action_overridden_sum += float(overridden.float().sum().item())
                            action_overridden_count += n

                # Active-dimension count for entropy diagnostics
                active_dim_sum += float(action_dim_mask.sum().item())
                active_dim_count += num_envs

                # Reward reconstruction error
                if hasattr(env.unwrapped, "get_reward_components"):
                    rcomp = env.unwrapped.get_reward_components()
                    # rcomp: (original_reward, collision_penalty, gripper_reward,
                    #         early_close_penalty, gripper_open_bonus)
                    if rcomp[0] is not None:
                        approach_reward_sum += float(rcomp[0].sum().item())
                        approach_reward_count += num_envs
                    if len(rcomp) > 1 and rcomp[1] is not None:
                        collision_rcomp_sum += float(rcomp[1].sum().item())
                        collision_rcomp_count += num_envs
                    if len(rcomp) > 2 and rcomp[2] is not None:
                        gripper_reward_sum += float(rcomp[2].sum().item())
                        gripper_reward_count += num_envs
                    if len(rcomp) > 3 and rcomp[3] is not None:
                        early_close_pen_sum += float(rcomp[3].sum().item())
                        early_close_pen_count += num_envs
                    if len(rcomp) > 4 and rcomp[4] is not None:
                        gripper_open_bonus_sum += float(rcomp[4].sum().item())
                        gripper_open_bonus_count += num_envs

                    # Reconstruct reward from components and compare to actual.
                    # The env normalizes dense reward by /5; components are
                    # pre-normalization, so we divide to match the actual reward.
                    if rcomp[0] is not None and rcomp[1] is not None and rcomp[2] is not None:
                        recon = (rcomp[0] - rcomp[1] + rcomp[2]) / 5.0
                        error = (reward - recon).abs().mean().item()
                        reward_recon_error_sum += error * num_envs
                        reward_recon_error_count += num_envs

                # --- Per-episode accumulator update ---
                ep_rewards += reward
                ep_lengths += 1

                # --- Handle completed episodes ---
                if done.any():
                    done_indices = done.nonzero(as_tuple=True)[0]
                    # final_info carries the episode-outcome dict for the
                    # episode that just finished (pre-reset).  Regular info
                    # at a done step already reflects the *new* episode
                    # (post-reset), so success/is_grasped etc. must be
                    # read from final_info when available.
                    final_info = info.get("final_info", None)
                    for idx in done_indices:
                        i = idx.item()
                        # Record completed episode stats
                        completed_rewards.append(float(ep_rewards[i].item()))
                        completed_lengths.append(int(ep_lengths[i].item()))
                        # Success / is_grasped from final_info (pre-reset)
                        s_source = final_info if final_info is not None else info
                        s = s_source.get("success", None)
                        if s is not None:
                            completed_successes.append(bool(s[i].item()))
                        else:
                            completed_successes.append(False)

                        # Reset per-env accumulators for the new episode
                        ep_rewards[i] = 0.0
                        ep_lengths[i] = 0

                # --- Time-limit bootstrap ---
                # For truncated sub-envs we need V(final_observation) as the
                # TD-error bootstrap target, NOT the value of the post-reset
                # observation (which belongs to the next episode).
                bootstrap_value = torch.zeros(num_envs, device=device)
                if truncated.any():
                    final_obs = info.get("final_observation", None)
                    if final_obs is not None:
                        if isinstance(final_obs, np.ndarray):
                            final_obs = torch.from_numpy(final_obs).float().to(device)
                        elif final_obs.device != device:
                            final_obs = final_obs.to(device)
                        with torch.no_grad():
                            _, _, _boot_val = policy.get_action(final_obs)
                        if _boot_val.ndim == 2 and _boot_val.shape[-1] == 1:
                            _boot_val = _boot_val.squeeze(-1)
                        # Only assign to truncated envs (terminated use 0)
                        trun_mask = truncated.bool()
                        bootstrap_value[trun_mask] = _boot_val[trun_mask]

                # --- Store in buffer ---
                # ON-POLICY FIX: store action_exec and log_prob_exec, NOT
                # the original (action_raw, log_prob_raw).  The reward and
                # next_obs came from action_exec, so log_prob must match.
                if value.ndim == 2 and value.shape[-1] == 1:
                    value = value.squeeze(-1)
                buffer.add(
                    obs=obs,
                    action=action_exec,
                    reward=reward,
                    value=value,
                    log_prob=log_prob_exec,
                    done=done,
                    bootstrap_value=bootstrap_value,
                    action_dim_mask=action_dim_mask,
                )

                # --- Advance ---
                obs = next_obs
                if isinstance(obs, np.ndarray):
                    obs = torch.from_numpy(obs).float().to(device)
                elif obs.device != device:
                    obs = obs.to(device)

                timestep += num_envs
                if timestep >= total_timesteps:
                    break

            # ---- Compute GAE ----
            with torch.no_grad():
                _, _, last_value = policy.get_action(obs)
            # Safe squeeze: value_net outputs (num_envs, 1) → (num_envs,)
            if last_value.ndim == 2 and last_value.shape[-1] == 1:
                last_value = last_value.squeeze(-1)
            last_done = done  # (num_envs,) bool
            buffer.compute_advantages(last_value, last_done)

            # ---- PPO update ----
            metrics = ppo_algo.update(buffer)

            # ---- Logging ----
            now = time.time()
            wall_time = now - t_start
            steps_since_log = timestep - last_log_timestep
            dt = now - last_log_time
            sps = steps_since_log / dt if dt > 0 else 0.0
            last_log_time = now
            last_log_timestep = timestep

            # Explained variance
            data = buffer.get_training_data()
            explained_var = compute_explained_variance(data["values"], data["returns"])

            # Episode stats (across all completed episodes)
            if completed_rewards:
                recent_n = min(100, len(completed_rewards))
                recent_rewards = completed_rewards[-recent_n:]
                ep_rew_mean = np.mean(recent_rewards)
                ep_rew_std = np.std(recent_rewards)
                recent_lengths = completed_lengths[-recent_n:]
                ep_len_mean = np.mean(recent_lengths)
                ep_len_std = np.std(recent_lengths)
                # Success rate over all completed episodes
                success_rate = np.mean(completed_successes) if completed_successes else 0.0
                n_completed = len(completed_rewards)
            else:
                ep_rew_mean = ep_rew_std = ep_len_mean = ep_len_std = 0.0
                success_rate = 0.0
                n_completed = 0

            # Collision / gripper / grasp rates
            table_collision_rate = (
                collision_steps_total / max(total_env_steps, 1)
            )
            table_collision_penalty_mean = (
                collision_penalty_sum / max(collision_penalty_count, 1)
            )
            early_close_rate = (
                early_close_steps_total / max(total_env_steps, 1)
            )
            mean_gripper_width = (
                gripper_width_sum / max(gripper_width_count, 1)
                if gripper_width_count > 0 else 0.0
            )
            grasp_rate = (
                grasp_steps_total / max(grasp_check_count, 1)
                if grasp_check_count > 0 else 0.0
            )

            # GPU stats
            gpu_util = get_gpu_utilization()
            gpu_mem = get_gpu_memory_mb()

            # Numerical health checks
            obs_mean_abs = float(data["observations"].abs().mean().item())
            reward_mean_abs = float(data["rewards"].abs().mean().item())

            # New on-policy diagnostics
            action_mismatch_rate = (
                action_mismatch_sum / max(action_mismatch_count, 1)
            )
            executed_gripper_mean = (
                executed_gripper_sum / max(action_mismatch_count, 1)
            )
            arm_action_mag = (
                arm_action_magnitude_sum / max(action_mismatch_count, 1)
            )
            mean_tcp_dist = (
                tcp_cube_distance_sum / max(tcp_cube_distance_count, 1)
            )
            approach_rew_mean = (
                approach_reward_sum / max(approach_reward_count, 1)
            )
            collision_rcomp_mean = (
                collision_rcomp_sum / max(collision_rcomp_count, 1)
            )
            gripper_rew_mean = (
                gripper_reward_sum / max(gripper_reward_count, 1)
            )
            early_close_pen_mean_val = (
                early_close_pen_sum / max(early_close_pen_count, 1)
            )
            gripper_open_bonus_mean_val = (
                gripper_open_bonus_sum / max(gripper_open_bonus_count, 1)
            )
            mask_active_rate_val = (
                mask_active_sum / max(mask_active_count, 1)
            )
            action_overridden_rate_val = (
                action_overridden_sum / max(action_overridden_count, 1)
            )

            reward_recon_error = (
                reward_recon_error_sum / max(reward_recon_error_count, 1)
            )
            active_dim_mean = (
                active_dim_sum / max(active_dim_count, 1)
            )

            log_entry = {
                "timestep": timestep,
                "total_env_steps": total_env_steps,
                "episode_reward_mean": round(ep_rew_mean, 4),
                "episode_reward_std": round(ep_rew_std, 4),
                "episode_length_mean": round(ep_len_mean, 1),
                "episode_length_std": round(ep_len_std, 1),
                "success_rate": round(success_rate, 6),
                "n_episodes_completed": n_completed,
                "table_collision_rate": round(table_collision_rate, 6),
                "table_collision_penalty_mean": round(table_collision_penalty_mean, 6),
                "early_close_rate": round(early_close_rate, 6),
                "mean_gripper_width": round(mean_gripper_width, 6),
                "grasp_rate": round(grasp_rate, 6),
                "policy_loss": round(metrics["policy_loss"], 6),
                "value_loss": round(metrics["value_loss"], 6),
                "entropy": round(metrics["entropy"], 6),
                "approx_kl": round(metrics["approx_kl"], 6),
                "explained_variance": round(explained_var, 4)
                    if not np.isnan(explained_var) else "nan",
                "sps": round(sps, 1),
                "gpu_util_pct": round(gpu_util.get("gpu_util_pct", 0), 1),
                "gpu_mem_allocated_mb": round(gpu_mem[0], 1)
                    if gpu_mem else 0,
                "gpu_mem_reserved_mb": round(gpu_mem[1], 1)
                    if gpu_mem else 0,
                "wall_time_s": round(wall_time, 1),
                "obs_mean_abs": round(obs_mean_abs, 4),
                "reward_mean_abs": round(reward_mean_abs, 4),
                # New on-policy diagnostics
                "action_mismatch_rate": round(action_mismatch_rate, 6),
                "executed_gripper_action_mean": round(executed_gripper_mean, 6),
                "arm_action_magnitude": round(arm_action_mag, 6),
                "mean_tcp_cube_distance": round(mean_tcp_dist, 6),
                "approach_reward_mean": round(approach_rew_mean, 6),
                "collision_penalty_rcomp_mean": round(collision_rcomp_mean, 6),
                "gripper_reward_mean": round(gripper_rew_mean, 6),
                "early_close_penalty_mean": round(early_close_pen_mean_val, 6),
                "gripper_open_bonus_mean": round(gripper_open_bonus_mean_val, 6),
                "mask_active_rate": round(mask_active_rate_val, 6),
                "action_overridden_rate": round(action_overridden_rate_val, 6),
                "reward_reconstruction_error": round(reward_recon_error, 10),
                "masked_entropy_mean": round(metrics["entropy"], 6),
                "active_dim_count_mean": round(active_dim_mean, 4),
            }
            save_csv_log(str(csv_log_path), log_entry)

            # Console output
            gpu_str = ""
            if gpu_util:
                gpu_str = (f" | GPU:{gpu_util.get('gpu_util_pct', 0):.0f}% "
                           f"mem:{gpu_mem[0]:.0f}MiB" if gpu_mem else "")
            print(f"  Step {timestep:>8,}/{total_timesteps:,} | "
                  f"ep_rew={ep_rew_mean:>7.3f}±{ep_rew_std:.3f} | "
                  f"ep_len={ep_len_mean:.0f} | "
                  f"succ={success_rate:.3f} ({n_completed} eps) | "
                  f"col={table_collision_rate:.3f} | "
                  f"ec={early_close_rate:.3f} | "
                  f"grasp={grasp_rate:.3f} | "
                  f"pol_loss={metrics['policy_loss']:.4f} | "
                  f"val_loss={metrics['value_loss']:.4f} | "
                  f"ent={metrics['entropy']:.4f} | "
                  f"mismatch={action_mismatch_rate:.3f} | "
                  f"app_rew={approach_rew_mean:.3f} | "
                  f"tcp_dist={mean_tcp_dist:.3f} | "
                  f"SPS={sps:.0f}"
                  f"{gpu_str}")

            # Numerical anomaly detection
            anomalies = []
            if obs_mean_abs > 100:
                anomalies.append(f"obs_mean_abs={obs_mean_abs:.1f} (large)")
            if reward_mean_abs > 100:
                anomalies.append(f"reward_mean_abs={reward_mean_abs:.1f} (large)")
            if np.isnan(explained_var):
                anomalies.append("explained_variance=NaN")
            if metrics["approx_kl"] > 0.1:
                anomalies.append(f"approx_kl={metrics['approx_kl']:.4f} (large)")
            if anomalies:
                print(f"  *** ANOMALY: {'; '.join(anomalies)}")

            # ---- Checkpoint (infrequent for tests) ----
            if (checkpoint_interval > 0
                    and timestep % checkpoint_interval < n_steps_per_rollout * num_envs):
                ckpt_path = output_dir / f"checkpoint_{timestep}.pt"
                save_checkpoint(
                    path=str(ckpt_path),
                    policy=policy,
                    ppo=ppo_algo,
                    timestep=timestep,
                    env_id=config["env_id"],
                    obs_mode=config["obs_mode"],
                    control_mode=env.unwrapped.control_mode,
                    config=config,
                )
                print(f"  -> Saved checkpoint: {ckpt_path}")

    except KeyboardInterrupt:
        print("\n\nInterrupted!")
        env.close()
        sys.exit(0)

    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        env.close()
        sys.exit(1)

    # =====================================================================
    # Final summary
    # =====================================================================
    total_wall = time.time() - t_start
    avg_sps = timestep / total_wall if total_wall > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"Training Complete: {run_name}")
    print(f"{'=' * 70}")
    print(f"  Total env-steps:        {timestep:,}")
    print(f"  Wall time:              {total_wall:.1f}s ({total_wall/60:.1f}min)")
    print(f"  Average SPS:            {avg_sps:.0f}")
    print(f"  Episodes completed:     {len(completed_rewards)}")
    if completed_rewards:
        print(f"  Final avg ep reward:    {np.mean(completed_rewards[-10:]):.3f}"
              if len(completed_rewards) >= 10 else
              f"  Avg ep reward:          {np.mean(completed_rewards):.3f}")
        print(f"  Success rate:           {np.mean(completed_successes):.4f}"
              f" ({sum(completed_successes)}/{len(completed_successes)})")
    print(f"  Table collision rate:   {table_collision_rate:.4f}")
    print(f"  Early close rate:       {early_close_rate:.4f}")
    print(f"  Grasp rate:             {grasp_rate:.4f}")

    # GPU memory final
    final_mem = get_gpu_memory_mb()
    if final_mem:
        print(f"  GPU mem final:          {final_mem[0]:.0f} MiB allocated, "
              f"{final_mem[1]:.0f} MiB reserved")

    # Final checkpoint
    final_path = output_dir / f"final_{run_name}.pt"
    save_checkpoint(
        path=str(final_path),
        policy=policy,
        ppo=ppo_algo,
        timestep=timestep,
        env_id=config["env_id"],
        obs_mode=config["obs_mode"],
        control_mode=env.unwrapped.control_mode,
        config=config,
    )
    print(f"  Final checkpoint:       {final_path}")

    # CSV log path for reference
    print(f"  CSV log:                {csv_log_path}")
    print(f"\nDone.")

    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
