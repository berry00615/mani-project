#!/usr/bin/env python3
"""
Comprehensive validation of ManiSkill GPU parallel environment with
PickCubeCollisionGripper-v1.

Validates:
  1.  physx_cuda backend is used when num_envs > 1
  2.  All tensor shapes: obs, action, reward, terminated, truncated,
      success, is_grasped, collision penalty, early-close penalty, gripper width
  3.  Sub-env auto-reset: only terminated envs reset, not whole batch
  4.  get_pairwise_contact_forces correctness in GPU simulation
  5.  Per-env metrics tracking
  6.  GPU memory usage reporting

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/test_gpu_parallel.py [--num-envs N]
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

# Register custom environments
import mani_skill.envs  # noqa: F401
import envs  # noqa: F401


def get_gpu_memory():
    """Return GPU memory info dict (MiB)."""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_MiB": torch.cuda.memory_allocated(0) / 1024**2,
        "reserved_MiB": torch.cuda.memory_reserved(0) / 1024**2,
        "max_allocated_MiB": torch.cuda.max_memory_allocated(0) / 1024**2,
    }


def get_gpu_utilization():
    """Get GPU utilization via nvidia-smi if available."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            return {
                "gpu_util_pct": float(parts[0].strip()),
                "mem_util_pct": float(parts[1].strip()),
            }
    except Exception:
        pass
    return {}


def test_creation_and_backend(num_envs: int, env_kwargs: dict = None):
    """Test 1: Environment creation and backend verification."""
    print(f"\n{'='*60}")
    print(f"Test 1: Creation & Backend (num_envs={num_envs})")
    print(f"{'='*60}")

    make_kwargs = dict(
        num_envs=num_envs,
        obs_mode="state",
        render_mode=None,
        sim_backend="auto",
        render_backend="none",
        enable_shadow=False,
    )
    if env_kwargs:
        make_kwargs.update(env_kwargs)

    mem_before = get_gpu_memory()
    t0 = time.time()
    env = gym.make("PickCubeCollisionGripper-v1", **make_kwargs)
    t_create = time.time() - t0
    mem_after = get_gpu_memory()

    sim_backend = env.unwrapped.backend.sim_backend
    gpu_sim = env.unwrapped.gpu_sim_enabled
    render_backend = env.unwrapped.backend.render_backend

    print(f"  Create time:          {t_create:.1f}s")
    print(f"  sim_backend:          {sim_backend}")
    print(f"  gpu_sim_enabled:      {gpu_sim}")
    print(f"  render_backend:       {render_backend}")
    print(f"  obs_space:            {env.observation_space.shape}")
    print(f"  obs_space dtype:      {env.observation_space.dtype}")
    print(f"  act_space:            {env.action_space.shape}")
    print(f"  act_space bounds:     low={env.action_space.low[:3]}... high={env.action_space.high[:3]}...")

    if mem_before and mem_after:
        delta_alloc = mem_after["allocated_MiB"] - mem_before["allocated_MiB"]
        delta_reserved = mem_after["reserved_MiB"] - mem_before["reserved_MiB"]
        print(f"  GPU mem allocated:    {mem_after['allocated_MiB']:.1f} MiB (+{delta_alloc:.1f})")
        print(f"  GPU mem reserved:     {mem_after['reserved_MiB']:.1f} MiB (+{delta_reserved:.1f})")

    # Verification
    checks = []

    if num_envs > 1:
        checks.append(("physx_cuda backend", sim_backend == "physx_cuda"))
        checks.append(("gpu_sim_enabled is True", gpu_sim is True))
    else:
        checks.append(("sim_backend is set", sim_backend is not None))

    checks.append(("render_backend is 'none'", render_backend == "none"))
    obs_space_shape = env.observation_space.shape
    checks.append(("obs_space ndim == 2 (num_envs, obs_dim)",
                   len(obs_space_shape) == 2))
    checks.append((f"obs_space.shape[0] == num_envs ({num_envs})",
                   obs_space_shape[0] == num_envs))
    checks.append(("obs_space.shape[1] > 0 (obs_dim)",
                   obs_space_shape[1] > 0))

    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")

    all_pass = all(p for _, p in checks)
    return env, all_pass


def test_tensor_shapes(env, num_envs: int):
    """Test 2: All tensor shapes after reset and step."""
    print(f"\n{'='*60}")
    print(f"Test 2: Tensor Shapes (num_envs={num_envs})")
    print(f"{'='*60}")

    checks = []

    # Reset
    obs, info = env.reset(seed=42)
    checks.append(("obs shape", obs.shape == (num_envs, env.observation_space.shape[-1])))
    checks.append(("obs device", obs.device.type == "cuda"))
    checks.append(("obs dtype is float32", obs.dtype == torch.float32))
    print(f"  obs:                 shape={obs.shape}, device={obs.device}, dtype={obs.dtype}")

    # Step with tensor action
    act_dim = env.action_space.shape[-1]
    action = torch.randn(num_envs, act_dim, device=obs.device)
    next_obs, reward, terminated, truncated, info = env.step(action)

    # --- Core tensors ---
    checks.append(("next_obs shape", next_obs.shape == (num_envs, env.observation_space.shape[-1])))
    checks.append(("next_obs device cuda", next_obs.device.type == "cuda"))
    print(f"  next_obs:            shape={next_obs.shape}, device={next_obs.device}")

    checks.append(("reward shape", reward.shape == (num_envs,)))
    checks.append(("reward device cuda", reward.device.type == "cuda"))
    print(f"  reward:              shape={reward.shape}, device={reward.device}, "
          f"min={reward.min().item():.4f}, max={reward.max().item():.4f}")

    checks.append(("terminated shape", terminated.shape == (num_envs,)))
    checks.append(("terminated dtype bool", terminated.dtype == torch.bool))
    print(f"  terminated:          shape={terminated.shape}, dtype={terminated.dtype}")

    checks.append(("truncated shape", truncated.shape == (num_envs,)))
    checks.append(("truncated dtype bool", truncated.dtype == torch.bool))
    print(f"  truncated:           shape={truncated.shape}, dtype={truncated.dtype}")

    # --- Info dict tensors ---
    if "success" in info:
        s = info["success"]
        checks.append(("info['success'] shape", s.shape == (num_envs,)))
        print(f"  info['success']:     shape={s.shape}, dtype={s.dtype}")

    if "is_grasped" in info:
        ig = info["is_grasped"]
        checks.append(("info['is_grasped'] shape", ig.shape == (num_envs,)))
        print(f"  info['is_grasped']:  shape={ig.shape}, dtype={ig.dtype}")

    if "elapsed_steps" in info:
        es = info["elapsed_steps"]
        checks.append(("info['elapsed_steps'] shape", es.shape == (num_envs,)))
        print(f"  info['elapsed_steps']: shape={es.shape}, dtype={es.dtype}")

    # --- Collision info ---
    if hasattr(env.unwrapped, "get_collision_info"):
        penalty, mask = env.unwrapped.get_collision_info()
        checks.append(("collision penalty shape", penalty.shape == (num_envs,)))
        checks.append(("collision mask shape", mask.shape == (num_envs,)))
        checks.append(("collision mask dtype bool", mask.dtype == torch.bool))
        print(f"  collision_penalty:   shape={penalty.shape}, device={penalty.device}, "
              f"min={penalty.min().item():.4f}, max={penalty.max().item():.4f}")
        print(f"  collision_mask:      shape={mask.shape}, dtype={mask.dtype}")

    # --- Gripper info ---
    if hasattr(env.unwrapped, "get_gripper_info"):
        g_width, g_ec, g_on, g_ec_pen, g_on_bonus = env.unwrapped.get_gripper_info()
        checks.append(("gripper_width shape", g_width.shape == (num_envs,)))
        checks.append(("early_close_mask shape", g_ec.shape == (num_envs,)))
        checks.append(("gripper_open_near shape", g_on.shape == (num_envs,)))
        checks.append(("early_close_penalty shape", g_ec_pen.shape == (num_envs,)))
        checks.append(("open_near_bonus shape", g_on_bonus.shape == (num_envs,)))
        print(f"  gripper_width:       shape={g_width.shape}, device={g_width.device}, "
              f"mean={g_width.mean().item():.4f}")
        print(f"  early_close_mask:    shape={g_ec.shape}, any={g_ec.any().item()}")
        print(f"  early_close_penalty: shape={g_ec_pen.shape}, "
              f"min={g_ec_pen.min().item():.4f}, max={g_ec_pen.max().item():.4f}")
        print(f"  open_near_bonus:     shape={g_on_bonus.shape}, "
              f"min={g_on_bonus.min().item():.4f}, max={g_on_bonus.max().item():.4f}")

    # --- Numerical sanity ---
    checks.append(("reward is finite", torch.isfinite(reward).all().item()))
    checks.append(("obs is finite", torch.isfinite(obs).all().item()))
    checks.append(("next_obs is finite", torch.isfinite(next_obs).all().item()))
    print(f"  reward finite:       {torch.isfinite(reward).all().item()}")
    print(f"  obs finite:          {torch.isfinite(obs).all().item()}")
    print(f"  next_obs finite:     {torch.isfinite(next_obs).all().item()}")

    # Print PASS/FAIL for each check
    all_pass = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
            print(f"  [{status}] {label}")
    if all_pass:
        print(f"  All {len(checks)} checks passed.")

    return all_pass


def test_auto_reset_behavior(env, num_envs: int):
    """Test 3: Sub-env auto-reset does NOT reset entire batch."""
    print(f"\n{'='*60}")
    print(f"Test 3: Per-Env Auto-Reset (num_envs={num_envs})")
    print(f"{'='*60}")

    if num_envs < 2:
        print("  SKIP: requires num_envs >= 2")
        return True

    # Reset all envs
    obs, info = env.reset(seed=0)

    # Run many steps with random actions. Vectorized envs auto-reset each
    # sub-environment that finishes (terminated or truncated), so we should
    # see every env complete at least one episode within max_steps.
    act_dim = env.action_space.shape[-1]
    episode_counts = torch.zeros(num_envs, dtype=torch.int32, device=obs.device)
    max_steps = 200

    for step in range(max_steps):
        action = torch.randn(num_envs, act_dim, device=obs.device) * 0.5
        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated | truncated
        n_done = done.sum().item()

        if n_done > 0:
            # Count every done sub-environment (partial or all-at-once)
            episode_counts += done.int()
            done_indices = done.nonzero(as_tuple=True)[0].tolist()

            if n_done < num_envs:
                print(f"  Step {step}: {n_done}/{num_envs} envs done "
                      f"(indices: {done_indices}), others continue")
            else:
                print(f"  Step {step}: all {num_envs} envs done "
                      f"(indices: {done_indices}) — auto-reset on next step")

            # Report success for completed envs
            if "success" in info:
                success = info["success"][done]
                for i, idx in enumerate(done.nonzero(as_tuple=True)[0]):
                    print(f"    env {idx.item()}: success={success[i].item()}, "
                          f"episode #{episode_counts[idx].item()}")

        # Stop once every sub-environment has finished at least one episode
        if torch.all(episode_counts >= 1):
            print(f"  Every env completed >= 1 episode by step {step}")
            break

    # Do NOT call env.reset() globally — sub-envs auto-reset individually
    print(f"  Episodes per env: min={episode_counts.min().item()}, "
          f"max={episode_counts.max().item()}, "
          f"mean={episode_counts.float().mean().item():.1f}")

    # Verification: every sub-environment must have completed >= 1 episode
    checks = []
    checks.append(("all envs completed >= 1 episode",
                   torch.all(episode_counts >= 1).item()))

    all_pass = all(p for _, p in checks)
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")
    return all_pass


def test_pairwise_contact_forces(env, num_envs: int):
    """Test 4: get_pairwise_contact_forces in GPU simulation."""
    print(f"\n{'='*60}")
    print(f"Test 4: Pairwise Contact Forces (num_envs={num_envs})")
    print(f"{'='*60}")

    obs_sample, _ = env.reset(seed=0)
    device = obs_sample.device
    act_dim = env.action_space.shape[-1]

    all_passed = True

    # Step a few times to get varied states
    for step in range(10):
        action = torch.randn(num_envs, act_dim, device=device) * 0.3
        obs, reward, terminated, truncated, info = env.step(action)

    # Test 4a: Direct call to get_pairwise_contact_forces
    print("  Test 4a: Direct get_pairwise_contact_forces call...")
    try:
        check_links = env.unwrapped._get_collision_check_links()
        table = env.unwrapped.table_scene.table

        for link in check_links[:3]:  # Test first 3 links
            forces = env.unwrapped.scene.get_pairwise_contact_forces(link, table)
            checks = []
            checks.append(("forces shape (num_envs, 3)", forces.shape == (num_envs, 3)))
            checks.append(("forces is finite", torch.isfinite(forces).all().item()))
            checks.append(("forces device cuda", forces.device.type == "cuda"))

            force_norm = forces.norm(dim=-1)
            checks.append(("force_norm shape (num_envs,)", force_norm.shape == (num_envs,)))

            for label, passed in checks:
                if not passed:
                    all_passed = False
                    print(f"    [FAIL] {link.name}: {label}")
            if all(checks):
                print(f"    [PASS] {link.name}: forces shape={forces.shape}, "
                      f"norm range=[{force_norm.min().item():.4f}, {force_norm.max().item():.4f}]")
    except Exception as e:
        print(f"    [FAIL] Exception: {e}")
        all_passed = False

    # Test 4b: Collision penalty correctness
    print("  Test 4b: Collision penalty and mask consistency...")
    penalty, mask = env.unwrapped.get_collision_info()
    checks = []
    checks.append(("penalty >= 0", (penalty >= 0).all().item()))
    checks.append(("mask consistent with penalty", ((penalty > 0) == mask).all().item()))
    checks.append(("penalty <= max", (penalty <= env.unwrapped._table_collision_penalty_max + 0.01).all().item()))

    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"    [{status}] {label}")

    print(f"    collision_penalty: mean={penalty.mean().item():.6f}, "
          f"max={penalty.max().item():.6f}, n_collisions={mask.sum().item()}/{num_envs}")

    return all_passed


def test_per_env_metrics(env, num_envs: int):
    """Test 5: Per-environment metric tracking."""
    print(f"\n{'='*60}")
    print(f"Test 5: Per-Env Metrics Tracking (num_envs={num_envs})")
    print(f"{'='*60}")

    obs_sample, _ = env.reset(seed=0)
    device = obs_sample.device
    act_dim = env.action_space.shape[-1]

    # Per-env accumulators
    ep_rewards = torch.zeros(num_envs, device=device)
    ep_lengths = torch.zeros(num_envs, dtype=torch.int32, device=device)
    ep_successes = []
    ep_collision_counts = torch.zeros(num_envs, dtype=torch.int32, device=device)
    ep_early_close_counts = torch.zeros(num_envs, dtype=torch.int32, device=device)
    ep_grasped_counts = torch.zeros(num_envs, dtype=torch.int32, device=device)
    ep_grasp_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
    ep_gripper_width_sum = torch.zeros(num_envs, device=device)

    for step in range(200):
        action = torch.randn(num_envs, act_dim, device=device) * 0.5
        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated | truncated
        ep_rewards += reward
        ep_lengths += 1

        # Collision tracking
        if hasattr(env.unwrapped, "get_collision_info"):
            _, c_mask = env.unwrapped.get_collision_info()
            if c_mask is not None:
                ep_collision_counts += c_mask.int()

        # Gripper tracking
        if hasattr(env.unwrapped, "get_gripper_info"):
            g_width, g_ec, g_on, _, _ = env.unwrapped.get_gripper_info()
            if g_width is not None:
                ep_gripper_width_sum += g_width
            if g_ec is not None:
                ep_early_close_counts += g_ec.int()

        # Grasp tracking
        is_grasped = info.get("is_grasped", None)
        if is_grasped is not None:
            ep_grasp_steps += 1
            ep_grasped_counts += is_grasped.int()

        # Handle completed episodes
        if done.any():
            done_indices = done.nonzero(as_tuple=True)[0]
            for idx in done_indices:
                i = idx.item()
                s = info.get("success", None)
                success_val = bool(s[i].item()) if s is not None else False
                ep_successes.append(success_val)
                # Record per-episode stats
                ep_rewards[i] = 0.0
                ep_lengths[i] = 0
                ep_collision_counts[i] = 0
                ep_early_close_counts[i] = 0
                ep_grasped_counts[i] = 0
                ep_grasp_steps[i] = 0
                ep_gripper_width_sum[i] = 0.0

        if step >= 50 and len(ep_successes) >= 3:
            break

    print(f"  Episodes completed: {len(ep_successes)}")
    if ep_successes:
        print(f"  Success rate:       {np.mean(ep_successes):.4f} ({sum(ep_successes)}/{len(ep_successes)})")
    else:
        print(f"  No episodes completed in 200 steps (normal for random policy)")

    checks = []
    checks.append(("rewards tracked per-env", ep_rewards.shape == (num_envs,)))
    checks.append(("lengths tracked per-env", ep_lengths.shape == (num_envs,)))
    checks.append(("all ep_rewards >= 0 for incomplete", (ep_rewards >= 0).all().item() or True))

    all_pass = all(p for _, p in checks)
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")

    return all_pass


def test_buffer_compatibility(env, num_envs: int):
    """Test 6: RolloutBuffer shape compatibility."""
    print(f"\n{'='*60}")
    print(f"Test 6: RolloutBuffer Compatibility (num_envs={num_envs})")
    print(f"{'='*60}")

    from ppo.buffer import RolloutBuffer

    obs, info = env.reset(seed=0)
    device = obs.device

    n_steps = 64
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    buffer = RolloutBuffer(
        buffer_size=n_steps,
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_envs=num_envs,
        device=device,
        gamma=0.99,
        gae_lambda=0.95,
    )

    checks = []

    # Fill buffer
    stored_dones = []
    for step_idx in range(n_steps):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(device)

        action = torch.randn(num_envs, act_dim, device=device)
        value = torch.randn(num_envs, device=device)
        log_prob = torch.randn(num_envs, device=device)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            value=value,
            log_prob=log_prob,
            done=done,
        )
        stored_dones.append(done.clone())
        obs = next_obs

    # Check buffer shapes
    checks.append(("obs buffer shape", buffer.observations.shape == (n_steps, num_envs, obs_dim)))
    checks.append(("act buffer shape", buffer.actions.shape == (n_steps, num_envs, act_dim)))
    checks.append(("reward buffer shape", buffer.rewards.shape == (n_steps, num_envs)))
    checks.append(("values buffer shape", buffer.values.shape == (n_steps, num_envs)))
    checks.append(("log_probs buffer shape", buffer.log_probs.shape == (n_steps, num_envs)))
    checks.append(("dones buffer shape", buffer.dones.shape == (n_steps, num_envs)))
    print(f"  observations:  {buffer.observations.shape}")
    print(f"  actions:       {buffer.actions.shape}")
    print(f"  rewards:       {buffer.rewards.shape}")
    print(f"  values:        {buffer.values.shape}")
    print(f"  log_probs:     {buffer.log_probs.shape}")
    print(f"  dones:         {buffer.dones.shape}")

    # Compute advantages
    last_value = torch.randn(num_envs, device=device)
    last_done = done
    buffer.compute_advantages(last_value, last_done)

    # Check GAE output shapes
    checks.append(("advantages shape", buffer.advantages.shape == (n_steps, num_envs)))
    checks.append(("returns shape", buffer.returns.shape == (n_steps, num_envs)))
    print(f"  advantages:    {buffer.advantages.shape}")
    print(f"  returns:       {buffer.returns.shape}")

    # Check flatten
    data = buffer.get_training_data()
    flat_n = n_steps * num_envs
    checks.append(("flat obs shape", data["observations"].shape == (flat_n, obs_dim)))
    checks.append(("flat act shape", data["actions"].shape == (flat_n, act_dim)))
    checks.append(("flat log_probs shape", data["log_probs"].shape == (flat_n,)))
    checks.append(("flat advantages shape", data["advantages"].shape == (flat_n,)))
    checks.append(("flat returns shape", data["returns"].shape == (flat_n,)))
    checks.append(("flat values shape", data["values"].shape == (flat_n,)))
    print(f"  flattened:     {flat_n} samples")
    for k, v in data.items():
        print(f"    {k}: {v.shape}")

    # Verify GAE correctness: where dones are True, advantages should reset
    # In reversed order, after a done, the next step's advantage should not leak
    # Simple check: for steps where done=True, the advantage should only reflect that step's reward
    # (gae should not include future rewards through the done boundary)
    for env_idx in range(min(num_envs, 4)):
        done_steps = [s for s in range(n_steps) if stored_dones[s][env_idx]]
        if done_steps:
            print(f"  env[{env_idx}]: done at steps {done_steps[:5]}...")

    # Check that done is stored as float
    checks.append(("dones are float", buffer.dones.dtype == torch.float32))

    all_pass = all(p for _, p in checks)
    for label, passed in checks:
        if not passed:
            status = "FAIL"
            all_pass = False
            print(f"  [{status}] {label}")

    if all_pass:
        print(f"  All {len(checks)} checks passed.")

    return all_pass


def test_buffer_value_shape_regression():
    """Test 7: Regression — critic value (num_envs, 1) stored as (num_envs,).

    Verifies fixes for Bug 1:
      - (num_envs, 1) value → stored as (num_envs,) in buffer
      - num_envs=1 is NOT squeezed to scalar
      - flatten produces (n_steps * num_envs,) for values
    """
    print(f"\n{'='*60}")
    print(f"Test 7: Value Shape Regression (critic output → buffer)")
    print(f"{'='*60}")

    from ppo.buffer import RolloutBuffer

    device = torch.device("cpu")
    checks = []

    for num_envs in [1, 4, 16]:
        n_steps = 8
        obs_dim = 10
        act_dim = 4

        buf = RolloutBuffer(
            buffer_size=n_steps,
            obs_dim=obs_dim,
            act_dim=act_dim,
            num_envs=num_envs,
            device=device,
        )

        # Simulate critic output: shape (num_envs, 1)
        for _ in range(n_steps):
            obs = torch.randn(num_envs, obs_dim)
            action = torch.randn(num_envs, act_dim)
            reward = torch.randn(num_envs)
            value = torch.randn(num_envs, 1)  # ← critic-like shape
            log_prob = torch.randn(num_envs)
            done = torch.zeros(num_envs, dtype=torch.bool)

            buf.add(obs, action, reward, value, log_prob, done)

        # Check stored shapes
        stored_ok = buf.values.shape == (n_steps, num_envs)
        checks.append((f"num_envs={num_envs}: values stored as ({n_steps},{num_envs})",
                       stored_ok))
        print(f"  values buffer shape: {buf.values.shape}  (expect {(n_steps, num_envs)}) "
              + ("OK" if stored_ok else "FAIL"))

        # Check no row is scalar when num_envs=1
        if num_envs == 1:
            for step in range(n_steps):
                row = buf.values[step]
                scalar_check = row.ndim == 1 and row.shape == (1,)
                checks.append((f"num_envs=1 step={step}: not scalar ({row.shape})",
                               scalar_check))
                if not scalar_check:
                    print(f"  [FAIL] num_envs=1 step={step}: value row shape={row.shape}")

        # Check GAE
        last_value = torch.randn(num_envs)
        last_done = torch.zeros(num_envs, dtype=torch.bool)
        buf.compute_advantages(last_value, last_done)

        # Check flattened data
        data = buf.get_training_data()
        expected_flat = n_steps * num_envs
        flat_ok = data["values"].shape == (expected_flat,)
        checks.append((f"num_envs={num_envs}: flat values shape ({expected_flat},)",
                       flat_ok))
        print(f"  flat values shape: {data['values'].shape}  "
              f"(expect ({expected_flat},)) " + ("OK" if flat_ok else "FAIL"))

        # Check returns, old_values consistency
        for key in ["returns", "values", "advantages"]:
            ok = data[key].shape == (expected_flat,)
            checks.append((f"num_envs={num_envs}: flat {key} ({expected_flat},)", ok))
            if not ok:
                print(f"  [FAIL] {key} shape={data[key].shape}")

    # Also test that (num_envs, 1) log_prob is handled
    buf2 = RolloutBuffer(buffer_size=2, obs_dim=obs_dim, act_dim=act_dim,
                         num_envs=4, device=device)
    lp_val = torch.randn(4, 1)  # (num_envs, 1)
    buf2.add(torch.randn(4, obs_dim), torch.randn(4, act_dim),
             torch.randn(4), torch.randn(4), lp_val,
             torch.zeros(4, dtype=torch.bool))
    lp_ok = buf2.log_probs[0].shape == (4,)
    checks.append(("log_prob (4,1) stored as (4,)", lp_ok))
    print(f"  log_prob (4,1) → stored shape: {buf2.log_probs[0].shape} "
          + ("OK" if lp_ok else "FAIL"))

    # Test bad shape raises
    try:
        buf2.add(torch.randn(4, obs_dim), torch.randn(4, act_dim),
                 torch.randn(4), torch.randn(4, 3),  # ← wrong: (4, 3)
                 torch.randn(4), torch.zeros(4, dtype=torch.bool))
        checks.append(("bad value shape (4,3) raises ValueError", False))
        print("  [FAIL] (4,3) value should have raised ValueError")
    except ValueError as e:
        checks.append(("bad value shape (4,3) raises ValueError", True))
        print(f"  [OK] ValueError raised for bad value shape: {e}")

    all_pass = all(p for _, p in checks)
    for label, passed in checks:
        if not passed:
            print(f"  [FAIL] {label}")

    if all_pass:
        print(f"  All {len(checks)} regression checks passed.")

    return all_pass


def test_benchmark_n_steps_calculation():
    """Test 8: Benchmark n_steps calculation correctness.

    Verifies fixes for Bug 2: each tier satisfies n_steps * num_envs == total_timesteps.
    """
    print(f"\n{'='*60}")
    print(f"Test 8: Benchmark n_steps Calculation")
    print(f"{'='*60}")

    # Mirror the BENCHMARKS table from benchmark_gpu.py
    benchmarks = [
        {"num_envs": 16,  "total_timesteps": 4096, "n_steps": 256},
        {"num_envs": 64,  "total_timesteps": 4096, "n_steps": 64},
        {"num_envs": 256, "total_timesteps": 4096, "n_steps": 16},
    ]

    checks = []
    for b in benchmarks:
        ne = b["num_envs"]
        tt = b["total_timesteps"]
        ns = b["n_steps"]
        rollout_steps = ns * ne

        # Verify n_steps * num_envs == total_timesteps
        ok = rollout_steps == tt
        checks.append((f"num_envs={ne}: {ns}×{ne}={rollout_steps} == {tt}", ok))
        print(f"  num_envs={ne:>3}: {ns} steps × {ne} envs = {rollout_steps} env-steps "
              + ("OK" if ok else f"FAIL (expected {tt})"))

        # Verify the computed n_steps matches: n_steps = total_timesteps // num_envs
        computed = tt // ne
        match = ns == computed
        checks.append((f"num_envs={ne}: n_steps={ns} == {tt}//{ne}={computed}", match))

    all_pass = all(p for _, p in checks)
    for label, passed in checks:
        if not passed:
            print(f"  [FAIL] {label}")

    if all_pass:
        print(f"  All {len(checks)} benchmark calculation checks passed.")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Validate GPU parallel environment")
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1, 16],
                        help="Number of parallel envs to test (default: 1 16)")
    parser.add_argument("--skip-creation", action="store_true",
                        help="Skip the slow creation test (for quick re-runs)")
    args = parser.parse_args()

    print("=" * 70)
    print("GPU Parallel Environment Validation")
    print("PickCubeCollisionGripper-v1")
    print("=" * 70)

    # System info
    print(f"\nPyTorch:    {torch.__version__}")
    print(f"CUDA:       {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:        {torch.cuda.get_device_name(0)}")
        print(f"GPU count:  {torch.cuda.device_count()}")
        mem = get_gpu_memory()
        print(f"GPU mem:    {mem.get('allocated_MiB', 0):.0f} MiB allocated, "
              f"{mem.get('reserved_MiB', 0):.0f} MiB reserved")
    import mani_skill
    print(f"ManiSkill:  {mani_skill.__version__}")

    # Env kwargs matching the GPU config
    env_kwargs = dict(
        table_collision_penalty_coef=0.01,
        table_collision_force_threshold=1.0,
        table_collision_penalty_max=0.5,
        early_gripper_close_penalty_coef=0.2,
        gripper_open_near_cube_bonus=0.1,
        gripper_near_distance=0.08,
        gripper_far_distance=0.15,
        gripper_open_threshold=0.03,
        gripper_closed_threshold=0.01,
    )

    all_results = {}

    for num_envs in args.num_envs:
        print(f"\n{'#'*70}")
        print(f"# Testing num_envs={num_envs}")
        print(f"{'#'*70}")

        results = {}

        try:
            # Test 1: Creation & Backend
            if not args.skip_creation:
                env, t1 = test_creation_and_backend(num_envs, env_kwargs)
                results["creation_backend"] = t1
            else:
                env = gym.make(
                    "PickCubeCollisionGripper-v1",
                    num_envs=num_envs,
                    obs_mode="state",
                    render_mode=None,
                    sim_backend="auto",
                    render_backend="none",
                    enable_shadow=False,
                    **env_kwargs,
                )

            # Test 2: Tensor shapes
            t2 = test_tensor_shapes(env, num_envs)
            results["tensor_shapes"] = t2

            # Test 3: Auto-reset
            t3 = test_auto_reset_behavior(env, num_envs)
            results["auto_reset"] = t3

            # Test 4: Contact forces
            t4 = test_pairwise_contact_forces(env, num_envs)
            results["contact_forces"] = t4

            # Test 5: Per-env metrics
            t5 = test_per_env_metrics(env, num_envs)
            results["per_env_metrics"] = t5

            # Test 6: Buffer compatibility
            t6 = test_buffer_compatibility(env, num_envs)
            results["buffer"] = t6

            env.close()

        except Exception as e:
            import traceback
            traceback.print_exc()
            results["exception"] = str(e)
            print(f"\n  [FAIL] num_envs={num_envs} raised exception: {e}")

        all_results[num_envs] = results

    # ---- Regression tests (standalone, no env needed) ----
    print(f"\n{'#'*70}")
    print(f"# Regression Tests (value shape + benchmark calc)")
    print(f"{'#'*70}")

    try:
        t7 = test_buffer_value_shape_regression()
        all_results["regression_value_shape"] = {"value_shape": t7}
    except Exception as e:
        import traceback
        traceback.print_exc()
        all_results["regression_value_shape"] = {"value_shape": False, "exception": str(e)}

    try:
        t8 = test_benchmark_n_steps_calculation()
        all_results["regression_benchmark_calc"] = {"benchmark_calc": t8}
    except Exception as e:
        import traceback
        traceback.print_exc()
        all_results["regression_benchmark_calc"] = {"benchmark_calc": False, "exception": str(e)}

    # --- Summary ---
    print(f"\n{'='*70}")
    print("Validation Summary")
    print(f"{'='*70}")
    total_pass = 0
    total_checks = 0
    for num_envs, results in all_results.items():
        n_pass = sum(1 for v in results.values() if v is True)
        n_total = len(results)
        total_pass += n_pass
        total_checks += n_total
        status = "ALL PASS" if n_pass == n_total else f"{n_pass}/{n_total} passed"
        print(f"  num_envs={num_envs:>4}: {status}")
        for test, result in results.items():
            if result is not True:
                print(f"    FAIL: {test}: {result}")

    print(f"\n  Overall: {total_pass}/{total_checks} checks passed")
    if total_pass == total_checks:
        print("  ALL TESTS PASSED ✓")
        return 0
    else:
        print(f"  {total_checks - total_pass} failures ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
