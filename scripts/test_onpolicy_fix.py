#!/usr/bin/env python3
"""
Unit tests for the on-policy action consistency fix.

Tests:
  1. Policy get_action_dist returns correct shapes
  2. get_log_prob matches evaluate_actions log_prob
  3. Executed action reconstruction: action_exec matches env's executed action
  4. Log_prob recomputation for executed action is correct
  5. Buffer stores (action_exec, log_prob_exec) pair
  6. Sanity: log_prob changes when gripper action changes
  7. PPO ratio stability with forced +1.0 action
  8. log_prob at action boundary (+1.0) is well-defined

Usage:
    python scripts/test_onpolicy_fix.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import envs  # noqa: F401
from ppo import ActorCritic, RolloutBuffer


def make_env(num_envs=1, env_kwargs=None, **kwargs):
    make_kwargs = dict(
        num_envs=num_envs,
        obs_mode="state",
        render_mode=None,
        render_backend="none",
        sim_backend="auto",
        enable_shadow=False,
    )
    make_kwargs.update(kwargs)
    if env_kwargs:
        make_kwargs.update(env_kwargs)
    return gym.make("PickCubeGripperCurriculum-v1", **make_kwargs)


def make_policy(obs_dim, act_dim, action_low, action_high, device=None):
    """Create policy on the specified device."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        policy_hidden_sizes=[64, 64],
        value_hidden_sizes=[64, 64],
        action_low=action_low,
        action_high=action_high,
    ).to(device)


def get_device(model):
    return next(model.parameters()).device


def test_get_action_dist_shapes():
    """Test 1: get_action_dist returns correct shapes."""
    print("=" * 60)
    print("Test 1: get_action_dist shapes")
    print("=" * 60)

    env = make_env(num_envs=4)
    obs, _ = env.reset(seed=42)
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    action_low = env.action_space.low[0] if env.action_space.low.ndim > 1 else env.action_space.low
    action_high = env.action_space.high[0] if env.action_space.high.ndim > 1 else env.action_space.high

    policy = make_policy(obs_dim, act_dim, action_low, action_high, device=obs.device)

    with torch.no_grad():
        action, log_prob, value, action_mean, std = policy.get_action_dist(obs)

    assert action.shape == (4, act_dim), f"action shape {action.shape}"
    assert log_prob.shape == (4,), f"log_prob shape {log_prob.shape}"
    assert value.shape == (4, 1), f"value shape {value.shape}"
    assert action_mean.shape == (4, act_dim), f"action_mean shape {action_mean.shape}"
    assert std.shape == (act_dim,), f"std shape {std.shape}"

    print(f"  action:     {action.shape}")
    print(f"  log_prob:   {log_prob.shape}")
    print(f"  value:      {value.shape}")
    print(f"  action_mean:{action_mean.shape}")
    print(f"  std:        {std.shape}")
    env.close()
    print("  ✅ PASSED\n")


def test_get_log_prob_matches_evaluate():
    """Test 2: get_log_prob returns same value as evaluate_actions log_prob."""
    print("=" * 60)
    print("Test 2: get_log_prob == evaluate_actions log_prob")
    print("=" * 60)

    env = make_env(num_envs=4)
    obs, _ = env.reset(seed=42)
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    action_low = env.action_space.low[0] if env.action_space.low.ndim > 1 else env.action_space.low
    action_high = env.action_space.high[0] if env.action_space.high.ndim > 1 else env.action_space.high

    policy = make_policy(obs_dim, act_dim, action_low, action_high, device=obs.device)

    with torch.no_grad():
        action, log_prob, value, action_mean, std = policy.get_action_dist(obs)

        # get_log_prob should match evaluate_actions
        log_prob_via_get = policy.get_log_prob(action_mean, std, action)
        log_prob_via_eval, _, _ = policy.evaluate_actions(obs, action)

    diff = (log_prob_via_get - log_prob_via_eval).abs().max().item()
    print(f"  log_prob via get_log_prob:     {log_prob_via_get[:2].tolist()}")
    print(f"  log_prob via evaluate_actions: {log_prob_via_eval[:2].tolist()}")
    print(f"  max absolute difference:       {diff:.10f}")

    assert diff < 1e-6, f"log_prob mismatch: max diff = {diff}"
    env.close()
    print("  ✅ PASSED\n")


def test_executed_action_reconstruction():
    """Test 3: action_exec matches env's executed_gripper_action."""
    print("=" * 60)
    print("Test 3: Executed action reconstruction")
    print("=" * 60)

    env = make_env(
        num_envs=4,
        env_kwargs={
            "force_gripper_open_enabled": True,
            "force_gripper_open_until_distance": 0.0,  # always far
            "table_collision_penalty_coef": 0.0,
            "early_gripper_close_penalty_coef": 0.0,
            "gripper_open_near_cube_bonus": 0.0,
        }
    )

    obs, _ = env.reset(seed=42)
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    action_low = env.action_space.low[0] if env.action_space.low.ndim > 1 else env.action_space.low
    action_high = env.action_space.high[0] if env.action_space.high.ndim > 1 else env.action_space.high

    policy = make_policy(obs_dim, act_dim, action_low, action_high, device=obs.device)

    with torch.no_grad():
        action_raw, _, _, action_mean, std = policy.get_action_dist(obs)

    # Send raw action to env (env will apply mask)
    next_obs, reward, term, trunc, info = env.step(action_raw)

    # Get the executed gripper action from the env
    mask_info = env.unwrapped.get_gripper_mask_info()
    executed_gripper = mask_info[4]  # executed_gripper_action

    # Reconstruct executed action
    action_exec = action_raw.clone()
    action_exec[..., -1] = executed_gripper

    for i in range(4):
        print(f"  env {i}: raw_gripper={action_raw[i, -1].item():.4f}, "
              f"exec_gripper={executed_gripper[i].item():.4f}, "
              f"reconstructed={action_exec[i, -1].item():.4f}")

    assert torch.allclose(action_exec[..., -1], executed_gripper), \
        "Reconstructed gripper action doesn't match executed"
    for i in range(4):
        assert action_exec[i, -1].item() > 0.9, \
            f"Env {i}: expected executed gripper ~+1.0, got {action_exec[i, -1].item()}"

    env.close()
    print("  ✅ PASSED\n")


def test_log_prob_recomputation_correct():
    """Test 4: Recomputing log_prob for executed action gives correct value."""
    print("=" * 60)
    print("Test 4: log_prob recomputation for executed action")
    print("=" * 60)

    # Use num_envs=2 to ensure GPU sim and tensor obs
    env = make_env(
        num_envs=2,
        env_kwargs={
            "force_gripper_open_enabled": True,
            "force_gripper_open_until_distance": 0.0,
            "table_collision_penalty_coef": 0.0,
            "early_gripper_close_penalty_coef": 0.0,
            "gripper_open_near_cube_bonus": 0.0,
        }
    )

    obs, _ = env.reset(seed=42)
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    action_low = env.action_space.low[0] if env.action_space.low.ndim > 1 else env.action_space.low
    action_high = env.action_space.high[0] if env.action_space.high.ndim > 1 else env.action_space.high

    policy = make_policy(obs_dim, act_dim, action_low, action_high, device=obs.device)

    with torch.no_grad():
        action_raw, log_prob_raw, value, action_mean, std = policy.get_action_dist(obs)

        # Manually set gripper to +1.0 (simulating the mask)
        action_exec = action_raw.clone()
        action_exec[..., -1] = 1.0

        # Recompute log_prob via get_log_prob (efficient)
        log_prob_exec_via_get = policy.get_log_prob(action_mean, std, action_exec)

        # Recompute log_prob via evaluate_actions (reference)
        log_prob_exec_via_eval, _, _ = policy.evaluate_actions(obs, action_exec)

    diff = (log_prob_exec_via_get - log_prob_exec_via_eval).abs().max().item()

    print(f"  action_raw[-1] (mean):    {action_raw[:, -1].mean().item():.6f}")
    print(f"  action_exec[-1] (mean):   {action_exec[:, -1].mean().item():.6f}")
    print(f"  log_prob_raw (mean):      {log_prob_raw.mean().item():.6f}")
    print(f"  log_prob_exec (get):      {log_prob_exec_via_get.mean().item():.6f}")
    print(f"  log_prob_exec (eval):     {log_prob_exec_via_eval.mean().item():.6f}")
    print(f"  log_prob max diff:        {diff:.10f}")

    assert diff < 1e-6, f"log_prob mismatch: {diff}"
    assert (log_prob_exec_via_get - log_prob_raw).abs().max().item() > 1e-8, \
        "log_prob should change when gripper action changes"

    env.close()
    print("  ✅ PASSED\n")


def test_buffer_stores_executed_pair():
    """Test 5: Buffer stores correct (action_exec, log_prob_exec) pair."""
    print("=" * 60)
    print("Test 5: Buffer stores executed action/log_prob pair")
    print("=" * 60)

    env = make_env(
        num_envs=2,
        env_kwargs={
            "force_gripper_open_enabled": True,
            "force_gripper_open_until_distance": 0.0,
            "table_collision_penalty_coef": 0.0,
            "early_gripper_close_penalty_coef": 0.0,
            "gripper_open_near_cube_bonus": 0.0,
        }
    )

    obs, _ = env.reset(seed=42)
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    action_low = env.action_space.low[0] if env.action_space.low.ndim > 1 else env.action_space.low
    action_high = env.action_space.high[0] if env.action_space.high.ndim > 1 else env.action_space.high

    policy = make_policy(obs_dim, act_dim, action_low, action_high, device=obs.device)
    device = get_device(policy)

    buffer = RolloutBuffer(
        buffer_size=4,
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_envs=2,
        device=device,
        gamma=0.99,
        gae_lambda=0.95,
    )

    with torch.no_grad():
        action_raw, log_prob_raw, value, action_mean, std = policy.get_action_dist(obs)

    next_obs, reward, term, trunc, info = env.step(action_raw)

    # Reconstruct executed action
    mask_info = env.unwrapped.get_gripper_mask_info()
    executed_gripper = mask_info[4]
    action_exec = action_raw.clone()
    action_exec[..., -1] = executed_gripper
    log_prob_exec = policy.get_log_prob(action_mean, std, action_exec)

    # Store in buffer
    if value.ndim == 2 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    buffer.add(
        obs=obs,
        action=action_exec,
        reward=reward,
        value=value,
        log_prob=log_prob_exec,
        done=term | trunc,
    )

    # Verify buffer contents
    data = buffer.get_training_data()
    stored_action = data["actions"]
    stored_log_prob = data["log_probs"]

    assert torch.allclose(stored_action, action_exec), \
        "Buffer action != executed action"
    assert torch.allclose(stored_log_prob, log_prob_exec), \
        "Buffer log_prob != computed log_prob"

    # Verify log_prob corresponds to stored action
    with torch.no_grad():
        verify_logp, _, _ = policy.evaluate_actions(obs, stored_action)
    diff = (stored_log_prob - verify_logp).abs().max().item()
    assert diff < 1e-6, f"Buffer log_prob doesn't match evaluate_actions: diff={diff}"

    mismatch = (stored_log_prob - log_prob_raw).abs().max().item()
    print(f"  stored_action[:, -1]:     {stored_action[:, -1].tolist()}")
    print(f"  action_raw[:, -1]:        {action_raw[:, -1].tolist()}")
    print(f"  stored_log_prob:          {stored_log_prob.tolist()}")
    print(f"  log_prob_raw:             {log_prob_raw.tolist()}")
    print(f"  |log_prob_diff| max:      {mismatch:.6f}")
    print(f"  verify log_prob match:    diff={diff:.10f}")

    env.close()
    print("  ✅ PASSED\n")


def test_log_prob_changes_with_gripper():
    """Test 6: log_prob genuinely changes when only gripper dimension changes."""
    print("=" * 60)
    print("Test 6: log_prob sensitivity to gripper dim change")
    print("=" * 60)

    env = make_env(num_envs=2)
    obs, _ = env.reset(seed=42)
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    action_low = env.action_space.low[0] if env.action_space.low.ndim > 1 else env.action_space.low
    action_high = env.action_space.high[0] if env.action_space.high.ndim > 1 else env.action_space.high

    policy = make_policy(obs_dim, act_dim, action_low, action_high, device=obs.device)

    with torch.no_grad():
        action_raw, log_prob_raw, value, action_mean, std = policy.get_action_dist(obs)

        gripper_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        log_probs = []
        for gv in gripper_values:
            action_mod = action_raw.clone()
            action_mod[..., -1] = gv
            lp = policy.get_log_prob(action_mean, std, action_mod)
            log_probs.append(lp.mean().item())

    print(f"  log_prob at gripper=-1.0: {log_probs[0]:.6f}")
    print(f"  log_prob at gripper=-0.5: {log_probs[1]:.6f}")
    print(f"  log_prob at gripper= 0.0: {log_probs[2]:.6f}")
    print(f"  log_prob at gripper=+0.5: {log_probs[3]:.6f}")
    print(f"  log_prob at gripper=+1.0: {log_probs[4]:.6f}")

    for i, lp in enumerate(log_probs):
        assert np.isfinite(lp), f"log_prob at gripper={gripper_values[i]} is not finite: {lp}"

    diffs = [abs(log_probs[i] - log_probs[i+1]) for i in range(len(log_probs)-1)]
    max_diff = max(diffs)
    print(f"  max pairwise diff:        {max_diff:.6f}")
    assert max_diff > 1e-10, "log_prob not responding to gripper changes"
    assert not any(np.isnan(lp) for lp in log_probs), "NaN in log_probs"
    assert not any(np.isinf(lp) for lp in log_probs), "Inf in log_probs"

    env.close()
    print("  ✅ PASSED\n")


def test_ppo_ratio_stability():
    """Test 7: PPO ratio stays numerically stable with forced +1.0 action."""
    print("=" * 60)
    print("Test 7: PPO ratio stability with forced gripper=+1.0")
    print("=" * 60)

    env = make_env(num_envs=2)
    obs, _ = env.reset(seed=42)
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    action_low = env.action_space.low[0] if env.action_space.low.ndim > 1 else env.action_space.low
    action_high = env.action_space.high[0] if env.action_space.high.ndim > 1 else env.action_space.high

    policy = make_policy(obs_dim, act_dim, action_low, action_high, device=obs.device)

    with torch.no_grad():
        _, _, _, action_mean, std = policy.get_action_dist(obs)
        action_exec = torch.randn(2, act_dim, device=obs.device)
        action_exec[..., :-1] = action_mean[..., :-1]
        action_exec[..., -1] = 1.0  # forced gripper open

        old_logp = policy.get_log_prob(action_mean, std, action_exec)

    new_logp, _, _ = policy.evaluate_actions(obs, action_exec)

    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)

    print(f"  old_logp (mean):          {old_logp.mean().item():.6f}")
    print(f"  new_logp (mean):          {new_logp.mean().item():.6f}")
    print(f"  log_ratio (mean):         {log_ratio.mean().item():.6f}")
    print(f"  ratio (mean):             {ratio.mean().item():.6f}")

    assert (ratio - 1.0).abs().max().item() < 1e-5, \
        f"Ratio should be ~1.0 before update, got {ratio.tolist()}"

    for name, val in [("old_logp", old_logp), ("new_logp", new_logp),
                       ("log_ratio", log_ratio), ("ratio", ratio)]:
        assert torch.isfinite(val).all(), f"{name} contains NaN/Inf"

    env.close()
    print("  ✅ PASSED\n")


def test_boundary_log_prob():
    """Test 8: log_prob at action boundary (+1.0) is well-defined."""
    print("=" * 60)
    print("Test 8: log_prob at action boundary (+1.0)")
    print("=" * 60)

    env = make_env(num_envs=2)
    obs, _ = env.reset(seed=42)
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    action_low = env.action_space.low[0] if env.action_space.low.ndim > 1 else env.action_space.low
    action_high = env.action_space.high[0] if env.action_space.high.ndim > 1 else env.action_space.high

    policy = make_policy(obs_dim, act_dim, action_low, action_high, device=obs.device)

    action_at_boundary = torch.zeros(2, act_dim, device=obs.device)
    action_at_boundary[:, -1] = 1.0  # at the boundary

    with torch.no_grad():
        logp_boundary, _, _ = policy.evaluate_actions(obs, action_at_boundary)

    print(f"  log_prob(gripper=+1.0):   {logp_boundary.tolist()}")
    print(f"  finite:                   {torch.isfinite(logp_boundary).all().item()}")

    assert torch.isfinite(logp_boundary).all(), \
        f"log_prob at boundary is not finite: {logp_boundary.tolist()}"
    assert logp_boundary.max().item() < 100, \
        f"log_prob at boundary suspiciously large"
    assert logp_boundary.min().item() > -1000, \
        f"log_prob at boundary extremely negative"

    env.close()
    print("  ✅ PASSED\n")


if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║    On-Policy Action Consistency Fix — Unit Tests      ║")
    print("╚" + "═" * 58 + "╝")
    print()

    all_passed = True

    tests = [
        ("get_action_dist shapes", test_get_action_dist_shapes),
        ("get_log_prob == evaluate_actions", test_get_log_prob_matches_evaluate),
        ("Executed action reconstruction", test_executed_action_reconstruction),
        ("log_prob recomputation correct", test_log_prob_recomputation_correct),
        ("Buffer stores executed pair", test_buffer_stores_executed_pair),
        ("log_prob changes with gripper", test_log_prob_changes_with_gripper),
        ("PPO ratio stability", test_ppo_ratio_stability),
        ("Boundary log_prob", test_boundary_log_prob),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"  ❌ FAILED [{name}]: {e}\n")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("  🎉 ALL TESTS PASSED")
    else:
        print("  ❌ SOME TESTS FAILED")
    print("=" * 60)
