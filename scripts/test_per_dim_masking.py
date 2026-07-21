#!/usr/bin/env python3
"""
Unit tests for per-dimension action masking in PPO.

Verifies that when action dimensions are masked (deterministically
overridden by the environment), those dimensions:
  - do NOT affect the masked log_prob
  - do NOT receive policy gradient
  - do NOT affect the PPO ratio

While arm dimensions continue to train normally.

Tests:
  1.  Mask active: changing gripper action does NOT change masked log_prob
  2.  Mask active: changing arm action DOES change masked log_prob
  3.  Mask inactive: changing gripper action DOES change log_prob
  4.  Mask active: gripper gradient is zero, arm gradient is non-zero
  5.  Mask inactive: both arm and gripper gradients are non-zero
  6.  Same params → PPO ratio = 1 (masked)
  7.  Batch with mixed masks (some envs masked, some not)
  8.  evaluate_actions_masked matches manual masking
  9.  Entropy only counts active dimensions
  10. Buffer stores and returns action_dim_mask correctly

Usage:
    python scripts/test_per_dim_masking.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ppo import ActorCritic, RolloutBuffer


def make_policy(obs_dim=10, act_dim=8, device=None):
    """Create a small test policy."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        policy_hidden_sizes=[32, 32],
        value_hidden_sizes=[32, 32],
        action_low=-np.ones(act_dim),
        action_high=np.ones(act_dim),
    ).to(device)


# ---------------------------------------------------------------------------
# Test 1: Mask active → gripper change does NOT affect masked log_prob
# ---------------------------------------------------------------------------

def test_mask_blocks_gripper_log_prob():
    print("=" * 60)
    print("Test 1: Mask blocks gripper from log_prob")
    print("=" * 60)

    act_dim = 8
    policy = make_policy(act_dim=act_dim)
    device = next(policy.parameters()).device

    obs = torch.randn(3, 10, device=device)
    action_mean = torch.zeros(3, act_dim, device=device)
    std = torch.ones(act_dim, device=device)

    # Set gripper mean to -0.5 (biased toward close) so that
    # log_prob(-1.0) != log_prob(+1.0).  Normal(0,1) is symmetric.
    action_mean[:, -1] = -0.5

    # All envs masked: gripper dim = 0
    mask = torch.ones(3, act_dim, device=device)
    mask[:, -1] = 0.0

    # Two actions differing only in gripper
    act1 = torch.zeros(3, act_dim, device=device)
    act1[:, -1] = -1.0  # close

    act2 = torch.zeros(3, act_dim, device=device)
    act2[:, -1] = +1.0  # open

    lp1 = policy.get_log_prob_per_dim(action_mean, std, act1)
    lp2 = policy.get_log_prob_per_dim(action_mean, std, act2)

    masked_lp1 = (lp1 * mask).sum(dim=-1)
    masked_lp2 = (lp2 * mask).sum(dim=-1)

    # Per-dim: gripper log_prob SHOULD differ (skewed mean)
    gripper_diff = (lp1[:, -1] - lp2[:, -1]).abs().max().item()
    print(f"  Per-dim gripper log_prob diff: {gripper_diff:.6f}")
    assert gripper_diff > 0.01, "Gripper per-dim log_prob should differ"

    # Masked total: should be IDENTICAL (mask blocks gripper)
    masked_diff = (masked_lp1 - masked_lp2).abs().max().item()
    print(f"  Masked total log_prob diff:    {masked_diff:.10f}")
    assert masked_diff < 1e-7, \
        f"Masked log_prob should be identical, got diff={masked_diff}"

    # Arm dimensions: should match (only gripper changed)
    arm_match = (lp1[:, :-1] - lp2[:, :-1]).abs().max().item()
    print(f"  Arm dims match:                {arm_match:.10f}")
    assert arm_match < 1e-7, "Arm log_probs should be identical"

    print("  ✅ PASSED\n")


# ---------------------------------------------------------------------------
# Test 2: Mask active → arm change DOES change masked log_prob
# ---------------------------------------------------------------------------

def test_mask_allows_arm_log_prob():
    print("=" * 60)
    print("Test 2: Arm changes still affect masked log_prob")
    print("=" * 60)

    act_dim = 8
    policy = make_policy(act_dim=act_dim)
    device = next(policy.parameters()).device

    obs = torch.randn(2, 10, device=device)
    action_mean = torch.zeros(2, act_dim, device=device)
    std = torch.ones(act_dim, device=device)
    action_mean[:, 0] = 0.5

    mask = torch.ones(2, act_dim, device=device)
    mask[:, -1] = 0.0  # gripper masked

    act1 = torch.zeros(2, act_dim, device=device)
    act1[:, 0] = +1.0

    act2 = torch.zeros(2, act_dim, device=device)
    act2[:, 0] = -1.0

    lp1 = policy.get_log_prob_per_dim(action_mean, std, act1)
    lp2 = policy.get_log_prob_per_dim(action_mean, std, act2)

    masked_lp1 = (lp1 * mask).sum(dim=-1)
    masked_lp2 = (lp2 * mask).sum(dim=-1)

    diff = (masked_lp1 - masked_lp2).abs().max().item()
    print(f"  Masked log_prob diff (arm changed): {diff:.6f}")
    assert diff > 0.01, "Arm change should affect masked log_prob"

    print("  ✅ PASSED\n")


# ---------------------------------------------------------------------------
# Test 3: Mask inactive → gripper change DOES change log_prob
# ---------------------------------------------------------------------------

def test_unmasked_gripper_affects_log_prob():
    print("=" * 60)
    print("Test 3: Unmasked gripper affects log_prob")
    print("=" * 60)

    act_dim = 8
    policy = make_policy(act_dim=act_dim)
    device = next(policy.parameters()).device

    obs = torch.randn(2, 10, device=device)
    action_mean = torch.zeros(2, act_dim, device=device)
    std = torch.ones(act_dim, device=device)
    action_mean[:, -1] = -0.5  # skewed mean

    # All-ones mask: nothing masked
    mask = torch.ones(2, act_dim, device=device)

    act1 = torch.zeros(2, act_dim, device=device)
    act1[:, -1] = -1.0
    act2 = torch.zeros(2, act_dim, device=device)
    act2[:, -1] = +1.0

    lp1 = policy.get_log_prob_per_dim(action_mean, std, act1)
    lp2 = policy.get_log_prob_per_dim(action_mean, std, act2)

    masked_lp1 = (lp1 * mask).sum(dim=-1)
    masked_lp2 = (lp2 * mask).sum(dim=-1)

    diff = (masked_lp1 - masked_lp2).abs().max().item()
    print(f"  Unmasked log_prob diff: {diff:.6f}")
    assert diff > 0.01, "Gripper change should affect unmasked log_prob"

    print("  ✅ PASSED\n")


# ---------------------------------------------------------------------------
# Test 4: Mask active → gripper gradient is zero
# ---------------------------------------------------------------------------

def test_mask_zeroes_gripper_gradient():
    print("=" * 60)
    print("Test 4: Mask zeroes gripper gradient")
    print("=" * 60)

    act_dim = 8
    policy = make_policy(act_dim=act_dim)
    device = next(policy.parameters()).device

    obs = torch.randn(2, 10, device=device)
    mask = torch.ones(2, act_dim, device=device)
    mask[:, -1] = 0.0  # gripper masked

    # Use evaluate_actions_masked and backprop through policy loss
    actions = torch.randn(2, act_dim, device=device)

    log_prob, entropy, value = policy.evaluate_actions_masked(obs, actions, mask)

    # Compute a simple loss and backprop
    loss = -log_prob.mean()  # maximize log_prob
    loss.backward()

    # Check gradients on actor_head (which produces action_mean)
    actor_grad = policy.actor_head.weight.grad  # (act_dim, hidden_dim)
    # Gripper row (last row) should be all zeros
    gripper_grad_norm = actor_grad[-1, :].norm().item()
    # Arm rows should have non-zero gradient
    arm_grad_norm = actor_grad[:-1, :].norm().item()

    print(f"  Actor head weight grad:")
    print(f"    Gripper row (last) norm:  {gripper_grad_norm:.10f}")
    print(f"    Arm rows norm:            {arm_grad_norm:.6f}")

    assert gripper_grad_norm < 1e-7, \
        f"Gripper gradient should be zero, got {gripper_grad_norm}"
    assert arm_grad_norm > 1e-7, \
        f"Arm gradient should be non-zero, got {arm_grad_norm}"

    print("  ✅ PASSED\n")


# ---------------------------------------------------------------------------
# Test 5: Mask inactive → gripper gradient is non-zero
# ---------------------------------------------------------------------------

def test_unmasked_gripper_has_gradient():
    print("=" * 60)
    print("Test 5: Unmasked gripper has non-zero gradient")
    print("=" * 60)

    act_dim = 8
    policy = make_policy(act_dim=act_dim)
    device = next(policy.parameters()).device

    obs = torch.randn(2, 10, device=device)
    mask = torch.ones(2, act_dim, device=device)  # nothing masked

    actions = torch.randn(2, act_dim, device=device)
    log_prob, entropy, value = policy.evaluate_actions_masked(obs, actions, mask)

    policy.zero_grad()
    loss = -log_prob.mean()
    loss.backward()

    actor_grad = policy.actor_head.weight.grad
    gripper_grad_norm = actor_grad[-1, :].norm().item()
    arm_grad_norm = actor_grad[:-1, :].norm().item()

    print(f"  Gripper row grad norm: {gripper_grad_norm:.6f}")
    print(f"  Arm rows grad norm:    {arm_grad_norm:.6f}")

    assert gripper_grad_norm > 1e-7, \
        f"Unmasked gripper gradient should be non-zero, got {gripper_grad_norm}"
    assert arm_grad_norm > 1e-7, \
        f"Arm gradient should be non-zero"

    print("  ✅ PASSED\n")


# ---------------------------------------------------------------------------
# Test 6: Same params → ratio=1 with masking
# ---------------------------------------------------------------------------

def test_ratio_is_one_same_params():
    print("=" * 60)
    print("Test 6: PPO ratio = 1 with same params (masked)")
    print("=" * 60)

    act_dim = 8
    policy = make_policy(act_dim=act_dim)
    device = next(policy.parameters()).device

    obs = torch.randn(4, 10, device=device)
    mask = torch.ones(4, act_dim, device=device)
    mask[0, -1] = 0.0  # env 0: gripper masked
    mask[1, -1] = 0.0  # env 1: gripper masked
    # env 2,3: all unmasked

    actions = torch.randn(4, act_dim, device=device)

    # Old log_prob
    with torch.no_grad():
        old_logp, _, _ = policy.evaluate_actions_masked(obs, actions, mask)

    # New log_prob (same params, no update)
    new_logp, _, _ = policy.evaluate_actions_masked(obs, actions, mask)

    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)

    print(f"  old_logp:      {old_logp.tolist()}")
    print(f"  new_logp:      {new_logp.tolist()}")
    print(f"  ratio:         {ratio.tolist()}")
    max_dev = (ratio - 1.0).abs().max().item()
    print(f"  max |ratio-1|: {max_dev:.10f}")

    assert max_dev < 1e-5, f"Ratio should be ~1.0, got deviation {max_dev}"

    print("  ✅ PASSED\n")


# ---------------------------------------------------------------------------
# Test 7: Batch with mixed masks
# ---------------------------------------------------------------------------

def test_mixed_mask_batch():
    print("=" * 60)
    print("Test 7: Mixed mask batch")
    print("=" * 60)

    act_dim = 8
    policy = make_policy(act_dim=act_dim)
    device = next(policy.parameters()).device

    obs = torch.randn(4, 10, device=device)
    action_mean = torch.randn(4, act_dim, device=device)
    std = torch.ones(act_dim, device=device) * 0.5

    # Mixed masks
    mask = torch.ones(4, act_dim, device=device)
    mask[0, -1] = 0.0  # env 0: gripper masked
    mask[3, -1] = 0.0  # env 3: gripper masked
    # env 1,2: fully controllable

    action = torch.randn(4, act_dim, device=device)

    lp_per_dim = policy.get_log_prob_per_dim(action_mean, std, action)
    masked_lp = (lp_per_dim * mask).sum(dim=-1)

    # Verify evaluate_actions_masked matches same forward pass
    with torch.no_grad():
        # Manually compute same as evaluate_actions_masked
        features = policy.features(obs)
        am_tanh = torch.tanh(policy.actor_head(features))
        am = policy._rescale_action(am_tanh)
        s = torch.exp(policy.log_std)
        d = torch.distributions.Normal(am, s)
        manual_lp = (d.log_prob(action) * mask).sum(dim=-1)

        eval_lp, eval_ent, _ = policy.evaluate_actions_masked(obs, action, mask)

    diff = (manual_lp - eval_lp).abs().max().item()
    print(f"  Manual vs eval_actions_masked diff: {diff:.10f}")
    assert diff < 1e-6, f"Mismatch: {diff}"

    # Check per-env
    for i in range(4):
        active_dims = int(mask[i].sum().item())
        is_gripper_masked = (mask[i, -1].item() == 0.0)
        print(f"  env {i}: active_dims={active_dims}, "
              f"gripper_masked={is_gripper_masked}, logp={eval_lp[i].item():.4f}")

    print("  ✅ PASSED\n")


# ---------------------------------------------------------------------------
# Test 8: evaluate_actions_masked matches manual computation
# ---------------------------------------------------------------------------

def test_evaluate_actions_masked_matches_manual():
    print("=" * 60)
    print("Test 8: evaluate_actions_masked == manual")
    print("=" * 60)

    act_dim = 8
    policy = make_policy(act_dim=act_dim)
    device = next(policy.parameters()).device

    obs = torch.randn(5, 10, device=device)
    mask = torch.ones(5, act_dim, device=device)
    mask[:, -1] = 0.0  # all gripper masked

    actions = torch.randn(5, act_dim, device=device)

    # Via masked API
    with torch.no_grad():
        lp_masked, ent_masked, val = policy.evaluate_actions_masked(
            obs, actions, mask)

    # Via manual: get per-dim, mask, sum
    features = policy.features(obs)
    action_mean_tanh = torch.tanh(policy.actor_head(features))
    action_mean = policy._rescale_action(action_mean_tanh)
    std = torch.exp(policy.log_std)
    dist = torch.distributions.Normal(action_mean, std)

    lp_manual = (dist.log_prob(actions) * mask).sum(dim=-1)
    ent_manual = (dist.entropy() * mask).sum(dim=-1)

    lp_diff = (lp_masked - lp_manual).abs().max().item()
    ent_diff = (ent_masked - ent_manual).abs().max().item()

    print(f"  log_prob max diff: {lp_diff:.10f}")
    print(f"  entropy max diff:  {ent_diff:.10f}")

    assert lp_diff < 1e-6, f"log_prob mismatch: {lp_diff}"
    assert ent_diff < 1e-6, f"entropy mismatch: {ent_diff}"

    print("  ✅ PASSED\n")


# ---------------------------------------------------------------------------
# Test 9: Entropy only counts active dimensions
# ---------------------------------------------------------------------------

def test_entropy_active_dims_only():
    print("=" * 60)
    print("Test 9: Entropy only counts active dims")
    print("=" * 60)

    act_dim = 8
    policy = make_policy(act_dim=act_dim)
    device = next(policy.parameters()).device

    obs = torch.randn(3, 10, device=device)

    # Full mask
    full_mask = torch.ones(3, act_dim, device=device)
    _, ent_full, _ = policy.evaluate_actions_masked(
        obs, torch.zeros(3, act_dim, device=device), full_mask)

    # Gripper masked
    partial_mask = torch.ones(3, act_dim, device=device)
    partial_mask[:, -1] = 0.0
    _, ent_partial, _ = policy.evaluate_actions_masked(
        obs, torch.zeros(3, act_dim, device=device), partial_mask)

    # With log_std_init=0, each dim entropy = 0.5*log(2πe) ≈ 1.419
    per_dim_expected = 0.5 * np.log(2 * np.pi * np.e)  # ≈ 1.419

    print(f"  Entropy full (8 dims):    {ent_full[0].item():.6f}")
    print(f"  Entropy partial (7 dims): {ent_partial[0].item():.6f}")
    print(f"  Ratio (7/8):              {ent_partial[0].item() / ent_full[0].item():.6f}")
    print(f"  Expected per-dim:         {per_dim_expected:.6f}")

    ratio = ent_partial[0].item() / ent_full[0].item()
    assert 0.85 < ratio < 0.90, \
        f"Entropy ratio should be ~7/8=0.875, got {ratio:.4f}"

    print("  ✅ PASSED\n")


# ---------------------------------------------------------------------------
# Test 10: Buffer stores and returns action_dim_mask correctly
# ---------------------------------------------------------------------------

def test_buffer_stores_mask():
    print("=" * 60)
    print("Test 10: Buffer stores and returns action_dim_mask")
    print("=" * 60)

    act_dim = 8
    obs_dim = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    buffer = RolloutBuffer(
        buffer_size=4, obs_dim=obs_dim, act_dim=act_dim,
        num_envs=2, device=device,
    )

    obs = torch.randn(2, obs_dim, device=device)
    action = torch.randn(2, act_dim, device=device)
    reward = torch.ones(2, device=device)
    value = torch.zeros(2, device=device)
    log_prob = torch.zeros(2, device=device)
    done = torch.zeros(2, device=device)

    # Custom mask: env 0 gripper masked, env 1 fully active
    mask = torch.ones(2, act_dim, device=device)
    mask[0, -1] = 0.0

    buffer.add(obs, action, reward, value, log_prob, done,
               action_dim_mask=mask)

    data = buffer.get_training_data()
    stored_mask = data["action_dim_masks"]

    print(f"  Original mask[0]: {mask[0].tolist()}")
    print(f"  Stored mask[0]:   {stored_mask[0].tolist()}")
    print(f"  Original mask[1]: {mask[1].tolist()}")
    print(f"  Stored mask[1]:   {stored_mask[1].tolist()}")

    assert torch.allclose(stored_mask, mask), "Mask not stored correctly"

    # Default mask (when not provided)
    buffer.reset()
    buffer.add(obs, action, reward, value, log_prob, done)
    data = buffer.get_training_data()
    default_mask = data["action_dim_masks"]
    expected = torch.ones(2, act_dim, device=device)
    print(f"  Default mask shape: {default_mask.shape}")
    print(f"  Expected shape:     {expected.shape}")
    print(f"  Default mask[0]:    {default_mask[0].tolist()}")
    print(f"  Expected[0]:        {expected[0].tolist()}")
    print(f"  All close:          {torch.allclose(default_mask, expected)}")
    assert torch.allclose(default_mask, expected), \
        f"Default mask should be all-ones, got {default_mask.tolist()}"

    print("  ✅ PASSED\n")


if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  Per-Dimension Action Masking — Unit Tests          ║")
    print("╚" + "═" * 58 + "╝")
    print()

    all_passed = True
    tests = [
        ("Mask blocks gripper log_prob", test_mask_blocks_gripper_log_prob),
        ("Arm changes affect masked log_prob", test_mask_allows_arm_log_prob),
        ("Unmasked gripper affects log_prob", test_unmasked_gripper_affects_log_prob),
        ("Mask zeroes gripper gradient", test_mask_zeroes_gripper_gradient),
        ("Unmasked gripper has gradient", test_unmasked_gripper_has_gradient),
        ("PPO ratio = 1 same params", test_ratio_is_one_same_params),
        ("Mixed mask batch", test_mixed_mask_batch),
        ("evaluate_actions_masked == manual", test_evaluate_actions_masked_matches_manual),
        ("Entropy active dims only", test_entropy_active_dims_only),
        ("Buffer stores mask", test_buffer_stores_mask),
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
