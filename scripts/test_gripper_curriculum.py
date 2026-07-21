#!/usr/bin/env python3
"""
Unit tests for PickCubeGripperCurriculum-v1 gripper action masking.

Tests:
  1. Far + policy close: mask_active=True, action_overridden=True
  2. Near + policy close: mask_active=False, action_overridden=False
  3. Far + policy open:  mask_active=True, action_overridden=False
     (mask is eligible but policy already wants open — not overridden)
  4. Overridden <= close invariant
  5. is_grasped gate in mask logic
  6. Continuous 200-step run with all metrics tracked
  7. Batch tensor with mixed actions
  8. Masking disabled

Usage:
    python scripts/test_gripper_curriculum.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import mani_skill.envs  # noqa: F401 — register ManiSkill envs
import envs  # noqa: F401 — register custom project envs


def make_env(env_kwargs=None, **kwargs):
    """Create a headless PickCubeGripperCurriculum-v1 environment.

    ``env_kwargs`` is flattened into the top-level ``gym.make()`` kwargs,
    matching the pattern used by ``train_ppo.py``.
    """
    make_kwargs = dict(
        num_envs=1,
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


def test_far_policy_close():
    """
    Test 1: When TCP is far AND policy requests close:
      - mask_active = True
      - action_overridden = True
      - executed action[-1] = +1.0
      - policy_requested_close = True
    """
    print("=" * 60)
    print("Test 1: Far + policy close → mask active & overridden")
    print("=" * 60)

    env = make_env(
        env_kwargs={
            "force_gripper_open_enabled": True,
            "force_gripper_open_until_distance": 0.0,  # always "far"
            "table_collision_penalty_coef": 0.0,
            "early_gripper_close_penalty_coef": 0.0,
            "gripper_open_near_cube_bonus": 0.0,
        }
    )

    env.reset(seed=42)

    action = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)
    obs, reward, term, trunc, info = env.step(action)

    (mask_active, action_overridden, policy_requested_close,
     policy_gripper, executed_gripper, tcp_dist, near_cube
     ) = env.unwrapped.get_gripper_mask_info()

    print(f"  TCP-cube distance:     {tcp_dist.item():.4f}")
    print(f"  mask_active:           {mask_active.item()}")
    print(f"  action_overridden:     {action_overridden.item()}")
    print(f"  policy_requested_close:{policy_requested_close.item()}")
    print(f"  policy_gripper:        {policy_gripper.item():.4f}")
    print(f"  executed_gripper:      {executed_gripper.item():.4f}")
    print(f"  near_cube:             {near_cube.item()}")

    assert mask_active.item() is True
    assert action_overridden.item() is True
    assert policy_requested_close.item() is True
    assert executed_gripper.item() > 0.9
    assert near_cube.item() is False  # distance=0 threshold → never near

    env.close()
    print("  ✅ PASSED\n")


def test_near_policy_close():
    """
    Test 2: When TCP is near AND policy requests close:
      - mask_active = False
      - action_overridden = False
      - executed == policy (no override)
    """
    print("=" * 60)
    print("Test 2: Near + policy close → mask inactive, not overridden")
    print("=" * 60)

    env = make_env(
        env_kwargs={
            "force_gripper_open_enabled": True,
            "force_gripper_open_until_distance": 100.0,  # always "near"
            "table_collision_penalty_coef": 0.0,
            "early_gripper_close_penalty_coef": 0.0,
            "gripper_open_near_cube_bonus": 0.0,
        }
    )

    env.reset(seed=42)

    action = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)
    obs, reward, term, trunc, info = env.step(action)

    (mask_active, action_overridden, policy_requested_close,
     policy_gripper, executed_gripper, tcp_dist, near_cube
     ) = env.unwrapped.get_gripper_mask_info()

    print(f"  TCP-cube distance:     {tcp_dist.item():.4f}")
    print(f"  mask_active:           {mask_active.item()}")
    print(f"  action_overridden:     {action_overridden.item()}")
    print(f"  policy_requested_close:{policy_requested_close.item()}")
    print(f"  near_cube:             {near_cube.item()}")

    assert mask_active.item() is False
    assert action_overridden.item() is False
    assert policy_requested_close.item() is True  # policy DID request close
    assert near_cube.item() is True

    # When mask is inactive, executed == policy
    assert abs(policy_gripper.item() - executed_gripper.item()) < 0.01

    env.close()
    print("  ✅ PASSED\n")


def test_far_policy_open():
    """
    Test 3: When TCP is far but policy ALREADY requests open:
      - mask_active = True  (rule IS eligible to fire)
      - action_overridden = False  (policy intent was NOT changed)
      - executed == policy (both ~+1.0)

    This is the critical distinction: mask_active != action_overridden.
    """
    print("=" * 60)
    print("Test 3: Far + policy open → mask active but NOT overridden")
    print("=" * 60)

    env = make_env(
        env_kwargs={
            "force_gripper_open_enabled": True,
            "force_gripper_open_until_distance": 0.0,  # always "far"
            "table_collision_penalty_coef": 0.0,
            "early_gripper_close_penalty_coef": 0.0,
            "gripper_open_near_cube_bonus": 0.0,
        }
    )

    env.reset(seed=42)

    action = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, +1.0]], dtype=np.float32)
    obs, reward, term, trunc, info = env.step(action)

    (mask_active, action_overridden, policy_requested_close,
     policy_gripper, executed_gripper, tcp_dist, near_cube
     ) = env.unwrapped.get_gripper_mask_info()

    print(f"  TCP-cube distance:     {tcp_dist.item():.4f}")
    print(f"  mask_active:           {mask_active.item()}")
    print(f"  action_overridden:     {action_overridden.item()}")
    print(f"  policy_requested_close:{policy_requested_close.item()}")
    print(f"  policy_gripper:        {policy_gripper.item():.4f}")
    print(f"  executed_gripper:      {executed_gripper.item():.4f}")

    # Key assertions: mask IS active, but action NOT overridden
    assert mask_active.item() is True, (
        "Mask should be active (far + not grasped)"
    )
    assert action_overridden.item() is False, (
        "Action should NOT be overridden — policy already requested open"
    )
    assert policy_requested_close.item() is False, (
        "Policy requested OPEN, so policy_requested_close must be False"
    )

    env.close()
    print("  ✅ PASSED\n")


def test_overridden_le_close_invariant():
    """
    Test 4: Run 100 steps with random actions and verify:
      action_overridden_rate <= policy_requested_close_rate
    at every step (the overridden set is a subset of close-request set).
    """
    print("=" * 60)
    print("Test 4: Invariant — overridden_rate <= close_rate")
    print("=" * 60)

    env = make_env(
        env_kwargs={
            "force_gripper_open_enabled": True,
            "force_gripper_open_until_distance": 0.10,
            "table_collision_penalty_coef": 0.0,
            "early_gripper_close_penalty_coef": 0.0,
            "gripper_open_near_cube_bonus": 0.0,
        }
    )

    obs, info = env.reset(seed=42)

    cum_overridden = 0
    cum_close = 0
    cum_mask_active = 0
    total = 0

    for step_i in range(100):
        action = np.random.uniform(-1.0, 1.0, size=(1, 8)).astype(np.float32)
        obs, reward, term, trunc, info = env.step(action)

        (mask_active, action_overridden, policy_requested_close,
         _, _, _, _) = env.unwrapped.get_gripper_mask_info()

        cum_mask_active += int(mask_active.sum().item())
        cum_overridden += int(action_overridden.sum().item())
        cum_close += int(policy_requested_close.sum().item())
        total += 1

        # Per-step invariant: overridden ⇒ close
        # (if overridden is True, close must also be True)
        if action_overridden.item():
            assert policy_requested_close.item(), (
                f"Step {step_i}: overridden=True but close=False — invariant broken"
            )
        # Per-step invariant: overridden ⇒ mask_active
        if action_overridden.item():
            assert mask_active.item(), (
                f"Step {step_i}: overridden=True but mask_active=False — invariant broken"
            )

        if term or trunc:
            done = term.item() if hasattr(term, 'item') else term
            if done:
                obs, info = env.reset()

    overridden_rate = cum_overridden / total
    close_rate = cum_close / total
    mask_rate = cum_mask_active / total

    print(f"  Total steps:           {total}")
    print(f"  mask_active_rate:      {mask_rate:.4f}")
    print(f"  overridden_rate:       {overridden_rate:.4f}")
    print(f"  close_rate:            {close_rate:.4f}")
    print(f"  overridden <= close:   {overridden_rate <= close_rate + 1e-9}")
    print(f"  overridden <= mask:    {overridden_rate <= mask_rate + 1e-9}")

    assert overridden_rate <= close_rate + 1e-9, (
        f"overridden_rate ({overridden_rate:.4f}) > close_rate ({close_rate:.4f})"
    )
    assert overridden_rate <= mask_rate + 1e-9, (
        f"overridden_rate ({overridden_rate:.4f}) > mask_active_rate ({mask_rate:.4f})"
    )

    env.close()
    print("  ✅ PASSED\n")


def test_grasped_gate():
    """
    Test 5: is_grasped correctly gates the mask.
    """
    print("=" * 60)
    print("Test 5: is_grasped gate in mask logic")
    print("=" * 60)

    env = make_env(
        env_kwargs={
            "force_gripper_open_enabled": True,
            "force_gripper_open_until_distance": 0.0,
            "table_collision_penalty_coef": 0.0,
            "early_gripper_close_penalty_coef": 0.0,
            "gripper_open_near_cube_bonus": 0.0,
        }
    )

    env.reset(seed=42)

    is_grasped = env.unwrapped.agent.is_grasping(env.unwrapped.cube)
    print(f"  Initial is_grasped: {is_grasped.item()}")
    assert not is_grasped.item(), "Expected not grasped at start"

    action = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)
    obs, reward, term, trunc, info = env.step(action)

    (mask_active, action_overridden, _, _, _, _, _) = \
        env.unwrapped.get_gripper_mask_info()

    # With threshold=0.0 and is_grasped=False:
    # mask_active = (dist > 0.0) & (~False) = True
    assert mask_active.item() is True
    assert action_overridden.item() is True

    env.close()
    print("  ✅ PASSED\n")


def test_continuous_200_steps():
    """
    Test 6: Continuous 200-step run, tracking all refined metrics.
    """
    print("=" * 60)
    print("Test 6: Continuous 200-step run with all metrics")
    print("=" * 60)

    env = make_env(
        env_kwargs={
            "force_gripper_open_enabled": True,
            "force_gripper_open_until_distance": 0.10,
            "table_collision_penalty_coef": 0.01,
            "early_gripper_close_penalty_coef": 0.05,
            "gripper_open_near_cube_bonus": 0.05,
        }
    )

    obs, info = env.reset(seed=42)

    mask_active_count = 0
    overridden_count = 0
    close_count = 0
    near_count = 0
    dist_sum = 0.0
    total_steps = 200

    for step_i in range(total_steps):
        action = np.random.uniform(-1.0, 1.0, size=(1, 8)).astype(np.float32)
        obs, reward, term, trunc, info = env.step(action)

        (mask_active, action_overridden, policy_requested_close,
         _, _, tcp_dist, near_cube) = env.unwrapped.get_gripper_mask_info()

        if mask_active is not None:
            mask_active_count += int(mask_active.sum().item())
        if action_overridden is not None:
            overridden_count += int(action_overridden.sum().item())
            # Invariant check
            if action_overridden.item():
                assert mask_active.item(), (
                    f"Step {step_i}: overridden but mask not active"
                )
                assert policy_requested_close.item(), (
                    f"Step {step_i}: overridden but policy didn't request close"
                )
        if policy_requested_close is not None:
            close_count += int(policy_requested_close.sum().item())
        if tcp_dist is not None:
            dist_sum += float(tcp_dist.sum().item())
        if near_cube is not None:
            near_count += int(near_cube.sum().item())

        # NaN check
        if isinstance(obs, torch.Tensor):
            assert not torch.isnan(obs).any(), f"NaN in obs at step {step_i}"

        if term or trunc:
            done = term.item() if hasattr(term, 'item') else term
            if done:
                obs, info = env.reset()

        if (step_i + 1) % 50 == 0:
            print(f"  ... {step_i + 1}/200 ok "
                  f"(mask={mask_active_count}, overridden={overridden_count}, "
                  f"close={close_count}, near={near_count})")

    print(f"\n  Total steps:            {total_steps}")
    print(f"  mask_active:            {mask_active_count} "
          f"({mask_active_count/total_steps:.3f})")
    print(f"  action_overridden:      {overridden_count} "
          f"({overridden_count/total_steps:.3f})")
    print(f"  policy_requested_close: {close_count} "
          f"({close_count/total_steps:.3f})")
    print(f"  near_cube:              {near_count} "
          f"({near_count/total_steps:.3f})")
    print(f"  mean_tcp_distance:      {dist_sum/total_steps:.4f}")
    print(f"  overridden <= close:    {overridden_count <= close_count}")
    print(f"  overridden <= mask:     {overridden_count <= mask_active_count}")

    assert mask_active_count > 0, "Expected some mask-active steps"
    assert overridden_count <= close_count, "Invariant: overridden <= close"
    assert overridden_count <= mask_active_count, "Invariant: overridden <= mask_active"

    env.close()
    print("  ✅ PASSED\n")


def test_batch_mixed_actions():
    """
    Test 7: Batch tensor with mixed gripper actions.
    - env 0: requests close (-1.0) → should be overridden
    - env 1: requests close (-0.5) → should be overridden
    - env 2: requests open  (+1.0) → mask active but NOT overridden
    - env 3: requests open  (+0.8) → mask active but NOT overridden

    All 4 envs are "far" (threshold=0.0), so mask is active everywhere.
    """
    print("=" * 60)
    print("Test 7: Batch tensor with mixed actions")
    print("=" * 60)

    env = make_env(
        num_envs=4,
        env_kwargs={
            "force_gripper_open_enabled": True,
            "force_gripper_open_until_distance": 0.0,
            "table_collision_penalty_coef": 0.0,
            "early_gripper_close_penalty_coef": 0.0,
            "gripper_open_near_cube_bonus": 0.0,
        }
    )

    env.reset(seed=42)

    action = np.zeros((4, 8), dtype=np.float32)
    action[0, -1] = -1.0   # close
    action[1, -1] = -0.5   # close
    action[2, -1] = +1.0   # open
    action[3, -1] = +0.8   # open

    obs, reward, term, trunc, info = env.step(action)

    (mask_active, action_overridden, policy_requested_close,
     policy_gripper, executed_gripper, tcp_dist, near_cube
     ) = env.unwrapped.get_gripper_mask_info()

    print(f"  mask_active:              {mask_active.tolist()}")
    print(f"  action_overridden:        {action_overridden.tolist()}")
    print(f"  policy_requested_close:   {policy_requested_close.tolist()}")
    print(f"  policy_gripper:           {[round(x, 2) for x in policy_gripper.tolist()]}")
    print(f"  executed_gripper:         {[round(x, 2) for x in executed_gripper.tolist()]}")

    # All envs: mask_active = True (threshold=0, all are "far")
    assert mask_active.sum().item() == 4, f"All envs far, got {mask_active.tolist()}"

    # Envs 0,1: close requested → overridden
    assert action_overridden[0].item() is True
    assert action_overridden[1].item() is True
    # Envs 2,3: open requested → NOT overridden
    assert action_overridden[2].item() is False
    assert action_overridden[3].item() is False

    # policy_requested_close check
    assert policy_requested_close[0].item() is True
    assert policy_requested_close[1].item() is True
    assert policy_requested_close[2].item() is False
    assert policy_requested_close[3].item() is False

    # Executed: all should be ~+1.0 (mask fires for all, but only changes 0,1)
    for i in range(4):
        assert executed_gripper[i].item() > 0.9, (
            f"Env {i}: executed should be ~+1.0, got {executed_gripper[i].item()}"
        )

    # Batch-level invariant
    overridden_sum = action_overridden.sum().item()
    close_sum = policy_requested_close.sum().item()
    mask_sum = mask_active.sum().item()
    assert overridden_sum <= close_sum, f"{overridden_sum} > {close_sum}"
    assert overridden_sum <= mask_sum, f"{overridden_sum} > {mask_sum}"
    print(f"  overridden_sum={overridden_sum} <= close_sum={close_sum} ✅")
    print(f"  overridden_sum={overridden_sum} <= mask_sum={mask_sum} ✅")

    env.close()
    print("  ✅ PASSED\n")


def test_disabled_masking():
    """
    Test 8: force_gripper_open_enabled=False → no mask info set.
    """
    print("=" * 60)
    print("Test 8: Masking disabled")
    print("=" * 60)

    env = make_env(
        env_kwargs={
            "force_gripper_open_enabled": False,
            "force_gripper_open_until_distance": 0.0,
            "table_collision_penalty_coef": 0.0,
            "early_gripper_close_penalty_coef": 0.0,
            "gripper_open_near_cube_bonus": 0.0,
        }
    )

    env.reset(seed=42)

    action = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)
    obs, reward, term, trunc, info = env.step(action)

    (mask_active, action_overridden, policy_requested_close,
     policy_gripper, executed_gripper, tcp_dist, near_cube
     ) = env.unwrapped.get_gripper_mask_info()

    print(f"  mask_active:             {mask_active}")
    print(f"  action_overridden:       {action_overridden}")
    print(f"  policy_requested_close:  {policy_requested_close}")

    # When disabled, no mask info is set (all None)
    assert mask_active is None, f"Expected None, got {mask_active}"

    env.close()
    print("  ✅ PASSED\n")


if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  PickCubeGripperCurriculum-v1 — Refined Metric Tests  ║")
    print("╚" + "═" * 58 + "╝")
    print()

    all_passed = True

    # NOTE: test_batch_mixed_actions MUST run first — multi-env GPU
    # must be created before any other GPU PhysX init.
    test_order = [
        ("Batch mixed actions", test_batch_mixed_actions),
        ("Far + policy close", test_far_policy_close),
        ("Near + policy close", test_near_policy_close),
        ("Far + policy open", test_far_policy_open),
        ("Overridden <= close invariant", test_overridden_le_close_invariant),
        ("is_grasped gate", test_grasped_gate),
        ("Continuous 200 steps", test_continuous_200_steps),
        ("Masking disabled", test_disabled_masking),
    ]

    for name, test_fn in test_order:
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
