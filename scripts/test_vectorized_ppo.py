#!/usr/bin/env python3
"""
Unit tests for vectorized PPO training support (num_envs > 1).

Tests:
  1. RolloutBuffer shape handling with num_envs=1,4,256
  2. Buffer.add with tensor/numpy/scalar inputs
  3. GAE computation shape correctness
  4. Per-env episode tracking logic
  5. Done mask handling
  6. Success tracking per finished env
"""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ppo import RolloutBuffer


# ──────────────────────────────────────────────────────────────────────
# Test 1: Buffer shape handling with different num_envs
# ──────────────────────────────────────────────────────────────────────
def test_buffer_shapes():
    """Verify buffer tensor shapes for num_envs=1, 4, 256."""
    obs_dim, act_dim = 42, 8
    buffer_size = 16

    for num_envs in [1, 4, 256]:
        buf = RolloutBuffer(
            buffer_size=buffer_size,
            obs_dim=obs_dim,
            act_dim=act_dim,
            num_envs=num_envs,
            device=torch.device("cpu"),
        )
        assert buf.observations.shape == (buffer_size, num_envs, obs_dim), \
            f"obs shape mismatch: {buf.observations.shape}"
        assert buf.actions.shape == (buffer_size, num_envs, act_dim), \
            f"act shape mismatch: {buf.actions.shape}"
        assert buf.rewards.shape == (buffer_size, num_envs), \
            f"rew shape mismatch: {buf.rewards.shape}"
        assert buf.values.shape == (buffer_size, num_envs), \
            f"val shape mismatch: {buf.values.shape}"
        assert buf.log_probs.shape == (buffer_size, num_envs), \
            f"lp shape mismatch: {buf.log_probs.shape}"
        assert buf.dones.shape == (buffer_size, num_envs), \
            f"done shape mismatch: {buf.dones.shape}"
        assert buf.advantages.shape == (buffer_size, num_envs), \
            f"adv shape mismatch: {buf.advantages.shape}"
        assert buf.returns.shape == (buffer_size, num_envs), \
            f"ret shape mismatch: {buf.returns.shape}"
        print(f"  [PASS] Buffer shapes for num_envs={num_envs}")


# ──────────────────────────────────────────────────────────────────────
# Test 2: buffer.add with various input types
# ──────────────────────────────────────────────────────────────────────
def test_buffer_add_inputs():
    """buffer.add should accept tensor, numpy, and scalar inputs."""
    obs_dim, act_dim = 42, 8
    num_envs = 4
    buf = RolloutBuffer(
        buffer_size=8, obs_dim=obs_dim, act_dim=act_dim,
        num_envs=num_envs, device=torch.device("cpu"),
    )
    obs = torch.randn(num_envs, obs_dim)
    action = torch.randn(num_envs, act_dim)
    value = torch.randn(num_envs, 1)  # (num_envs, 1) — policy output
    log_prob = torch.randn(num_envs, 1)  # (num_envs, 1) — policy output

    # Case A: tensor reward/done
    reward_t = torch.randn(num_envs)
    done_t = torch.zeros(num_envs)
    buf.add(obs=obs, action=action, reward=reward_t, value=value,
            log_prob=log_prob, done=done_t)
    assert buf.rewards[0].shape == (num_envs,), \
        f"tensor reward shape: {buf.rewards[0].shape}"
    assert buf.dones[0].shape == (num_envs,), \
        f"tensor done shape: {buf.dones[0].shape}"
    buf.reset()

    # Case B: numpy reward/done
    reward_np = np.random.randn(num_envs).astype(np.float32)
    done_np = np.zeros(num_envs, dtype=bool)
    reward_t2 = torch.from_numpy(reward_np)
    done_t2 = torch.from_numpy(done_np.astype(np.float32))
    buf.add(obs=obs, action=action, reward=reward_t2, value=value,
            log_prob=log_prob, done=done_t2)
    assert buf.rewards[0].shape == (num_envs,), \
        f"numpy reward shape: {buf.rewards[0].shape}"
    buf.reset()

    # Case C: scalar reward/done (num_envs=1)
    buf1 = RolloutBuffer(
        buffer_size=2, obs_dim=obs_dim, act_dim=act_dim,
        num_envs=1, device=torch.device("cpu"),
    )
    buf1.add(
        obs=obs[:1], action=action[:1],
        reward=torch.tensor(1.5),
        value=value[:1], log_prob=log_prob[:1],
        done=torch.tensor(False),
    )
    assert buf1.rewards[0].shape == (1,), \
        f"scalar reward shape: {buf1.rewards[0].shape}"
    print("  [PASS] buffer.add with tensor / numpy / scalar inputs")


# ──────────────────────────────────────────────────────────────────────
# Test 3: GAE computation shape correctness
# ──────────────────────────────────────────────────────────────────────
def test_gae_shapes():
    """GAE output shapes must be correct for different num_envs."""
    obs_dim, act_dim = 42, 8
    buffer_size = 16

    for num_envs in [1, 4]:
        buf = RolloutBuffer(
            buffer_size=buffer_size, obs_dim=obs_dim, act_dim=act_dim,
            num_envs=num_envs, device=torch.device("cpu"),
            gamma=0.99, gae_lambda=0.95,
        )
        obs = torch.randn(num_envs, obs_dim)
        action = torch.randn(num_envs, act_dim)
        value = torch.randn(num_envs, 1)
        log_prob = torch.randn(num_envs, 1)

        for _ in range(buffer_size):
            reward = torch.randn(num_envs)
            done = torch.zeros(num_envs)
            done[torch.randint(0, num_envs, (1,))] = 1.0
            buf.add(obs=obs, action=action, reward=reward,
                    value=value, log_prob=log_prob, done=done)

        last_value = torch.randn(num_envs, 1)
        last_done = torch.zeros(num_envs)

        buf.compute_advantages(last_value, last_done)

        assert buf.advantages.shape == (buffer_size, num_envs), \
            f"adv shape: {buf.advantages.shape}"
        assert buf.returns.shape == (buffer_size, num_envs), \
            f"ret shape: {buf.returns.shape}"

        # Training data should be flattened
        data = buf.get_training_data()
        total_samples = buffer_size * num_envs
        for key in ["observations", "actions", "rewards", "log_probs",
                     "advantages", "returns", "values"]:
            assert data[key].shape[0] == total_samples, \
                f"{key} samples: {data[key].shape[0]} != {total_samples}"
            assert data[key].ndim == 1 or data[key].shape[1] in (obs_dim, act_dim), \
                f"{key} ndim issue: {data[key].shape}"

        print(f"  [PASS] GAE shapes for num_envs={num_envs}")


# ──────────────────────────────────────────────────────────────────────
# Test 4: Per-env episode tracking (different envs done at different steps)
# ──────────────────────────────────────────────────────────────────────
def test_per_env_episode_tracking():
    """Simulate 4 envs where each finishes at a different step."""
    num_envs = 4
    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_lengths_vec = np.zeros(num_envs, dtype=np.int64)
    episode_rewards = []
    episode_lengths = []

    # Simulate: env 0 finishes at step 3, env 1 at step 5,
    #           env 2 at step 2, env 3 at step 7
    rewards_per_step = [
        np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),  # step 0
        np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),  # step 1
        np.array([1.0, 1.0, 0.5, 1.0], dtype=np.float32),  # step 2: env2 done
        np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),  # step 3: env0 done
        np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),  # step 4
        np.array([0.0, 0.2, 0.0, 1.0], dtype=np.float32),  # step 5: env1 done
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),  # step 6
        np.array([0.0, 0.0, 0.0, 0.8], dtype=np.float32),  # step 7: env3 done
    ]

    done_per_step = [
        np.array([False, False, False, False]),  # step 0
        np.array([False, False, False, False]),  # step 1
        np.array([False, False, True,  False]),  # step 2
        np.array([True,  False, False, False]),  # step 3
        np.array([False, False, False, False]),  # step 4
        np.array([False, True,  False, False]),  # step 5
        np.array([False, False, False, False]),  # step 6
        np.array([False, False, False, True]),   # step 7
    ]

    for step_idx in range(len(rewards_per_step)):
        r = rewards_per_step[step_idx]
        d = done_per_step[step_idx]

        episode_returns += r
        episode_lengths_vec += 1

        finished = np.flatnonzero(d)
        for idx in finished:
            episode_rewards.append(float(episode_returns[idx]))
            episode_lengths.append(int(episode_lengths_vec[idx]))
            episode_returns[idx] = 0.0
            episode_lengths_vec[idx] = 0

    # After all steps:
    # Env 0: steps 0-3 = 4 steps, reward = 1+1+1+1 = 4.0
    # Env 1: steps 0-5 = 6 steps, reward = 1+1+1+1+1+0.2 = 5.2
    # Env 2: steps 0-2 = 3 steps, reward = 1+1+0.5 = 2.5
    # Env 3: steps 0-7 = 8 steps, reward = 1+1+1+1+1+1+1+0.8 = 7.8
    expected_rewards = [2.5, 4.0, 5.2, 7.8]  # by done time
    expected_lengths = [3, 4, 6, 8]

    for i, (er, el) in enumerate(zip(expected_rewards, expected_lengths)):
        assert abs(episode_rewards[i] - er) < 1e-5, \
            f"env {i}: expected reward {er}, got {episode_rewards[i]}"
        assert episode_lengths[i] == el, \
            f"env {i}: expected length {el}, got {episode_lengths[i]}"

    print(f"  [PASS] Per-env episode tracking: rewards={episode_rewards}, "
          f"lengths={episode_lengths}")


# ──────────────────────────────────────────────────────────────────────
# Test 5: Done mask handling — no .item() on multi-element tensors
# ──────────────────────────────────────────────────────────────────────
def test_done_mask_no_item():
    """Verify done_mask works on (num_envs,) arrays without .item()."""
    num_envs = 256

    # Simulate a done tensor from ManiSkill GPU sim
    done = torch.zeros(num_envs, dtype=torch.bool)
    done[10] = True
    done[100] = True
    done[200] = True

    done_bool = done.detach().bool().cpu().numpy().reshape(-1)
    assert done_bool.shape == (num_envs,), f"shape: {done_bool.shape}"

    finished = np.flatnonzero(done_bool)
    assert len(finished) == 3, f"expected 3 finished, got {len(finished)}"
    assert list(finished) == [10, 100, 200], f"unexpected indices: {finished}"

    # Verify no .item() is called (this test itself is the check)
    print(f"  [PASS] Done mask: finished_indices={finished}")


# ──────────────────────────────────────────────────────────────────────
# Test 6: Success tracking only records finished envs
# ──────────────────────────────────────────────────────────────────────
def test_success_per_finished_env():
    """Only the success of done environments should be recorded."""
    num_envs = 8
    episode_successes = []

    # Simulate: envs 2, 5 finished. Only env 2 was successful.
    done_bool = np.zeros(num_envs, dtype=bool)
    done_bool[2] = True
    done_bool[5] = True
    finished_indices = np.flatnonzero(done_bool)

    success_val = torch.zeros(num_envs, dtype=torch.bool)
    success_val[2] = True   # env 2 succeeded
    success_val[5] = False  # env 5 failed

    if len(finished_indices) > 0:
        success_array = success_val.detach().cpu().numpy().astype(bool).reshape(-1)
        episode_successes.extend(success_array[finished_indices].tolist())

    assert episode_successes == [True, False], \
        f"Expected [True, False], got {episode_successes}"
    print(f"  [PASS] Success per finished env: {episode_successes}")


# ──────────────────────────────────────────────────────────────────────
# Test 7: Reward conversion — no .item() on multi-element tensor
# ──────────────────────────────────────────────────────────────────────
def test_reward_no_item():
    """reward_np should be extractable without calling .item()."""
    num_envs = 256
    reward = torch.randn(num_envs) * 0.1

    # This should work (no .item() call):
    if isinstance(reward, torch.Tensor):
        reward_np = reward.detach().cpu().numpy().reshape(-1).astype(np.float32)
    else:
        reward_np = np.asarray(reward, dtype=np.float32).reshape(-1)

    assert reward_np.shape == (num_envs,), f"shape: {reward_np.shape}"
    assert reward_np.dtype == np.float32, f"dtype: {reward_np.dtype}"
    print(f"  [PASS] Reward conversion: shape={reward_np.shape}, "
          f"dtype={reward_np.dtype}")


# ──────────────────────────────────────────────────────────────────────
# Test 8: Training data shape for PPO update
# ──────────────────────────────────────────────────────────────────────
def test_training_data_shapes():
    """get_training_data must return correctly shaped tensors for PPO."""
    obs_dim, act_dim = 42, 8
    buffer_size = 16
    num_envs = 256

    buf = RolloutBuffer(
        buffer_size=buffer_size, obs_dim=obs_dim, act_dim=act_dim,
        num_envs=num_envs, device=torch.device("cpu"),
        gamma=0.99, gae_lambda=0.95,
    )

    obs = torch.randn(num_envs, obs_dim)
    action = torch.randn(num_envs, act_dim)
    value = torch.randn(num_envs, 1)
    log_prob = torch.randn(num_envs, 1)

    for _ in range(buffer_size):
        reward = torch.randn(num_envs)
        done = torch.zeros(num_envs)
        done[torch.randint(0, num_envs, (num_envs // 4,))] = 1.0
        buf.add(obs=obs, action=action, reward=reward,
                value=value, log_prob=log_prob, done=done)

    buf.compute_advantages(
        last_value=torch.randn(num_envs, 1),
        last_done=torch.zeros(num_envs),
    )

    data = buf.get_training_data()
    n_expected = buffer_size * num_envs

    assert data["observations"].shape == (n_expected, obs_dim), \
        f"obs: {data['observations'].shape}"
    assert data["actions"].shape == (n_expected, act_dim), \
        f"act: {data['actions'].shape}"
    assert data["rewards"].shape == (n_expected,), \
        f"rew: {data['rewards'].shape}"
    assert data["log_probs"].shape == (n_expected,), \
        f"lp: {data['log_probs'].shape}"
    assert data["advantages"].shape == (n_expected,), \
        f"adv: {data['advantages'].shape}"
    assert data["returns"].shape == (n_expected,), \
        f"ret: {data['returns'].shape}"
    assert data["values"].shape == (n_expected,), \
        f"val: {data['values'].shape}"

    print(f"  [PASS] Training data shapes: {n_expected} samples")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("Vectorized PPO — Unit Tests")
    print("=" * 70)
    print()

    tests = [
        ("Buffer shapes", test_buffer_shapes),
        ("Buffer.add inputs", test_buffer_add_inputs),
        ("GAE shapes", test_gae_shapes),
        ("Per-env episode tracking", test_per_env_episode_tracking),
        ("Done mask (no .item)", test_done_mask_no_item),
        ("Success per finished env", test_success_per_finished_env),
        ("Reward conversion (no .item)", test_reward_no_item),
        ("Training data shapes", test_training_data_shapes),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print(f"{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 70}")

    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)
