# Task 6: TwoRobotPickCube-v1

## Status

Complete. TRPC-O5 final is the promoted model after five training methods,
full checkpoint selection, independent-seed retesting, and a matched-seed
1,000-environment formal evaluation.

## Native environment

- ManiSkill 3.0.1, reference commit
  `a4a4f9272ad64b1564035874b605ceb687b63ed8`.
- Environment: native `TwoRobotPickCube-v1`.
- Robots: two `panda_wristcam` agents controlled jointly.
- Control: `pd_joint_delta_pos`.
- Episode horizon: 100.
- Strict success: cube center is within `0.025 m` of the goal and the right
  arm satisfies the native static predicate with threshold `0.2`.
- Randomization: the cube starts in the left robot's workspace; the goal is
  sampled in the right robot's workspace with randomized height.
- Normalized dense reward divides the five-stage native reward by 21.

## Method comparison

1. **TRPC-O1 official:** `gamma=0.8`, `gae_lambda=0.9`, 50M steps.
2. **TRPC-O2 long-credit:** `gamma=0.99`, `gae_lambda=0.95`, 50M steps.
3. **TRPC-O3 moderate-credit:** `gamma=0.95`, `gae_lambda=0.95`, 50M steps.
4. **TRPC-O4:** O3 `ckpt_476`, learning rate `1e-4`, 10M-step continuation.
5. **TRPC-O5:** O3 `ckpt_476`, learning rate `5e-5`, 10M-step continuation.

## Artifact policy

- Server runs: `runs/two_robot_pick_cube_*`
- Logs/evaluations: `logs/two_robot_pick_cube/`
- Videos: `videos/two_robot_pick_cube/`
- Reports: `docs/two_robot_pick_cube_*`
- Config records: `configs/two_robot_pick_cube/`

Every output is uniquely named. All periodic and final checkpoints are
retained. Online evaluation is a health signal only and never automatically
promotes the final checkpoint.

## Selection protocol

1. Random-action and deterministic untrained official-actor baselines on 100
   fixed vector environments.
2. Sweep every checkpoint on selection seed 0.
3. Re-evaluate finalists on independent seed 10000.
4. Compare finalists on matched seed 20260723 with 1,000 environments.
5. Promote only a confirmed improvement, then verify local/server SHA-256.

Diagnostics retain phase transitions and mutually exclusive failures:
cube never crossed the handoff line, right arm never grasped, grasped but
never entered the goal, entered the goal but never became static, cube fell
off the table, and other.

## Final result

Promoted local model:

```text
checkpoints/ppo_two_robot_pick_cube_o5_lr5e5_10m/best_o5_final_1000of1000_matchedseed.pt
```

Promoted server model:

```text
runs/two_robot_pick_cube_TRPC-O5_o3ckpt476_lr5e5_10m_seed0_20260723/best_o5_final_1000of1000_matchedseed.pt
```

SHA-256:

```text
ebe5d6865c4d7282f428931258144ac73ec37add7b94c9578de3e8448d1aa63c
```

- Selection seed 0: 100/100.
- Independent seed 10000: 100/100.
- Matched seed 20260723: 1,000/1,000 strict success.
- Handoff, right-arm grasp, placement, and placed-static rates: all 100%.
- Mean minimum cube-goal distance: 2.827 mm.
- Mutually exclusive failures: zero.

O3 reached 998/1,000 and O4 reached 997/1,000. O5 was promoted only after the
matched-seed comparison; all inferior checkpoints remain retained.

## Video and report

- Twenty deterministic strict-success videos:
  `videos/two_robot_pick_cube/o5_best_20_successes_20260724`.
- All videos were validated as H.264, 512×512, 30 fps, 101 frames, and
  3.3667 seconds.
- Final Chinese report:
  `docs/two_robot_pick_cube_final_report_zh.docx`.
- Full append-only record:
  `docs/two_robot_pick_cube_experiment_log.md`.
