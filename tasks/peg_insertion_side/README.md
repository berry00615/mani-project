# Task 3: PegInsertionSide-v1

## Status

Complete. PIS-O2 `ckpt_551` is the promoted model after full checkpoint
selection, independent-seed retest, and matched-seed 1,000-environment formal
evaluation.

## Native environment

- ManiSkill 3.0.1, reference commit
  `a4a4f9272ad64b1564035874b605ceb687b63ed8`.
- Environment: native `PegInsertionSide-v1`, `panda_wristcam`.
- State observation: 43 floats. Action: 8 floats in `[-1, 1]`.
- Control: `pd_joint_delta_pos`. Episode horizon: 100.
- Strict success: in the box-hole frame, the peg head must satisfy
  `x >= -0.015 m`, `|y| <= hole_radius`, and `|z| <= hole_radius`, where
  `hole_radius = peg_radius + 0.003 m`.
- Dense reward: reach the peg-tail grasp pose; grasp with at most 20-degree
  finger/object angle; align peg center and head in goal-frame YZ; insert;
  strict success sets raw dense reward to 10. Normalized dense divides by 10.

## Official PPO-fast reference

Configuration: `configs/peg_insertion_side/official_ppo_fast_75m.yaml`.

- 2,048 environments, rollout 16, 8 epochs, 32 minibatches.
- gamma 0.97, GAE lambda 0.95, 75M environment steps.
- 100 evaluation steps, 16 online evaluation environments.
- Physical GPU 1 only unless a later log explicitly records a safe, genuine
  multi-GPU topology.

## Namespace and artifact policy

- Server runs: `runs/peg_insertion_side_*`
- Logs/evaluation: `logs/peg_insertion_side/`
- Videos: `videos/peg_insertion_side/`
- Reports: `docs/peg_insertion_side_*`
- Every run and output directory is unique. No PickCube or StackCube result
  directory may be modified. All checkpoints, including periodic and final,
  are retained. Final is never automatically promoted.

## Selection protocol

1. Random-action and deterministic untrained official-actor baselines on a
   fixed 100-environment vector seed set.
2. Sweep every saved checkpoint on selection seed 0 (100 environments).
3. Re-evaluate finalists on independent seed 10000 (100 environments).
4. Compare candidates on matched seed 20260722 (1,000 environments).
5. Promote only a confirmed improvement by copying, then verify server/local
   SHA-256. Produce strict-success videos and validate them with `ffprobe`.

Diagnostics include ever grasped, peg-axis orientation, pre-insertion YZ
alignment, hole-entry lateral alignment, insertion depth, strict success, and
mutually exclusive failure classes.

## Final results

### PIS-O1: official 75M from scratch

- Run:
  `runs/ppo-PegInsertionSide-v1-state-seed0-official75m_20260722_1630`.
- Runtime: about 3 h 27 min on physical GPU 1.
- Retained: 92 periodic checkpoints plus `final_ckpt.pt`.
- Selection seed 0: `ckpt_2251` 26/100, `ckpt_2151` 21/100,
  `ckpt_2126` 22/100, final 11/100.
- Independent seed 10000: 23/100, 19/100, and 14/100 respectively.
- Matched seed 20260722, 1,000 environments: `ckpt_2151` 175/1000,
  `ckpt_2251` 160/1000, `ckpt_2126` 160/1000, final 89/1000.
- O1 best is therefore `ckpt_2151`, not the selection-set leader and not
  final. Its main remaining failure was insufficient insertion depth.

### PIS-O2: low-learning-rate continuation

- Run:
  `runs/ppo-PegInsertionSide-v1-state-seed0-official_ckpt2151_lr1e4_25m_20260723`.
- Source: immutable copy of O1 `ckpt_2151`; fresh optimizer.
- Additional 25M steps with learning rate `1e-4`, linearly annealed to zero.
- Runtime: about 1 h 50 min on physical GPU 1.
- Retained: 31 periodic checkpoints, final, and source copy.
- Selection seed 0: five candidates reached 99/100; final reached 97/100.
- Independent seed 10000: `ckpt_551` and `ckpt_701` both reached 100/100.
- Matched seed 20260722, 1,000 environments:
  `ckpt_551` 991/1000 and `ckpt_701` 990/1000.
- `ckpt_551` diagnostics: 99.9% ever grasped, 99.9% ever oriented within
  10 degrees, 99.8% ever pre-insert aligned, 100% ever entry aligned,
  0.836-degree mean minimum axis angle, 1.306 mm mean minimum lateral error,
  and +26.46 mm mean maximum insertion x.
- Its nine mutually exclusive failures were one no-grasp, one no-pre-insert
  alignment, and seven insufficient-depth episodes.

The improvement over the O1 best is 81.6 percentage points. The original
overlapping failure masks were corrected and both finalists were rerun into
new mutually exclusive evaluation directories; strict success and ranking did
not change.

## Videos and report

- Twenty deterministic strict-success videos:
  `videos/peg_insertion_side/showcase_best_o2_ckpt551_20_successes_20260723/`.
- First five trajectories in four independent views plus five 2x2 grids:
  `videos/peg_insertion_side/multiview_best_o2_ckpt551_first5_20260723/`.
- All delivered MP4 files were validated with `ffprobe`; manifests are stored
  beside the videos.
- Detailed Chinese final report:
  `docs/peg_insertion_side_final_report_zh.docx`.
- Running experiment record:
  `docs/peg_insertion_side_experiment_log.md`.

## Local test environment

The Apple Silicon local test environment is `.venv-peg-insertion-side` and
contains ManiSkill 3.0.1, SAPIEN, PyTorch, and video dependencies. The test
entry point is `scripts/test_peg_insertion_side_local.py`; it uses CPU PhysX
because the training `physx_cuda` backend is not available on macOS.

Quick strict-success test:

```bash
.venv-peg-insertion-side/bin/python \
  scripts/test_peg_insertion_side_local.py --episodes 10 --seed 0
```

Single deterministic episode with MP4:

```bash
.venv-peg-insertion-side/bin/python \
  scripts/test_peg_insertion_side_local.py \
  --episodes 1 --seed 1 \
  --video videos/peg_insertion_side/local_test/seed1.mp4
```

The promoted model is:

```text
checkpoints/ppo_peg_insertion_side_official_ckpt2151_lr1e4_25m/best_o2_ckpt551_991of1000_matchedseed.pt
```

SHA-256:

```text
b48a4e0732de5e1e68bc906897166be95f2cecb3a4c8d245680dc62e2e7a6c49
```
