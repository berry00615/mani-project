# PegInsertionSide experiment log

This is the running record for task 3. Times use Asia/Shanghai. Experiments
are non-destructive: established artifacts are never overwritten, each run
has a unique namespace, and negative results remain part of the record.

## PIS-O0: upstream investigation and untrained baselines

- Date: 2026-07-22.
- Source: ManiSkill 3.0.1; requested reference commit
  `a4a4f9272ad64b1564035874b605ceb687b63ed8`.
- Native environment: `PegInsertionSide-v1`, state (43), action (8),
  `pd_joint_delta_pos`, horizon 100, normalized dense reward.
- Official baseline arguments: 2,048 envs, rollout 16, 8 update epochs,
  32 minibatches, gamma 0.97, GAE lambda 0.95, 75M steps, evaluation horizon
  100 and 16 online evaluation envs.
- Hardware inspection: physical GPU 0 had two active compute processes;
  physical GPU 1 was idle. All task-3 execution is therefore assigned only
  to physical GPU 1.
- Baselines to run before training: random actions and a deterministically
  initialized official PPO-fast actor on selection vector seed 0 with 100
  environments.

### Fixed-seed baseline results

| Policy | Strict success | Ever grasped | Pre-insert aligned | Entry aligned | Mean return |
|---|---:|---:|---:|---:|---:|
| Random actions | 0/100 | 0% | 0% | 0% | 2.089 |
| Untrained official actor | 0/100 | 0% | 0% | 0% | 2.593 |

All 200 failures were classified as failure to grasp. Server artifacts are in
the unique `logs/peg_insertion_side/baseline_*_20260722_*` directories.

### Native success and diagnostic definitions

The native predicate transforms the peg head into the hole frame. Success is
`head_x >= -0.015`, `abs(head_y) <= hole_radius`, and
`abs(head_z) <= hole_radius`. The report additionally records grasp, peg-axis
angular error to the goal axis, upstream pre-insertion YZ alignment, lateral
hole-entry error, maximum signed insertion depth, and failure phase.

## PIS-O1: official 75M PPO-fast training

- Run: `runs/ppo-PegInsertionSide-v1-state-seed0-official75m_20260722_1630`.
- Console and launch records:
  `logs/peg_insertion_side/training_official75m_20260722_1630/`.
- Python PID: 3098367. Single process on physical GPU 1 only.
- Official task parameters exactly match the reference configuration. Online
  video capture is disabled; final strict-success videos will be generated
  separately from the selected best checkpoint.
- Early health check at iteration 87/2,288: GPU 1 used 4.76 GiB at 68%; four
  checkpoints retained; online mean return rose from 2.50 to 30.74 while
  strict online success remained zero.

### O1 completion and checkpoint selection

- Runtime: about 3 h 27 min.
- Retained artifacts: 92 periodic checkpoints and `final_ckpt.pt`.
- Selection seed 0, 100 environments:
  `ckpt_2251` 26/100, `ckpt_2151` 21/100, `ckpt_2126` 22/100, final 11/100.
- Independent seed 10000, 100 environments:
  `ckpt_2251` 23/100, `ckpt_2151` 19/100, `ckpt_2126` 14/100.
- Matched seed 20260722, 1,000 environments:
  `ckpt_2151` 175/1000, `ckpt_2251` 160/1000,
  `ckpt_2126` 160/1000, final 89/1000.
- `ckpt_2151` was promoted as the O1 best only after the formal comparison.
  Its SHA-256 is
  `a350b940961c592d46729b7e9cc82bcedb8049cd1d114722d658c3e53afe2689`.
- Failure analysis showed that grasp and broad alignment were mostly learned;
  insufficient positive insertion depth remained the dominant bottleneck.

## PIS-O2: continuation from the O1 best

- Date: 2026-07-23.
- Run:
  `runs/ppo-PegInsertionSide-v1-state-seed0-official_ckpt2151_lr1e4_25m_20260723`.
- Initialization: immutable copy of O1 `ckpt_2151`; fresh optimizer.
- Training: additional 25M environment steps, learning rate `1e-4` linearly
  annealed to zero; all other official environment and PPO parameters kept.
- Hardware: one process on physical GPU 1 only.
- Runtime: about 1 h 50 min.
- Retained artifacts: 31 periodic checkpoints, final, and source checkpoint
  copy.
- Online final evaluation: 16/16 strict success, mean return 85.62. This was
  treated as a health metric, not as the model-selection result.

### O2 full selection sweep

Selection seed 0, 100 environments:

| Checkpoint | Strict success |
|---|---:|
| `ckpt_551` | 99/100 |
| `ckpt_626` | 99/100 |
| `ckpt_701` | 99/100 |
| `ckpt_726` | 99/100 |
| `ckpt_751` | 99/100 |
| final | 97/100 |

Independent seed 10000, 100 environments:

| Checkpoint | Strict success |
|---|---:|
| `ckpt_551` | 100/100 |
| `ckpt_701` | 100/100 |
| `ckpt_626` | 99/100 |
| `ckpt_751` | 99/100 |
| `ckpt_726` | 98/100 |

### O2 matched-seed formal evaluation

Matched seed 20260722, 1,000 environments:

| Model | Strict success | Rate |
|---|---:|---:|
| O2 `ckpt_551` | 991/1000 | 99.1% |
| O2 `ckpt_701` | 990/1000 | 99.0% |
| O1 `ckpt_2151` | 175/1000 | 17.5% |

`ckpt_551` formal diagnostics:

- Ever grasped: 99.9%.
- Ever oriented within 10 degrees: 99.9%.
- Ever pre-insert aligned: 99.8%.
- Ever entry aligned: 100%.
- Mean minimum axis angle: 0.836 degrees.
- Mean minimum lateral error: 1.306 mm.
- Mean maximum insertion x: +26.46 mm.
- Mean normalized-dense return: 83.154.
- Mutually exclusive failures: one no-grasp, one no-pre-insert alignment,
  seven insufficient-depth.

The original evaluation script allowed failure masks to overlap. Before
modification, the server script was backed up to
`backups/peg_insertion_side_eval_before_mutually_exclusive_fix_20260723`.
Failure masks were made mutually exclusive in the order no grasp, no
orientation, no pre-insert alignment, no entry alignment, insufficient depth,
and other. Both O2 finalists were rerun into new
`*_mutually_exclusive_20260723` directories. Strict success and ranking were
unchanged.

## PIS-O3: promotion, local handoff, and visual evidence

- Promoted model:
  `checkpoints/ppo_peg_insertion_side_official_ckpt2151_lr1e4_25m/best_o2_ckpt551_991of1000_matchedseed.pt`.
- Server promoted copy:
  `runs/ppo-PegInsertionSide-v1-state-seed0-official_ckpt2151_lr1e4_25m_20260723/best_o2_ckpt551_991of1000_matchedseed.pt`.
- Server/local SHA-256, verified identical:
  `b48a4e0732de5e1e68bc906897166be95f2cecb3a4c8d245680dc62e2e7a6c49`.
- Independent local environment: `.venv-peg-insertion-side`.
- Local test entry point: `scripts/test_peg_insertion_side_local.py`.
- Local smoke verification: seed 0 and seed 1 each strict success 1/1.
- Twenty seed 0-19 showcase videos were all strict success and validated as
  H.264, 512x512, 30 fps, 101 frames, 3.3667 seconds.
- Seeds 0-4 additionally have synchronized hero, front, side, and top
  individual views plus a 1024x1024 four-view grid. All media parameters were
  validated with `ffprobe`.
- Detailed report:
  `docs/peg_insertion_side_final_report_zh.docx`.

## Final status

Task 3 is complete. The promoted model improved the matched-seed formal result
from the O1 best 175/1000 to 991/1000, an absolute gain of 81.6 percentage
points. All checkpoints remain retained; final was not treated as best.
