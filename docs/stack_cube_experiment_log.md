# StackCube model iteration log

This document is the running source of truth for the second task. Times are in
Asia/Shanghai unless otherwise stated. Existing checkpoints are never
overwritten; promoted models will be copied to descriptive filenames.

## SC-S1: native dense-reward baseline

- Date: 2026-07-22
- Goal: establish a clean upstream `StackCube-v1` PPO baseline before adding
  task-specific reward shaping or curriculum stages.
- Config: `configs/ppo_stack_cube_stage1_baseline_gpu1_2m.yaml`
- Initialization: from scratch; no PickCube weights transferred.
- Environment: `StackCube-v1`, state observations, 50-step horizon,
  `pd_joint_delta_pos`, PhysX CUDA, 256 parallel environments.
- PPO: 2,000,000 environment steps, rollout 64 x 256, learning rate 3e-4,
  batch size 1024, 4 epochs, gamma 0.99, GAE lambda 0.95, clip 0.2,
  entropy coefficient 0, separate 256-256-256 actor and critic networks.
- Hardware: NVIDIA A100-SXM4-80GB, physical GPU 1 selected with
  `CUDA_VISIBLE_DEVICES=1`.
- Server PID: 2974631.
- Server backup: `backups/stack_cube_stage1_before_20260722_124227`.
- Checkpoints: every approximately 100k environment steps in
  `checkpoints/ppo_stack_cube_stage1_baseline_gpu1_2m/`.

### Observations during training

| Environment steps | Mean episode reward | Strict success | Note |
|---:|---:|---:|---|
| 16,384 | 1.858 | 0.0% | Random-policy region |
| 327,680 | 7.9 (approx.) | 0.0% | Dense-reward phases learned rapidly |
| 671,744 | 8.362 | 0.0% | Potential plateau before full stack/release success |
| 2,000,128 | 18.725 | 0.0% | Finished; 0 successes in 39,936 trajectories |

The generic trainer's `grasp_rate` column is not valid for native StackCube:
the environment reports `is_cubeA_grasped`, whereas the historical PickCube
diagnostic expects `is_grasped`. This affects reporting only, not observations,
rewards, PPO updates, or the strict `success` metric.

### Planned post-training analysis

1. Evaluate saved checkpoints on identical fixed seeds.
2. Select by strict success rate, using episode length and return only as
   tie-breakers.
3. Inspect failures by phase: reach/grasp, lift/transport, alignment, release,
   and post-release stability.
4. Use the failure distribution—not dense return alone—to define SC-S2.

### Fixed-seed and phase evaluation

- Deterministic seeds 0-99: 0/100 strict successes, mean return 22.312.
- Ever grasped: 99%; final still grasping: 96%.
- Ever inside the complete on-target predicate: 0%.
- Mean minimum XY error: 13.37 cm; only 7% ever reached 3 cm XY error.
- Diagnosis: SC-S1 solved acquisition but not directed transport toward cube B.

## SC-S2: grasped transport curriculum

- Date: 2026-07-22
- Initialization: SC-S1 `final_seed0_nenvs256.pt` at timestep 2,000,128.
- New environment: `StackCubeTransportCurriculum-v1`, 100-step horizon.
- Goal: convert the 99% grasp capability into motion toward cube B before
  introducing a release-focused stage.
- Reward additions: grasp-conditioned goal progress x120 (2 cm clipped),
  grasp-conditioned goal proximity x4, drop penalty 4, strict success bonus
  50, and time cost 0.01. Upstream dense reward is retained.
- PPO changes: 1M additional environment steps; learning rate 1e-4; clip 0.15;
  batch size 1024; all other core settings retained.
- Output: `checkpoints/ppo_stack_cube_stage2_transport_gpu1_1m/`.
- Server backup: `backups/stack_cube_stage2_before_20260722_125726`.
- Server PID: 2980878; physical GPU 1.

### SC-S2 result

- Completed at timestep 3,000,320 in 4.8 minutes.
- Training: 1/9,984 strict successes; final rolling rate 0.01%.
- Fixed seeds 0-99 across all 21 saved candidates: every checkpoint scored
  0/100 strict successes and 0% complete on-target rate.
- Mean minimum XY error remained 13-14 cm. The best 3 cm reach rate was 11%
  at the first checkpoint and later declined to 4% at final.
- Decision: reject SC-S2 as a model improvement. Preserve all artifacts, but
  branch SC-S3 from SC-S1 rather than from the degraded SC-S2 policy.

## SC-S3: local-target transport curriculum

- Initialization: branch from SC-S1 final at timestep 2,000,128.
- Cube B starts in a 7-11 cm collision-free ring around cube A; the physical
  stack geometry and strict success predicate are unchanged.
- Removes SC-S2's farmable per-step goal-proximity reward.
- Adds grasp-conditioned progress x300 and best-so-far improvement x150,
  one-time drop/premature-release penalties, on-target bonus 10, strict success
  bonus 100, and time cost 0.01.
- At timestep 2,262,272, a partial auto-reset exposed a pose batch mismatch
  (one reset index versus 256 poses). Training stopped cleanly after saving
  `checkpoint_2262272.pt`. The fix writes only `env_idx` poses; continuation
  uses a distinct run/log name and exactly the remaining 737,856 steps.

### SC-S3 result

- Best placement checkpoint: `checkpoint_2606336.pt`.
- Fixed seeds 0-99: 3% ever exact on-target, 8% within 3 cm, 0% success.
- Final reached 3 cm in 17% but exact on-target was lower at 2%.

## SC-S4: exact-placement auto-release

- Initialization: SC-S3 checkpoint 2,606,336.
- Exact upstream on-target geometry triggers forced gripper opening; only that
  gripper dimension is masked out of PPO.
- Training: 17/4,878 strict successes (0.35%).
- Best deterministic checkpoint: 2,802,944, with 3% exact placement and 3%
  released on target, but 0% on-target-and-static and 0% strict success.

## SC-S5: stable release

- Initialization: SC-S4 checkpoint 2,802,944.
- Auto-release requires both exact placement and upstream static predicate.
- Adds on-target low-speed shaping, on-target static bonus, and motion penalty
  so the cube is braked before the gripper opens.

### SC-S5 result

- Training: 1/4,865 strict successes (0.02%), worse than SC-S4.
- Final deterministic seeds 0-99: 0% success, 1% exact on-target, 0% released.
- Decision: reject this sparse static-gated curriculum and preserve it as a
  negative ablation rather than spending more steps on it.

## SC-O1: official ManiSkill PPO-fast baseline

- Rationale: the upstream v3.0.1 baseline specifies 50M steps for StackCube;
  SC-S1 used only 2M and inherited PickCube-oriented gamma/GAE settings.
- Source: ManiSkill tag v3.0.1, commit
  `a4a4f9272ad64b1564035874b605ceb687b63ed8`.
- Script: `third_party/ManiSkill-v3.0.1-official/ppo/ppo_fast.py` (unmodified).
- Environment: native `StackCube-v1`, normalized dense reward,
  `pd_joint_delta_pos`, 4096 GPU environments, 16-step rollouts.
- PPO: 50,000,000 steps, 8 update epochs, 32 minibatches, learning rate 3e-4,
  gamma 0.8, GAE lambda 0.9, clip 0.2.
- Evaluation: 16 environments every 25 iterations; unique checkpoint per
  evaluation plus `final_ckpt.pt`.
- Isolated Python environment:
  `/vepfs-mlp2/queue013/public/huangborui/envs/stackcube_official`.
- GPU: physical GPU 1 via `CUDA_VISIBLE_DEVICES=1`.
- Run: `runs/ppo-StackCube-v1-state-seed0-official50m_20260722`.
- Console log: `logs/ppo-StackCube-v1-state-seed0-official50m_20260722.log`.
- Server PID: 3037176.

### SC-O1 result

- Completed all 50M steps (762 iterations) in 29.5 minutes.
- Saved 31 periodic checkpoints plus `final_ckpt.pt`; no checkpoint was
  overwritten.
- Upstream 16-env evaluation rose from 0% to a peak of 81%; final evaluation
  was 75%.
- Fixed 100-env checkpoint sweep selected `ckpt_726.pt`:
  - selection seeds: 80/100;
  - independent seeds: 84/100;
  - combined model-selection evidence: 164/200 (82%).
- `final_ckpt.pt` scored 68/100 on selection seeds and 80/100 on independent
  seeds, confirming that final was not consistently best.
- Promoted copy: `best_official_ckpt726_164of200.pt`; original `ckpt_726.pt`
  remains unchanged. SHA-256:
  `53490c52e78d12f2a656f7ed45da6ce552f87c358879760ebbedf3556fac6263`.

### Formal 1,000-environment evaluation

- Fixed vector seed: 20260722.
- Strict success: 821/1,000 (82.1%).
- Ever grasped: 99.7%; ever exact on-target: 87.9%; ever released on-target:
  85.8%; ever on-target-and-static: 82.2%.
- Mean minimum XY error: 2.362 cm; mean minimum Z error: 4.326 mm.
- Failure classification (179 total): 3 no-grasp, 118 never exact on-target,
  21 on-target but never released, 36 released but never static, and 1 other
  timing/intersection failure.
- Primary residual bottleneck: exact placement; secondary bottleneck:
  post-release stability.
- Formal artifacts:
  `logs/evaluation/stack_official50m_best_formal1000.{json,csv,log}`.

## SC-O2: checkpoint-726 low-learning-rate continuation

- Initialization: a preserved copy of SC-O1 `ckpt_726.pt`; the original run,
  checkpoint, logs, and promoted model were not modified.
- Run: `runs/ppo-StackCube-v1-state-seed0-official_ckpt726_lr1e4_10m_20260722`.
- GPU: physical GPU 1 via `CUDA_VISIBLE_DEVICES=1`.
- Environment and PPO structure: unchanged from SC-O1 (`StackCube-v1`, 4,096
  environments, 16-step rollouts, 8 epochs, 32 minibatches, gamma 0.8, GAE
  lambda 0.9).
- Continuation: 10,000,000 new environment steps with a fresh optimizer and
  learning rate 1e-4 linearly annealed to zero.
- Runtime: approximately 7 minutes 51 seconds (152 iterations).
- Preservation: source copy, launch command, console log, seven periodic
  checkpoints, and `final_ckpt.pt` are all retained independently.

### SC-O2 model selection

- Fixed seeds 0-99: final 88/100, checkpoint 151 88/100, source 80/100.
- Independent seeds 10000-10099: final 95/100, checkpoint 151 90/100,
  source 84/100.
- The final checkpoint ranked first on both sets: 183/200 (91.5%), compared
  with the unchanged source checkpoint's 164/200 (82.0%).
- Promoted canonical copy:
  `best_official_ckpt726_lr1e4_10m_894of1000_matchedseed.pt`.
- SHA-256:
  `4b42502f287c62a0eddbc3f2c61924bcb226f4838ddd14ba76038143a87c2edc`.

### SC-O2 formal evaluation

- Matched comparison seed 20260722: 894/1,000 strict successes (89.4%), a
  7.3 percentage-point absolute improvement over SC-O1's 82.1%.
- Matched-seed diagnostics: 99.8% ever grasped, 90.5% ever exact on-target,
  90.2% ever released on-target, and 89.4% ever on-target-and-static.
- Mean minimum XY error: 1.889 cm; mean minimum Z error: 3.315 mm.
- Failure classification (106 total): 2 no-grasp, 93 never exact on-target,
  3 on-target but never released, 8 released but never static, 0 other.
- Additional seed-0 formal set: 920/1,000 (92.0%); 100% ever grasped, 93.1%
  ever exact on-target, 92.7% released, and 92.0% static.
- The remaining bottleneck is still exact final placement, but both placement
  precision and post-release stability improved substantially.
- Artifacts: `logs/evaluation/stack_ckpt726_lr1e4_*`.
