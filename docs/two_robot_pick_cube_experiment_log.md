# TwoRobotPickCube experiment log

This is the append-only research record for task 6. Times use Asia/Shanghai.
Established artifacts are never overwritten; every run, baseline, evaluation,
and monitor stream uses a unique directory.

## TRPC-O0: investigation and baselines

- Date: 2026-07-23.
- Native environment: `TwoRobotPickCube-v1`, two Panda robots, state
  observation, joint `pd_joint_delta_pos` control, horizon 100.
- Native strict success: cube-goal distance at most 0.025 m and right arm
  static under the environment's 0.2 threshold.
- Native normalized dense reward has five stages: left reach/handoff, right
  reach/grasp plus left clearance, goal transport plus left reset, near-goal,
  and placed/static.
- Official PPO-fast reference: 1,024 environments, rollout 100, 8 update
  epochs, 32 minibatches, 50M environment steps, 16 online evaluation
  environments, CUDA graphs.
- Required pre-training baselines: random actions and deterministic untrained
  official actor, each on selection vector seed 0 with 100 environments.
- Initial method pair: official PPO-fast versus an otherwise matched
  `gamma=0.99`, `gae_lambda=0.95` long-credit variant.

### Environment interface and fixed-seed results

- The native multi-agent action space is a Gym `Dict`. The official
  `FlattenActionSpaceWrapper` converts it to a 16-dimensional joint action.
- The flattened state observation has 66 floats.
- Random actions, seed 0, 100 environments: 0/100 strict success; 0% crossed
  the handoff line; 0% right-arm grasp; mean return 1.1652. All 100 failures
  were classified as no handoff.
- Deterministic untrained official actor, seed 0, 100 environments: 0/100
  strict success; 0% crossed the handoff line; 0% right-arm grasp; mean
  return 1.2980. All 100 failures were classified as no handoff.
- Two startup negatives are retained. The first stalled while attempting to
  download PhysX into the bounded runtime directory; the already-installed
  server library was copied into that directory with matching SHA-256. The
  second exposed the unflattened multi-agent action-space incompatibility.
  The evaluator was backed up, then fixed to use the official wrapper.

### Persistence and evidence policy

Each training run retains its immutable configuration, exact command, source
version, Python/package versions, physical GPU mapping, PID and tmux session,
complete console stream, periodic three-minute resource monitor, all periodic
checkpoints, final checkpoint, online metrics, later fixed-seed sweeps, and
failure classifications. Disconnecting the local Mac must not terminate a
server job.

## TRPC-O1: official PPO-fast, 50M

- Physical GPU 1; 1,024 environments; rollout 100; 488 iterations.
- Runtime: 1 h 22 min; exit status 0.
- Retained 20 periodic checkpoints, final, TensorBoard, videos, full console
  streams, manifests, and GPU monitoring.
- Selection seed 0: final 91/100; `ckpt_476` 90/100.

## TRPC-O2: long-credit PPO-fast, 50M

- `gamma=0.99`, `gae_lambda=0.95`; other main settings matched O1.
- Runtime: about 1 h 7 min; exit status 0.
- `ckpt_476`: selection 100/100; independent 99/100.
- Matched seed 20260723: 992/1,000.
- Failures: one no-right-grasp and seven placed but not static.
- Mean return 78.715; mean minimum goal distance 5.051 mm.

## TRPC-O3: moderate-credit PPO-fast, 50M

- `gamma=0.95`, `gae_lambda=0.95`; exit status 0.
- `ckpt_476`: selection 100/100; independent 100/100.
- Matched seed 20260723: 998/1,000.
- Failures: one no-right-grasp and one placed but not static.
- Mean return 81.796; mean minimum goal distance 5.307 mm.

The official discount 0.8 was too short for the multi-stage handoff. Gamma
0.95 learned a more reliable complete sequence than 0.99.

## TRPC-O4 and O5: low-learning-rate continuation

Both began from immutable copies of O3 `ckpt_476`, used fresh optimizers,
retained `gamma=0.95` and `gae_lambda=0.95`, added 10M steps, and linearly
annealed their learning rates.

### O4: learning rate 1e-4

- Runtime: about 15 min; exit status 0.
- Final: 100/100 on both 100-environment seed sets.
- Matched seed 20260723: 997/1,000.
- Failures: one fell and two no-right-grasp.
- Mean return 83.148; mean minimum goal distance 3.326 mm.
- Rejected because strict success regressed below O3 despite higher return.

### O5: learning rate 5e-5

- Runtime: about 14.5 min; exit status 0.
- Final: 100/100 on selection seed 0 and independent seed 10000.
- Matched seed 20260723: 1,000/1,000 strict success.
- Handoff, right-arm grasp, placement, and placed-static: all 100%.
- Mean return 83.308; mean minimum goal distance 2.827 mm.
- All mutually exclusive failure counts are zero.

## Promotion

- Server:
  `runs/two_robot_pick_cube_TRPC-O5_o3ckpt476_lr5e5_10m_seed0_20260723/best_o5_final_1000of1000_matchedseed.pt`.
- Local:
  `checkpoints/ppo_two_robot_pick_cube_o5_lr5e5_10m/best_o5_final_1000of1000_matchedseed.pt`.
- Server/local SHA-256:
  `ebe5d6865c4d7282f428931258144ac73ec37add7b94c9578de3e8448d1aa63c`.
- All source, periodic, final, and rejected checkpoints remain retained.

## Video and report evidence

- Deterministic seeds 0–19 were recorded from the promoted model; all 20
  episodes satisfied the strict native success predicate.
- Local `ffprobe` verification confirmed every MP4 is H.264, 512×512,
  30 fps, 101 frames, and 3.3667 seconds.
- Video directory:
  `videos/two_robot_pick_cube/o5_best_20_successes_20260724`.
- Final Chinese report:
  `docs/two_robot_pick_cube_final_report_zh.docx`.
- The DOCX passed structural checks for page geometry, fixed-DXA table
  geometry, required metrics, artifact paths, and model hash. PNG render QA
  could not be performed because LibreOffice/`soffice` is not installed.
