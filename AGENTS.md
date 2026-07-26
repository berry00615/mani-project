# Agent startup instructions

This repository contains four completed ManiSkill tasks and is in
semi-closed maintenance. Adjust and verify existing work only; do not start a
new task unless the project owner explicitly changes this status. Before
changing or training anything, read these files in order:

1. `PROJECT_STATE.yaml`
2. `WORKFLOW.md`
3. `tasks/pick_cube/README.md`
4. `tasks/stack_cube/README.md`
5. `tasks/peg_insertion_side/README.md`
6. `tasks/two_robot_pick_cube/README.md`

Mandatory operating rules:

- **Server write boundary:** the only server path that may be changed is
  `/vepfs-mlp2/queue013/public/huangborui` and its descendants. Every other
  server path is read-only: do not create, modify, move, rename, or delete
  anything outside this boundary, even if permissions allow it. Resolve and
  verify the target path before every server-side write or destructive action.
- Never delete or overwrite existing configs, checkpoints, logs, videos,
  reports, or experiment directories.
- Back up a file before modifying an established server-side artifact.
- Start every new experiment in a uniquely named config/run/output directory.
- The server has two A100 GPUs. Default to physical GPU 1 because GPU 0 may be
  occupied. Multi-GPU use is allowed when it materially benefits the task,
  but first verify both GPUs are free and will not interfere with another
  process. Use an implementation that genuinely supports multi-GPU execution
  (for example DDP or explicit job-level parallelism); setting
  `CUDA_VISIBLE_DEVICES=0,1` alone does not make a single-GPU trainer scale.
  Record GPU allocation and launch topology for every run.
- For long jobs, monitor no more frequently than once every three minutes
  unless diagnosing a startup failure.
- Do not promote the final training checkpoint automatically. Sweep all saved
  checkpoints on fixed seeds, verify finalists on an independent seed set,
  and compare on a matched 1,000-environment formal evaluation.
- A visually plausible episode is not enough. Use the task's strict native
  success predicate and retain failure classifications.
- Preserve negative experiments in the report; they are part of the research
  record.
- Before handing off, synchronize the best model, evaluation JSON/CSV/logs,
  videos, and reports to both the local workspace and server, then verify
  hashes for promoted model files.

The canonical code layout (`envs/`, `configs/`, `scripts/`, `ppo/`) is shared
by all tasks. Task folders are indexes and handoff documents; do not relocate
shared source files merely for cosmetic organization because existing configs
and commands depend on their current paths.
