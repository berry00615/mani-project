# StackCube training plan

## Success criterion

For ManiSkill `StackCube-v1`, cube A must be centered on cube B, its vertical
offset must match one cube width within 5 mm, cube A must be static, and the
robot must no longer be grasping it. The default horizon is 50 control steps.

## Training sequence

1. **Native baseline (2M environment steps).** Train PPO from scratch with the
   upstream dense reward. This establishes whether custom shaping is needed
   without introducing reward loopholes.
2. **Fixed-seed evaluation.** Sweep saved checkpoints on seeds 0-99 and retain
   copies of the best checkpoint; never rename or overwrite the originals.
3. **Failure diagnosis.** Classify failures as no grasp, drop in transit,
   inaccurate placement, failure to release, or post-release instability.
4. **Targeted curriculum.** Only if the baseline plateaus, create new registered
   environments and independent configs for grasp/lift, placement precision,
   and stable release. Each stage gets a new output directory and resumes from
   a copied best checkpoint.
5. **Final validation.** Evaluate at least 100 fixed seeds and record successful
   and failed videos for visual inspection.

## Isolation and safety

- All StackCube files and outputs use `stack_cube` names.
- Existing PickCube configs, checkpoints, logs, and videos remain unchanged.
- Server training is pinned to physical GPU 1 via `CUDA_VISIBLE_DEVICES=1`.
- Before synchronization, changed server-side source/config files are copied
  into a timestamped backup directory with SHA-256 manifests.
