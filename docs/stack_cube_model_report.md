# StackCube model training report

## Outcome

The current best model is a 10M-step low-learning-rate continuation of
ManiSkill official PPO-fast checkpoint 726 on native `StackCube-v1`. It
achieved **894/1,000 (89.4%)** strict successes on the same fixed-seed formal
set used by the previous model, up from 82.1%. An additional seed-0 set scored
920/1,000 (92.0%). Strict success requires exact stacking, release, and cube
staticity.

Best model:

`checkpoints/ppo_stack_cube_official_ckpt726_lr1e4_10m/best_official_ckpt726_lr1e4_10m_894of1000_matchedseed.pt`

SHA-256:

`4b42502f287c62a0eddbc3f2c61924bcb226f4838ddd14ba76038143a87c2edc`

## Training configuration

- ManiSkill v3.0.1, commit `a4a4f9272ad64b1564035874b605ceb687b63ed8`
- Native `StackCube-v1`, state observations, `pd_joint_delta_pos`
- Physical A100 GPU 1 (`CUDA_VISIBLE_DEVICES=1`)
- 4,096 parallel environments, 16-step rollouts
- 50,000,000-step official run followed by 10,000,000 new continuation steps
- Continuation PPO: learning rate 1e-4 linearly annealed to zero, gamma 0.8,
  GAE lambda 0.9, clip 0.2,
  8 update epochs, 32 minibatches
- Continuation runtime: approximately 7 minutes 51 seconds
- Initialization: official checkpoint 726 at approximately 47.5M steps

## Model selection

All 32 official model files were evaluated on identical 100-environment fixed
starts. The top selection results were checkpoint 726 (80%), checkpoint 651
(78%), checkpoint 676 (76%), checkpoint 751 (71%), and final (68%). On a
second independent 100-environment set, checkpoint 726 again ranked first at
84%. Its combined selection evidence was 164/200 (82%). A separate 10M-step
continuation from that checkpoint was then swept without overwriting the
baseline. Its final checkpoint ranked first on both fixed sets at 88/100 and
95/100, for 183/200 (91.5%); the unchanged source scored 80/100 and 84/100.

## Formal evaluation and failure analysis

On the same 1,000 fixed evaluation environments (vector seed 20260722), the
continued model scored 89.4%:

| Metric | Result |
|---|---:|
| Strict success | 894/1,000 (89.4%) |
| Ever grasped | 99.8% |
| Ever exact on-target | 90.5% |
| Ever released on-target | 90.2% |
| Ever on-target and static | 89.4% |
| Mean minimum XY error | 1.889 cm |
| Mean minimum Z error | 3.315 mm |

The 106 failures comprised 2 grasp failures, 93 failures to reach the exact
stacking predicate, 3 exact placements without release, and 8 releases without
staticity. On a separate seed-0 formal set the same weights scored 92.0%.
Future improvement should still focus first on final placement precision.

## Iteration history

Five custom curriculum experiments preceded the official baseline. They
identified grasp, transport, release, and staticity bottlenecks but topped out
at sparse successes. The decisive changes were adopting the upstream
StackCube training scale and hyperparameters, then continuing the strongest
checkpoint for 10M steps at one-third the original learning rate. Full
per-stage settings, negative results, backups, and diagnostic evidence are
retained in `stack_cube_experiment_log.md`.
