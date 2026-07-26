# ManiSkill 四任务 PPO 项目

基于 ManiSkill 3、SAPIEN 和 PyTorch 的 Panda 机械臂强化学习项目。本地 Mac
用于开发和归档，A100 开发机用于大规模并行训练、评估和离屏渲染。项目目前
处于半结项维护状态，已完成 PickCube、StackCube、PegInsertionSide 与
TwoRobotPickCube 四个任务；后续以归档、复验和小范围修正为主，不再扩展新任务。

## 重启接管入口

下次开始工作时依次阅读：

1. `AGENTS.md`
2. `PROJECT_STATE.yaml`
3. `WORKFLOW.md`
4. `tasks/pick_cube/README.md`
5. `tasks/stack_cube/README.md`
6. `tasks/peg_insertion_side/README.md`
7. `tasks/two_robot_pick_cube/README.md`

任务总索引位于 `tasks/README.md`。共享源码保持在原目录，以避免破坏现有配置
和复现命令。

## 环境安装

推荐 Python 3.10/3.11：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

本机也可以激活已有环境：

```bash
source ~/projects/maniskill-local/.venv/bin/activate
```

## 主要入口

```bash
# 校验默认训练配置，不开始训练
python train.py --config configs/ppo_pick_cube.yaml --dry-run

# A100 并行训练
CUDA_VISIBLE_DEVICES=1 python train.py \
  --config configs/ppo_pick_cube_efficient_center_stage6.8_500k.yaml \
  --run-name seed0_nenvs256_stage6.8

# 恢复训练
CUDA_VISIBLE_DEVICES=1 python train.py --config CONFIG.yaml --resume CHECKPOINT.pt

# 无界面固定种子评估
python render.py --checkpoint CHECKPOINT.pt --seeds 0,1,2,3,4 --no-render

# 离屏录像（不会打开 GUI）
python render.py --checkpoint CHECKPOINT.pt --seeds 0,1,2 --record \
  --record-dir videos/evaluation

# 检查 checkpoint 元数据
python scripts/inspect_checkpoint.py --checkpoint CHECKPOINT.pt
```

`train.py` 是唯一推荐训练入口；它调用带 GPU 并行、动作覆盖屏蔽、诊断日志和恢复训练支持的实现。`scripts/train_ppo.py` 仅作为指向同一实现的兼容入口。

## 课程环境

环境从基础碰撞/夹爪控制逐级推进：

1. `PickCubeCollisionPenalty-v1`
2. `PickCubeCollisionGripper-v1`
3. `PickCubeGripperCurriculum-v1`
4. `PickCubeLiftCurriculum-v1`
5. `PickCubeStableLiftCurriculum-v1`
6. `PickCubeTargetTransportCurriculum-v1`
7. `PickCubeDirectedTransportCurriculum-v1`
8. `PickCubeGoalBrakeCurriculum-v1`
9. `PickCubePrecisionCarryCurriculum-v1`
10. `PickCubePostureStableCarry-v1`
11. `PickCubeCenterPrecisionCarry-v1`
12. `PickCubeEfficientCenterCarry-v1`

训练环境在距离方块较远时可能强制保持夹爪打开。训练循环会通过逐维 action mask 从 PPO log-prob 和 entropy 中排除被覆盖的夹爪动作，避免破坏 on-policy 假设。

## 测试

```bash
python -m unittest discover -s tests -v
python -m compileall -q envs ppo scripts train.py render.py
```

## 当前最佳模型

PickCube Stage 6.9：

`checkpoints/ppo_pick_cube_final_precision_stage6.9_gpu1_300k/best_seed0_nenvs256_stage6.9_90of100.pt`

固定 100 seeds：90/100。

StackCube SC-O2：

`checkpoints/ppo_stack_cube_official_ckpt726_lr1e4_10m/best_official_ckpt726_lr1e4_10m_894of1000_matchedseed.pt`

同种子 1,000 环境正式评估：894/1000（89.4%）；额外 seed 0 正式评估：
920/1000（92.0%）。详细信息见各任务 README 和 `docs/`；checkpoint、日志和
视频属于实验产物，默认不纳入 Git。

PegInsertionSide PIS-O2：

`checkpoints/ppo_peg_insertion_side_official_ckpt2151_lr1e4_25m/best_o2_ckpt551_991of1000_matchedseed.pt`

独立 100 seeds：100/100；同种子 1,000 环境正式评估：991/1000（99.1%）。
详细报告见 `docs/peg_insertion_side_final_report_zh.docx`。

TwoRobotPickCube TRPC-O5：

`checkpoints/ppo_two_robot_pick_cube_o5_lr5e5_10m/best_o5_final_1000of1000_matchedseed.pt`

独立 100 seeds：100/100；同种子 1,000 环境正式评估：1000/1000。
详细报告见 `docs/two_robot_pick_cube_final_report_zh.docx`。

## 平台说明

- A100：使用 CUDA 与 GPU 并行仿真训练。
- macOS：CUDA 不可用；策略推理可用 MPS，仿真/渲染由 ManiSkill 选择兼容后端。
- `pinocchio` 缺失在当前任务中通常只是警告。
- SAPIEN 找不到系统 Vulkan 时会尝试内置实现；录像和 GUI 仍应通过实际输出验证。
