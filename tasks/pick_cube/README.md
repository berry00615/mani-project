# Task 1：PickCube 运输

## 状态

已完成，最终阶段 Stage 6.9，固定 100 seeds 严格成功 90/100。

## 最佳模型

```text
checkpoints/ppo_pick_cube_final_precision_stage6.9_gpu1_300k/best_seed0_nenvs256_stage6.9_90of100.pt
```

SHA-256：

```text
35d61a031d20dcba73ef9fe60be9398ddb0a8bd83c6eb81889c2df852a0fdbad
```

成功视频：

```text
videos/stage6.9_best_20_successes/
```

正式报告：

```text
docs/mani_project_full_research_report_stage6_9.docx
```

## 文件归属

- 配置：`configs/*pick_cube*.yaml`
- 环境：`envs/pick_cube_*.py`
- 模型：`checkpoints/ppo_pick_cube*/` 以及早期根级 `final_*stage*.pt`
- 日志：`logs/ppo_pick_cube*/`、早期 collision/gripper/finetune 日志
- 视频：除 `videos/stackcube_*` 外的 PickCube、课程阶段和早期诊断目录
- 通用训练入口：`train.py`、`scripts/train_ppo_gpu.py`
- 通用渲染入口：`render.py`

## 关键结论

Stage 6.8 已能高概率完成运输，肉眼接近成功的失败主要来自严格 success
predicate 的精确位置、静止和 dwell 条件。Stage 6.9 针对最终精度进行续训，
固定种子提升至 90/100。后续任务不应覆盖或继续写入 Stage 6.9 目录。

## 复验提示

使用 `render.py` 的确定性策略和固定 seeds 评估。注意自定义 checkpoint 与官方
ManiSkill PPO-fast checkpoint 格式不同，不能混用 StackCube 官方权重加载器。
