# Task 2：StackCube 堆叠

## 状态

已完成，最终阶段 SC-O2。同种子 1,000 环境严格成功 894/1000（89.4%），
额外 seed 0 正式评估 920/1000（92.0%）。

## 最佳模型

本地：

```text
checkpoints/ppo_stack_cube_official_ckpt726_lr1e4_10m/best_official_ckpt726_lr1e4_10m_894of1000_matchedseed.pt
```

服务器：

```text
runs/ppo-StackCube-v1-state-seed0-official_ckpt726_lr1e4_10m_20260722/best_official_ckpt726_lr1e4_10m_894of1000_matchedseed.pt
```

SHA-256：

```text
4b42502f287c62a0eddbc3f2c61924bcb226f4838ddd14ba76038143a87c2edc
```

## 文件归属

- 自定义课程配置：`configs/ppo_stack_cube_stage*.yaml`
- 自定义课程环境：`envs/stack_cube_*.py`
- 官方权重：`checkpoints/ppo_stack_cube_official*/`
- 筛选与诊断：`scripts/*stack_cube*`
- 评估结果：`logs/evaluation/stack_*`
- 最终视频：`videos/stackcube_best_sc_o2_894of1000/`
- 最终中文报告：`docs/stack_cube_final_report_zh.md`
- 完整实验日志：`docs/stack_cube_experiment_log.md`
- 模型摘要：`docs/stack_cube_model_report.md`
- 初始方案：`docs/stack_cube_training_plan.md`

## 最终训练参数

- ManiSkill v3.0.1，commit
  `a4a4f9272ad64b1564035874b605ceb687b63ed8`
- 原生 `StackCube-v1`，state，`pd_joint_delta_pos`
- 官方 PPO-fast：4,096 envs、rollout 16、8 epochs、32 minibatches
- gamma 0.8、GAE lambda 0.9
- SC-O1：50M steps，学习率 3e-4
- SC-O2：从 `ckpt_726.pt` 新增 10M steps，学习率 1e-4 线性退火至 0
- 只使用物理 GPU 1

## 最终评估

SC-O2 final 在两组固定 100 seeds 上分别为 88/100 和 95/100，累计
183/200；源 checkpoint 726 为 80/100 和 84/100，累计 164/200。

matched seed `20260722` 的 1,000 环境评估：

- 严格成功：894/1000
- 曾抓取：99.8%
- 曾精确到位：90.5%
- 曾到位释放：90.2%
- 曾到位且静止：89.4%
- 平均最小 XY 误差：1.889 cm
- 平均最小 Z 误差：3.315 mm

106 个失败中，2 个未抓住、93 个从未精确到位、3 个到位未释放、8 个释放后
未静止。当前剩余瓶颈是最终精确放置。

## 复验入口

官方 checkpoint 必须用服务器隔离环境和官方 actor 结构加载：

```text
/vepfs-mlp2/queue013/public/huangborui/envs/stackcube_official/bin/python
third_party/ManiSkill-v3.0.1-official/ppo/ppo_fast.py
```

批量评估使用：

```text
scripts/sweep_official_stackcube_checkpoints.py
```

不要用本项目自定义 `render.py` 直接加载官方 PPO-fast state dict。
