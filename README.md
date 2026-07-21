# ManiSkill 本地开发项目

本项目用于在 Apple Silicon Mac 上开发、调试、渲染和录制 ManiSkill 环境。当前 Mac 负责本地交互、渲染与录像；后续使用 A100 服务器进行强化学习训练。项目结构便于在两端同步代码、配置和 checkpoint。

## 环境

激活现有 ManiSkill 虚拟环境：

```bash
source ~/projects/maniskill-local/.venv/bin/activate
```

## 运行脚本

在项目根目录执行：

```bash
python scripts/explore_env.py
python scripts/random_policy.py --env-id PickCube-v1 --steps 1000 --seed 0
python scripts/record_random_episode.py --env-id PickCube-v1 --episodes 3 --seed 0 --output-dir videos/random
```

- `explore_env.py`：打印 observation 与动作空间结构，并用随机动作运行 300 步。
- `random_policy.py`：在 GUI 中运行可配置步数的随机策略，并汇总 episode reward。
- `record_random_episode.py`：无 human 窗口录制随机策略 RGB MP4，每个 episode 最多 200 步。

GUI 窗口中可直接关闭窗口退出；也可在启动脚本的终端按 `Ctrl+C`，脚本会通过 `finally` 关闭环境。

## 常见问题

- **Vulkan 环境变量缺失**：确认当前终端已配置 Vulkan SDK / MoltenVK 所需的环境变量，再重新运行。SAPIEN 提示回退到内置 Vulkan 时，通常仍可工作，但应确认渲染结果。
- **没有激活虚拟环境**：若出现 `ModuleNotFoundError`，先执行上面的 `source` 命令，或直接使用 `~/projects/maniskill-local/.venv/bin/python`。
- **首次运行较慢**：首次创建环境、加载资产和初始化渲染器可能耗时较长，属于正常现象。
- **pinocchio warning**：若当前任务不使用相关机器人学功能，`pinocchio package is not installed` 警告可暂时忽略。

## 后续计划

- 在 A100 上接入 PPO 训练
- 在 Mac 上进行 checkpoint 回放与可视化
- 完善 MP4 导出流程
- 维护 A100 与 Mac 的代码、配置和依赖版本同步
