# 任务索引

本目录按任务组织项目状态，但不移动共享源码和既有实验产物。

| 任务 | 状态 | 当前最佳 | 入口 |
|---|---|---|---|
| PickCube（运输） | 已完成 | 90/100 | `pick_cube/README.md` |
| StackCube（堆叠） | 已完成 | 894/1000 matched seed | `stack_cube/README.md` |
| PegInsertionSide（侧向插销） | 已完成 | 991/1000 matched seed | `peg_insertion_side/README.md` |
| TwoRobotPickCube（双臂协作） | 已完成 | 1000/1000 matched seed | `two_robot_pick_cube/README.md` |

共享实现仍位于根目录的 `envs/`、`ppo/`、`scripts/`、`configs/`。模型、日志和
视频分别位于 `checkpoints/`、`logs/`、`videos/`，任务 README 给出精确清单。

项目当前处于半结项维护状态，不再开始新任务。维护已有任务时，不要修改或复用
四个已完成任务的输出目录；任何复验或调整都使用新的唯一目录。
