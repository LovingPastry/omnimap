# OmniMap 文档

仓库根目录只保留 [README.md](../README.md)（安装、数据准备、引用），其余文档都在这里。

## 索引

| 文档 | 内容 |
| --- | --- |
| [info_flow.md](info_flow.md) | ROS 三环架构（tracking / planning / servo）的运行方式与话题约定 |
| [execution_side_setup.md](execution_side_setup.md) | 执行侧（机器人主机）轻量环境搭建，不依赖完整 InfoFlow 环境 |
| [sim.md](sim.md) | 闭环仿真入口与参数说明 |
| [nbv_fisher.md](nbv_fisher.md) | Fisher 信息场计算与 NBV 视点规划 |
| [velocity_cmd_algorithm.md](velocity_cmd_algorithm.md) | 由策略场生成 Twist 速度指令的数学推导 |

## 相关约定

- 面向 AI 助手的仓库规则见 [../CLAUDE.md](../CLAUDE.md) 与 [../AGENTS.md](../AGENTS.md)
- 环境与构建脚本在 [../scripts/](../scripts/)
- 离线工具脚本在 [../tools/](../tools/)
