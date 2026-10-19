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
| [nbv_ablation.md](nbv_ablation.md) | 主动视角策略四项可切换修复（`nbv_policy`）的定义、配置回退与仿真消融报告 |
| [active_view_experiments.html](active_view_experiments.html) | 交互网页：09-05 至 10-18 全部消融与仿真实验的思维链（现象→诊断→选型→实验→结论），区分算法改进 / 超参数 / 实验设置 / 诊断 |
| [eps_contrib_report.md](eps_contrib_report.md) | 仿真报告：探测步长 `grad_eps` 扫描 + Fisher 分数来源诊断（更正消融报告的"近视"结论） |
| [ur5_nbv_kinematics_report.html](ur5_nbv_kinematics_report.html#three-loops) | UR5 主动感知三环复盘：改造顺序、源码证据、可执行信息动作与最小真机对照；附论文精读 |
| [ACTIVE_PERCEPTION_REVIEW.md](../ACTIVE_PERCEPTION_REVIEW.md) | 三环复盘的完整评审记录、执行/观测反馈定义、反例检查与结果—主张边界 |

## 相关约定

- 面向 AI 助手的仓库规则见 [../CLAUDE.md](../CLAUDE.md) 与 [../AGENTS.md](../AGENTS.md)
- 环境与构建脚本在 [../scripts/](../scripts/)
- 离线工具脚本在 [../tools/](../tools/)
