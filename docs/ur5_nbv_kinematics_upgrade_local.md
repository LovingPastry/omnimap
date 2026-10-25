# OmniMap 计算端运动学反馈升级

更新：2026-10-25。本文按当前源码维护，替代 2026-10-22 的实施前清单。
当前工作区为 `/home/sumi/experience/omnimap`，代码通过 Git 同步到算力端；部署路径可不同。
执行端见 [sumi 升级](ur5_nbv_kinematics_upgrade_sumi.md)，模式参数、日志和测试命令见
[本机验收](ur5_nbv_kinematics_validation_local.md)。

## 已实现

| 源码 | 职责 |
|---|---|
| `ros_ws/src/omnimap_msgs/msg/KinematicStatus.msg` | 独立 14 字段反馈，消息已生成 |
| `info_flow/info_flow_planning_node.py` | 接收反馈，发布原始 Fisher 命令和最终命令 |
| `info_flow/kinematic_feedback.py` | 双时钟超时、frame/指标/状态校验、持续奇异减速及停车 |
| `info_flow/kinematic_projection.py` | 二维最小半空间投影；尚未接入控制 |
| `info_flow/info_flow_servo_runtime.py` | 生成相机参考点 Twist，发布实际命令追踪，失效零速 |
| `ros_ws/src/omnimap_kinematics` | 执行侧桥接与 MoveIt RobotState monitor |

Planning 不加载 RobotModel，不从 status 码推导 Jacobian 或修正方向。
默认 observe 记录反馈、保留 Fisher 输出；enforce 才启用状态干预：code 1 持续 0.1 秒后默认缩放至 0.5，
code 2–5、未知码或失效反馈停车。原始命令独立发布到 `/omnimap/spherical_cmd_raw`，可回放。
反馈在 Fisher 计算完成后重新读取；原始/最终命令保留相同 tick。

## 坐标和指标契约

Servo 在 world_frame 计算速度，要求 cmd_frame 相同；UR5 链路均为 base_link 表达。
`/servo_server/delta_twist_camera` 以 cam_1_color_optical_frame 为参考点，桥接输出
`/servo_server/delta_twist_cmds` 以 tool0 为参考点，仅做一次：

```text
v_T^B = v_C^B - omega^B × (p_C^B - p_T^B)
omega_T^B = omega_C^B
```

Planning/Servo 不重复参考点变换；没有使用 cam_1_baselink。
monitor 使用实际 Servo 模型、group、关节顺序及 tool0 Jacobian，计算
`Jbar = inverse(Dx) J_B Dq`、`m = sigma_min(Jbar)`。
反馈 condition 已归一化；日志 raw_condition 用于比较下游原始条件数，17/30 不能直接用于归一化阈值。

## 连续修正仍需完成

目标仍为两个球面速率上的 `min 0.5 ||u-u_info||², subject to a.T u >= c`。
投影函数已拒绝非有限输入和不可用修正方向，但当前消息没有 a/c。
尚需执行侧按同一模型、参考点和速度映射生成并验证 a/c、现场标定 m_min，再接入控制。
状态减速不能替代这一目标，status code 不能推导 a。
无效/不可行方向必须零速或重新规划，不复用旧方向；加入速度等额外约束后需联合求解，
投影后任意裁剪可能破坏半空间约束。

## 构建和完整验收

```bash
conda activate InfoFlow
./scripts/build_ros_ws.sh
source /opt/ros/noetic/setup.bash
source ros_ws/devel/setup.bash
python tests/test_kinematic_feedback.py
python tests/test_kinematic_runtime.py
```

计算端默认仅构建消息，执行端使用 --with-kinematics；消息修改后两端重建并 source。
完整升级仍须满足：

- 同 tick 的原始/最终命令、实际 Twist、桥接输入/输出可关联，joint/status 按时间关联。
- sigma/condition 的模型、参考点、frame 和关节顺序与实际 Servo 一致。
- 区分奇异减速/急停、collision、joint bound；保留原始 Fisher 独立回放能力。
- 无效方向、TF/关节/status 或桥接失效时反馈无效、enforce 零速，执行桥接另有 watchdog。
- 标定和连续投影接入后验证约束、恢复及实际运动效果。

Int8 status 没有源时间戳，不能声称与 Twist 属于同一 MoveIt 内部周期。
2026-10-25 已有真实关节状态零速诊断；持续对齐、减速标定、硬件断链和非零闭环仍未验收。
