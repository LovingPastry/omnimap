# OmniMap 计算端运动学反馈升级

更新：2026-10-26。本文按当前源码维护，替代 2026-10-22 的实施前清单。
当前工作区为 `/home/sumi/experience/omnimap`，代码通过 Git 同步到算力端；部署路径可不同。
执行端见 [sumi 升级](ur5_nbv_kinematics_upgrade_sumi.md)，模式参数、日志和测试命令见
[本机验收](ur5_nbv_kinematics_validation_local.md)。

## 已实现

| 源码 | 职责 |
|---|---|
| `ros_ws/src/omnimap_msgs/msg/KinematicStatus.msg` | 独立 14 字段反馈，消息已生成 |
| `info_flow/info_flow_planning_node.py` | 接收状态与约束反馈，记录投影建议，发布原始 Fisher 命令和最终命令 |
| `info_flow/kinematic_feedback.py` | 双时钟超时、frame/指标/状态校验、持续奇异减速及停车 |
| `info_flow/kinematic_constraint.py` | 关联最近执行命令，验证约束来源/新鲜度/参考球/局部范围，计算只读投影建议 |
| `info_flow/kinematic_projection.py` | 二维半空间、线性限值和仿射范数限值的联合最小投影，以及 Servo 局部速度模型；尚未接入控制 |
| `info_flow/info_flow_servo_runtime.py` | 生成相机参考点 Twist，发布实际命令追踪，失效零速 |
| `ros_ws/src/omnimap_kinematics` | 执行侧桥接、MoveIt RobotState monitor 与二维构型导数诊断 |

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
投影函数已拒绝非有限输入和不可用修正方向。新增独立 KinematicConstraint 消息，
执行侧按实际 Servo 映射和同一 MoveIt 模型生成 a、drift；显式配置 m_min 后才生成有限 c。
接口、尺度、失效条件见 [二维构型约束诊断](ur5_nbv_kinematic_constraint.md)。
Planning 已订阅该接口，默认 `--kinematic_projection_mode observe` 只计算/记录建议，不修改命令；
`off` 可关闭建议计算。消费者的双时钟、命令来源、参考球及局部范围检查已有回调测试。
联合求解器已支持线性关节限值、仿射线速度/每周期速度变化限值和局部球；Planning 当前接入半空间、局部球与实际 RobotModel 关节速度上下限。
执行侧通过 KinematicConstraint 发布带关节名称的 qdot=A u+bq，以及源 Servo 周期的相机线速度、速度变化限值和死区。
Planning 已将这些限值联合求解。仍需执行时重新校验、制动距离与实际关节加速度验证、现场标定、停车后的模型恢复和非零闭环验收，再接入控制。
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
python tests/test_kinematic_constraint.py
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

2026-10-26 再次只读检查本机默认 ROS master 不可访问，现场数据门槛仍未完成。
这不影响代码与隔离回调验证，不能据此假定实机链路已经运行。

### 源周期关节位置与加速度边界（2026-10-26）

ServoTwistTrace 传输实际 `servo_dt_sec`；KinematicConstraint 传输对应关节样本的
`joint_position`、`joint_velocity`，以及 RobotModel 的位置/加速度上下限和各关节 bounded 标记。
Planning 对有界关节联合求解 `q_min <= q + dt*(A*u+bq) <= q_max`，以及
`dq + a_min*dt <= A*u+bq <= dq + a_max*dt`。模型未配置的加速度上限不填猜测值；
日志 `acceleration_limits_complete` 明确标记是否六个关节均有上限。
这是一周期 Euler 预测和相对测量速度的约束，不证明真实执行加速度或制动距离安全；
源关节样本与 Twist 允许的小量时间偏差也仍需在执行重校验中处理。
两条消息 MD5 均改变，算力端和执行端必须重新构建并重启相关节点。

已补充 RobotModel 集成覆盖：隔离 `kinematics.test` 仅为 j1/j3 设置不同加速度上限，
验证六关节名称顺序、正负上限和缺失标记，以及乱序 JointState 的位置/非零速度传输；
生产 launch 测试保留完全未配置加速度上限的路径。两套各 9 项通过，均为合成模型证据。

### 停车后模型查询基础接口（2026-10-26）

monitor 新增私有只读服务 `~evaluate_model`，类型 `omnimap_msgs/EvaluateKinematicModel`。
默认节点名下路径为 `/omnimap_kinematic_monitor/evaluate_model`。请求携带
`reason=preview` 的候选 ServoTwistTrace；其中 command 是候选重建值，不表示已发布或已执行。
服务从新鲜关节历史选择与候选时间相近且 TF 覆盖的样本，检查实际 MoveIt 状态、模型/TF、
映射重建、导数和可用限值，返回标记 `advisory_preview:` 的 KinematicConstraint。
查询不依赖旧 BridgeStatus，不向任何控制/反馈 topic 发布；失效在响应中显式标记。
这补齐了停车后不借助先行运动即可重新获取模型的接口；Servo 尚未生成/调用候选预览，
自动恢复与执行前重校验仍未接入。调用结果不能直接作为执行许可。
新增服务需两端重建消息包并重启 monitor；本次既有消息字段及 MD5 不变。
