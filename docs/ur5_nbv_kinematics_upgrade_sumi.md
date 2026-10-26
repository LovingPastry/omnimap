# UR5 执行端升级：桥接与 MoveIt 运动学反馈

更新：2026-10-26。当前主机就是 sumi，允许修改 `~/ros/catkin_ws/src`。
Git 管理的实现来源是仓库 `ros_ws/src/omnimap_kinematics`，部署脚本复用到机器人工作区。
旧的“仅 SSH 只读检查”限制不再适用。详细参数和证据见 [sumi 验收](ur5_nbv_kinematics_validation_sumi.md)。

## 实现状态

| 交付项 | 当前状态 |
|---|---|
| 普通/KF 桥接 | 已实现参考点变换、BridgeStatus、状态停车、墙钟 watchdog |
| MoveIt C++ monitor | 已实现实际模型/group/限速、SVD、原始及归一化 condition、实测速度诊断 |
| 命令关联 | ServoTwistTrace 按序号、时间、frame、数值关联 tick；无匹配标记 tick_unavailable |
| 部署 | 包链接、旧桥接入口备份/替换、运行依赖补齐，整机默认启动不变 |
| 启动恢复 | 等待 Servo 参数；robot_description_param 指向全局 URDF，避免私有参数遮蔽 |
| 采样对齐 | 有界关节历史，选择 TF 已覆盖且与当前命令相近的新鲜样本；不修改 Twist |
| 二维导数接口 | 已发布 KinematicConstraint；Planning 已消费并记录只读投影建议，尚未控制执行 |
| m_min 和二维修正 | 尚未完成现场标定或控制接入，不能宣称全计划完成 |

## frame、模型与时间契约

```text
OmniMap Servo → /servo_server/delta_twist_camera
  → camera_twist_bridge.py（none 或 kalman）
  → /servo_server/delta_twist_cmds → MoveIt Servo
```

输入输出均在 base_link 表达，相机参考点为 cam_1_color_optical_frame，末端为 tool0。
桥接仅做 `v_T = v_C - omega × (p_C-p_T)`，不改成 tool0 frame，不重复变换。
两处 TF 使用输入 Twist 采样时间，KF 只滤波变换后速度，零输入/失效清除滤波残留。
不要同时启动普通、KF 和 launch 内的多个桥接 publisher。

monitor 从实际 Servo 参数读取模型相关配置与 topics，使用 RobotModel、SRDF、joint-limit overrides。
缺失 Servo 参数时等待；模型/配置不兼容时失败。关节按名称装配，tool0 原点 Jacobian 转至 base_link。
`Jbar = inverse(Dx) J_B Dq`：Dq 来自模型速度上限，Dx 默认为
`[0.10,0.10,0.10,0.15,0.15,0.15]`，前三项 m/s、后三项 rad/s。
归一化 condition 不共用原始条件数阈值，实际 Servo 阈值以运行参数为准。

monitor 最多缓存 200 个关节样本，并限制接收年龄；选取 TF 覆盖、新鲜度和命令时间差均合格的最新样本。
RobotState 与 TF 验证使用同一选中时间，日志同时记录 latest_joint_stamp 和 joint_stamp。
最新关节流仍必须有效；畸形或重复/倒退时间样本清空历史，断流仍按墙钟超时。
无法选取时反馈无效，不拿最新 TF 冒充同时间 TF，不放宽模型一致性检查。

## 构建、部署与验收

```bash
./scripts/build_ros_ws.sh --with-kinematics
python3 scripts/deploy_sumi_kinematics.py
source /opt/ros/noetic/setup.bash
source ~/ros/catkin_ws/devel/setup.bash
source ros_ws/devel/setup.bash --extend
roslaunch omnimap_kinematics diagnostics.launch start_bridge:=true
```

已有驱动但没有 Servo 时，最后命令增加 start_servo:=true，复用标准 ur5_servo.launch。
此选项适用于默认 Servo 命名空间，不启动驱动或切换控制器；只运行 monitor 则省略 start_bridge。
自定义命名空间用 servo_ns，状态 topic 解析实际 Servo 配置，可用 status_topic 显式覆盖。

```bash
python3 scripts/collect_sumi_kinematics.py
rostest omnimap_kinematics kinematics.test use_sim_time:=true
rostest omnimap_kinematics diagnostics_launch.test
```

集成测试使用隔离 master、合成模型和 /test/* topics。
现场须另录 SphericalCommand/raw、ServoTwistTrace、BridgeStatus、KinematicStatus、
桥接输入/输出、MoveIt status、joint_states 和 TF；完整命令见验收文档。
Int8 status 只有接收时间，不能证明同一内部计算周期。

## 剩余门槛

1. 连续诊断对齐与真实模型/标定/关节顺序验证。2026-10-25 零速探测 60 条反馈中 10 条有效，尚不稳定。
2. 记录真实减速时 raw condition、sigma_min 和 status 的关系，标定 m_min；零速日志不足以标定。
3. 执行侧已生成二维导数诊断；消费者已联合求解半空间、局部球、RobotModel 关节速度界限及源 Servo 周期的相机线速度/速度变化/死区约束。消息包含带关节名称的 qdot=A u+bq、模型上下限及相机参考点映射；仍需执行时重新校验、关节加速度/位置可行性处理及现场映射验证，再接入控制，保留原始命令和失效零速。
4. 硬件断链、恢复和非零闭环验收；离线测试不能代替这些证据。

桥接 code 0/1 允许有效输入，2–5 及未知码停车。
joint_states 失效由 monitor 标记，Planning enforce 与 MoveIt 承担相应停车；桥接不拥有 RobotState。
Planning 默认仍为 observe，尚未按未标定指标启用连续修正。

二维接口的数值定义、参数、消息兼容性和剩余控制门槛见 [构型约束诊断](ur5_nbv_kinematic_constraint.md)。
