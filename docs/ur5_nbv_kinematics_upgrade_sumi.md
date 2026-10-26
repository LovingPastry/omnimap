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
3. 执行侧已生成二维导数诊断；消费者已联合求解半空间、局部球、RobotModel 关节速度界限及源 Servo 周期的相机线速度/速度变化/死区约束。消息包含带关节名称的 qdot=A u+bq、模型上下限及相机参考点映射；仍需执行时重新校验、制动距离与实际关节加速度验证及现场映射验证，再接入控制，保留原始命令和失效零速。
4. 硬件断链、恢复和非零闭环验收；离线测试不能代替这些证据。

桥接 code 0/1 允许有效输入，2–5 及未知码停车。
joint_states 失效由 monitor 标记，Planning enforce 与 MoveIt 承担相应停车；桥接不拥有 RobotState。
Planning 默认仍为 observe，尚未按未标定指标启用连续修正。

二维接口的数值定义、参数、消息兼容性和剩余控制门槛见 [构型约束诊断](ur5_nbv_kinematic_constraint.md)。

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
这补齐了停车后不借助先行运动即可重新获取模型的接口；Servo 现已具备无副作用候选计算、显式查询及可选 observe 调度；
自动恢复与执行前重校验仍未接入。调用结果不能直接作为执行许可。
新增服务需两端重建消息包并重启 monitor；本次既有消息字段及 MD5 不变。

### Servo 候选计算与显式查询（2026-10-26）

`_compute_servo_candidate` 与实际控制共用同一计算，实现径向 PI 抗饱和、角速度、
速度/slew/deadband 及映射构造；返回候选积分状态，实际周期才提交积分及上一命令。
`evaluate_preview(cmd, pose, pose_stamp, dt, query)` 检查来源时间/frame，构造 preview trace，
调用传入的 EvaluateKinematicModel service proxy；不会发布命令或追踪，也不会推进积分、
slew 或周期时钟。控制锁保护候选计算与实际周期之间的状态读取，RPC 在锁外执行。
方法可用于停车命令的只读评估；可选 observe 调度见后文，尚无运动恢复入口；
响应已检查候选关联、返回年龄、控制器周期变化和返回后的最新 TF；实际周期建议重算及最新关节复核见后文。

### 预览响应关联与周期失效（2026-10-26）

`evaluate_preview` 对请求命令作独立拷贝，检查返回的 preview 标记、有效性、frame、
tick/model、command/twist 时间与序号、参考球、速率、相机映射及所有源周期线速度限值。
响应 header/joint/twist/command 时间均须新鲜；超过 pose_stale_timeout_sec 才返回的 RPC 结果被丢弃。
查询期间控制器周期编号或 PI/上一线角速度发生变化时拒绝结果，包括数值相同的新周期。
这只是返回结果的关联和失效检查：RPC 等待现有调用端截止时间，但底层阻塞操作不能强制取消；
查询后最新 TF 已复核，最新关节已增加本机冻结状态复核，实际周期 dt 现已用于重建建议约束，尚未执行投影或恢复运动。
相关回调测试现为 14 项通过。

### 有界查询等待和返回后 TF 复核（2026-10-26）

`--kinematic_preview_timeout_sec` 默认 0.1 秒，须有限且大于零；实际等待预算还受
pose_stale_timeout_sec 限制。`SingleFlightQuery` 在单个 daemon worker 中执行只读 RPC；
调用端到期抛 TimeoutError。底层 RPC 不能强制取消，超时 worker 返回前拒绝新请求，
不创建替代 worker，晚到结果不复用。RPC 异常原样传递，实际返回后才释放占用。
返回后立即读取最新 TF（零等待），要求时间新鲜且与请求位姿在 1e-9 数值容差内一致；
这适用于冻结状态的预览，不能当作运动中模型误差的硬件安全裕量。
RPC 等待期间不持有控制锁；候选计算和末尾控制器状态比较持锁。命令/位姿/请求均独立拷贝。
新增 2 项有界查询测试及 15 项回调测试通过。执行 dt 复核、
连续投影执行和恢复运动仍未接入。

### 本机关节流复核（2026-10-26）

Servo 以 `--kinematic_joint_topic`（默认 `/joint_states`，部署时须与实际 Servo/monitor
配置一致）订阅本机关节状态，只用于只读预览校验。响应返回后，最新样本必须在 ROS 和
接收墙钟两种年龄上均新鲜，时间不早于模型选用样本，六关节按名称与响应 q/dq 对齐。
位置和速度要求在 1e-9 数值容差内一致；这是冻结状态检查，不是允许运动误差的安全裕量。
畸形、重复/倒退时间样本使旧好样本失效；更新且合格的样本才恢复检查。
本次关节状态 5 项、实际回调 16 项测试通过，包括 TF 不变但关节变化的拒绝。
本机样本复核仍是返回后的单次观察，不能证明后续执行期间状态不变；
实际周期建议重算已接入，投影执行与恢复仍未接入。

未来时间的本机关节样本在接收时拒绝，不推进已接受时间基线；后续正常新样本可恢复，相关测试已覆盖。

### 自动只读 observe 调度（2026-10-26）

Servo 参数 `--kinematic_preview_mode observe` 显式启用独立 ROS Timer（默认 off，
`--kinematic_preview_hz 2`）；`--kinematic_preview_service` 默认
`/omnimap_kinematic_monitor/evaluate_model`。调度订阅 `--spherical_cmd_raw_topic`
（默认 `/omnimap/spherical_cmd_raw`，须与最终命令 topic 不同）。
查询前后均核对原始/最终命令的 tick、model、来源时间/frame、参考球和接收年龄，
原始策略 stop 禁止查询；最终状态反馈 stop 可做只读评估。原始/最终 Header.seq
由不同 publisher 分配，不用于二者关联。查询期间任一命令变化，结果即失效。
结果只存为 `latest_preview_constraint` 并写 kinematic_preview 日志，不发布任何控制/追踪消息。
每次调度先清除旧建议，错误或超时不会保留上次结果。调度不推进 Servo 周期时钟。
预览 dt 使用 1/servo_hz，日志标记 `dt_source=nominal`，不代表测得的未来执行周期。
同样保留查询期间 Servo 周期变化失效检查，因此查询跨过控制周期时会被拒绝；
当前是冻结状态诊断，并不保证运动中持续取得可用模型。18 项回调测试通过。

```bash
python info_flow/info_flow_servo_runtime.py --kinematic_preview_mode observe \
  --kinematic_preview_hz 2 --kinematic_preview_timeout_sec 0.1 \
  --kinematic_joint_topic /joint_states
```

命令是部署选项，未在实机执行；实际 joint topic 应与 Servo/monitor 配置一致。
实际周期建议重算见后文；下一步仍需投影执行、恢复状态机和现场标定验收。

### 实际周期建议重算（2026-10-26）

observe 调度保留带来源命令、完整相机仿射映射和冻结位姿的 PreviewEvaluation。
实际 Servo 周期以本周期有效 dt（含现有启动默认/上下限处理）重算 PI、角速度与限值，
复核 raw/final 来源、本机关节样本和位姿。完整仿射映射变化时拒绝旧模型，要求重新查询；
仅 dt/上一线速度限值变化时，重建位置、关节加速度、相机 speed/slew/deadband 约束，
联合求解并再次核对投影后的实际计算输出没有被限幅改变。
KinematicConstraintGate 显式区分 preview 与 execution 来源，拒绝角色混用。
数值求解使用独立控制器快照及单个后台 worker，调用端等待预算默认
`--kinematic_cycle_projection_budget_sec 0.003`，并受名义周期四分之一上限约束。
超时仍占用唯一 worker，晚结果丢弃，不创建替代求解；每个实际周期先清空旧建议。
运动分支只记录建议，原输出不变；停车分支先发布零速/清理状态，再做零等待 TF 和有界诊断，
不会借诊断恢复运动。日志为 `kinematic_cycle_projection ... dt_source=actual`。
本阶段消费者 16 项、回调 21 项通过，包括 dt 变短后加速度约束改变建议速率、
PI 映射变化拒绝、运动输出不变、停车保持零速和阻塞求解无替代任务。
这仍是冻结模型的建议重算；原始候选处于非仿射限幅分支时仍拒绝查询/使用，
连续投影执行、完整恢复状态机、现场标定及实机验收仍未完成。
