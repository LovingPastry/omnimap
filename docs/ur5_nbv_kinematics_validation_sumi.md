# sumi 运动学诊断：实现、部署与验收

更新：2026-10-25。对应 [上位机方案](ur5_nbv_kinematics_upgrade_sumi.md)。

## 2026-10-25 实机 ROS 启动修复

本次现场 master 和机器人驱动已运行，整机使用的是默认 `launch_robot.launch`，未启用 Servo。
修复了两处启动问题：

1. 缺少 `/servo_server` 参数时，monitor 现在等待参数就绪，不再以 required process 失败结束整个 launch。
2. 原 launch 创建了私有 `robot_description="robot_description"`，MoveIt 的参数搜索误把这个字符串当作 URDF。
   改用独立的 `robot_description_param` 指向全局 `/robot_description`，避免遮蔽真实模型。

整机没有启动 Servo 时，诊断终端使用：

```bash
roslaunch omnimap_kinematics diagnostics.launch start_bridge:=true start_servo:=true
```

若整机已使用 `start_servo:=true` 启动，则诊断终端仍使用 `start_bridge:=true` 即可，不重复启动 Servo。
`start_servo` 选项复用 `ur5_moveit_config/launch/ur5_servo.launch`，启动标准 `/servo_server`；
该选项需要 source 机器人 catkin workspace，仅适用于默认 Servo 命名空间。
只启动诊断时不启动机器人驱动，也不切换控制器。

本次实际启动后 `/servo_server`、桥接、monitor 均存活，完整启动日志无 ERROR/FATAL/异常退出。
读取了当前 UR5 模型和真实 joint_states，纯零速输入探测收到 60 条反馈，其中 10 条链有效；
有效样本示例为 `sigma_min=2.37855`、归一化 condition `16.917`、MoveIt status `0`。
仍存在 joint_states 比 TF 更新更快导致的采样时刻外插等待，不能把这轮称为连续诊断已稳定。
没有发送非零命令、切换控制器或执行机械臂运动，也没有据此标定 `m_min`。
测试结束后，本次启动的 Servo、桥接、monitor 和零速探测节点均已退出，原整机节点保留。
现场日志：`/tmp/omnimap_diagnostics_live_test.log`。

新增 `rostest omnimap_kinematics diagnostics_launch.test`，直接覆盖正式 launch 的模型参数接线，
并验证 Servo 参数延迟到达及 `/clock` 未启动时 monitor 仍能存活和恢复。6 项测试通过。

## 已落地的代码

实现放在仓库的 `ros_ws/src/omnimap_kinematics`，供 sumi 的现有工作区复用：

| 文件 | 行为 |
|---|---|
| `scripts/camera_twist_bridge.py` | base 表达下的 camera → tool0 参考点变换、普通/KF 两种模式、墙钟 watchdog |
| `src/omnimap_kinematic_monitor.cpp` | MoveIt RobotModel/RobotState、按名称装配关节状态、归一化 Jacobian 的 SVD；只发布诊断 |
| `config/kinematics.yaml` | 任务速度尺度、样本超时、时间对齐容差和发布频率 |
| `launch/diagnostics.launch` | 独立 monitor；`start_bridge:=true` 启动升级桥接，`start_servo:=true` 可选启动标准 UR5 Servo；不启动驱动 |
| `omnimap_msgs/msg/BridgeStatus.msg` | 每次桥接输出的完整输入/输出 Twist、topic、参考点、TF 年龄、status 和失效原因 |
| `omnimap_msgs/msg/ServoTwistTrace.msg` | 实际 Twist 的序号/采样时间、Planning tick、模型版本及命令时间 |

Servo runtime 默认向 `/omnimap/servo_twist_trace` 发布追踪消息，可用 `--twist_trace_topic` 修改。
控制 Twist 的 six components 和 TF 采样时间保持原语义。ROS 给顶层 Header 分配 `seq`，
因此 tick 有独立字段；嵌套的 `command.header.seq` 保留实际发送的 Twist 序号。
monitor 同时按 Twist 的序号、采样时间、frame 和 six components 关联 tick。
未匹配到追踪时标为 `tick_unavailable`，不会把最新 Planning tick 当作本条 Twist 的来源。

桥接继续只计算一次：

```text
v_T^B = v_C^B − ω^B × (p_C^B − p_T^B)
ω_T^B = ω_C^B
```

两条 TF 都查询输入 Twist 的采样时刻。TF 缺失立即输出零速，下一周期重试；
查询不使用依赖 ROS 时间的等待，以免 `/clock` 暂停后 watchdog 卡住。

## 指标与有效性

monitor 从实际 Servo 参数命名空间读取 `move_group_name`、`planning_frame`、
`robot_link_command_frame`、`ee_frame_name`、joint/status/Cartesian input topics。
缺失 Servo 参数时等待；command/planning frame 不一致、非六自由度串联链、group 末 link 与 EE 不一致时启动失败。
机器人描述默认使用相同的 `robot_description`，含 SRDF 和 planning joint-limit overrides。
不会加载另一份 UR5 DH 参数，也不会从 status 码推导 sigma。

```text
J     = RobotState.getJacobian(group, tool0, reference_point=0)
J_B   = 将 group-root-parent 表达的 J 的 linear/angular 两块旋转至 base_link
Jbar  = inverse(Dx) J_B Dq
m     = sigma_min(Jbar)
kappa = sigma_max(Jbar) / sigma_min(Jbar)
```

`Dq` 使用 RobotModel 逐关节最大速度，列顺序来自 group variable names。
`Dx` 默认对角元为 `[0.10, 0.10, 0.10, 0.15, 0.15, 0.15]`，单位分别为 m/s、rad/s；
可用 `task_velocity_scales` 配置六个正有限值。启动日志记录 joint order、Dx、Dq、模型/group/frame。
反馈的 condition 是归一化结果；日志 `raw_condition` 来自未缩放的同一个 J，供比较 MoveIt 的阈值。
不能把 17/30 直接用作归一化指标阈值。
`command_speed_norm=||inverse(Dx) twist_tool||`，
`actual_speed_norm=||inverse(Dx) J_B qdot||`，后者来自实测关节速度。
最小奇异值小于等于 `1e-12` 时报告 0，condition 为正无穷。

每周期检查 joint/status/bridge 的源时间与接收墙钟时间，默认超时 0.3 s；
joint 与桥接输入采样时间差、两处 status 接收时间差默认最多 0.1 s。
`Int8` status 没有源时间戳，记录的是接收时间，不能声称与某条 Twist 属于同一个 MoveIt 计算周期。
group 及上游可动关节缺失、重复名称、非有限数据或缺少关节速度均使反馈无效。
`tool0` TF 与同一关节样本的 RobotState 位姿须一致，默认容差为 0.01 m / 0.03 rad。
相机 optical frame 可由手眼标定 TF 提供，不要求嵌入 URDF；存在对应模型 link 时也检查位姿。
TF 年龄是查询样本的年龄，静态 TF 记为 0；它本身不证明硬件采集或手眼标定正确。

| 桥接条件 | 输出 |
|---|---|
| 有效新鲜输入，code 0/1 | 普通模式保持原公式；KF 模式只滤波转换后的 six components |
| code 2/3/4/5 或未知码 | 零速，记录状态名/数值，重置滤波状态 |
| 输入缺失/超时、空或未来时间戳、frame 错误、NaN/Inf | 零速，不复用旧运动 |
| TF 不可用/过期、status 缺失/超时 | 零速，记录原因 |
| 明确零输入、退出节点 | 直接零速，绕过滤波残留 |

桥接默认 `publish_hz=50`、`stale_timeout=0.3`、`status_timeout=0.3`。
`filter_mode` 允许 `none` / `kalman`；KF 线/角 process noise 默认 `1e-4`、measurement noise 为 `1e-3`。
频率、超时、尺度和噪声要求正有限值。桥接用 monotonic 墙钟，monitor 用 WallTimer 和 SteadyTime。
joint_states 断流时 monitor 报无效；桥接不拥有 RobotState，该条件的停车仍由 Planning enforce 和 MoveIt 负责。

## 部署与启动

执行端完整构建需要 ROS Noetic 的 `moveit_core`、`moveit_ros_planning` development packages 和 Eigen。
构建脚本默认仍只构建消息，计算端不必安装 MoveIt；执行端显式启用 monitor：

```bash
cd /home/sumi/experience/omnimap
./scripts/build_ros_ws.sh --with-kinematics
python3 scripts/deploy_sumi_kinematics.py
source /opt/ros/noetic/setup.bash
source ~/ros/catkin_ws/devel/setup.bash
source ros_ws/devel/setup.bash --extend
```

部署把两个仓库内 ROS 包链接到 `~/ros/catkin_ws/src`，将现有
`robot_else_nodes/scripts/camera_twist_bridge.py` / `camera_twist_bridge_kf.py`
改成仓库实现的入口，分别默认 `none` / `kalman`，私有参数仍可覆盖。
同时为现有包的 `package.xml` 补充 `omnimap_msgs`、`omnimap_kinematics` 运行依赖。
替换前保存 `.before_omnimap_<UTC时间>` 原文件；重复执行不修改相同入口，同名的其他 package 会拒绝覆盖。
没有启动节点，`launch_robot.launch` 的默认行为未改动。本次实际备份后缀为 `20261024T094638984386Z`。
`package.xml` 的备份后缀为 `20261024T095835143359Z`。

驱动、MoveIt Servo、TF 和 joint_states 使用现场已有启动流程。仅启动 monitor：

```bash
roslaunch omnimap_kinematics diagnostics.launch
```

桥接未启动时 monitor 持续报无效。确认输出 topic 没有其他桥接 publisher 后，启动本套桥接与 monitor：

```bash
roslaunch omnimap_kinematics diagnostics.launch start_bridge:=true
# KF 替代方式：在上面的命令增加 filter_mode:=kalman
```

不要同时启动普通桥接、KF 桥接和 launch 内的桥接。Servo 参数命名空间变化时传
`servo_ns:=/实际命名空间`；输入/输出 topic 用 `input_topic`、`output_topic` 配置。
桥接默认从该 Servo 命名空间的 `status_topic` 参数读取实际状态话题，配置中的相对名称也在该命名空间解析；
只有参数尚不存在时才回退为 `<servo_ns>/status`。可用 `status_topic:=/实际话题` 显式覆盖。
桥接启动日志包含最终订阅的状态话题。
Planning 保持 `--kinematic_feedback_mode observe`，真实诊断与标定完成后再启用 enforce。
本次没有猜测 `m_min`，也没有启用二维半空间控制修正。

## 复核与本次证据

```bash
conda activate InfoFlow
./scripts/build_ros_ws.sh --with-kinematics
source /opt/ros/noetic/setup.bash
source ros_ws/devel/setup.bash
python -m py_compile info_flow/info_flow_servo_runtime.py \
  ros_ws/src/omnimap_kinematics/scripts/camera_twist_bridge.py scripts/*sumi_kinematics.py
python tests/test_kinematic_feedback.py
python tests/test_kinematic_runtime.py
python tests/test_camera_twist_bridge.py
python tests/test_sumi_deploy.py
python tests/test_sumi_inventory.py
# rostest 自建隔离 master，只使用 /test/* 控制话题，不启动机器人驱动。
rostest omnimap_kinematics kinematics.test use_sim_time:=true
```

本次：消息与 C++ 构建通过；InfoFlow 数值/反馈 8 项、Planning/Servo 回调 9 项、桥接失效/配置测试 7 项、部署保留/幂等测试 3 项、有效配置采集测试 1 项通过；
真实 ROS 桥接 + MoveIt monitor 的隔离集成测试 6 项通过。
集成测试以独立 NumPy FK 有限差分校验 SVD、归一化、实测速度、坐标系和参考点，
覆盖关节倒序、外部相机 TF、奇异构型、状态码、失效停车/恢复，以及暂停 `/clock` 后的墙钟 watchdog。
状态话题使用自定义命名空间中的相对名称，验证桥接与 monitor 都遵循实际配置。
回调测试隔离地图/Fisher/GPU/绘图模块，不验证 Fisher 目标函数或渲染效果。

以下命令只读现场状态、保存 inventory，并记录对齐数据：

```bash
python3 scripts/collect_sumi_kinematics.py
rosrun tf tf_echo base_link cam_1_color_optical_frame
rosrun tf tf_echo base_link tool0
rostopic hz /joint_states
rostopic echo -n 5 /servo_server/status
rostopic echo -n 5 /omnimap/bridge_status
rostopic echo -n 5 /omnimap/kinematic_status
rosbag record -O kinematic_chain.bag /omnimap/spherical_cmd_raw /omnimap/spherical_cmd \
  /omnimap/servo_twist_trace /omnimap/bridge_status /omnimap/kinematic_status /omnimap/kinematic_constraint \
  /servo_server/delta_twist_camera /servo_server/delta_twist_cmds \
  /servo_server/status /joint_states /tf /tf_static
```

采集脚本的 `--servo-ns` 同样控制状态、关节和 Servo 输出话题的配置读取与相对名称解析；
camera 输入可用 `--input-topic` 指定。若 master 在线但 Servo 参数缺失，记录 `CONFIG_UNVERIFIED` 并返回 2，
不会用猜测的 `/servo_server/status` 替代实际配置。

本机默认 master 的只读 inventory 位于 `/home/sumi/omnimap_kinematic_inventory_20261024_174813.txt`；
仓库配置 `http://172.19.93.154:11311` 的复核位于 `/home/sumi/omnimap_kinematic_inventory_20261024_175526.txt`。
以下为 2026-10-24 历史记录，master 已于次日恢复。两次 `rosnode list` 均返回 `Unable to communicate with master`，inventory 以退出码 2 标记运行时未验证。
18:03 的再次复核保存于 `/home/sumi/omnimap_kinematic_inventory_20261024_180336.txt`，结果相同。
虽仍有 Servo 进程存活，但其进程环境指向的也是上述不可访问 master，不能以进程存在替代运行链路证据。
隔离测试不构成真实机器人验收。实际 topic 连通性、模型/标定对齐、开始减速时 raw condition 与 sigma_min 的关系、
`m_min` 标定、硬件断链停车仍待现场数据。完成诊断门槛后才进入方案第 5 步的二维半空间控制修正。

| 方案步骤 | 当前证据和剩余门槛 |
|---|---|
| 1. status/frame/速度记录 | BridgeStatus、ServoTwistTrace 和 monitor 日志已实现；有效命令数值由参考点公式测试校验 |
| 2. TF 时间和参考点平移项 | 同采样时间查询及有限差分/刚体速度测试通过；真实 TF 年龄与标定仍待现场记录 |
| 3. MoveIt 诊断节点 | C++ 节点、构建、部署和真实 RobotState 隔离测试完成；已有真实模型/joint_states 零速样本，持续运行验收待完成 |
| 4. 标定 m_min / 比较减速时机 | raw/normalized condition、sigma_min、status 的对齐接口已就绪；需要真实减速日志 |
| 5. 二维最小半空间控制修正 | 执行侧局部映射和 a/drift/c 诊断已实现，控制尚未接入；已实现只读建议的联合限值/时效约束，仍需步骤 4 标定及连续执行/实际残差验证 |
| 6. stale/TF/status 零速策略 | 普通/KF 桥接 watchdog、状态停车和失效恢复测试通过；硬件断链验收待完成 |

## 2026-10-25 关节与 TF 延迟对齐修复

monitor 不再只查询最新 joint_states 时刻：缓存最多 200 个且未超过 sample_timeout 的接收样本，
选择两条 TF 均已覆盖、源时间/接收时间新鲜且符合 max_sample_skew 的最新样本。
该样本同时驱动 RobotState、实测速度和 TF 模型一致性检查；日志区分 latest_joint_stamp 与 joint_stamp。
最新关节消息仍必须有效，畸形或时间倒退/重复输入清空历史；关节断流、TF 缺失或历史过期仍反馈无效。
此改动不修改桥接 Twist，不放宽 TF/模型误差容差，也不证明现场持续对齐已验收。

本轮验证：`./scripts/build_ros_ws.sh --with-kinematics` 成功；隔离 ROS 集成测试 7 项全部通过，
新增 TF 持续落后 80 ms 的有效反馈与关节断流失效/恢复检查。日志在 `/tmp/omnimap_alignment_test.log`。

## 二维导数接口（2026-10-25）

新增 `/omnimap/kinematic_constraint`，详细数值定义见 [构型约束诊断](ur5_nbv_kinematic_constraint.md)。
ServoTwistTrace 的消息 MD5 已改变，两端须重建消息并重启相关节点。
当前不发布修正控制命令；阈值默认未配置，c=NaN，不以测试阈值代替实机标定。

本轮验证：消息/C++ 构建通过；InfoFlow 数值与模型测试 11 项、Planning/Servo 回调 9 项、
inventory 1 项通过；直接节点和正式 diagnostics.launch 两套隔离 ROS 测试各 9 项通过。
独立 NumPy FK 有限差分核对 a/drift；显式测试阈值只在隔离合成模型中设置，正式 launch 验证未配置时 c 为 NaN。
日志：`/tmp/omnimap_constraint_test.log`、`/tmp/omnimap_constraint_launch_test.log`。

## 2026-10-26 自动预览链路验证

新增 `rostest omnimap_kinematics servo_preview.test`，真实运行 Servo 的订阅/Timer、
EvaluateKinematicModel RPC、C++ RobotModel monitor 和普通桥接。输入是合成 URDF、
静态 q/TF、合成原始球面命令和始终停车的最终命令，所有 topic 均在 `/test/*`。
4 项通过：自动查询与当前周期建议；MoveIt halt 清缓存及新模型恢复；关节断流/恢复；
原始 policy stop 阻止查询。完整捕获的控制输出均为零。
该验证不替换网络发布、TF 查询或 RPC，但没有运行真实 Fisher/地图/GPU，也没有实机运动。
它证明冻结合成场景中的自动只读恢复链，不证明运动中的模型连续更新、投影执行或物理安全。

```bash
source /opt/ros/noetic/setup.bash
source ros_ws/devel/setup.bash
rostest omnimap_kinematics servo_preview.test
```

测试构造运行参数后实例化实际 Servo 类，不调用生产 CLI 的配置加载；源码从包的真实路径
解析仓库根目录，因此支持机器人 catkin_ws 内的符号链接部署。新用例已加入 CMake rostest 清单。
