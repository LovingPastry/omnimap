# 本机运动学反馈接入与验收

更新：2026-10-26（保留 2026-10-24 起的阶段记录）。对应 [本机升级计划](ur5_nbv_kinematics_upgrade_local.md)。

执行侧桥接与 MoveIt monitor 现已实现并部署源文件入口；构建、启动、指标定义及待完成的硬件验收见
[sumi 诊断实现与验收](ur5_nbv_kinematics_validation_sumi.md)。本文下方“现场验收仍待完成”保留此前本机接入核查记录。

## 运行模式

Planning 默认使用 `--kinematic_feedback_mode observe`，符合先发布诊断、对齐现场数据后再启用控制干预的顺序。
此模式计算、发布原始 Fisher 指令，并记录反馈验证结果与建议速度比例。
确认诊断链后使用 `--kinematic_feedback_mode enforce`，启用以下行为：

| 反馈 | enforce 输出 |
|---|---|
| 正常，MoveIt code 0 | 保留 Fisher 输出 |
| 连续 code 1 达到 0.1 秒 | 两个球面速率乘以 0.5 |
| code 2 | 零速 stop，原因含 `halt_for_singularity` |
| code 3 / 4 | 零速 stop，分别标记 collision 减速 / 急停 |
| code 5 | 零速 stop，标记 `joint_bound` |
| 未知 code | 零速 stop，保留数值 code 与 `UNKNOWN` 名称 |
| 反馈缺失、过期、未来时间戳、frame 错误、无效 TF/桥接链或无效指标 | 零速 stop，给出具体原因 |

重新收到可用反馈后，下一次规划按新状态处理。Fisher 自己的 stop 在减速时保留。
反馈在 Fisher 计算完成后重新读取，避免长时间计算前缓存的状态覆盖最新急停信息。

参数均为 Planning 的 CLI 参数：

| 参数 | 默认值 |
|---|---|
| `--kinematic_status_topic` | `/omnimap/kinematic_status` |
| `--kinematic_status_timeout_sec` | `0.5` |
| `--kinematic_source_frame` | `base_link` |
| `--kinematic_task_frame` | `tool0` |
| `--kinematic_deceleration_duration_sec` | `0.1` |
| `--kinematic_deceleration_scale` | `0.5` |
| `--raw_spherical_cmd_topic` | `/omnimap/spherical_cmd_raw` |

反馈新鲜度同时检查 ROS 的 `header.stamp` 和本进程的 monotonic 接收时间。
过期消息重新送达不会变成新反馈；ROS 时钟停止时，接收超时仍生效。
空时间戳无效，允许的未来时钟偏差为 50 ms。frame 比较允许前导 `/`，不允许留空以跳过检查。

## 原始命令与执行记录

每次 Fisher 正常返回后，先向 `/omnimap/spherical_cmd_raw` 发布未经反馈修改的 `SphericalCommand`，
再向 `/omnimap/spherical_cmd` 发布最终指令。两个消息的 tick、pose stamp、模型版本与参考球一致。
反馈缺失或急停也不会阻止原始 Fisher 输出留存。两个 topic 不能经 ROS 命名解析/重映射后指向同一名称。
默认 Servo 只订阅最终指令 topic。

Planning 的 `kinematic_feedback:` 日志包含当前 tick、command/feedback stamp、feedback tick、
source/task frame、MoveIt code/标准名称/上游文本、sigma/condition/manipulability、速度范数、链有效性和处理结果。
记录时保留 `profile` 日志节，例如 `--log_section all --log_file`。

Servo 的 `servo_twist:` INFO 日志记录每一次实际发布的 six components、frame、相机参考点、
planner tick、模型版本、command stamp、Twist stamp、topic 与原因，包含超时、TF 失败、异常和 stop 的零速输出。
使用 `--log_file` 可将它保存到 `--output` 目录。Twist 的 stamp 延续 TF 采样时间。
Servo 在 `world_frame` 计算速度，要求 `cmd_frame` 与其相同，并校验球面指令的 frame。
camera 到 tool0 的参考点变换继续由现场桥接承担；本机没有再做一次 `p×ω` 变换。

记录原始指令并独立回放到专用 topic：

```bash
source ros_ws/devel/setup.bash
rosbag record -O fisher_raw.bag /omnimap/spherical_cmd_raw
# 结束录制后，在独立的回放会话执行：
rosbag play fisher_raw.bag /omnimap/spherical_cmd_raw:=/omnimap/replay/spherical_cmd
```

原始指令的回放接收方使用 `/omnimap/replay/spherical_cmd`；这个 topic 默认没有机器人控制订阅者。
回放测试若需要运行 Servo，将其 `--cmd_topic` 设置为 `/omnimap/replay/twist`。

## 运动学计算边界

KinematicStatus 保持计划列出的 14 个字段；另新增 KinematicConstraint 及扩展 ServoTwistTrace 的局部速度模型。`std_msgs` 依赖覆盖 build/export/runtime。
Planning 不加载机器人模型、不计算 Jacobian、不从 MoveIt code 推导 `a`。
执行侧 monitor 使用与实际 Servo 相同的 RobotModel、planning group、关节顺序、`base_link` 表达和 `tool0` 参考点，
并在 TF、joint_states、status 或桥接链失效时将反馈标为无效；实机模型一致性仍需验收。
`KinematicStatus` 没有单独的 joint/status/TF 时间戳字段；这些样本的关联证据来自现场 rosbag 和 monitor 日志，
不能仅凭 `transform_chain_valid=true` 声明已经验证关节顺序。

`kinematic_projection.py` 保留二维最小半空间投影定义，要求 `u_info/a` 为两个有限速率、`c` 为有限标量。
需要修正而方向范数不可用时抛出 `ValueError`。新的 KinematicConstraint 提供执行侧 `a/drift` 及显式阈值下的 `c`，Planning 已验证并记录投影建议，仍未接入控制路径；
将来接入时，调用方必须把该异常转成零速或重新规划，禁止复用上一次修正方向。
当前 enforce 只使用真实反馈进行状态停车和持续奇异减速。

## 本机验证

在 `InfoFlow` 环境运行以下检查；消息构建脚本会自行选择系统 ROS Python：

```bash
conda activate InfoFlow
./scripts/build_ros_ws.sh
source /opt/ros/noetic/setup.bash
source ros_ws/devel/setup.bash
python -m py_compile info_flow/info_flow_planning_node.py info_flow/kinematic_feedback.py \
  info_flow/kinematic_projection.py info_flow/info_flow_servo_runtime.py info_flow/servo_runtime_common.py
python tests/test_kinematic_feedback.py
python tests/test_kinematic_constraint.py
python tests/test_kinematic_runtime.py
python tests/test_kinematic_projection.py
python tests/test_bounded_query.py
python tests/test_preview_joint_state.py
python tests/test_projection_readiness.py
```

纯数值测试验证投影的约束与最短距离、不可用方向、双时钟过期、frame、指标、状态码、持续减速和恢复。
运行时测试使用生成的 ROS 消息和真实 Planning/Servo 回调，替换地图加载、Fisher 计算、TF 查询和网络发布。
覆盖 observe 原样输出、enforce 停车、原始命令留存、消息序列化、计算期间收到的急停、真实 Twist 记录和失效零速。
这些测试验证本机处理逻辑，不验证机器人上的 Jacobian 数值或 MoveIt 实际减速时机。

2026-10-24 阶段结果（历史计数）：ROS 消息构建退出码 0；InfoFlow 的 Python 编译通过；纯数值/反馈测试 8 项、
Planning/Servo 回调测试 9 项全部通过；`git diff --check` 通过。

## 现场验收仍待完成

2026-10-24 的 master 不可访问是历史记录。2026-10-25 已恢复 master，
执行端已完成启动修复并取得真实关节状态下的零速诊断，详见 sumi 验收记录。
这不证明连续诊断稳定或非零控制有效；当前工作允许部署到本机机器人 ROS 工作区。
后续现场记录命令：

```bash
source scripts/source_env.sh
rostopic echo -n 1 /omnimap/kinematic_status
rostopic info /servo_server/delta_twist_camera
rostopic info /servo_server/delta_twist_cmds
rosnode info <实际运行的桥接节点名>
rosrun tf tf_echo base_link cam_1_color_optical_frame
rosrun tf tf_echo base_link tool0
rosbag record -O kinematic_chain.bag /omnimap/spherical_cmd_raw /omnimap/spherical_cmd \
  /omnimap/kinematic_constraint /omnimap/kinematic_status /servo_server/delta_twist_camera /servo_server/delta_twist_cmds \
  /servo_server/status /joint_states /tf /tf_static
```

完整验收需要以下证据：

| 计划要求 | 当前证据 | 待补证据 |
|---|---|---|
| 同 tick 关联球面命令、桥接输入/输出、MoveIt status、joint_states | 原始/最终命令保留相同 tick，Servo 输出日志记录 tick 和两个时间戳 | 现场五类 topic 的 rosbag 与 monitor/bridge 日志 |
| sigma/condition 的参考点、frame 和关节顺序与 Servo 一致 | Planning 校验 source/task frame，未自行建立另一套模型 | monitor 源码/参数、RobotModel/group/关节顺序与实际 Servo 对照 |
| 区分奇异减速/急停、collision、joint bound | 本机处理并记录 code 0–5 的标准名称，未知码停车 | 现场 `/servo_server/status` 与反馈转发的真实样本 |
| 原始 Fisher 命令可单独回放 | 独立原始命令 topic，测试验证反馈急停时仍保留其数值；已提供录制/回放命令 | 现场录制文件的独立回放记录 |
| 无效方向、TF 过期或桥接中断时降级 | 投影拒绝无效输入；enforce 对失效反馈停车；Servo 失效输出零速 | 现场 monitor 的失效检测以及断链/超时的零速验证 |

确认以上数据对齐后再使用 enforce；连续投影需完成 [构型约束诊断](ur5_nbv_kinematic_constraint.md) 所列的现场标定和控制接入门槛。

## 2026-10-26 消费者回归

Planning 新增独立 KinematicConstraint 订阅及只读投影日志；投影建议与最终控制命令明确分开。
该阶段消费者测试 10 项、Planning/Servo 回调 10 项通过（历史计数）。后续已接入联合可用关节/相机限值、只读模型重新查询、实际周期建议重算和健康恢复门槛；连续投影执行、运动中模型更新和现场验收仍未完成。
参数、失效原因及日志定义见 [约束接口](ur5_nbv_kinematic_constraint.md)。

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

## 2026-10-26 连续模型流验证

新增 KinematicModel，同一 MoveIt monitor 连续发布同采样 J、相机位姿和完整指标梯度。直接节点及生产 launch 的两套 ROS 用例覆盖独立 FK/梯度比较、关节变化后更新、无桥接依赖、停车失效，以及快照到二维约束的转换与非法数据拒绝。自动只读 Servo 链路另有 4 项回归，部署脚本检查新生成文件。此为合成关节/TF 数值变化的证据，已增加 Servo 流式 observe 验证，但未验证实际运动或流式投影执行；细节见升级文档。

## 2026-10-26 流式 observe 接入

Servo 现可选择 kinematic_preview_source=stream，在模型同采样位姿上按实际周期生成建议，无 RPC/名义查询 Timer。servo_stream.test 覆盖变化的合成关节、模型/映射更新、MoveIt 停车、关节断流和 policy stop（5 项），全程最终 stop，捕获控制输出全为零。query 来源保留单独回归，仍使用冻结检查。模型年龄及相对最新 q/dq/TF 的差值仅作诊断，不能替代执行时延/残差裕量或实机验收。升级文档现按当前实现重新整理。
