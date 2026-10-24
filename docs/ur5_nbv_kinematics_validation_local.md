# 本机运动学反馈接入与验收

日期：2026-10-24。对应 [本机升级计划](ur5_nbv_kinematics_upgrade_local.md)。

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

本机消息包生成了计划列出的 14 个字段。`std_msgs` 依赖覆盖 build/export/runtime。
Planning 不加载机器人模型、不计算 Jacobian、不从 MoveIt code 推导 `a`。
执行侧仍需使用与实际 Servo 相同的 RobotModel、planning group、关节顺序、`base_link` 表达和 `tool0` 参考点，
并在 TF、joint_states、status 或桥接链失效时将反馈标为无效。
`KinematicStatus` 没有单独的 joint/status/TF 时间戳字段；这些样本的关联证据来自现场 rosbag 和 monitor 日志，
不能仅凭 `transform_chain_valid=true` 声明已经验证关节顺序。

`kinematic_projection.py` 保留二维最小半空间投影定义，要求 `u_info/a` 为两个有限速率、`c` 为有限标量。
需要修正而方向范数不可用时抛出 `ValueError`。当前消息未提供 `a/c`，连续投影没有接入控制路径；
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
python tests/test_kinematic_runtime.py
```

纯数值测试验证投影的约束与最短距离、不可用方向、双时钟过期、frame、指标、状态码、持续减速和恢复。
运行时测试使用生成的 ROS 消息和真实 Planning/Servo 回调，替换地图加载、Fisher 计算、TF 查询和网络发布。
覆盖 observe 原样输出、enforce 停车、原始命令留存、消息序列化、计算期间收到的急停、真实 Twist 记录和失效零速。
这些测试验证本机处理逻辑，不验证机器人上的 Jacobian 数值或 MoveIt 实际减速时机。

本次实际结果：ROS 消息构建退出码 0；InfoFlow 的 Python 编译通过；纯数值/反馈测试 8 项、
Planning/Servo 回调测试 9 项全部通过；`git diff --check` 通过。

## 现场验收仍待完成

2026-10-24 在沙箱内和沙箱外执行 `rostopic info /omnimap/kinematic_status`，均返回
`ERROR: Unable to communicate with master!`。随后在沙箱外加载仓库 `scripts/source_env.sh`，
使用其中的 `ROS_MASTER_URI=http://172.19.93.154:11311` 重试，结果相同。
尚未收到真实诊断消息；不能标记整个计划验收完成。
本机升级不包含对 sumi 的写入或启动机器人。现场恢复后，按计划执行：

```bash
source scripts/source_env.sh
rostopic echo -n 1 /omnimap/kinematic_status
rostopic info /servo_server/delta_twist_camera
rostopic info /servo_server/delta_twist_cmds
rosnode info <实际运行的桥接节点名>
rosrun tf tf_echo base_link cam_1_color_optical_frame
rosrun tf tf_echo base_link tool0
rosbag record -O kinematic_chain.bag /omnimap/spherical_cmd_raw /omnimap/spherical_cmd \
  /omnimap/kinematic_status /servo_server/delta_twist_camera /servo_server/delta_twist_cmds \
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

确认以上数据对齐后再使用 enforce；连续投影还需执行侧提供经验证的 `a/c` 接口。
