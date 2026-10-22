# OmniMap 本机升级方案：把 MoveIt 运动学反馈接入 Planning

更新时间：2026-10-22

本文只描述本机仓库 `/home/fuyx/lanzc/omnimap` 的改动。上位机 `sumi` 的改动见 [`ur5_nbv_kinematics_upgrade_sumi.md`](ur5_nbv_kinematics_upgrade_sumi.md)。本次检查中 `ssh sumi` 在 SSH 密钥交换阶段被远端关闭，因此远端文件名和行号不能伪装成已核实事实。

## 已核实的当前链路

| 文件与行 | 当前行为 | 升级含义 |
|---|---|---|
| `info_flow/info_flow_planning_node.py:48` | Planning 节点只保存 pose、snapshot 和 Fisher policy | 当前没有关节状态、MoveIt status 或 Jacobian |
| `info_flow/info_flow_planning_node.py:129` | 发布 `SphericalCommand`，订阅 pose 和 snapshot | 可在消息中增加诊断字段，或单独增加反馈 topic |
| `info_flow/info_flow_planning_node.py:425` | `_publish_spherical_command()` 写入球面速率、参考半径、Fisher 分数和 stop 原因 | 适合增加 `m`, `sigma_min`, `condition`, `projection` 等规划诊断 |
| `info_flow/info_flow_planning_node.py:517` | 定时器执行 pose/snapshot 检查并调用 `next_pose_from_c2w()` | 连续构型修正应位于策略输出与消息发布之间 |
| `info_flow/info_flow_servo_runtime.py:49` | 执行侧读取 TF 和 `SphericalCommand` | 这里有实时相机位姿，但没有 MoveIt RobotState/Jacobian |
| `info_flow/info_flow_servo_runtime.py:271` | 以 50 Hz 将球面速率转换为线速度和角速度 | 这里必须记录“实际送往桥接节点的 Twist”的 frame 和速度 |
| `info_flow/info_flow_servo_runtime.py:466` | 发布 `TwistStamped`，frame 为 `args.cmd_frame` | 该 frame 必须与桥接节点的输入约定一致 |
| `info_flow/servo_runtime_common.py:148` | `lookup_latest_pose_from_tf()` 查询 `world_frame -> camera_frame` | 可复用时间戳，但不能由此得到关节 Jacobian |
| `info_flow/servo_runtime_common.py:171` | `publish_motion_components()` 只填 `TwistStamped` 的 six components | 不包含桥接后的 `tool0` 速度或 MoveIt status |
| `ros_ws/src/omnimap_msgs/msg/SphericalCommand.msg:1` | 当前消息只有 11 个字段 | 不建议把执行侧实时 Jacobian 塞进规划命令；增加独立反馈消息更清楚 |
| `docs/execution_side_setup.md:123` | 执行侧输出配置为 `/servo_server/delta_twist_camera` | 必须确认桥接节点是否订阅这个 topic，避免重复变换 |

## MoveIt Noetic 的事实基线

以官方 `noetic-devel` 的 `moveit_ros/moveit_servo/src/servo_calcs.cpp` 为准：

- `434–510`：Cartesian Twist 进入后，MoveIt 将输入 frame 旋转到 planning frame；普通 incoming frame 使用 `planning_frame -> incoming_frame` 的旋转矩阵，`tool0`/EE frame 走已有变换。
- `512–534`：计算 `delta_x`，调用 `current_state_->getJacobian(joint_model_group_)`，做薄 SVD 和伪逆，再得到关节增量。
- `536–540`：先执行关节速度限制，再执行奇异性/碰撞缩放。
- `693–756`：用 `condition = sigma_max / sigma_min`，取 SVD 的最后一个 `U` 列作为奇异方向，利用一次小的前向状态扰动解决 SVD 符号不确定性；当命令方向朝向奇异区时，在 lower threshold 与 hard-stop threshold 之间线性减速，超过 hard-stop 则缩放为 0。
- `759–800`：逐关节检查 velocity bounds，并对 `delta_theta` 做比例缩放。
- `268–273`：每次循环发布 `std_msgs/Int8` status。
- `status_codes.h:46–64`：奇异相关状态是 `DECELERATE_FOR_SINGULARITY=1` 和 `HALT_FOR_SINGULARITY=2`。

因此 MoveIt 并没有现成的可查询 `sigma_min` topic。Planning 需要自己用同一 RobotModel 和同一末端参考点计算 `m(q)`；MoveIt status 只作为“下游实际是否开始减速/急停”的反馈。

## 推荐的本机改动顺序

### 1. 新增运动学诊断消息

新增 `ros_ws/src/omnimap_msgs/msg/KinematicStatus.msg`：

```text
std_msgs/Header header
uint32 planner_tick
string source_frame
string task_frame
float64 sigma_min
float64 sigma_max
float64 condition_number
float64 manipulability
float64 command_speed_norm
float64 actual_speed_norm
int8 moveit_status
string moveit_status_text
bool transform_chain_valid
string reason
```

同时修改 `ros_ws/src/omnimap_msgs/CMakeLists.txt` 的 message 列表和 `ros_ws/src/omnimap_msgs/package.xml` 的消息生成依赖。编译后再修改 Python 导入处。

### 2. Planning 保持球面命令接口，接收反馈

在 `info_flow/info_flow_planning_node.py:121–147` 增加 `KinematicStatus` 订阅和缓存。不要在 Planning 里假设它能直接访问 URDF 或 MoveIt RobotState。

在 `info_flow/info_flow_planning_node.py:634–690` 的策略输出之后增加两类处理：

1. 读取最近一条 `KinematicStatus`，检查 `source_frame/task_frame`、时间年龄和 `transform_chain_valid`。
2. 若反馈显示持续 `DECELERATE_FOR_SINGULARITY`，下一 tick 降低命令速度或触发动作方向重评估；若为 `HALT_FOR_SINGULARITY`，发布带原因的 stop，并等待恢复策略。

诊断字段可以先写入 Planning 日志；不要在没有真实反馈时调大速度阈值。

### 3. 保持连续修正的计算定义

在新文件 `info_flow/kinematic_projection.py` 中实现纯 NumPy 函数，避免 ROS 依赖：

```python
def project_spherical_command(u_info, a, c):
    # min 0.5 ||u-u_info||^2  s.t. a.T @ u >= c
    violation = float(c - np.dot(a, u_info))
    if violation <= 0.0:
        return np.asarray(u_info, dtype=float), 0.0
    denom = float(np.dot(a, a))
    if denom <= 1e-12:
        raise ValueError("no usable kinematic correction direction")
    delta = (violation / denom) * np.asarray(a, dtype=float)
    return np.asarray(u_info, dtype=float) + delta, float(np.linalg.norm(delta))
```

该函数只处理两个球面速率。`a` 和 `c` 应由执行侧根据同一桥接规则计算并反馈，或由一个拥有 MoveIt RobotModel 的执行侧诊断节点直接修正 Twist。不要把 `a` 从 MoveIt 的 status code 反推出来。

### 4. 重新编译和验证消息链

修改完成后按以下顺序验证：

```bash
./scripts/build_ros_ws.sh
source ros_ws/devel/setup.bash
python3 -m py_compile info_flow/info_flow_planning_node.py info_flow/kinematic_projection.py
rostopic echo -n 1 /omnimap/kinematic_status
```

先只发布诊断，不修改真实 Twist；确认 frame、时间戳、关节顺序和 MoveIt status 对齐后再启用修正。

## frame 与桥接的具体判定

当前本机代码默认 `cmd_frame=base_link`，但用户提供的实际链路还包含 `cam_1_baselink -> tool0` 桥接节点。必须先确认桥接节点的输入输出：

```bash
rosnode info <bridge_node>
rostopic info /servo_server/delta_twist_camera
rostopic info <bridge_output_topic>
rosrun tf tf_echo cam_1_baselink tool0
```

如果桥接节点已经将 Twist 变换到 `tool0` 并发布给 MoveIt Servo，Planning/Servo 侧只应记录原始命令和桥接后的命令，不能再次做 camera-to-tool0 变换。如果桥接节点只发布 TF 而 MoveIt Servo 直接订阅 camera-frame Twist，则让 `TwistStamped.header.frame_id` 明确为 `cam_1_baselink`，由 MoveIt 做一次 frame 旋转。

`cam_1_baselink` 与 `tool0` 之间有平移时，要确认桥接实现是否使用完整 Twist adjoint：

```text
v_tool = R v_cam + p × (R ω_cam)
ω_tool = R ω_cam
```

MoveIt Noetic 的 `servo_calcs.cpp:472–500` 对普通 incoming frame 只旋转 linear/angular 两个向量；如果桥接节点已经加入 `p × ω`，再把它当作普通 frame 交给 MoveIt 会重复处理。这个细节决定 `J_camera` 应取相机参考点，还是取 `J_tool0`。

## 接受标准

本机升级完成前必须看到：

- 同一 tick 的 `SphericalCommand`、桥接输入、桥接输出、MoveIt status 和 `joint_states` 可以按时间戳关联；
- `sigma_min`/condition number 的计算使用的末端参考点、frame 和关节顺序与实际 Servo 一致；
- MoveIt 状态 topic 能区分 `DECELERATE_FOR_SINGULARITY`、`HALT_FOR_SINGULARITY`、collision 和 joint bound；
- 开启连续修正前，原始 Fisher 命令仍可单独回放；
- 任何 `a` 不可用、TF 过期或桥接链断开都降级为零速或重新规划，而不是继续发送旧命令。
