# UR5 上位机（sumi）升级方案：桥接节点与 MoveIt Servo 诊断

更新时间：2026-10-24

实现与部署记录见 [sumi 运动学诊断验收](ur5_nbv_kinematics_validation_sumi.md)。
普通/KF 桥接入口已升级，MoveIt C++ monitor 已构建；以下清单保留 2026-10-22 的方案基线。
真实 ROS master 当前不可访问，尚未标定 `m_min` 或启用二维半空间控制修正。

## 现场核查状态

SSH 已恢复，以下文件和行号已经在 `sumi` 的 `~/ros/catkin_ws/src` 中只读核对。源码中没有找到 `cam_1_baselink`；实际相机 frame 是 `cam_1_color_optical_frame`。当前桥接输入和输出都标记为 `base_link`，桥接通过 TF 读取 camera/tool0 的位置差，修正同一 `base_link` 表达下的参考点速度。

方案基线核查时 ROS master 不可访问，节点运行情况尚未核实；当时没有写入远端工作区。
2026-10-24 的实际代码和本机工作区部署变化以链接的验收记录为准。

## 1. 已核实的桥接和启动链路

已核实的文件：

| 文件与行 | 已确认行为 |
|---|---|
| `robot_else_nodes/scripts/camera_twist_bridge.py:33–37` | 输入 `/servo_server/delta_twist_camera`，输出 `/servo_server/delta_twist_cmds`，相机为 `cam_1_color_optical_frame`，末端为 `tool0` |
| `camera_twist_bridge.py:64–66` | 订阅输入并发布输出；`74–80` 只接受 `header.frame_id == base_link` |
| `camera_twist_bridge.py:84–90` | 查询 `base_link→camera` 和 `base_link→tool0` 的平移，构造 `r_tc=p_C−p_T` |
| `camera_twist_bridge.py:111–138` | 使用 `v_T=v_C−ω×r_tc`、`ω_T=ω_C`，输出 frame 仍为 `base_link` |
| `robot_else_nodes/scripts/camera_twist_bridge_kf.py:47–52, 169–208` | KF 版本使用相同 frame、TF 和参考点公式，只是在输出前对 six components 做滤波 |
| `robot_else_nodes/launch/launch_robot.launch:12–15` | `start_servo` 默认 `false` |
| `launch_robot.launch:78–85` | Servo 可启动，但 `camera_twist_bridge.py` 节点目前被注释，没有自动启动 |
| `universal_robot/ur5_moveit_config/config/ur5_servo.yaml:29–43` | MoveIt 使用 `base_link` planning/command frame、`tool0` EE、输入 `/servo_server/delta_twist_cmds`、status `/servo_server/status`，奇异阈值为 17/30 |

若需要重复生成清单，使用：

```bash
cd ~/ros/catkin_ws/src
printf '%s\n' '=== bridge candidates ==='
rg -n --hidden -g '*.cpp' -g '*.h' -g '*.py' -g '*.launch' -g '*.yaml' \
  'cam_1_baselink|cam_1_color_optical_frame|tool0|delta_twist_camera|TwistStamped|lookupTransform|lookup_transform|tf2' .

printf '%s\n' '=== MoveIt Servo source/config ==='
find . -path '*moveit_servo*' -type f | sort
rg -n 'condition|singular|lower_singularity_threshold|hard_stop_singularity_threshold|status_topic|planning_frame|robot_link_command_frame|ee_frame_name' \
  . -g '*.cpp' -g '*.h' -g '*.yaml'

printf '%s\n' '=== runtime graph ==='
rosnode list | sort
rostopic list | rg 'servo|twist|joint|status|cam_1|tool0'
rosnode list | while read n; do rosnode info "$n" 2>/dev/null | rg -q 'cam_1_baselink|cam_1_color_optical_frame|tool0|delta_twist_camera' && { echo "--- $n"; rosnode info "$n"; }; done
```

把输出保存为 `~/omnimap_kinematic_inventory_$(date +%Y%m%d_%H%M%S).txt`。运行时还要先 `source /opt/ros/noetic/setup.bash` 和 `source ~/ros/catkin_ws/devel/setup.bash`；本次检查发现 ROS master 不可访问，所以没有把空的 `rosnode list` 当作运行链路证据。

## 2. MoveIt Servo 需要记录的指标

官方 Noetic `moveit_ros/moveit_servo/src/servo_calcs.cpp` 的关键位置如下：

| 官方源文件位置 | 计算 | 对项目的作用 |
|---|---|---|
| `servo_calcs.cpp:434–510` | 将 incoming Twist frame 旋转到 planning frame | 检查桥接是否已经做过一次 frame 变换 |
| `servo_calcs.cpp:512–534` | `getJacobian()`、SVD、伪逆、关节增量 | MoveIt 实际使用的 Jacobian 入口 |
| `servo_calcs.cpp:536–540` | 速度限幅后应用奇异性缩放 | `sigma_min` 低并不会自动发布给上游 |
| `servo_calcs.cpp:693–756` | `condition = σmax/σmin`、奇异方向、线性减速/急停 | MoveIt 的内部奇异性指标和触发逻辑 |
| `servo_calcs.cpp:759–800` | 逐关节 velocity bound | 不能只看整体 condition number |
| `servo_calcs.cpp:268–273` | 发布 `std_msgs/Int8` status | 订阅该 topic 作为实际执行反馈 |
| `status_codes.h:46–64` | status code 定义 | 1=decelerate singularity，2=halt singularity，3/4=collision，5=joint bound |

在 sumi 上查询实际 status topic 和阈值：

```bash
rosparam get /servo_server/status_topic
rosparam get /servo_server/lower_singularity_threshold
rosparam get /servo_server/hard_stop_singularity_threshold
rostopic echo -n 1 /servo_server/status
```

实际命名空间可能不是 `/servo_server`，以 `rosnode info /servo_server` 和参数服务器输出为准。MoveIt 状态码只说明它采取了什么动作，不包含 condition 数值；要得到 `sigma_min` 或 condition，需要在拥有 MoveIt `RobotState` 的节点中调用同一 Jacobian 计算。

## 3. 桥接节点的推荐改法

### 3.1 已确认输入和输出

当前真实链路是：

```text
OmniMap Servo Runtime
  /servo_server/delta_twist_camera  (TwistStamped, frame_id=base_link)
    → camera_twist_bridge.py 或 camera_twist_bridge_kf.py
  /servo_server/delta_twist_cmds   (TwistStamped, frame_id=base_link)
    → MoveIt Servo (ur5_servo.yaml:34)
```

桥接不是把消息改成 `tool0` frame，也不是把相机轴旋转到基座轴；输入和输出都用 `base_link` 表达。它利用 `base_link→cam_1_color_optical_frame` 与 `base_link→tool0` 的平移，改变 Twist 的参考点。`launch_robot.launch:82–85` 当前把普通桥接节点注释掉，所以必须确认实际启动命令是否另行运行了 bridge/KF 节点。

### 3.2 检查 Twist 变换是否包含平移项

这里的代码已经实现了平移项。它定义 `r_tc=p_C−p_T`，并在 `camera_twist_bridge.py:122–130` 使用：

```text
v_T^B = v_C^B − ω^B × (p_C^B − p_T^B)
ω_T^B = ω_C^B
```

这与完整刚体速度关系一致，因为所有向量已经在 `base_link` 中表达；不应再给 MoveIt 发送一次 `cam→tool0` 变换。Planning 侧的 `J̄(q)` 应使用与 Servo 一致的 `base_link` 表达和 `tool0` 参考点。KF 版本只对变换后的 six components 做滤波，不能把滤波器当作奇异性诊断。

### 3.3 在桥接节点增加诊断输出

在真实桥接源文件中新增：

- `std_msgs/Float64` 或项目自定义 `KinematicStatus` 发布 `sigma_min`、condition、速度范数；
- `std_msgs/Int8` 订阅或转发 MoveIt `/servo_server/status`；
- `TwistStamped` 的输入/输出 frame、时间戳和变换年龄日志；
- 输入速度、桥接后速度、MoveIt status 的同一序号或时间戳关联。

诊断输出建议放在桥接节点或独立 monitor；当前桥接能看到输入命令、变换后的 tool-reference 速度和最终输出 topic，但它不拥有 MoveIt `RobotState`，不能自行计算可靠的 Jacobian。MoveIt 内部 condition number 仍由 Servo 自己计算；桥接只负责收集，不应从 status code 伪造数值。

## 4. 运动学指标的落地位置

有两种实现方式，优先使用第 A 种：

### A. 在 sumi 新增一个 MoveIt 诊断节点

新增文件建议：`~/ros/catkin_ws/src/<现有_robot_pkg>/src/omnimap_kinematic_monitor.cpp`。

节点订阅 `/joint_states`，读取 `robot_description` 和 MoveIt planning group，构造 `moveit::core::RobotState`，用与 Servo 相同的 group 和参考 link 调用 `getJacobian()`，计算：

```text
J̄ = Dx⁻¹ J Dq
m(q) = σmin(J̄)
κ(q) = σmax(J̄) / σmin(J̄)
```

它同时订阅桥接输入和 MoveIt status，发布 `KinematicStatus`。这样每个 Planning tick 可通过 ROS 反馈得到真实 q、真实末端参考点和真实 Servo 状态。

### B. 直接修改桥接节点

只有在桥接已经链接 MoveIt C++ API、能拿到 `RobotState` 时才采用。否则不要在普通 TF bridge 中复制 RobotModel 和 Jacobian 逻辑，避免产生第二套关节顺序、末端 link 或 URDF 参数。

## 5. 建议的上位机改动顺序

1. 只增加 status、frame 和速度日志，不改变 Twist 数值。
2. 确认 `base_link -> cam_1_color_optical_frame`、`base_link -> tool0` 的 TF 时间戳和 `p×ω` 项；不要再寻找不存在的 `cam_1_baselink`。
3. 新增 `omnimap_kinematic_monitor.cpp`，发布 `sigma_min`/condition，不参与控制。
4. 用日志标定 Planning 的 `m_min`，并比较 MoveIt 开始减速时的 condition 与本机 `sigma_min`。
5. 仅在诊断稳定后，让 Planning 的二维球面命令使用最小半空间修正。
6. 在 bridge 或 Servo 侧增加明确的 stale、TF failure、status failure 零速策略。

## 6. 上位机验收命令

```bash
cd ~/ros/catkin_ws
catkin_make --pkg <现有_robot_pkg>
source devel/setup.bash
rosrun tf tf_echo base_link cam_1_color_optical_frame
rosrun tf tf_echo base_link tool0
rostopic info /servo_server/delta_twist_camera
rostopic info /servo_server/delta_twist_cmds
rostopic hz /joint_states
rostopic echo -n 5 /servo_server/status
rostopic echo -n 5 <bridge_input_topic>
rostopic echo -n 5 <bridge_output_topic>
```

验收记录至少包含：输入/输出 frame、TF 年龄、输入输出 linear/angular 速度、joint state 时间戳、MoveIt status、`sigma_min`、condition、是否发生减速或急停。没有这些对齐数据时，不应把停机归因于奇异点。
