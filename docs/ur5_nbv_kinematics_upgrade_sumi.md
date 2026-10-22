# UR5 上位机（sumi）升级方案：桥接节点与 MoveIt Servo 诊断

更新时间：2026-10-22

## 现场核查状态

本轮从算力侧执行 `ssh sumi` 时，远端在 SSH key exchange 阶段返回 `Connection closed by remote host`；`sumi` 解析到的两个地址均无法进入。因此下面的“桥接节点文件名和行号”不能写成已确认事实，必须先在上位机执行第 1 节的 inventory 命令。MoveIt 官方 Noetic 源码行号已经核对，远端若使用未修改的 noetic-devel，应与这些位置一致；若有本地 fork，以 inventory 输出为准。

## 1. 在 sumi 上先生成精确文件和行号清单

不要先改代码。登录后运行：

```bash
cd ~/ros/catkin_ws/src
printf '%s\n' '=== bridge candidates ==='
rg -n --hidden -g '*.cpp' -g '*.h' -g '*.py' -g '*.launch' -g '*.yaml' \
  'cam_1_baselink|tool0|delta_twist_camera|TwistStamped|lookupTransform|lookup_transform|tf2' .

printf '%s\n' '=== MoveIt Servo source/config ==='
find . -path '*moveit_servo*' -type f | sort
rg -n 'condition|singular|lower_singularity_threshold|hard_stop_singularity_threshold|status_topic|planning_frame|robot_link_command_frame|ee_frame_name' \
  . -g '*.cpp' -g '*.h' -g '*.yaml'

printf '%s\n' '=== runtime graph ==='
rosnode list | sort
rostopic list | rg 'servo|twist|joint|status|cam_1|tool0'
rosnode list | while read n; do rosnode info "$n" 2>/dev/null | rg -q 'cam_1_baselink|tool0|delta_twist_camera' && { echo "--- $n"; rosnode info "$n"; }; done
```

把输出保存为 `~/omnimap_kinematic_inventory_$(date +%Y%m%d_%H%M%S).txt`。这一步会给出桥接源文件的真实文件名和行号，也能确认桥接节点究竟是发布给 MoveIt Servo，还是直接发布给 UR 控制器。

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

### 3.1 先确认输入和输出

从第 1 节 inventory 找到桥接源文件后，在该文件当前行号附近确认以下四件事：

1. 订阅的输入 topic 和 `TwistStamped.header.frame_id`；
2. TF 查询的父子 frame，是 `cam_1_baselink -> tool0` 还是反方向；
3. 输出 topic 是否是 MoveIt Servo 的 `cartesian_command_in_topic`；
4. 输出是 `TwistStamped`，还是已经转换成 `JointJog`/厂商速度命令。

桥接必须只做一次 frame 变换。若桥接已经输出 `tool0` frame 的 Twist，则输出消息的 `header.frame_id` 要写成 `tool0`，并在诊断消息中记录 `source_frame=cam_1_baselink`、`task_frame=tool0`、TF 时间戳和变换年龄。

### 3.2 检查 Twist 变换是否包含平移项

刚体变换的完整空间速度变换为：

```text
v_tool = R v_cam + p × (R ω_cam)
ω_tool = R ω_cam
```

如果现有桥接只执行 `R*v` 和 `R*ω`，它相当于只变换轴方向，没有变换参考点。此时要么把参考点固定为同一个 frame 并在 Planning 侧使用对应 Jacobian，要么在桥接中补上 `p × (Rω)`。补项前必须用静态 TF 和一组手工角速度命令验证符号。

### 3.3 在桥接节点增加诊断输出

在真实桥接源文件中新增：

- `std_msgs/Float64` 或项目自定义 `KinematicStatus` 发布 `sigma_min`、condition、速度范数；
- `std_msgs/Int8` 订阅或转发 MoveIt `/servo_server/status`；
- `TwistStamped` 的输入/输出 frame、时间戳和变换年龄日志；
- 输入速度、桥接后速度、MoveIt status 的同一序号或时间戳关联。

诊断输出建议放在桥接节点，因为它同时看到 camera-frame 命令、tool0-frame 命令和最终发布 topic。MoveIt 内部 condition number 仍由 Servo 自己计算；桥接只负责收集，不应从 status code 伪造数值。

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
2. 确认 `cam_1_baselink -> tool0` 的方向、静态 TF、时间戳和是否含 `p×ω`。
3. 新增 `omnimap_kinematic_monitor.cpp`，发布 `sigma_min`/condition，不参与控制。
4. 用日志标定 Planning 的 `m_min`，并比较 MoveIt 开始减速时的 condition 与本机 `sigma_min`。
5. 仅在诊断稳定后，让 Planning 的二维球面命令使用最小半空间修正。
6. 在 bridge 或 Servo 侧增加明确的 stale、TF failure、status failure 零速策略。

## 6. 上位机验收命令

```bash
cd ~/ros/catkin_ws
catkin_make --pkg <现有_robot_pkg>
source devel/setup.bash
rosrun tf tf_echo cam_1_baselink tool0
rostopic hz /joint_states
rostopic echo -n 5 /servo_server/status
rostopic echo -n 5 <bridge_input_topic>
rostopic echo -n 5 <bridge_output_topic>
```

验收记录至少包含：输入/输出 frame、TF 年龄、输入输出 linear/angular 速度、joint state 时间戳、MoveIt status、`sigma_min`、condition、是否发生减速或急停。没有这些对齐数据时，不应把停机归因于奇异点。
