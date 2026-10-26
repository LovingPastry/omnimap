# UR5 真实运动标定与验收 SOP

适用工作区：`/home/sumi/experience/omnimap`。日期：2026-10-26。
执行端 sumi；Tracking/Planning 在算力端。本文是操作流程，不代表已完成运动验收。

目标：证明真实机械臂按受限命令运动，投影确实修改实际输出，失去输入/模型时停车，
恢复后只执行新命令，并保存可复核的数据。先完成 A–D，再进行 E–G；未通过的步骤不能跳过。

## A. 人员、工位和停止方式

- 至少明确一名示教器操作员和一名终端记录员；操作员能够随时停止机器人，双方确认后才启动每次运动。
- 核对实际机器人型号/控制器/软件版本、TCP、相机与夹具安装、负载及重心。
  使用本工位已批准的安全配置和降速设置；不能把 ROS 的速度限制当成机器人安全功能。
- 夹爪不抓物、不动作。清空计划轨迹及停车余量内的人员、物品、桌边和线缆；不得靠近关节极限或奇异姿态做首次试验。
- **正常结束**：先停止 Planning 产生新命令，保持 Servo/桥接/monitor 运行，
  等待命令超时后的零输出，确认真实关节速度归零，再结束本轮 Servo。不要通过关闭驱动完成普通停车。
- **出现非预期运动、越界或有人进入运动范围**：操作员立即使用工位规定的实体停止措施；
  不等待命令超时，不把 Ctrl+C 当作安全停止按钮。
- 出现 Protective Stop 后停止本轮，保留错误码，调查原因；按该机器人的操作规程由人员恢复。
  不自动解锁/重启程序，不通过降低安全限制、伪造 code 0 或增大超时让测试继续。

保护停机恢复原则参见 [UR 官方说明](https://docs.universal-robots.com/tutorials/controlling-robot-externally/stop-recovery.html)。
其界面和具体步骤需匹配现场控制器版本，本文不下发 Dashboard 解锁命令。

## B. 固定版本、配置和记录

先停止已有 OmniMap 运动会话；保留机器人正常 bringup。执行端每个终端先执行：

```bash
cd /home/sumi/experience/omnimap
conda activate InfoFlow
source /opt/ros/noetic/setup.bash
source ~/ros/catkin_ws/devel/setup.bash
source ros_ws/devel/setup.bash --extend
```

记录员创建独立目录，不覆盖旧记录；将打印出的绝对路径填入其他终端的 RUN：

```bash
RUN=$(mktemp -d "$HOME/omnimap_acceptance_$(date +%Y%m%d_%H%M%S)_XXXXXX")
export RUN
printf '%s\n' "$RUN"
git rev-parse HEAD > "$RUN/git_head.txt"
git status --porcelain > "$RUN/git_status.txt"
cp config/rtabmap_config.yaml "$RUN/rtabmap_config.yaml"
rosparam dump "$RUN/effective_rosparams.yaml"
python3 scripts/collect_sumi_kinematics.py --output "$RUN/inventory.txt"
```

算力端也保存 Git HEAD、Tracking/Planning 完整命令和有效配置。两端消息重建且节点重启后再测。
记录机器人安全配置校验值、操作员、起始关节角、TCP/负载、相机标定文件版本、现场照片和试验范围。
ROS 参数归档可能包含设备信息，按工位数据权限保存。

检查控制链，只允许下列各一条：

```text
Planning → /omnimap/spherical_cmd → OmniMap Servo
  → /servo_server/delta_twist_camera → 普通桥接
  → /servo_server/delta_twist_cmds → MoveIt Servo → 实际生效的机器人控制器
```

```bash
rostopic info /omnimap/spherical_cmd
rostopic info /servo_server/delta_twist_camera
rostopic info /servo_server/delta_twist_cmds
rosparam get /servo_server
rosservice call /controller_manager/list_controllers
```

**通过条件**：控制 topic 没有第二个运动 publisher；实际运行控制器与 Servo 的 command_out_topic、
消息类型和关节顺序相符；不是仅发布到无人订阅的 topic。若 controller_manager 不在上述命名空间，
用 `rosservice list` 找实际的 list_controllers，不猜测或自动切换控制器。
有效模型的关节边界与实际参数一致。2026-10-26 曾发现磁盘 YAML 与运行参数不同，不能只检查磁盘文件。

在单独终端持续录制，所有运动和故障测试都不能中断记录：

```bash
rosbag record -O "$RUN/kinematic_chain.bag" \
  /omnimap/spherical_cmd_raw /omnimap/spherical_cmd \
  /omnimap/servo_twist_trace /omnimap/bridge_status \
  /omnimap/kinematic_status /omnimap/kinematic_constraint /omnimap/kinematic_model \
  /servo_server/delta_twist_camera /servo_server/delta_twist_cmds \
  /servo_server/status /joint_states /tf /tf_static \
  /ur_hardware_interface/safety_mode /ur_hardware_interface/robot_mode \
  /ur_hardware_interface/robot_program_running
```

额外录制**实际生效**的 Servo 输出和控制器反馈 topic，其名称从上一步有效参数/控制器查询得到。
Bridge、monitor、Planning 和 Servo 日志均保留，不用 `rostopic echo` 截图替代完整记录。

## C. 当天零速复核

不启动 OmniMap Servo/Planning。确认无同名 monitor 或桥接，再运行：

```bash
roslaunch omnimap_kinematics diagnostics.launch \
  start_bridge:=true start_servo:=false \
  input_topic:=/omnimap/acceptance_zero_input \
  > "$RUN/zero_diagnostics.log" 2>&1
```

若 Servo 尚未启动，先按工位既有程序启动并确认控制器；不能在已有 Servo 时再启动第二个。
专用输入不发消息时，桥接 watchdog 只输出零速，可以唤醒 low_latency_mode 的 Servo 状态发布。

观察至少 30 秒：真实关节速度接近静止噪声基线，Servo 输入 six components 全为零，
KinematicModel 在启动稳定后持续 valid，sigma/导数有限，status 新鲜且为 0，threshold_configured 可为 false。
不允许故障后继续沿用上一个 valid 样本。若相机/模型校验失败，先修复标定/TF，再进入运动阶段。

注意：没有输入时 BridgeStatus 的 input_missing 是预期行为，此时不能把它当作完整桥接链已通过。
既往完整零速链检查记录见 [现场验收](ur5_nbv_kinematics_validation_sumi.md#2026-10-26-2109-真实零速链路验收)。
结束本轮专用零速 launch 后，才启动下一阶段使用正常输入 topic 的 launch。

## D. 第一次真实运动：只采数据，不启用投影执行

本阶段使用已有 MoveIt/机器人限制、Planning 状态停车和小范围低速控制，
**Servo preview=observe 不会约束实际命令**。若工位尚未批准这种基线运动，先完成工位调试，不能用本步骤代替。

1. 示教器操作员在空旷范围选定正常工作构型。回到外部控制前确认没有示教器/其他程序与 Servo 同时控制。
2. 启动正常 topic 的诊断链：

   ```bash
   roslaunch omnimap_kinematics diagnostics.launch start_bridge:=true start_servo:=false \
     > "$RUN/motion_diagnostics.log" 2>&1
   ```

3. 算力端按 [三环入口](info_flow.md#三环解耦运行)启动 Tracking；Planning 的原启动参数追加：

   ```text
   --kinematic_feedback_mode enforce --kinematic_projection_mode off
   ```

   不同时保留两份 Planning。先让 Servo 缺席，只观察新鲜的 raw/final 命令。
   核验参考球位于本次批准范围、初始半径接近当前相机距离、所选语义目标正确、方向可预期。
   若目标缺失，应看到 should_stop；不得通过关闭语义掩码来强行启动运动。

4. 操作员确认后在执行端运行下列**首轮试验限速**，并用外部计时只观察首个约 1 秒运动段后正常停车：

   ```bash
   python info_flow/info_flow_servo_runtime.py \
     --config config/rtabmap_config.yaml \
     --kinematic_preview_mode observe --kinematic_preview_source stream \
     --linear_vel_max 0.005 --radial_vel_max 0.002 \
     --linear_accel_max 0.01 --linear_deadband 0.001 \
     --no-enable_angular --radial_i_gain 0 \
     --log_file --output "$RUN/baseline_servo"
   ```

   5 mm/s 是本 SOP 的初始测试上限建议，**不是安全认证值，也不是标定参数**；
   现场批准上限更低时使用更低值。相机速度上限不等于 tool0、肘部或所有连杆速度上限。
   shell 启动时间不等于实际运动时间；以 trace/实测数据确定运动段，不能只靠 sleep 计时。
   停车按 A 执行。若状态尚未就绪而不动，查具体失效原因，不改为 observe 状态门绕过。

5. 检查方向、位移、关节速度、TF/模型误差和停车后位移。通过后，才将单段扩大至约 3 秒，
   每次回到安全范围重新确认；在批准的小范围内覆盖两个切向方向、正常径向误差和至少三个构型。
   需要朝向控制时另做一轮：去掉 --no-enable_angular，明确加
   `--enable_angular --angular_speed_max 0.02 --angular_accel_max 0.05`，先核验初始朝向误差，
   再以同样的短段顺序进行；不直接继承 YAML 中非正 angular_speed_max 的不限速语义。

**每段通过条件**：命令和实测运动方向一致；相机/关节实际速度不超过本轮批准范围；
非零段模型/TF持续有效或失效后及时停车；没有保护停机、明显抖动、突跳、线缆牵拉。
没有实际运动只能记作“未覆盖”，不能因为日志无报错判通过。

## E. 用采样数据确定四项执行门槛

这一步由开发/记录人员分析 bag，现场操作员不需要凭经验填内部参数。
先划分采样段与独立验证段；固定 RobotModel、Dx/Dq、TCP/相机标定和测试速度，变更后重新核验。

| 配置项 | 通俗含义 | 必须提交的依据 |
|---|---|---|
| constraint_m_min | 距离不良构型多近时开始修正 | 多构型 sigma/归一化指标误差、批准工作范围与停车余量；与真实减速记录的比较（如有） |
| kinematic_execution_max_age | 模型最多允许旧多久，秒 | 模型源时间/接收时间/位姿年龄分布及最坏端到端时延；独立验证段内的覆盖情况 |
| kinematic_execution_joint_delta | 模型采样后关节最多允许变化多少，rad | 按关节名重排后的六维位置差 L2、模型线性化误差随差值的变化 |
| kinematic_execution_residual | 本周期待发命令与模型预测最多允许差多少，无量纲 | 同一参考点/坐标系下 Dx 归一化 task 差；不是实测跟踪误差 |

另行报告实际跟踪误差、停机时间/距离和速度/加速度。任务命令残差合格不能替代实测跟踪合格。
对每项给出单位、样本范围、最坏值、独立验证结果、采用余量、批准人和适用速度/构型范围。
不能只取平均值或 p99 便宣称覆盖最坏情况；不能为了减少停车把门槛一直放大。

特别是 m_min：不能直接填 MoveIt raw condition 的 17/30，不能把某个静止姿态的 sigma 当成阈值，
也不能仅凭 code 1 对应的一个值建立通用换算。只在批准工作范围内靠近其边缘采样，
**不以主动撞障碍、逼近机械极限或制造 Protective Stop 获取阈值**。
没有足够数据时标定未完成，停在本阶段；不使用合成测试的阈值。

形成 `$RUN/calibration.env`，四个值来自上述报告（此处故意不提供虚构数值）：

```text
M_MIN=<已批准的指标阈值>
MODEL_MAX_AGE_S=<已批准的秒数>
JOINT_DELTA_RAD=<已批准的六关节差L2上限>
TASK_RESIDUAL=<已批准的归一化命令残差上限>
```

这是一份待填模板，不可直接 source；替换占位内容为正有限数字后才成为 shell 配置。
后续 observe 流式日志记录 model_age、joint_delta 等用于交叉验证；
若 m_min 未配置，投影建议不可用是预期状态，不把它误判成求解器故障。

## F. 开启投影执行，做非零闭环验收

停止 Planning 新命令并确认零速后，加载批准参数，重启 monitor：

```bash
source "$RUN/calibration.env"
rosparam set /omnimap_kinematic_monitor/constraint_m_min "$M_MIN"
```

**注意**：diagnostics.launch 会加载基础 YAML，但不设置 m_min；私有参数必须在 monitor 启动前已设置。
退出本轮诊断 launch，再按 D.2 启动，确认阈值实际生效、threshold_configured=true。
同时确认普通桥接 filter_mode=none、input_topic 正确、模型六关节加速度边界均有效。
运行参数只存在内存时，重启 master 后必须重新设置；验收配置应作为部署配置另行保存。

Planning 保持 feedback=enforce、projection=off；执行端替换原 Servo 会话，不并行启动：

```bash
python info_flow/info_flow_servo_runtime.py \
  --config config/rtabmap_config.yaml \
  --kinematic_preview_mode enforce --kinematic_preview_source stream \
  --kinematic_execution_max_age "${MODEL_MAX_AGE_S:?missing approved value}" \
  --kinematic_execution_joint_delta "${JOINT_DELTA_RAD:?missing approved value}" \
  --kinematic_execution_residual "${TASK_RESIDUAL:?missing approved value}" \
  --linear_vel_max 0.005 --radial_vel_max 0.002 \
  --linear_accel_max 0.01 --linear_deadband 0.001 \
  --no-enable_angular --radial_i_gain 0 \
  --log_file --output "$RUN/enforce_servo"
```

沿 D 的短段顺序复测，不能在首次 enforce 时提高速度。若批准参数对应不同测试限速，先统一适用范围。
通过后再做单独的受限朝向控制轮次。

验收至少包含：

1. 正常构型真实非零运动；raw/final/trace/模型序号及各采样时间可关联，真实关节确有运动。
2. 在已批准范围内至少出现一次**约束实际激活**，trace 的实际速率与 Planning final 不同，
   实际 Twist 与投影结果一致。仅相机限速裁剪或一直无修正不能证明奇异指标半空间生效；
   须同时留存指标约束激活、修正前违反/修正后满足、关节/相机联合限值满足的分析。
   若批准范围内未触发此情况，记为未覆盖并补充受控测试设计，不能把 m_min 随意调高以凑结果。
3. 每个实际发布的非零周期都有当时有效的模型、状态和执行检查；没有退回原始未投影命令。
4. 相机/tool0 只变换一次，且实际运动跟踪误差、速度/加速度、停车距离满足本轮批准标准。

## G. 失效停车与恢复（逐项做，逐项记录）

先在静止时验证操作方法，再在相同的受限短段运动中做软件失效测试。
每次故障前记录 t0，记录第一条零命令 t_cmd0、实测关节静止 t_rest、停止期间最大位移。
静止判据用当天噪声基线预先确定；不要只凭一帧 velocity=0。
测试前填写允许的命令停止延迟、物理停止时间和距离，须小于工位余量；没有标准不能判通过。

| 测试 | 现场动作 | 通过条件 |
|---|---|---|
| Planning 断流 | Ctrl+C 本轮 Planning，保留执行端和录制 | 命令超时后零输出、实机停住；无新命令时持续保持零 |
| 仅模型断流 | 单独故障轮次使用下述模型 relay；停止 relay，保留 monitor、桥接和健康状态 | enforce 拒绝过期模型并停车，不沿用故障前方向 |
| monitor/健康反馈断流 | 停止本轮诊断 launch | 模型和健康失效后停车；此 launch 的 required 节点退出会连带关闭桥接，按联合断链记录 |
| 桥接退出 | Ctrl+C 本轮包含桥接的诊断 launch | 验证桥接退出零速及下游 watchdog 的实际停车；以实测为准，不假定永远收到了退出消息 |
| 上游相机/Tracking 中断 | 停止本轮 Tracking 的软件进程；不现场拔机器人控制/安全线缆 | 数据过期传播到 stop、执行端停车；重新建图后只使用新快照 |
| 恢复 | 保持 Planning 停止，恢复诊断/Tracking并确认健康；操作员重新确认后才启动 Planning | 新模型、健康窗口和新命令齐备才恢复；不使用故障前缓存；恢复后重做短段 |

仅模型断流测试的准备：静止时新开终端运行
`rosrun topic_tools relay /omnimap/kinematic_model /omnimap/acceptance_model_feed`，
本轮 F 的 Servo 命令额外加 `--kinematic_model_topic /omnimap/acceptance_model_feed`，
其他参数不改。确认完整新模型已收到后才进行短段运动，随后 Ctrl+C relay。
relay 只转发原消息，不更新时间戳、不伪造健康状态。不得把停止 required monitor 后整个 launch
退出的联合故障说成“仅模型断流”；缺少 topic_tools 时该单项记为未覆盖。

超时使用实际有效参数计算预期范围，包含采样、调度和链路余量，并分别报告各环节实测值。
例如 Servo incoming_command_timeout 是下游断流保护，不等于 Planning 超时或机械臂实际停止时间。
模型就绪代码默认健康窗口 0.1 秒也不是整个系统停止时间承诺。

真实硬件急停/安全停机测试应由工位负责人按该型号的既有验收规程执行，
先在静止状态检查，再按已批准方法测试运动停车；记录机器人安全模式和实际停止。
**不故意制造碰撞/保护停机、不在运动时杀驱动或拔控制/安全线缆来代替规定的硬件验收**。
如该部分未执行，软件断流通过也不能标记“硬件停车已通过”。

软件健康恢复允许自动继续接收新命令，因此每项故障后先保持 Planning 停止，
避免恢复诊断节点时未经人员确认便再次运动。Protective Stop 的人工排障/恢复与这套软件恢复门槛分开验收。

## H. 交付与判定

停止产生运动的程序，确认关节静止，最后结束 rosbag，执行 `rosbag info` 核验文件可读。
归档配置、两端版本、四项门槛报告、日志、bag、现场观察、故障事件表和签字。

每个用例填写：构型/速度、开始时间、预期、实际、是否真正运动/激活约束、t_cmd0、t_rest、
停止位移、残差、失败原因、是否通过。至少覆盖：零速、基线运动、投影非零及约束激活、
三类软件断流、恢复、实际跟踪、按工位规程的硬件停止。

判定只能是“通过 / 失败 / 未覆盖”；未覆盖不视为通过。
仅当适用范围内的所有用例、标定和硬件要求通过后，才能把本轮配置列为可部署配置。
若只通过静止/低速/有限构型，应明确限定范围，不外推到更高速度、不同负载或全工作空间。
