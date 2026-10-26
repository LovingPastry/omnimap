# 执行侧二维构型约束诊断

更新：2026-10-26。该接口已实现局部方向导数和半空间参数的诊断输出，尚未接入控制。
它补充 [执行端升级](ur5_nbv_kinematics_upgrade_sumi.md)，不替代现场标定和非零闭环验收。

## 数据链和计算定义

1. Servo 使用当前 TF、球面命令、径向 PI 状态和上一周期速度执行原有控制计算。
2. `ServoTwistTrace` 随实际 Twist 携带冻结状态下的相机参考点映射 `v_camera = B r + b_camera`。
   B 为 row-major 6×2，r 是原始两个球面速率，单位 rad/s。
   当前实现 B 的前三行为 `radius * [e_theta, e_phi]`，后三行为零；
   b 包含实际径向 PI 速度和限幅/斜率限制后的实际角速度。
   姿态误差控制在当前固定 pose 下独立于 r，不能用理想 look-at 角速度代替。
3. 线速度限幅、slew 或 deadband 生效及其边界处，`rate_model_valid=false`。
   停车、超时、TF 失败等输出没有运动模型；不复用上一次模型。
4. monitor 按实际 Twist 的 seq/stamp/frame/六分量匹配追踪消息，使用相同 TF 参考点规则
   将 B、b 一起移到 tool0。必须同时复现桥接输出；KF 修改输出后不匹配时拒绝模型。
5. 设 `S=diag(spherical_rate_scales)`，物理速率 `r=S u`，u 是无量纲坐标。
   在同一个 RobotState 的 base_link/tool0 Jacobian 上解 `J [A,bq] = [B_tool S,b_tool]`。
   要求 J 满秩，并检查 `||inverse(Dx)(J[A,bq]-[B_tool S,b_tool])|| <= 1e-6`。
   这没有复制 UR5 DH 参数，也没有由 status code 推测构型导数。
6. 对 `m(q)=sigma_min(inverse(Dx) J(q) Dq)` 沿 A 两列及 bq 中心差分，得到 a 和 drift：

```text
qdot = A u + bq
a_i = D_(A_i) m
drift = D_bq m
c = -alpha * (m - m_min) - drift
局部要求：a.T u >= c
```

每个非零方向先用关节扰动范数 1e-4 rad，再用一半步长复核。
两步长误差超过 `1e-5 + 0.01*abs(fine_derivative)` 时拒绝导数。
最小归一化奇异值 ≤1e-6，或两个最小奇异值间隙 ≤`1e-5*max(1,sigma_max)` 时拒绝，
避免把非可微点/精确奇异点的零中心差分当作可用逃逸方向。
最多额外进行 12 次扰动 Jacobian/SVD（两个速率方向及非零漂移，各四次），不做候选渲染或迭代 IK。

## 消息、参数与有效性

输出 `/omnimap/kinematic_constraint`，类型 `omnimap_msgs/KinematicConstraint`。
包含实际 joint/Twist/command 时间、Twist seq、tick/model version、参考球、速率尺度及线性化速率，
以及 a、drift、sigma_min、逆映射残差、差分误差、c 和失效原因。
源/末端 frame 为 base_link/tool0。`header.stamp` 是诊断发布时间，不能代替 joint_stamp 或 twist_stamp。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `spherical_rate_scales` | `[0.1, 0.1]` rad/s | 将两个速率转换为无量纲动作，改变它会改变最小距离的度量 |
| `constraint_alpha` | `1.0` s^-1 | 半空间衰减系数，正有限值 |
| `constraint_m_min` | 未设置 | 对当前 RobotModel、Dx/Dq 的指标阈值，需现场标定，设置时要求正有限值 |
| `kinematic_constraint_topic` | `/omnimap/kinematic_constraint` | 诊断输出 topic |

`valid=true` 仅表示当前局部导数通过上述检查；
`threshold_configured=true` 仅表示显式配置了阈值，**不证明该阈值经过标定**。
未设置阈值时仍可记录 a 和 drift，但 c=NaN、threshold_configured=false，禁止用于投影控制。
链无效、trace 缺失/不匹配、限速、模型不匹配、非可微指标和不可逆任务都发布 valid=false。
该诊断失败不会关闭原有 KinematicStatus：状态反馈和桥接 watchdog 继续独立工作。

## 构建与记录

本次系列改动扩展了 ServoTwistTrace 和 KinematicConstraint，ROS 消息 MD5 会改变。两端需同步代码、重建消息并重启使用它的节点，
旧节点和新节点不可混用。计算端无需 MoveIt，只构建消息；执行端构建 monitor：

```bash
./scripts/build_ros_ws.sh --with-kinematics
source /opt/ros/noetic/setup.bash
source ros_ws/devel/setup.bash
rostopic echo -n 1 /omnimap/kinematic_constraint
```

完整链录包时增加 `/omnimap/kinematic_constraint`，保留原始/最终球面命令、trace、bridge、status、joint 和 TF。
不能只录 a/c 而丢失线性化点及控制状态。inventory 脚本已采集新 topic。

## 控制接入前仍需完成

本接口描述冻结 pose/PI/slew 状态下的局部映射；有限速率修正可能进入限速或 deadband 分支，
后续周期的姿态、积分器和下游 MoveIt 也会变化。Planning 已订阅此接口并计算只读建议，未把投影结果发送到机器人。
消费者已校验双时钟年龄、命令来源/参考球和局部范围，并联合求解源周期的速度/slew 约束；控制接入仍须执行时重新校验、验证制动距离、实际关节加速度与真实执行残差。
不能投影后任意裁剪再宣称仍满足半空间；不可行时零速/重新规划，不复用旧方向。
恢复停车还需定义模型重新获取机制，避免等不到运动 trace 而无法恢复。

当前 0.1 s 的关节/命令容差只支撑诊断关联，不等于已验证运动时的局部模型精度。
现场需验证持续对齐、真实导数稳定性、m_min 标定、硬件失效停车与非零闭环效果。

本轮验证：消息/C++ 构建通过；InfoFlow 数值与模型测试 11 项、Planning/Servo 回调 9 项、
inventory 1 项通过；直接节点和正式 diagnostics.launch 两套隔离 ROS 测试各 9 项通过。
独立 NumPy FK 有限差分核对 a/drift；显式测试阈值只在隔离合成模型中设置，正式 launch 验证未配置时 c 为 NaN。
日志：`/tmp/omnimap_constraint_test.log`、`/tmp/omnimap_constraint_launch_test.log`。

## Planning 只读消费（2026-10-26）

`--kinematic_projection_mode observe` 为默认，只记录候选命令与投影建议；`off` 关闭计算。
此参数没有 enforce 选项；已有 `--kinematic_feedback_mode enforce` 仍仅控制状态停车/缩放。
原始 Fisher 与最终执行命令的发布顺序和数值不受投影观察影响。

| Planning 参数 | 默认值 |
|---|---|
| `--kinematic_constraint_topic` | `/omnimap/kinematic_constraint` |
| `--kinematic_constraint_timeout_sec` | `0.3` s |
| `--kinematic_constraint_max_sample_skew_sec` | `0.1` s |
| `--kinematic_projection_local_radius` | `0.25`（无量纲速率空间） |

收到约束后缓存独立副本；在 Fisher 计算及状态缩放完成后，按最新反馈计算建议，再发布本 tick 最终命令。
校验包括：

- 接收 monotonic 时间与 header/joint/twist 源时间均不过期；禁止零/非有限时间、超过 50 ms 的未来时间。
- joint 与 twist 采样时间差、source/task/header frame、上游 valid、显式阈值、有限尺度和指标。
- c 与 `-alpha*(sigma_min-m_min)-drift` 一致，逆映射残差不超过 1e-6。
- 来源 tick/model/command stamp 和速率必须对应 Planning **最近一次最终执行命令**，不能拿 raw Fisher 代替。
  新发布的 stop/命令会立即使旧约束来源不再匹配。
- 当前地图版本、当前候选参考球、源执行命令参考球一致；候选处于线性化速率的配置半径内；联合求解半空间与局部球交集中的最小修正。
- 策略已经 stop、状态反馈不可用/急停时不计算逃逸命令；无可用投影方向时只记录失败。

日志 `kinematic_projection:` 记录 tick、source tick、三个时间、有效性/原因、候选/建议速率和归一化修正量。
该局部半径是诊断筛选参数，未证明它足以控制真实线性化误差；不能作为速度或加速度上限。

InfoFlow 消费者测试 10 项、Planning/Servo 回调 10 项通过，覆盖建议不改控制、过期/未来消息、
源命令替换、参考球变化、未配置阈值、约束不一致、局部范围和恢复。未发送任何机器人运动命令。

## 联合可行集求解器（2026-10-26）

`project_feasible_spherical_command` 在无量纲 u 空间最小化 `0.5*||u-u_info||²`，
同时支持 `a.T u >= c`、`L u <= h`、`||M u+b|| <= r` 和局部球。
关节速度可用 `qdot=A u+bq` 的正负两组线性不等式表示；相机速度和每周期速度变化可用两组仿射范数约束表示。
半径必须为正有限值；禁用某限值应省略该约束，不传零或无穷大。

如果解析半空间投影已经满足全部约束，直接使用它；否则用 SciPy SLSQP 联合求解。
SciPy 已列在 environment.yaml 和执行侧依赖中。求解后独立检查归一化可行性残差（1e-9）、
非负乘子的 KKT 驻点与互补条件（1e-7 相对尺度），不只依赖求解器成功标志。
不可行、失败、非有限结果或无法确认最优性时返回失败，不裁剪结果。这些是数值容差，不是硬件安全裕量。

Planning 观察模式已使用半空间与局部球联合求解：旧实现会拒绝半空间投影在球外、但交集内仍有解的情况。
RobotModel 关节速度约束和源 Servo 周期的相机线速度/slew 约束现已通过 KinematicConstraint 接入；
因此没有增加连续控制 enforce 模式。源周期关节边界见下节，执行时仍须重新验证。

新增求解器测试 8 项通过，覆盖解析交点、限速后破坏半空间的反例、仿射偏置、不可行交集和伪成功结果；
消费者现为 11 项，实际 Planning/Servo 回调 10 项通过。均为软件验证，不代表真实机器人验收。

### 关节速度反馈链

KinematicConstraint 新增 `joint_names[6]`、row-major `joint_rate_matrix[12]`、
`joint_rate_bias[6]`、`joint_velocity_lower[6]` 和 `joint_velocity_upper[6]`。
矩阵就是生成方向导数时用同一个 Jacobian 求出的 A，偏置是 bq；输入是无量纲 u，输出为 rad/s。
上下限直接读取该 MoveIt RobotModel 的 min/max velocity，不能把 Dq 的正上限当作不对称下限。
顺序与模型 group 完全一致，不依赖 JointState 消息顺序。

Planning 检查名称数量/唯一性、数组形状、有限值及上下限符号，构造
`A u <= upper-bq` 与 `-A u <= bq-lower`，和半空间、局部球一起求解。
该约束是模型级速度界限，不包含 MoveIt 的动态碰撞减速、状态缩放或控制器实际跟踪误差，仍只输出观察建议。
此阶段随后补入下述源周期位置/加速度边界，仍不能解释为完整执行可行集。

### 源 Servo 周期的相机线速度限值

ServoTwistTrace 新增显式 `linear_limits_valid`、`previous_linear[3]`、`linear_speed_limit`、
`linear_step_limit`、`linear_deadband`。previous 是更新前的命令；step 是本周期实际 `linear_accel_max*servo_dt_sec`，
不是 Planning 周期推算值。零 speed/step 表示禁用，负 speed 的旧禁用语义在 trace 中规范化为零。
monitor 拒绝缺失有效标记、非有限或负限值，并把前三行相机映射乘以速率尺度后发布为
`camera_linear_matrix[6]`、`camera_linear_bias[3]`，和上一速度及限值一起传入 KinematicConstraint。
这组矩阵保留相机参考点；关节映射仍用经杠杆臂修正的 tool0 Twist，不能混用。

Planning 同时加入 `||M u+b||<=speed` 和 `||M u+b-previous||<=step`。
死区外部不是凸集，故沿源速度单位方向 n 加入保守半空间 `n.T(M u+b)>=deadband`；
该半空间可能拒绝转向很大的可行速度，因此建议是这个保守凸集合内的最小修正，不是整个非凸死区外部的全局最优。
实际执行来源的源速度已处于死区时拒绝模型；preview 的保守逃逸处理见后文。角速度在冻结 pose 下不随两个球面速率变化，其实际限幅后数值已包含于偏置。

这些上限只描述源 trace 的周期。下一执行周期的 pose、PI、previous 和 dt 都可能变化；
数值容差也可能在限幅边界产生微小误差。执行时必须重新获取并验证模型，不能把当前观察建议直接作为闭环保证。
实际命令本身已限幅、slew 或 deadband 生效时，motion trace 仍标记无效局部模型；preview 改为查询限幅前数学模型。自动运动恢复机制尚未完成。

本阶段消息/C++ 构建通过，InfoFlow 消费者 14 项、Planning/Servo 回调 11 项、求解器 8 项、
原有反馈/局部模型 11 项通过；隔离节点及正式 launch 两套 ROS 测试各 9 项通过。
新增测试验证实际周期 step、更新前 previous、相机参考点矩阵、死区保守投影及缺失限值标记拒绝。
日志：`/tmp/omnimap_linear_limits_test.log`、`/tmp/omnimap_linear_limits_launch_test.log`。

消息/C++ 构建通过；消费者现为 12 项、实际 Planning/Servo 回调 10 项通过。
直接节点和正式 diagnostics.launch 两套隔离 ROS 测试各 9 项通过，使用独立 NumPy FK/Jacobian
核对 A、bq、关节顺序和模型速度上下限。日志为 `/tmp/omnimap_joint_bounds_test.log` 和
`/tmp/omnimap_joint_bounds_launch_test.log`。合成模型验证不代表 UR5 实机验收。

### 源周期关节位置与加速度边界（2026-10-26）

ServoTwistTrace 传输实际 `servo_dt_sec`；KinematicConstraint 传输对应关节样本的
`joint_position`、`joint_velocity`，以及 RobotModel 的位置/加速度上下限和各关节 bounded 标记。
Planning 对有界关节联合求解 `q_min <= q + dt*(A*u+bq) <= q_max`，以及
`dq + a_min*dt <= A*u+bq <= dq + a_max*dt`。模型未配置的加速度上限不填猜测值；
日志 `acceleration_limits_complete` 明确标记是否六个关节均有上限。
这是一周期 Euler 预测和相对测量速度的约束，不证明真实执行加速度或制动距离安全；
源关节样本与 Twist 允许的小量时间偏差也仍需在执行重校验中处理。
两条消息 MD5 均改变，算力端和执行端必须重新构建并重启相关节点。

本阶段构建通过：消费者 15 项、Planning/Servo 回调 11 项、隔离 ROS 两套各 9 项通过。ROS 合成模型验证了位置边界传输与缺失加速度上限标记；加速度已配置的消费者数值分支通过，已补充 RobotModel 部分关节配置加速度上限的集成覆盖：j1=0.7、j3=1.3，其余关节保持缺失；按名称核对上下限、标记和 q/dq。生产 launch 的无上限配置路径也通过。未操作真实机器人。

### 只读候选查询

`omnimap_msgs/EvaluateKinematicModel.srv` 接收 `ServoTwistTrace candidate`，返回
`KinematicConstraint constraint`。monitor 以私有 `~evaluate_model` 提供服务。
候选必须使用 `reason=preview`、正确 command frame 和新鲜 command/Planning 时间戳；
关节选择使用候选时间而非旧桥接输入时间。查询沿用同一 RobotModel、TF、MoveIt 状态、
映射重建和导数检查；省略实际桥接输出一致性检查，因为候选尚未发布。
响应 reason 的 `advisory_preview:` 前缀区分候选结果与实际命令追踪结果。
服务不发布任何命令或反馈消息，不检查碰撞轨迹，也不授权控制；阈值未配置仍返回
`threshold_configured=false` 和不可用的 c。执行方仍须生成正确候选并在使用时重新校验。

候选查询构建通过，两套隔离 ROS 测试各 10 项通过；新增用例覆盖旧桥接过期后重新查询、模型/新关节样本一致性、请求过期、错误 frame 和 MoveIt 停车拒绝。未运行实机，Servo 已有显式只读预览方法，已有可选 observe 调度，尚未恢复运动。

Servo 候选计算已从实际周期拆出，preview 与 actual 复用计算及 trace 构造；PI/slew 状态仅在实际周期提交。回调测试覆盖重复预览无状态变化/无发布、同状态实际输出一致、过期/frame/无效数学模型在 RPC 前拒绝。预览查询在控制锁外执行，不构成自动恢复或执行许可。

本次 Servo 重构：回调 13 项、投影求解器 8 项、反馈/局部模型 11 项软件测试通过。包含非零径向积分和角速度 slew 的预览/实际值一致性；不代表实机控制验收。

预览响应现已逐项关联请求的 tick/model、时间/序号、frame、参考球、速率及相机模型/限值；以返回年龄和单调耗时拒绝过期结果，并以控制器周期编号和 PI/slew 状态拒绝并发周期后的结果。回调 14 项通过，含错配、过期、控制状态变化、数值相同的新周期及晚返回。后续已补入调用端有界等待与返回后最新 TF 复核，底层 RPC 强制取消、最新关节复核见后文，真正执行仍未实现。

SingleFlightQuery 现在以单个后台 worker 限制 RPC 调用端等待；超时不启动替代请求，晚返回结果丢弃。默认 preview timeout=0.1 秒。evaluate_preview 对输入做独立拷贝，并在返回后以零等待查询最新 TF，拒绝过期或与候选位姿不一致的结果（1e-9 数值容差）。有界查询 2 项、回调 15 项通过，包括真实阻塞、占用拒绝、晚结果丢弃、异常恢复及 TF 变化。未提供底层强制取消或自动执行；本机关节复核已在后续补齐。

本机关节预览复核已接入 evaluate_preview：PreviewJointState 按名称保存独立 q/dq、源时间和接收时间；双时钟新鲜度、重复/倒退/畸形最新样本失效、模型样本不晚于本机样本以及 q/dq 冻结一致性均需通过。topic 参数 kinematic_joint_topic 默认 /joint_states，应匹配实际 Servo/monitor。关节测试 5 项、回调 16 项通过；随后已补实际周期 dt 建议重算，仍无连续执行证明。

未来时间的本机关节样本在接收时拒绝，不推进已接受时间基线；后续正常新样本可恢复，相关测试已覆盖。

可选 kinematic_preview_mode=observe（默认 off）以独立 2Hz ROS Timer 关联 raw/final 命令并查询模型，使用名义 1/servo_hz 预览 dt，日志明确 dt_source=nominal。结果只存为建议且每轮先失效旧值，无命令发布；原始 policy stop、错配、陈旧或查询期间命令变化均拒绝。跨 Servo 周期仍拒绝，是冻结诊断而非连续执行；实际周期建议重算现已接入，投影执行/恢复仍未接入。回调 18 项通过。

实际周期 observe 现以 PreviewEvaluation 保留的完整 affine 映射/位姿进行重验证，并按当前有效 dt 和上一线速度重建约束。PI/角速度映射变化要求重新查询；投影结果再次通过真实计算输出一致性检查。preview/execution 来源角色分离。独立快照 worker 默认等待 3ms（最多名义周期四分之一），超时单个占用、不复用晚结果；停止分支先零速再诊断。消费者 16 项、回调 21 项通过，仍不执行建议或自动恢复。

preview 的 camera_rate_matrix/bias 现表示限幅前数学候选；command 可以不满足线速度/slew/deadband 限值，且 seq 是请求关联值。实际 motion trace 的语义保持原样。consumer 对 preview 死区内源按半空间偏好选择支撑方向（退化时尝试映射方向），仍求解保守凸子集；actual 来源死区内模型保持拒绝。投影后必须核对实际输出，不能执行超限候选。

本批构建通过：回调 23 项、反馈/局部模型 12 项、消费者 16 项、求解器 8 项及两套隔离 ROS 各 10 项，共 79 项软件检查通过。ROS 用例覆盖桥接过期后查询超限数学候选并传回限值。未执行实机运动。

新增独立 ProjectionReadinessGate：基于 KinematicStatus 的双时钟新鲜度、frame/链/指标和 MoveIt 0/1 检查，以至少两条递增反馈和默认 0.1s 健康窗口允许建议。异常立即阻断，坏样本/断流重置恢复；有效时间的新停车反馈建立基线，阻止旧健康重放恢复。实际周期求解前后均检查，晚到停车丢弃结果。状态机 6 项、回调 26 项通过；仅是控制恢复的健康门槛，不执行/恢复运动。

健康准备未就绪时，回调立即清除缓存模型/建议并推进健康代次；查询与求解在同一代次才可保存结果，存储检查与反馈更新由同一锁串行化。因此即使工作期间先停车再快速恢复健康，也不能复用停车前结果，须重新查询。测试已覆盖该交错顺序。

新增 servo_preview.test 端到端验证（4 项通过）：真实 Servo 订阅/Timer、RPC、monitor 和桥接在合成 /test/* 场景自动生成当前周期建议，覆盖 MoveIt halt、关节断流和原始 policy stop。最终命令始终 stop，捕获输出全为零；未运行真实地图/Fisher或实机，不证明运动中连续控制。构建和 CMake rostest 注册通过。
