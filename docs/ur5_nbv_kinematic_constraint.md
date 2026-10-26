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
消费者已校验双时钟年龄、命令来源/参考球和局部范围，并联合求解源周期的速度/slew 约束；控制接入仍须执行时重新校验、处理关节加速度/位置约束与真实执行残差。
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
因此没有增加连续控制 enforce 模式。关节加速度需要对应采样的上一执行速度和周期，不能凭任意 dt 推导。

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
没有新增关节加速度或位置约束，不能把这一阶段解释为完整执行可行集。

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
源速度已处于死区时拒绝模型。角速度在冻结 pose 下不随两个球面速率变化，其实际限幅后数值已包含于偏置。

这些上限只描述源 trace 的周期。下一执行周期的 pose、PI、previous 和 dt 都可能变化；
数值容差也可能在限幅边界产生微小误差。执行时必须重新获取并验证模型，不能把当前观察建议直接作为闭环保证。
源命令本身已限幅、slew 或 deadband 生效时，仍发布无效局部模型；启动/停车恢复机制尚未完成。

本阶段消息/C++ 构建通过，InfoFlow 消费者 14 项、Planning/Servo 回调 11 项、求解器 8 项、
原有反馈/局部模型 11 项通过；隔离节点及正式 launch 两套 ROS 测试各 9 项通过。
新增测试验证实际周期 step、更新前 previous、相机参考点矩阵、死区保守投影及缺失限值标记拒绝。
日志：`/tmp/omnimap_linear_limits_test.log`、`/tmp/omnimap_linear_limits_launch_test.log`。

消息/C++ 构建通过；消费者现为 12 项、实际 Planning/Servo 回调 10 项通过。
直接节点和正式 diagnostics.launch 两套隔离 ROS 测试各 9 项通过，使用独立 NumPy FK/Jacobian
核对 A、bq、关节顺序和模型速度上下限。日志为 `/tmp/omnimap_joint_bounds_test.log` 和
`/tmp/omnimap_joint_bounds_launch_test.log`。合成模型验证不代表 UR5 实机验收。
