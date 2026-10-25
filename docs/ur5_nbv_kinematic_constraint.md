# 执行侧二维构型约束诊断

更新：2026-10-25。该接口已实现局部方向导数和半空间参数的诊断输出，尚未接入控制。
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

本次扩展了 ServoTwistTrace，ROS 消息 MD5 会改变。两端需同步代码、重建消息并重启使用它的节点，
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
后续周期的姿态、积分器和下游 MoveIt 也会变化。Planning 尚未订阅此接口，也未把投影结果发送到机器人。
接入时必须校验双时钟年龄、命令来源/参考球和线性化点，处理速度/加速度约束与真实执行残差。
不能投影后任意裁剪再宣称仍满足半空间；不可行时零速/重新规划，不复用旧方向。
恢复停车还需定义模型重新获取机制，避免等不到运动 trace 而无法恢复。

当前 0.1 s 的关节/命令容差只支撑诊断关联，不等于已验证运动时的局部模型精度。
现场需验证持续对齐、真实导数稳定性、m_min 标定、硬件失效停车与非零闭环效果。

本轮验证：消息/C++ 构建通过；InfoFlow 数值与模型测试 11 项、Planning/Servo 回调 9 项、
inventory 1 项通过；直接节点和正式 diagnostics.launch 两套隔离 ROS 测试各 9 项通过。
独立 NumPy FK 有限差分核对 a/drift；显式测试阈值只在隔离合成模型中设置，正式 launch 验证未配置时 c 为 NaN。
日志：`/tmp/omnimap_constraint_test.log`、`/tmp/omnimap_constraint_launch_test.log`。
