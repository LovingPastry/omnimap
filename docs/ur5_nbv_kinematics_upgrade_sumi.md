# UR5 执行端运动学反馈升级

更新：2026-10-26。当前主机为 sumi，允许修改 `~/ros/catkin_ws/src`。
Git 管理实现位于工作区 `ros_ws/src/omnimap_kinematics`，通过包链接部署到机器人工作区。
旧的“仅 SSH 只读检查”限制不再适用。[计算端分工](ur5_nbv_kinematics_upgrade_local.md)、
[接口细节](ur5_nbv_kinematic_constraint.md)和[现场验收记录](ur5_nbv_kinematics_validation_sumi.md)分别维护。
目前已有诊断、状态停车和只读投影建议；连续投影执行与完整实机验收尚未完成。

## 执行链与单一模型来源

```text
OmniMap Servo → delta_twist_camera
  → camera_twist_bridge.py → delta_twist_cmds → MoveIt Servo
                        ↘ BridgeStatus / KinematicStatus
MoveIt monitor → KinematicConstraint / EvaluateKinematicModel / KinematicModel
```

两路 Twist 均以 base_link 表达，相机参考点为 cam_1_color_optical_frame，末端为 tool0。
桥接只做 `v_T=v_C-omega×(p_C-p_T)`，不改为 tool0 frame，不重复参考点变换。
普通/KF 桥接只能保留一个控制 publisher。KF 的动态滤波不包含在当前候选仿射模型中。
桥接 code 0/1 允许有效输入，2–5/未知或过期状态停车，另有墙钟 watchdog。

monitor 使用实际 Servo 的模型/group/frame/关节边界和 topic 配置；缺少参数时等待，不猜另一套模型。
URDF 参数使用 robot_description_param 指向全局，避免私有 robot_description 字符串遮蔽 URDF。
最多缓存 200 个新鲜关节样本，选取 TF 覆盖的同采样 RobotState；最新流异常仍立即失效。
raw/normalized condition 分开记录，归一化指标为 `sigma_min(Dx^-1 J_B Dq)`；
Dx 默认为 `[.10,.10,.10,.15,.15,.15]`，Dq 来自实际模型速度上限。

## 三类只读接口

| 接口 | 当前用途 |
|---|---|
| KinematicConstraint | 关联实际 Twist 追踪，发布二维导数和可用约束；Planning 只记录建议 |
| `~evaluate_model` | 冻结候选查询，不依赖旧桥接输入，不发布命令；用于 query 来源 |
| KinematicModel | 按 publish_hz 连续发布同采样 J、相机位姿/偏移、完整梯度和边界；用于 stream 来源 |

KinematicModel 的 header 为发布时间，joint_stamp 为 J/相机位姿/关节样本时间；
status_received_stamp 仍只是 Int8 状态接收时间，不能宣称同一 MoveIt 内部周期。
model_sequence 递增用于缓存/重启检测。关节、TF、状态失效或指标不可微时发送 valid=false。
模型流不依赖原始命令/旧桥接；KinematicStatus 的完整链验证仍独立执行。
m_min 默认未配置，此时 c 不可用于投影，不以合成测试阈值代替真实标定。

## Servo observe 模式

默认 `--kinematic_preview_mode off`。observe 也不修改实际运动命令，来源二选一：

- query（默认来源）：独立 Timer 查询冻结数学候选，检查请求/响应关联、返回后的 TF/关节及控制器周期。
- stream：每个实际周期在模型快照自己的位姿上重算候选和约束，不发 RPC、不启动名义预览 Timer。

原始/最终球面命令须同 tick/model、来源时间/frame 和参考球，并在求解前后仍相同。
原始 policy stop 禁止建议，最终状态 stop 可评估；停车分支先发布零速和清理状态，再做诊断。
候选允许描述限幅前数学值，投影后必须用真实控制计算核对；超限候选不会发往控制 topic。

当前联合约束含半空间、局部球、实际模型速度、可用位置/加速度边界和相机 speed/slew/deadband。
位置是一步 Euler 预测，加速度是相对测量速度的一周期界限；不是制动距离或真实加速度证明。
死区使用保守凸支撑半空间；不可行、失败或数值校验不通过时建议失效，不复用旧方向。
数值 worker 默认等待 3ms（最多名义周期四分之一）；超时保留唯一在途槽，晚结果丢弃。

ProjectionReadinessGate 在求解前后检查 KinematicStatus 的 frame/链/指标、双时钟年龄与 code 0/1。
恢复至少需两条递增健康反馈和默认 0.1 秒健康窗口；故障清缓存并改变健康代次。
stream 还检查模型序号/故障代次及源时间晚于最近健康故障，防止延迟旧模型跨故障复用。
日志分别记录模型年龄、相对最新关节/TF 的差值和实际周期 dt；这些差值尚未成为经标定的执行裕量。

## 构建与部署

```bash
conda activate InfoFlow
./scripts/build_ros_ws.sh --with-kinematics
python3 scripts/deploy_sumi_kinematics.py
source /opt/ros/noetic/setup.bash
source ~/ros/catkin_ws/devel/setup.bash
source ros_ws/devel/setup.bash --extend
```

部署脚本先检查构建和冲突，保留旧桥接脚本备份，再设置包链接/兼容入口与运行依赖。
它会检查 KinematicModel 和 EvaluateKinematicModel 的生成文件，不能拿旧消息构建替代完整部署。
修改消息/新增类型后，两端重建并重启相关节点。快速检查使用 InfoFlow，构建脚本自行选择 ROS Python。

```bash
roslaunch omnimap_kinematics diagnostics.launch start_bridge:=true
```

已有驱动但没有 Servo 时增加 start_servo:=true，复用标准 ur5_servo.launch；不启动驱动或切换控制器。
仅 monitor 时省略 start_bridge。自定义命名空间用 servo_ns；状态 topic 解析实际 Servo 配置，必要时显式覆盖。
不要同时运行旧桥接和 launch 内桥接。

在已有运行参数上启用流式诊断的选项：

```bash
python info_flow/info_flow_servo_runtime.py --kinematic_preview_mode observe \
  --kinematic_preview_source stream --kinematic_model_topic /omnimap/kinematic_model \
  --kinematic_joint_topic /joint_states --kinematic_status_topic /omnimap/kinematic_status
```

以上为部署命令示例，本轮未在真实机器人上执行；topic 必须匹配实际配置。

## 隔离测试与现场证据

```bash
source /opt/ros/noetic/setup.bash
source ros_ws/devel/setup.bash
rostest omnimap_kinematics kinematics.test use_sim_time:=true
rostest omnimap_kinematics diagnostics_launch.test
rostest omnimap_kinematics servo_preview.test
rostest omnimap_kinematics servo_stream.test
```

前两套用独立 NumPy FK/有限差分校验实际 C++ J、完整梯度、a/drift、参考点与关节顺序，
并覆盖时间/TF/状态失效。后两套实际运行 Servo 的订阅/Timer、monitor 和桥接，
覆盖查询或流式建议、关节断流、MoveIt 停车及新模型恢复；stream 另验证关节变化后的连续建议更新。
所有输入是合成关节/TF/球面命令，最终命令始终 stop，捕获控制输出全部为零。
这不是实际机器人运动或真实地图/Fisher/GPU 的验收。

现场需录制 raw/final SphericalCommand、ServoTwistTrace、BridgeStatus、KinematicStatus、
KinematicConstraint、KinematicModel、实际 MoveIt status、joint_states 和 TF，并保留有效配置。
可用 `python3 scripts/collect_sumi_kinematics.py` 辅助采集，完整记录要求见验收文档。
2026-10-25 的真实关节零速样本不能证明持续对齐或非零控制有效。

## 剩余门槛

1. 将流式采样模型用于执行前，处理运动时延、模型/线性化和实际执行残差及可用裕量。
2. 完成连续投影执行、完整恢复状态机和实际桥接/滤波映射验证，保留原始命令及失效零速。
3. 用现场真实模型/减速样本标定 m_min，验证关节顺序、相机标定与模型一致性。
4. 完成硬件断链、恢复和非零闭环验收；不绕过 MoveIt 停车或机器人保护停机。
