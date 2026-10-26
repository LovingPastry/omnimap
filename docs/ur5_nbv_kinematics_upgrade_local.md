# OmniMap 计算端运动学反馈升级

更新：2026-10-26。工作区 `/home/sumi/experience/omnimap`，通过 Git 同步到算力端。
本文描述当前实现；阶段数值和历史验证见[约束接口](ur5_nbv_kinematic_constraint.md)。
执行端部署见 [sumi 升级](ur5_nbv_kinematics_upgrade_sumi.md)，现场要求见[本机验收](ur5_nbv_kinematics_validation_local.md)。
软件侧已接入最终 Twist 的流式投影执行及故障恢复；默认关闭。实机标定与验收尚未完成。
现场操作顺序、启动命令、标定交付物及停车/恢复判据见[真实运动验收 SOP](ur5_nbv_real_motion_sop.md)。

## 当前分工

| 组件 | 当前职责 |
|---|---|
| Planning | Fisher 原始命令、最终命令、状态门及只读约束建议；不加载另一套 RobotModel |
| Servo runtime | 本机 TF、径向 PI/朝向/速度/slew 控制；实际 Twist 追踪；可选查询或流式建议 |
| C++ monitor | 实际 MoveIt 模型/group/边界、归一化指标、二维导数、连续同采样模型流 |
| 普通/KF 桥接 | 相机到 tool0 参考点变换、状态停车、墙钟 watchdog；只能运行一个控制 publisher |

主要源码：`info_flow/info_flow_planning_node.py`、`info_flow/info_flow_servo_runtime.py`、
`info_flow/kinematic_feedback.py`、`info_flow/kinematic_constraint.py`、
`info_flow/kinematic_projection.py`、`info_flow/kinematic_model.py`、
`ros_ws/src/omnimap_kinematics/src/omnimap_kinematic_monitor.cpp`。

## 模式及默认值

| 参数 | 默认 | 行为 |
|---|---|---|
| Planning `--kinematic_feedback_mode` | observe | 记录状态；enforce 才对持续 code 1 减速、对 2–5/未知/失效停车 |
| Planning `--kinematic_projection_mode` | observe | 只计算/记录建议；off 禁用 |
| Servo `--kinematic_preview_mode` | off | observe 只记录；enforce 将流式投影接入最终 Twist |
| Servo `--kinematic_preview_source` | query | query 为冻结候选 RPC；stream 为每个实际周期使用连续模型快照 |

原始 Fisher 命令独立发布到 `/omnimap/spherical_cmd_raw`；最终命令为 `/omnimap/spherical_cmd`。
两者按 tick/model、来源时间/frame 和参考球关联，Header.seq 由不同 publisher 分配，不能据此关联。
原始策略 stop 禁止建议；最终状态反馈 stop 可以只读评估，但绝不借此恢复运动。
Planning 的状态 enforce 不等于连续投影 enforce，不从状态码推导 Jacobian 或逃逸方向。

## 坐标与模型契约

相机参考点为 cam_1_color_optical_frame，末端参考点为 tool0；速度均以 base_link 表达。
`/servo_server/delta_twist_camera` 经桥接到 `/servo_server/delta_twist_cmds`，只做一次：

```text
v_T^B = v_C^B - omega^B × (p_C^B - p_T^B)
omega_T^B = omega_C^B
```

没有改为 tool0 frame，也不使用 cam_1_baselink。monitor 从实际 Servo 参数读取模型/group/topics，
关节按名称装配，并检查 TF 与模型一致。指标为 `Jbar=Dx^-1 J_B Dq`、`m=sigma_min(Jbar)`，
Dq 来自实际 RobotModel 速度上限；归一化 condition 不能共用 raw condition 的 17/30 阈值。
m_min 必须显式标定配置，默认未配置时 threshold_configured=false、c=NaN。
Int8 status 只有接收时间，不能证明它与 Twist 属于同一 MoveIt 内部周期。

## 查询与流式建议

query 使用 `~evaluate_model` 服务（默认 `/omnimap_kinematic_monitor/evaluate_model`）。
候选计算不提交 PI/slew 状态、不推进 Servo 时钟、不发布命令。名义预览 dt 为 1/servo_hz，
日志标记 dt_source=nominal。请求和响应关联、新鲜度、返回后 TF/关节及控制器周期均检查；
冻结条件变化会拒绝结果。单个后台 RPC worker 到期使调用端返回，不能强制取消底层阻塞操作；
它实际结束前不创建替代任务，晚结果不复用。

stream 订阅 `/omnimap/kinematic_model`，不使用查询 RPC 或名义预览 Timer。
同一 monitor 连续发布同关节/TF 采样的 J、相机 c2w/参考点偏移、完整指标梯度和实际模型边界。
每个实际 Servo 周期用快照自己的位姿重新计算相机数学候选，再重建二维约束。
Python 不加载另一套机器人模型；不会给旧位姿候选仅改时间标签。
模型、关节和 status 源时间以及接收时间都必须新鲜，且不能早于最近健康故障门槛。
重放、失效样本或发布器序号重启使旧缓存/在途结果失效；重启后的递增新样本才重新使用。

流式建议描述采样状态，不冒充最新或未来物理状态。日志记录 model_seq/model_age、
joint_delta、velocity_delta、joint_stamp_delta、pose_delta、rotation_delta 和实际有效 dt。
模型与最新本机样本可以不同；差值用于观察，尚未作为经标定的执行误差裕量。

```bash
python info_flow/info_flow_servo_runtime.py --kinematic_preview_mode observe \
  --kinematic_preview_source stream --kinematic_model_topic /omnimap/kinematic_model \
  --kinematic_joint_topic /joint_states --kinematic_status_topic /omnimap/kinematic_status
```

这是部署选项，未在真实机器人上执行。joint/status/model topic 应匹配实际执行端配置。

## 当前周期约束与恢复门槛

目标仍为二维速率上的最小修正；物理速率 `rates=rate_scales*u`，约束包括：

- `a.T*u >= c`，以及局部模型球；a/drift 来自实际模型，c 依赖显式 m_min。
- RobotModel 关节速度界限，以及有界关节的一周期位置预测。
- 模型提供的关节加速度界限；缺失时明确标记，不填猜测值。
- 源控制器状态下的相机线速度、每周期 slew 和死区约束。

实际周期按 Servo 使用的有效 dt 重建位置/加速度/slew 限值；dt 有既有启动默认及上下限处理。
PI/角速度偏置变化时 query 旧模型拒绝，stream 在该周期重新构造候选。
速度/slew/deadband 生效的源可以作为限幅前数学预览，绝不作为超限执行命令发布。
死区外部为非凸集，当前求保守支撑半空间内的最小修正，不保证整个非凸集合全局最优。
投影后再通过真实控制计算核对输出，不允许任意裁剪修复结果。

数值求解使用独立控制器快照/单个 worker，等待预算默认 3ms，最多名义周期的四分之一。
超时或不可行则建议失效，不保留上一周期建议。停车分支先发零速，再进行有界诊断。
ProjectionReadinessGate 要求新鲜且有效的 KinematicStatus、MoveIt 0/1、至少两条递增健康样本，
以及默认 0.1 秒健康窗口。故障清模型/建议并推进健康代次；求解前后均检查，旧工作不能跨故障恢复。
observe 仅记录建议；enforce 通过下节额外门槛后才发布投影结果，不解除或绕过保护停机。


## 显式 enforce 执行

`--kinematic_preview_mode enforce --kinematic_preview_source stream` 将投影接到最终 Twist。
默认仍为 off；enforce 不支持 query，并要求显式提供三个正数限值（没有实机猜测默认值）：

| 参数 | 单位/含义 |
|---|---|
| `--kinematic_execution_max_age` | 秒；模型/位姿源时间和模型接收年龄上限 |
| `--kinematic_execution_joint_delta` | rad；最新六关节位置相对模型样本的 L2 差上限 |
| `--kinematic_execution_residual` | 无量纲；实际 task Twist 与模型预测之差的 Dx 归一化 L2 上限 |

还要求 monitor 显式配置 constraint_m_min（私有参数，重启 monitor 生效），模型提供全部六关节加速度界限，且实际桥接启动配置为 none。
`--kinematic_execution_bridge_node` 默认 /omnimap_camera_twist_bridge；启动时检查该节点的
filter_mode 和 input_topic，拒绝 KF、缺失配置或错误输入 topic。改配置后应重启相关节点。

enforce 以 Planning 状态减速后的最终速率为目标；约束投影仍可改变幅值/方向，最终 should_stop 始终停车。
发布前在本周期最新相机位姿重算投影命令，拒绝额外限幅，获取同时间 tool0 TF，检查 task 残差，
将该 task Twist 经模型 J 反解后再次检查关节速度、带最新 q/dq 差值裕量的位置/加速度及指标约束。
命令、健康/模型代次和时间在最终发布处再校验；失败、超时、不可行、未标定或缺界限都发零速，
不会退回未经投影的原速率。PI/slew 仅在成功发布周期提交。

原始/Planning 最终 SphericalCommand 保持不变；ServoTwistTrace 记录实际投影速率及线角速度。
kinematic_execution 日志用 tick、model_seq、command/joint/twist 时间关联模型与实际输出。
使用本地 enforce 时，Planning 建议设置 `--kinematic_projection_mode off`，避免重复建议；
Planning 的反馈状态门仍可保留。

这里的 task 残差比较本周期待发布命令与模型预测，不是机械臂实测跟踪误差。
这些是离散采样模型上的执行检查，不是实机制动距离或连续时间硬安全证明。
实机使用前仍须测定三个限值和 m_min，核验模型/标定及实际执行误差。

## 构建与证据

```bash
conda activate InfoFlow
./scripts/build_ros_ws.sh
source /opt/ros/noetic/setup.bash
source ros_ws/devel/setup.bash
python tests/test_kinematic_runtime.py
python tests/test_kinematic_model_cache.py
python tests/test_projection_readiness.py
```

算力端默认只构建消息；执行端用 `--with-kinematics`。新增 KinematicModel 需两端生成并重启相关节点。
ROS 验证命令见执行端文档：独立 FK/完整梯度/移动样本转换、查询链、流式链和故障恢复已有合成测试。
observe 测试保持零输出；新增 servo_execution.test 用非零投影命令驱动独立 FK/Jacobian 理想关节模型，
验证限速、故障零速和健康/新模型恢复。上述测试未运行真实地图/Fisher/GPU 或实际机器人。
既往真实关节零速诊断不能替代持续对齐、标定或非零闭环验收。

## 实机阶段仍需完成

2026-10-26 20:50（Asia/Shanghai）只读现场检查：ROS master、UR 驱动和 `/servo_server`
在线，`/joint_states` 有真实零速样本；但 monitor、桥接和 OmniMap Twist trace 尚无发布者，
`constraint_m_min` 未设置。因此当前现场还没有投影执行或闭环验收证据。
运行中的 `/robot_description_planning/joint_limits` 为六关节速度上限 π rad/s、
加速度界限开启且上限 5 rad/s²，与磁盘 YAML 不同；这些是读取到的配置值，不是实测标定。
执行端证据和下一步见 [sumi 现场核对](ur5_nbv_kinematics_upgrade_sumi.md#本轮只读现场核对)。

21:09 后续零速现场验证已通过短时模型/TF 和桥接链检查：220 条有效模型、217 条有效链反馈，
两轮 Servo 输出全零、实测关节速度全零。20:55 的状态缺失由 Servo 低延迟等待输入造成，
启动零速桥接后恢复；无需放宽健康门槛。此证据只覆盖当前静止构型约 12 秒的两轮检查，
不替代运动中对齐、m_min 标定、执行误差和硬件停车验收。

1. 现场标定 m_min 和执行年龄/关节变化/任务残差限值，验证适用工况与真实误差裕量。
2. 核验真实模型、关节顺序、相机标定、普通桥接配置和执行反馈；确认实际速度/加速度及停车效果。
3. 完成硬件断链、保护停机后的人工恢复边界和非零闭环验收，保留完整数据与有效配置。
