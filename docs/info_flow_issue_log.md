# InfoFlow 问题记录与后续分析计划

更新时间：2026-10-16

本文记录对 `info_flow/` 真机三节点（Tracking、Planning、Servo Runtime）的静态审阅结果。当前只记录问题，不在本轮修改实现。

## 一、工程与部署问题

### P0：必须先处理

1. **Planning 位姿过期阈值被硬编码为 `1e6`**
   - 位置：`info_flow/info_flow_planning_node.py:60-63, 580-628`
   - 影响：Tracking 停止或位姿冻结后，Planning 仍会持续发布基于旧位姿的命令。
   - 建议：使用 `pose_stale_timeout_sec` 的配置值；同时检查 ROS 时间和本地接收时间，并在 Fisher 计算完成后再次检查。

2. **Servo 只按本地接收时间判断命令新鲜度**
   - 位置：`info_flow/info_flow_servo_runtime.py:209-238`
   - 影响：旧命令在网络重连、队列延迟后仍可能被视为新命令。
   - 建议：校验 `header.stamp`、`planner_tick`、`model_version` 的单调性，并设置独立的硬超时上限。

3. **球坐标转换把下半球错误映射到赤道**
   - 位置：`info_flow/servo_runtime_common.py:193-207`、`info_flow/distributed_common.py:232-246`、`omnimap/gaussian/renderer/nbv/motion_policy.py:219-234`
   - 影响：相机越过场景中心水平面后，切向和径向方向不再对应真实位置。
   - 建议：统一使用 `clip(z, -1, 1)`，或在系统设计中明确禁止越过半球边界并停车。

4. **配置中的 `angular_speed_max: -0.05` 会关闭角速度限幅**
   - 位置：`config/rtabmap_config.yaml:82`；非正值语义见 `motion_policy.py:424-432` 和 `info_flow_servo_runtime.py:156-190`
   - 影响：角速度只受加速度斜率限制，可能逐步升到危险值。
   - 建议：配置使用正值；启动时拒绝非正值、NaN 和 Inf。

5. **Servo 异常和退出路径不保证发零速度**
   - 位置：`info_flow/info_flow_servo_runtime.py:308-337, 600-601`
   - 影响：TF 解析异常或进程退出时，最后一个非零 Twist 可能继续存在，系统依赖下游 watchdog。
   - 建议：TF 查询统一捕获异常并立即发零；注册 `rospy.on_shutdown`；最好再加独立的 wall-clock watchdog。

6. **控制消息和配置缺少有限性、范围校验**
   - 位置：`info_flow/info_flow_planning_node.py:421-453`、`info_flow/info_flow_servo_runtime.py:370-472`
   - 影响：NaN/Inf 半径、中心、角速度或增益可能绕过比较逻辑并进入 Twist 消息。
   - 建议：所有输入和最终线/角速度都做 `isfinite` 与上下限检查，失败时停车。

### P1：应在联调前处理

7. **快照没有心跳或年龄检查**
   - 位置：`info_flow/info_flow_planning_node.py:350-404, 513-578`
   - 影响：Tracking 停止后，Planning 会永久复用旧快照。
   - 建议：增加 snapshot receipt timeout、版本连续性和 `run_id` 校验；超时发布 stop。

8. **Fisher 计算前后没有重新检查位姿新鲜度**
   - 位置：`info_flow/info_flow_planning_node.py:580-690`
   - 影响：一次耗时较长的规划可能发布基于过期位姿的命令。
   - 建议：发布前重新读取位姿代数/时间戳，并验证计算耗时没有超过 deadline。

9. **世界坐标速度可能被标记为其他控制坐标系**
   - 位置：`info_flow/info_flow_servo_runtime.py:466-471`
   - 影响：当 `cmd_frame != world_frame` 时，线速度和角速度方向错误。
   - 建议：强制两者相等，或显式使用 TF 旋转 Twist。

10. **Planning 与 Servo 的 look-at 实现不完全一致**
    - 位置：`info_flow/servo_runtime_common.py:219-246` 与 `omnimap/util/lookat.py:40-80`
    - 影响：奇异方向附近 fallback 阈值不同，可能产生相机 roll 跳变。
    - 建议：两端直接复用同一个 look-at 实现。

11. **Tracking 关键帧门控在图像解码/入队成功前就提交状态**
    - 位置：`info_flow/slam_frontend.py:52-116`、`info_flow/info_flow_tracking_node.py:329-372`
    - 影响：解码失败的帧仍成为下一次运动比较基准，可能跳过真正需要的关键帧。
    - 建议：成功入队后再提交 gate 状态，或失败时回滚。

12. **Tracking 发布图像时间戳，而不是实际 TF 返回时间戳**
    - 位置：`info_flow/info_flow_tracking_node.py:284-318`
    - 影响：可能掩盖 TF 数据的实际延迟，导致规划端错误判断新鲜度。
    - 建议：保留并传播 `lookup_pose_from_tf` 返回的 transform stamp，同时记录图像与 TF 的时间差。

13. **Tracking worker 的张量转换在保护性 try 外部**
    - 位置：`info_flow/info_flow_tracking_node.py:395-406`
    - 影响：异常任务可能直接杀死后台线程，主节点仍保持运行但不再更新地图。
    - 建议：把任务解包放进 try，并监控 worker 是否退出。

14. **缺少输入时 Planning 不主动发送 stop**
    - 位置：`info_flow/info_flow_planning_node.py:528-578`
    - 影响：短时间内依赖 Servo 超时停车；若超时配置过大，残留运动持续时间过长。
    - 建议：初始化阶段可节流发送 stop，运行中丢失输入时立即发送 stop。

15. **Servo 配置文件错误会静默回退到默认值**
    - 位置：`info_flow/servo_runtime_common.py:271-282`
    - 影响：执行侧配置缺失或 YAML 错误时，实际控制参数可能与 Planning 端不一致。
    - 建议：至少打印明确告警；真机模式下配置加载失败应拒绝启动。

16. **同名话题在单体节点和三节点版本中消息类型不同**
    - 位置：`info_flow/info_flow_node.py:314-315` 与分布式 Planning 的 `SphericalCommand`
    - 影响：两套节点若同时运行在同一 ROS master，会发生 `/omnimap/spherical_cmd` 类型冲突。
    - 建议：单体节点改用独立话题，或明确禁止两套入口并存。

## 二、算法漏洞待验证

1. **半球约束在 Planner 和 Servo 中不一致**：策略允许大圆路径接近/穿越边界，但 Servo 的坐标恢复会把下半球压到赤道。需要统一“允许跨越”还是“边界停车”。
2. **Fisher 梯度与实际执行方向可能失配**：Planning 在冻结快照上计算梯度，Servo 使用实时 TF 和另一个姿态重建；坐标系、时间和 look-at 差异会让“评分最高方向”不等于实际移动方向。
3. **只优化单步 Fisher 增益，缺少运动代价和风险项**：当前输出主要依据局部梯度，可能偏向遮挡、碰撞、奇异姿态或过大视角变化。
4. **快照更新与规划频率不匹配**：地图版本按关键帧更新，而规划按固定频率重复使用同一快照；需要验证重复规划是否产生有效增益，还是只是在旧模型上抖动。
5. **缺少闭环控制效果与建图质量的联合目标**：当前日志更偏向“是否发出命令”，尚未证明运动确实带来几何、外观和语义信息增益。
6. **没有对失败场景做显式算法评估**：包括低纹理、深度空洞、快速运动、TF 延迟、遮挡、上下半球边界和规划计算超时。

## 三、后续实验评价框架

建议至少同时报告以下四组指标：

- **建图质量**：相机轨迹 ATE/RPE；几何 Chamfer/截断距离；TSDF 完整率与空洞率；3DGS 渲染 PSNR/SSIM/LPIPS。
- **主动感知收益**：单位时间新增可观测体素/高斯数量、未知区域减少量、Fisher 增益曲线、视角覆盖率、语义目标覆盖率。
- **控制质量**：实际轨迹跟踪误差、线/角速度峰值、加速度峰值、停止响应延迟、命令超时停车成功率、碰撞/越界次数。
- **系统代价**：Tracking/Planning/Servo 延迟分位数、快照加载时间、丢帧率、队列丢弃率、GPU/CPU/内存占用。

实验必须包含以下对照：

1. 固定轨迹或随机游走；
2. 只看几何的 NBV；
3. 只看 Fisher 的 NBV；
4. 当前三模态方法；
5. 当前方法去掉语义、去掉几何、去掉外观的消融；
6. 不同规划频率、快照频率、延迟和噪声条件下的鲁棒性测试。

每个场景至少多随机种子运行，报告均值、标准差或置信区间，并画“信息收益—时间”和“建图质量—运动代价”曲线，而不是只报告最终单点指标。

## 四、外部方法线索（待深入复现）

- **Fisher 信息主动视角选择**：[Next Best Sense](https://arm.stanford.edu/next-best-sense)（Strong et al., ICRA 2025）将 FisherRF 扩展到机器人在线 3DGS，并结合深度/法向对齐，报告在线真实机器人上的渲染与深度质量改进。它与当前 Fisher + 3DGS 路线最接近。
- **语义与动态 3DGS 的 Fisher 选择**：[Li, Jiang, Daniilidis, 3DV 2026](https://arxiv.org/abs/2512.22771) 同时对语义 Gaussian 参数和变形网络计算 Fisher 信息，比较随机和 uncertainty baseline，并报告渲染质量与语义分割性能。
- **传统体素 NBV**：[Isler et al., ICRA 2016](https://rpg.ifi.uzh.ch/docs/ICRA16_Isler.pdf) 把可见性、新表面发现概率与移动代价组合成 utility，而不是单独最大化信息增益。这提示当前方法需要显式加入路径长度、可达性和风险代价。
- **NBV 综述与评价共识**：[2025 NBV taxonomy](https://doi.org/10.1016/j.ophoto.2025.100098) 将方法分为规则、uncertainty、sampling、learning、prediction 五类；常见评价同时看覆盖率、信息增益、视图数量/时间、运动代价和重建准确度。
- **安全主动感知**：[Splat-CBF](https://arxiv.org/abs/2609.23100)（2026 preprint）把 Fisher 信息作为软目标，把碰撞/不确定区域风险作为硬 Control Barrier Function 约束，说明真机算法通常需要“安全硬约束 + 信息收益软目标”的结构。
- **候选视角与效率**：[GenNBV](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_GenNBV_Generalizable_Next-Best-View_Policy_for_Active_3D_Reconstruction_CVPR_2024_paper.pdf)（CVPR 2024）比较 random、uniform hemisphere、uncertainty、RL 和多目标 ActiveRMap，报告 coverage sufficiency 和 view efficiency，说明只给最终重建质量不足以评价 NBV。

初步判断：当前 OmniMap 的 Fisher 局部梯度可以作为信息收益项，但还不能单独证明三模态闭环有效。下一步应把 geometry/appearance/semantic 三类收益分别记录，并与运动代价、碰撞风险、视角冗余和真实重建质量联合评估。
