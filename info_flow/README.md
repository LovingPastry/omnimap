# InfoFlow ROS 节点

`info_flow/` 目录用于仓库内 ROS 对接节点，不走单独 `catkin` 包骨架。

## 当前功能

- `tf_native` 模式（默认）：
  - 输入为 `RGB(compressed) + Depth(compressed)`。
  - 位姿由 TF 查询 `world_frame -> camera_frame`（默认 `base_link -> cam_1_color_optical_frame`）。
  - Tracking 入队采用自主门控：
    - 首帧通过；
    - `dt` 小于阈值拒绝；
    - 平移和旋转增量均小于阈值拒绝；
    - 连续拒绝达到 30 帧，强制当前帧作为 keyframe 入队。
  - 仅通过门控才解码 RGBD 并入 Tracking 队列。

- `rtabmap_native` 模式：
  - 兼容别名，内部映射到 `tf_native`，并在 `main` 分区打印迁移提示。

- `legacy_tf` 模式：
  - 保留原有串行回滚路径：`RGBD -> TF pose -> track -> live planning -> publish`。

- Tracking/Planning 解耦（`tf_native`）：
  - Tracking worker 独立线程消费队列，执行 `OMNI.track(...)`，完成后构建并原子切换 `PlannerSnapshot`。
  - Planning loop 使用 `rospy.Timer` 固定频率运行，始终基于“当前快照 + 最新位姿”。
  - 当 `--planner_output_mode spherical_delta` 时，额外启用高频 Servo timer：
    - planner 低频输出球坐标增量命令；
    - servo 高频在线转换为笛卡尔 `TwistStamped`。

## 数据流（tf_native）

1. 同步输入 `RGB + Depth`。
2. 用图像时间戳查询 TF 得到 `c2w/w2c`，更新 `latest_pose_state`。
3. 自主门控判定是否入 Tracking。
4. 通过门控才解码 RGBD 并入队；队列满时丢最旧帧（`DropOldestQueue`）。
5. Tracking 成功后构建新快照，Planning 自动切到新快照。

## 启动 CLI

### tf_native（推荐）

```bash
source source_env.sh
python info_flow/info_flow_node.py \
  --config config/rtabmap_config.yaml \
  --slam_frontend_mode tf_native \
  --rgb_topic /cam_1/color/image_raw/compressed \
  --depth_topic /cam_1/aligned_depth_to_color/image_raw/compressed \
  --camera_info_topic /cam_1/color/camera_info \
  --world_frame base_link \
  --camera_frame cam_1_color_optical_frame \
  --sync_slop_sec 0.12 \
  --planner_hz 30 \
  --pose_stale_timeout_sec 0.2 \
  --track_queue_size 2 \
  --keyframe_min_interval_sec 0.10 \
  --keyframe_min_translation_m 0.01 \
  --keyframe_min_rotation_deg 1.0 \
  --cmd_topic /servo_server/delta_twist_camera \
  --cmd_frame base_link \
  --max_frames 500
```
### 速度场相关参数
```bash
python info_flow/info_flow_node.py \
  --planner_output_mode spherical_delta \
  --planner_hz 10 \
  --servo_hz 50 \
  --spherical_cmd_timeout_sec 0.25 \
  --keyframe_min_interval_sec 0.10 \
  --keyframe_min_translation_m 0.01 \
  --keyframe_min_rotation_deg 1.0 \
  --track_queue_size 2 \
  --fisher_step_scale 1e-5 \
  --grad_eps 0.01 \
  --dt 1.0 \
  --spherical_speed_min 0.0 \
  --linear_vel_max 0.05 \
  --angular_speed_max 0.5 \
  --enable_angular \
  --log_section fisher --log_section planner \
  --log_min_level INFO
```


### 兼容别名（内部仍走 tf_native）

```bash
python info_flow/info_flow_node.py \
  --config config/rtabmap_config.yaml \
  --slam_frontend_mode rtabmap_native
```

### legacy_tf 回滚

```bash
python info_flow/info_flow_node.py \
  --config config/rtabmap_config.yaml \
  --slam_frontend_mode legacy_tf \
  --rgb_topic /cam_1/color/image_raw/compressed \
  --depth_topic /cam_1/aligned_depth_to_color/image_raw/compressed
```

## 关键参数

- 前端与位姿：
  - `--slam_frontend_mode {tf_native,rtabmap_native,legacy_tf}`
  - `--rgb_topic`
  - `--depth_topic`
  - `--camera_info_topic`
  - `--world_frame`（默认 `base_link`）
  - `--camera_frame`（默认 `cam_1_color_optical_frame`）
  - `--sync_slop_sec`（默认 `0.12`，RGBD 同步窗口）

- 关键帧门控：
  - `--keyframe_min_interval_sec`（默认 `0.10`）
  - `--keyframe_min_translation_m`（默认 `0.01`）
  - `--keyframe_min_rotation_deg`（默认 `1.0`）
  - 强制关键帧间隔固定 30 帧（代码常量）

- 规划与实时性：
  - `--planner_hz`（默认 `30.0`）
  - `--servo_hz`（默认 `50.0`，仅 `spherical_delta` 模式生效）
  - `--planner_output_mode {cartesian_legacy,spherical_delta}`（默认 `cartesian_legacy`）
  - `--spherical_cmd_timeout_sec`（默认 `0.25`）
  - `--pose_stale_timeout_sec`（默认 `0.2`）
  - `--track_queue_size`（默认 `2`）
  - `--max_frames`（默认 `500`）

- 兼容保留但已弃用（tf_native 下不生效）：
  - `--slam_odom_topic`
  - `--odom_info_topic`

## 验收与诊断

- 控制输出频率（目标 `>=20Hz`）：

```bash
rostopic hz /servo_server/delta_twist_camera
```

- TF 可用性：

```bash
rosrun tf tf_echo base_link cam_1_color_optical_frame
```

- 关注日志统计字段：
  - 门控：`gate_passed / gate_interval / gate_motion / gate_forced`
  - 队列：`enqueued / dropped`
  - 跟踪：`track_ok / track_fail / snapshot_ok / snapshot_fail`
  - 规划：`planner_nonzero / planner_zero` 及零速原因细分

## 退出行为

- 达到 `--max_frames` 自动停止
- 退出前始终发布零速
- `--terminate` 开启时执行 `omni.terminate()`，并按需保存轨迹
- `--save_fisher_snapshots` 开启时保存首帧/末帧 Fisher 快照

## 三环解耦运行

1. 在算力端运行Tracking节点
```bash
source source_env.sh
python info_flow/info_flow_tracking_node.py \
  --config config/rtabmap_config.yaml \
  --log_profile debug \
  --log_section profile \
  --log_section main \
  --log_min_level DEBUG
```

2. 在算力端运行Planning节点
```bash
source source_env.sh
python info_flow/info_flow_planning_node.py \
  --config config/rtabmap_config.yaml \
  --log_section planner \
  --log_section profile \
  --log_min_level DEBUG \
  --log_profile debug
```

3. 在执行侧运行Servo节点
```bash
source source_servo_env.sh
python info_flow/info_flow_servo_runtime.py \
  --servo_hz 50   \
  --spherical_cmd_timeout_sec 0.25   \
  --pose_stale_timeout_sec 0.2   \
  --linear_vel_max 0.05   \
  --angular_speed_max 0.5   \
  --enable_angular   \
  --log_level INFO
```

## 执行侧最小化部署

如果执行侧主机已经具备 `ROS Noetic + MoveIt + ur_robot_driver`，建议不要继续复用算力侧的完整 `InfoFlow` 大环境，而是直接使用轻量执行入口：

- 运行入口：[info_flow_servo_runtime.py](/home/fuyx/lanzc/omnimap/info_flow/info_flow_servo_runtime.py)
- 环境脚本：[source_servo_env.sh](/home/fuyx/lanzc/omnimap/source_servo_env.sh)
- 完整指南：[EXECUTION_SIDE_SETUP.md](/home/fuyx/lanzc/omnimap/info_flow/EXECUTION_SIDE_SETUP.md)

轻量执行侧只保留：

- `rospy`
- `tf2_ros`
- `numpy`
- `scipy`
- `omnimap_msgs`

不再依赖：

- `torch`
- `OMNI.track`
- `Fisher planner`
- snapshot / gaussian backend
