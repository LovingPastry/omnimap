# InfoFlow ROS 节点

`info_flow/` 目录用于存放仓库内和 ROS 对接的 Python 节点，不走单独的 `catkin` 包骨架。

当前主节点是：

- `info_flow_node.py`

它会：

1. 订阅 RGB、Depth、`CameraInfo`
2. 从 TF 查询 `world_frame -> camera_frame`
3. 把图像、深度和位姿喂给 `OMNI.track(...)`
4. 用 `omnimap/gaussian/renderer/nbv/motion_policy.py` 里的 `FisherMotionPolicy`
5. 向 `/servo_server/delta_twist_camera` 发布 `geometry_msgs/TwistStamped`

- [ ]下游节点需要还需要做一层 `cam_1_color_frame -> tool0` 的转化

## 运行方式

```bash
source source_env.sh
python info_flow/info_flow_node.py \
  --config config/rtabmap_config.yaml \
  --fisher_step_scale 1e-5 \
  --linear_vel_max 0.05 \
  --angular_gain 2.0 \
  --radial_gain 0.2 \
  --angular_speed_max 1.0 \
  --dt 1.0 \
  --save_fisher_snapshots \
  --max_frames 500
```

## 开发期最常改的参数

- `--fisher_step_scale`: Fisher 控制主缩放，决定切向推进强度
- `--linear_vel_max`: 最大线速度，直接限制机械臂平移速度
- `--angular_gain`: 角速度增益，控制朝向修正力度
- `--radial_gain`: 径向修正增益，用于球面约束修正
- `--grad_eps`: Fisher 梯度中心差分步长
- `--spherical_speed_min`: 球面速度下限，低于该值时直接输出零速度
- `--enable_angular`: 是否输出角速度
- `--angular_speed_max`: 角速度上限，限制输出 `omega` 范数
- `--max_frames`: 最多处理多少帧，默认 `500`
- `--terminate`: 退出时是否执行后处理；关闭时不会进入保存帧和 `omni.terminate()` 阶段
- `--save_fisher_snapshots`: 仅保存首帧和尾帧的 Fisher 点云快照，避免逐帧球形速度场可视化计算

一个更接近日常调参的启动示例：

```bash
python info_flow/info_flow_node.py \
  --config config/rtabmap_config.yaml \
  --fisher_step_scale 5e-6 \
  --linear_vel_max 0.03 \
  --angular_gain 1.5 \
  --radial_gain 0.1 \
  --angular_speed_max 0.8 \
  --grad_eps 0.01 \
  --save_fisher_snapshots \
  --max_frames 300
```

## 低频接口参数

- `--config`: OmniMap 配置文件
- `--rgb_topic`: RGB 图像 topic
- `--depth_topic`: 深度图 topic
- `--camera_info_topic`: 相机内参 topic
- `--world_frame`: TF 世界坐标系
- `--camera_frame`: TF 相机坐标系
- `--cmd_topic`: 输出速度命令 topic，默认 `/servo_server/delta_twist_camera`
- `--cmd_frame`: `TwistStamped.header.frame_id`，默认 `base_link`
- `--depth_scale`: 深度缩放系数

## 失效保护

以下场景会发布零 `TwistStamped`：

- TF 查询失败
- 图像转换失败
- Fisher 状态尚未就绪
- 控制器判定应停止

## 退出行为

- 默认最多处理 `500` 帧，到达上限后自动停止
- 默认 `--no-terminate`，按下 `Ctrl+C` 或达到帧上限时不会进入 `saving frames`
- 如果显式传入 `--terminate`，节点退出时才会执行 `omni.terminate()`，并在 `--output` 有效时保存轨迹与 RGBD
- 如果显式传入 `--save_fisher_snapshots`，节点会在首帧和退出时各导出一次 Fisher 点云快照到 `output/nbv_vis/`
