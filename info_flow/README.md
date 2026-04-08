# InfoFlow ROS 节点

`info_flow/` 目录用于存放仓库内和 ROS 对接的 Python 节点，不走单独的 `catkin` 包骨架。

当前主节点是：

- `info_flow_node.py`

它会：

1. 订阅 RGB、Depth、`CameraInfo`
2. 从 TF 查询 `world_frame -> camera_frame`
3. 把图像、深度和位姿喂给 `OMNI.track(...)`
4. 复用 `sim/motion_policy.py` 里的 `FisherMotionPolicy`
5. 向 `/servo_server/delta_twist_cmds` 发布 `geometry_msgs/TwistStamped`

## 运行方式

先进入项目环境：

```bash
source source_env.sh
```

再启动节点：

```bash
python info_flow/info_flow_node.py \
  --config config/rtabmap_config.yaml \
  --rgb_topic /cam_1/color/image_raw \
  --depth_topic /cam_1/aligned_depth_to_color/image_raw \
  --camera_info_topic /cam_1/color/camera_info \
  --world_frame base_link \
  --camera_frame cam_1_color_optical_frame \
  --cmd_topic /servo_server/delta_twist_cmds \
  --cmd_frame base_link
```

## 主要参数

- `--config`: OmniMap 配置文件
- `--rgb_topic`: RGB 图像 topic
- `--depth_topic`: 深度图 topic
- `--camera_info_topic`: 相机内参 topic
- `--world_frame`: TF 世界坐标系
- `--camera_frame`: TF 相机坐标系
- `--cmd_topic`: 输出速度命令 topic，默认 `/servo_server/delta_twist_cmds`
- `--cmd_frame`: `TwistStamped.header.frame_id`，默认 `base_link`
- `--fisher_step_scale`: Fisher 控制主缩放
- `--linear_vel_max`: 最大线速度
- `--angular_gain`: 角速度增益
- `--radial_gain`: 径向修正增益
- `--dt`: 控制积分时间步长
- `--enable_angular`: 是否输出角速度

## 失效保护

以下场景会发布零 `TwistStamped`：

- TF 查询失败
- 图像转换失败
- Fisher 状态尚未就绪
- 控制器判定应停止
