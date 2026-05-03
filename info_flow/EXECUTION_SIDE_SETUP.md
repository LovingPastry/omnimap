# 执行侧最小化环境指南

这份指南面向已经具备以下基础的执行侧主机：

- 已安装并可运行 `ROS Noetic`
- 已有 `MoveIt`、`ur_robot_driver`、`servo_server` 等机器人执行栈
- 本机 `/tf`、`/tf_static`、机械臂控制 topic 已正常工作

目标是只在执行侧运行轻量 `servo runtime`：

- 订阅 `/omnimap/spherical_cmd`
- 读取本机 TF
- 发布最终 `TwistStamped`

不在执行侧运行：

- `OMNI.track`
- `Fisher planner`
- snapshot 加载
- `torch` / GPU 依赖

---

## 1. 需要部署哪些内容

执行侧只需要这些内容：

- `info_flow/info_flow_servo_runtime.py`
- `info_flow/servo_runtime_common.py`
- `ros_ws/src/omnimap_msgs/`
- `build_ros_ws.sh`
- `source_servo_env.sh`

如果你直接同步整个仓库也可以，但执行侧运行时只依赖上面这些。

---

## 2. 最小 Conda 环境

推荐使用独立的极简 conda 环境，而不是算力侧的 `InfoFlow` 大环境。

创建 conda 环境（推荐环境名：`infoflow-servo`）：

```bash
conda create -n infoflow-servo python=3.10 -y
conda activate infoflow-servo
python -m pip install --upgrade pip
python -m pip install -r info_flow/requirements-servo.txt
```

如果当前机器没有可写的 conda 默认 env 目录，可改用前缀路径：

```bash
conda create -p /path/to/.conda_envs/infoflow-servo python=3.10 -y
conda activate /path/to/.conda_envs/infoflow-servo
python -m pip install -r info_flow/requirements-servo.txt
```

这一步故意不安装：

- `torch`
- `opencv`
- `munch`
- `matplotlib`
- 任何 tracking / planning 相关依赖

---

## 3. 编译 ROS 自定义消息

执行侧只需要 `omnimap_msgs`，不需要编译 tracking / planning 功能。

在仓库根目录运行：

```bash
./build_ros_ws.sh
```

这个脚本会强制使用系统 ROS Python 来编译消息，避免被 conda 环境里的 `empy` 干扰。

编译完成后，执行侧会得到：

- `ros_ws/devel/setup.bash`
- `omnimap_msgs.msg.SphericalCommand`

---

## 4. 配置执行侧环境

推荐先设置 conda 环境名：

```bash
export SERVO_CONDA_ENV=infoflow-servo
```

如果你使用的是前缀路径环境，则设置为绝对路径：

```bash
export SERVO_CONDA_ENV=/path/to/.conda_envs/infoflow-servo
```

然后加载环境：

```bash
source source_servo_env.sh
```

这个脚本只做三件事：

- 激活你指定的极简 conda 环境（兼容旧版 venv 变量）
- source `/opt/ros/noetic/setup.bash`
- source `ros_ws/devel/setup.bash`

如果执行侧不是本机 `roscore`，再按实际网络设置：

```bash
export ROS_MASTER_URI=http://<master_ip>:11311
export ROS_IP=<exec_host_ip>
```

---

## 5. 启动轻量 Servo Runtime

最小启动命令：

```bash
source source_servo_env.sh
python info_flow/info_flow_servo_runtime.py \
  --spherical_cmd_topic /omnimap/spherical_cmd \
  --cmd_topic /servo_server/delta_twist_camera \
  --cmd_frame base_link \
  --world_frame base_link \
  --camera_frame cam_1_color_optical_frame \
  --servo_hz 50 \
  --spherical_cmd_timeout_sec 0.25 \
  --pose_stale_timeout_sec 0.2 \
  --linear_vel_max 0.05 \
  --angular_speed_max 0.5 \
  --enable_angular \
  --log_level INFO
```

说明：

- `world_frame` / `camera_frame` 必须与执行侧本机 TF 一致
- `cmd_topic` 必须与下游执行器一致
- 这个轻量 runtime 不读取 `config`，`--config` 仅保留兼容占位

---

## 6. 启动前检查

检查 TF：

```bash
rosrun tf tf_echo base_link cam_1_color_optical_frame
```

检查上游 spherical command 是否到达：

```bash
rostopic echo -n 1 /omnimap/spherical_cmd
```

检查执行侧控制输出：

```bash
rostopic hz /servo_server/delta_twist_camera
```

---

## 7. 运行时预期

正常情况下：

- 有新鲜 TF
- 有未过期的 `/omnimap/spherical_cmd`
- runtime 会持续输出非零 `TwistStamped`

以下情况会主动发零速：

- 没收到球坐标命令
- 球坐标命令超时
- 本机 TF 查失败
- 位姿过期
- planner 明确发来 `should_stop=True`

---

## 8. 和当前重版 Servo 节点的区别

执行侧推荐用：

- `info_flow/info_flow_servo_runtime.py`

不推荐在执行侧继续用：

- `info_flow/info_flow_servo_node.py`

因为后者仍依赖仓库里的共享重模块，环境明显更重，更适合开发/联调阶段，不适合作为执行侧最小化交付。
