# sim Fisher 调试手册

`sim/` 目录的目标，是在不接 ROS 和真机的前提下，构造一条可重复、可观察、可调参的 Fisher 信息场调试链路。

这套工程分成两条主线：

- 基础生成链：先确认点云渲染和 OmniMap 接入本身没问题
- Fisher 调试链：再专门看热力图、速度场、控制缩放和闭环轨迹

## 目录说明

```text
sim/
├── README.md
├── TODO.md
├── __init__.py
├── assets.py
├── scene_simulator.py
├── pose_utils.py
├── omnimap_runner.py
├── motion_policy.py
├── test_phase1_scene_simulator.py
├── test_phase2_omnimap_runner.py
└── test_phase3_motion_policy.py

sim_fisher_closed_loop.py
config/sim_rtabmap_config.yaml
```

关键文件：

- [scene_simulator.py](/home/fuyx/lanzc/omnimap/sim/scene_simulator.py)
  负责静态点云场景和 RGBD 渲染
- [omnimap_runner.py](/home/fuyx/lanzc/omnimap/sim/omnimap_runner.py)
  负责把仿真 RGBD / pose 喂给 `OMNI.track(...)`
- [motion_policy.py](/home/fuyx/lanzc/omnimap/sim/motion_policy.py)
  负责读取 Fisher 梯度并计算下一步相机位姿
- [test_phase1_scene_simulator.py](/home/fuyx/lanzc/omnimap/sim/test_phase1_scene_simulator.py)
  Phase 1：验证渲染链路
- [test_phase2_omnimap_runner.py](/home/fuyx/lanzc/omnimap/sim/test_phase2_omnimap_runner.py)
  Phase 2：验证 OmniMap 接入
- [test_phase3_motion_policy.py](/home/fuyx/lanzc/omnimap/sim/test_phase3_motion_policy.py)
  Phase 3：单步 Fisher 调试
- [sim_fisher_closed_loop.py](/home/fuyx/lanzc/omnimap/sim_fisher_closed_loop.py)
  Phase 4：真正闭环主循环
- [sim_rtabmap_config.yaml](/home/fuyx/lanzc/omnimap/config/sim_rtabmap_config.yaml)
  仿真专用配置，包含 Fisher 调试默认项

## 运行前置条件

建议先进入项目环境：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate InfoFlow
```

最少检查：

```bash
python3 -c "import open3d as o3d; print(o3d.__version__)"
python3 -c "import torch, scipy; print('torch/scipy ok')"
```

## 核心控制口径

### angular 模式

`--no-cartesian` 时，控制器走球面角度控制：

1. 计算当前视角的 `dF/dtheta` 和 `dF/dphi`
2. 乘 `--fisher_step_scale` 得到球坐标速度
3. 若球坐标速度模长小于 `--spherical_speed_min`，直接停止
4. 否则按球坐标速度更新 `(theta, phi)`
5. 再重建 `next_c2w`

这是“球面参数空间里的控制器”。

### cartesian 模式

`--cartesian` 时，控制器走“线速度 + 角速度积分”：

1. 计算当前视角的 `dF/dtheta` 和 `dF/dphi`
2. 乘 `--fisher_step_scale` 得到球坐标切向速度
3. 若球坐标速度模长小于 `--spherical_speed_min`，直接停止
4. 将球坐标切向速度映射成笛卡尔切向速度 `v_t`
5. 根据当前相机到参考球面的半径误差计算径向修正速度 `v_n`
6. 合成 `v_raw = v_t + v_n`
7. 用 `--linear_vel_max` 对最终线速度做一次模长限幅
8. 位置积分：`p_{t+1} = p_t + dt * v`
9. 姿态误差：由“当前位置朝向球心”的期望姿态 `R_des` 和当前姿态 `R_cur` 计算 `R_err = R_des R_cur^{-1}`
10. 角速度命令：`omega = angular_gain * rotvec(R_err)`
11. 姿态积分：`R_{t+1} = Exp(omega * dt) R_t`

这条链已经取消了“每一步强制 look-at 球心”的老机制。现在是：

- 位置靠线速度积分更新
- 姿态靠角速度误差积分更新

如果你关闭 `--enable_angular`，则 `omega=0`，只保留位置积分。

## 常用参数说明

下面这些参数在 Phase 3 和 Phase 4 里最常用。

### 场与控制

- `--fisher_step_scale`
  Fisher 主缩放因子。把原始 `dF/dtheta, dF/dphi` 映射成控制器使用的球坐标速度。
- `--cartesian`
  开启笛卡尔控制模式。关闭时走球面角度控制；开启时走“线速度 + 角速度积分”。
- `--dt`
  控制积分时间步长，单位秒。`cartesian` 模式下同时作用在线速度积分和姿态积分。
- `--spherical_speed_min`
  球坐标速度模长的最小阈值。小于这个值时，控制器认为已经收敛并停止，不再继续推进。
- `--grad_eps`
  Fisher 梯度有限差分步长，影响 `dF/dtheta, dF/dphi` 的数值稳定性。

### 线速度与径向回正

- `--radial_gain`
  径向修正增益。只在 `cartesian` 模式下生效，用来把相机拉回参考球面。
- `--linear_vel_max`
  笛卡尔模式最终 3D 线速度上限。控制的是 `v_raw` 合成之后的实际执行速度，不是球坐标速度。

### 角速度与姿态积分

- `--angular_gain`
  角速度增益。把姿态误差 `rotvec(R_err)` 映射成角速度命令 `omega`。
- `--enable_angular` / `--no-enable_angular`
  是否输出角速度命令。关闭时只做位置积分，不做姿态积分。

### Fisher 可视化

- `--vis_gui`
  打开 Open3D / OmniMap GUI。
- `--show_fisher_heatmap`
  显示彩色 Fisher 信息场。
- `--show_fisher_arrows`
  显示球形速度场箭头。
- `--fisher_arrow_length`
  速度场箭头长度，只影响显示，不影响控制。
- `--fisher_window_mode combined|split`
  `combined` 表示热力图和速度场叠在一个窗口；`split` 表示分成两个窗口。
- `--fisher_heatmap_window_name`
  split 模式下热力图窗口标题。
- `--fisher_velocity_window_name`
  split 模式下速度场窗口标题。

### 半球采样与插值

- `--fisher_num_samples`
  半球原始采样点数。数值越大，原始 Fisher 采样和梯度采样越密。
- `--fisher_num_dense_points`
  半球稠密插值点数。数值越大，热力图更细，速度场可视化也更密。
- `--fisher_idw_power`
  球面 IDW 插值幂次。
- `--fisher_display_radius_scale`
  热力图显示半径缩放，避免球面挡住点云和轨迹。
- `--fisher_arrow_radius_scale`
  箭头显示半径缩放，避免箭头挡住点云和轨迹。

### GUI 观察与结果保存

- `--step_delay_sec`
  Phase 4 每一步推进后的暂停时间，方便观察 GUI。
- `--hold_gui_sec`
  脚本结束前额外停留一段时间，避免窗口一闪而过。
- `--save_frames`
  保存每一步 `rgb / depth / c2w`。
- `--terminate`
  Phase 2/4 结束后调用 `omni.terminate()` 输出最终结果。

### 场景与渲染

- `--pcd_path`
  输入点云路径。
- `--point_scale`
  点云缩放比例。毫米单位点云通常需要 `0.001`。
- `--config`
  OmniMap 配置文件。建议用 `config/sim_rtabmap_config.yaml`。
- `--save_dir` / `--output`
  脚本输出目录。
- `--scene`
  传给 OmniMap GUI 分支的场景名。

## Phase 1：生成仿真 RGBD

### 启动命令

```bash
python3 sim/test_phase1_scene_simulator.py \
  --pointcloud replica/log_pcd_0/对比结果/RoboSeg_geo/Kettle.ply \
  --point_scale 0.001 \
  --ground \
  --ground_size 1.0 \
  --coord_frame \
  --output_dir sim/sim_outputs/phase1_kettle_scaled
```

### 关键参数

- `--pointcloud`
  输入点云路径。
- `--point_scale 0.001`
  点云缩放比例。`Kettle.ply` 这类毫米单位点云通常需要 `0.001`。
- `--ground`
  是否添加地面。
- `--ground_size 1.0`
  地面边长。
- `--coord_frame`
  是否添加坐标轴。
- `--output_dir`
  输出目录，保存 `rgb / depth / c2w`。

### 期望结果

- 输出目录里至少有 3 组 `view_*_rgb.png / *_depth.npy / *_c2w.npy`
- 控制台出现 3 次 `[Render i] ...`
- 如果方向检查开启，应看到 `[SelfCheck i] PASS`

## Phase 2：把仿真 RGBD 接入 OmniMap

### 启动命令

```bash
python3 sim/test_phase2_omnimap_runner.py \
  --input_dir sim/sim_outputs/phase1_kettle_scaled \
  --config config/sim_rtabmap_config.yaml \
  --output sim/sim_outputs/phase2 \
  --vis_gui \
  --terminate
```

### 关键参数

- `--input_dir`
  Phase 1 输出目录。
- `--config`
  OmniMap 配置文件。
- `--output`
  输出目录。
- `--vis_gui`
  打开 OmniMap GUI。
- `--terminate`
  重放结束后执行 `omni.terminate()`。

### 期望结果

- 控制台连续打印 `[OmniMapRunner] idx=...`
- `initialized=True`
- `keyframes` / `gaussians` 不再长期为 0

## Fisher 调试链（Phase 3 + Phase 4）

Phase 3 和 Phase 4 用的是同一套控制器与同一套参数面，区别只在于：

- Phase 3：只做一次 `next pose` 计算，适合看单步数值
- Phase 4：持续闭环推进，适合看轨迹、热力图和速度场随时间变化

如果你已经确认 Phase 1 和 Phase 2 没问题，当前最推荐的默认启动方式是这条 Phase 4 命令：

```bash
python3 sim_fisher_closed_loop.py \
  --pcd_path replica/log_pcd_0/对比结果/RoboSeg_geo/Kettle.ply \
  --point_scale 0.001 \
  --config config/sim_rtabmap_config.yaml \
  --num_steps 50 \
  --save_dir sim/sim_outputs/phase4_dense \
  --vis_gui \
  --show_fisher_arrows \
  --fisher_window_mode split \
  --fisher_num_samples 128 \
  --fisher_num_dense_points 1024 \
  --step_delay_sec 0.1 \
  --hold_gui_sec 2.0 \
  --cartesian \
  --dt 0.1 \
  --fisher_step_scale 1e-4 \
  --linear_vel_max 0.5 \
  --radial_gain 0.2 \
  --angular_gain 2.0
```

这组参数的设计意图是：

- 线速度主导仍然来自 Fisher 梯度
- 径向回正只做轻度约束，不强行把相机锁死在球面上
- 姿态大致朝向球心，但不再每步硬性强制 look-at
- split 模式下同时看热力图窗口和速度场窗口

### Phase 3：单步调试

这个脚本会：

1. 用 Phase 1 生成的 RGBD warm-up OmniMap
2. 计算一次 Fisher 梯度
3. 执行一次控制器
4. 写出 `phase3_next_c2w.npy`、`phase3_motion_result.json`、`phase3_debug.csv`

#### 最小命令

```bash
python3 sim/test_phase3_motion_policy.py \
  --input_dir sim/sim_outputs/phase1_kettle_scaled \
  --config config/sim_rtabmap_config.yaml \
  --output sim/sim_outputs/phase3
```

#### 推荐 GUI 命令

```bash
python3 sim/test_phase3_motion_policy.py \
  --input_dir sim/sim_outputs/phase1_kettle_scaled \
  --config config/sim_rtabmap_config.yaml \
  --output sim/sim_outputs/phase3_cartesian \
  --vis_gui \
  --show_fisher_arrows \
  --fisher_window_mode split \
  --fisher_num_samples 128 \
  --fisher_num_dense_points 1024 \
  --cartesian \
  --dt 0.1 \
  --fisher_step_scale 1e-4 \
  --linear_vel_max 0.5 \
  --radial_gain 0.2 \
  --angular_gain 2.0 \
  --hold_gui_sec 5
```

参数说明：

- `--vis_gui`
  打开 GUI。
- `--show_fisher_arrows`
  显示速度场箭头。
- `--fisher_window_mode split`
  拆成两个窗口。
- `--hold_gui_sec 5`
  让一次性脚本结束前多停 5 秒，避免看起来像“闪退”。

split 模式下两个窗口都会显示：

- 观测到的 TSDF 点云
- 相机轨迹
- 当前相机姿态

区别只在叠加层：

- A 窗口显示彩色 Fisher 信息场
- B 窗口显示球形速度场

#### 关键参数

- `--cartesian`
  开启“线速度 + 角速度积分”控制。
- `--dt 0.1`
  控制积分时间步长。
- `--fisher_step_scale 1e-4`
  Fisher 主缩放因子。这个值越大，切向控制越激进。
- `--linear_vel_max 0.5`
  最终线速度上限。这个值过大时，很容易下一帧直接看不到物体。
- `--radial_gain 0.2`
  轻度径向回正。当前推荐值的含义是“允许偏离球面，但不要完全漂走”。
- `--angular_gain 2.0`
  姿态误差增益。当前推荐值会让相机大致对准球心，但不再每步强制锁头。
- `--fisher_num_samples 128`
  提高原始采样点数量。
- `--fisher_num_dense_points 1024`
  提高稠密插值点数量。

#### 期望结果

- 输出目录中生成：
  - `phase3_next_c2w.npy`
  - `phase3_motion_result.json`
  - `phase3_debug.csv`
- 控制台出现：
  - `[FisherMotionPolicy] ...`
  - `[Phase3] Saved next pose ...`

重点看 [phase3_motion_result.json](/home/fuyx/lanzc/omnimap/sim/sim_outputs/phase3/phase3_motion_result.json)：

- `grad_theta_raw`
- `grad_phi_raw`
- `spherical_speed_scaled`
- `spherical_speed_applied`
- `should_stop`
- `vt_world`
- `vn_world`
- `velocity_raw_world`
- `velocity_world`
- `desired_c2w`
- `rotvec_error`
- `angular_velocity_world`
- `angular_speed_raw`
- `angular_speed_applied`
- `next_position`
- `next_c2w`

### Phase 4：完整闭环主循环

这个脚本会持续执行：

`render -> track -> Fisher policy -> next pose -> log`

#### 最小闭环命令

```bash
python3 sim_fisher_closed_loop.py \
  --pcd_path replica/log_pcd_0/对比结果/RoboSeg_geo/Kettle.ply \
  --point_scale 0.001 \
  --config config/sim_rtabmap_config.yaml \
  --num_steps 20 \
  --save_dir sim/sim_outputs/phase4
```

#### 推荐默认闭环命令

```bash
python3 sim_fisher_closed_loop.py \
  --pcd_path replica/log_pcd_0/对比结果/RoboSeg_geo/Kettle.ply \
  --point_scale 0.001 \
  --config config/sim_rtabmap_config.yaml \
  --num_steps 50 \
  --save_dir sim/sim_outputs/phase4_dense \
  --vis_gui \
  --show_fisher_arrows \
  --fisher_window_mode split \
  --fisher_num_samples 128 \
  --fisher_num_dense_points 1024 \
  --step_delay_sec 0.1 \
  --hold_gui_sec 2.0 \
  --cartesian \
  --dt 0.1 \
  --fisher_step_scale 1e-4 \
  --linear_vel_max 0.5 \
  --radial_gain 0.2 \
  --angular_gain 2.0
```

参数说明：

- `--vis_gui`
  打开 GUI。
- `--show_fisher_arrows`
  显示速度场箭头。
- `--fisher_window_mode split`
  拆成两个窗口。
- `--fisher_num_samples 128`
  提高半球原始采样点数量。
- `--fisher_num_dense_points 1024`
  提高半球稠密插值点数量。
- `--step_delay_sec 0.1`
  每一步推进后暂停 0.1 秒，方便观察。
- `--hold_gui_sec 2.0`
  所有步骤结束后再停 2 秒。
- `--cartesian`
  开启“线速度 + 角速度积分”控制。
- `--dt 0.1`
  控制积分时间步长。
- `--fisher_step_scale 1e-4`
  控制 Fisher 梯度映射出来的球坐标切向速度大小。
- `--linear_vel_max 0.5`
  最终线速度上限。当前推荐值属于比较保守、稳定的量级。
- `--radial_gain 0.2`
  轻度径向回正，不强制把相机严格锁在球面上。
- `--angular_gain 2.0`
  姿态误差增益，让镜头大致跟住球心方向。

#### 保存每一步渲染结果

```bash
python3 sim_fisher_closed_loop.py \
  --pcd_path replica/log_pcd_0/对比结果/RoboSeg_geo/Kettle.ply \
  --point_scale 0.001 \
  --config config/sim_rtabmap_config.yaml \
  --num_steps 10 \
  --save_dir sim/sim_outputs/phase4_frames \
  --save_frames \
  --vis_gui \
  --show_fisher_arrows
```

参数说明：

- `--save_frames`
  保存每一步 `rgb / depth / c2w`。

#### 期望结果

- 控制台连续出现：
  - `[OmniMapRunner] idx=...`
  - `[FisherMotionPolicy] ...`
  - `[ClosedLoop] step=...`
- 输出目录至少包含：
  - `loop_log.jsonl`
  - `loop_debug.csv`
  - `trajectory_c2w_last.npy`
- 闭环结束后还会自动导出最后时刻的 Fisher/速度场可视化：
  - `nbv_vis/final_fisher_heatmap.png`
  - `nbv_vis/final_fisher_velocity.png`

重点看 [loop_log.jsonl](/home/fuyx/lanzc/omnimap/sim/sim_outputs/phase4/loop_log.jsonl)：

- `grad_theta_raw`
- `grad_phi_raw`
- `spherical_speed_scaled`
- `vt_world`
- `vn_world`
- `velocity_raw_world`
- `velocity_world`
- `desired_c2w`
- `rotvec_error`
- `angular_velocity_world`
- `angular_speed_raw`
- `angular_speed_applied`
- `num_keyframes`
- `num_gaussians`

## 调参指南

现象：相机步子太大，第二帧就容易看不到物体
处理：先降 `--linear_vel_max`，再降 `--fisher_step_scale`

现象：热力图有变化，但闭环几乎不动
处理：提高 `--fisher_step_scale`

现象：姿态转得太慢，画面中心总是跟不上
处理：提高 `--angular_gain`

现象：只想测位置积分，不想让姿态自动转
处理：加 `--no-enable_angular`

现象：速度场箭头太稀
处理：提高 `--fisher_num_samples`

现象：Fisher 彩色场不够细
处理：提高 `--fisher_num_dense_points`

现象：箭头太长太乱
处理：减小 `--fisher_arrow_length`

现象：日志里 `should_stop=true`
处理：说明球坐标速度模长已经低于 `--spherical_speed_min`，控制器认为已经收敛

## 建议验收顺序

1. 先跑 Phase 1，确认尺度和朝向正确
2. 再跑 Phase 2，确认 OmniMap 后端真的在长图
3. 再跑 Phase 3，确认原始梯度、线速度、角速度、next pose 都清楚
4. 最后跑 Phase 4，确认闭环多步推进时热力图、速度场和轨迹一致

不要一上来就闭环。基础链路没过时直接盯 Fisher 轨迹，错误会叠在一起，很难区分到底是场的问题、控制的问题，还是输入的问题。
