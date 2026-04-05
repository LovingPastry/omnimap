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
  负责读取 Fisher 梯度并计算 `next pose`
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

## 一组核心概念

后面命令里反复出现的 Fisher 参数，先统一解释：

- `--fisher_step_scale`
  控制器主缩放。角度模式下把原始梯度映射成 `delta_theta / delta_phi`；笛卡尔模式下把原始梯度映射成切向速度分量
- `--fisher_step_scale_theta`
  单独覆盖 theta 方向控制缩放
- `--fisher_step_scale_phi`
  单独覆盖 phi 方向控制缩放
- `--cartesian`
  是否启用笛卡尔速度场控制。关闭时维持原有球面角度控制；开启时改用 `v_t + v_n` 的 3D 速度积分
- `--dt`
  笛卡尔速度控制的时间步长，单位秒
- `--radial_gain`
  笛卡尔速度控制里的径向回正增益，用来把相机拉回初始参考球面
- `--fisher_arrow_length`
  速度场箭头长度，只影响显示，不影响控制
- `--show_fisher_heatmap`
  是否显示彩色 Fisher 信息场
- `--show_fisher_arrows`
  是否显示球形速度场箭头
- `--fisher_window_mode`
  `combined` 表示热力图和箭头叠在一个窗口；`split` 表示分成两个窗口
- `--fisher_heatmap_window_name`
  split 模式下热力图窗口标题
- `--fisher_velocity_window_name`
  split 模式下速度场窗口标题
- `--fisher_num_samples`
  半球采样点数。数值越大，速度场箭头越多，原始 Fisher 采样也越密
- `--fisher_num_dense_points`
  半球稠密插值点数。数值越大，彩色 Fisher 场越细腻，速度场箭头也越密
- `--fisher_idw_power`
  稠密热力图的球面 IDW 插值幂次
- `--grad_eps`
  有限差分步长，影响梯度估计
- `--max_delta_theta`
  单步 theta 最大允许变化量
- `--max_delta_phi`
  单步 phi 最大允许变化量
- `--fallback_delta_theta`
  梯度过小时 theta 的回退步长
- `--fallback_delta_phi`
  梯度过小时 phi 的回退步长

重要约定：

- 热力图表示 Fisher 信息场强度
- 红色箭头表示速度场方向，也就是“缩放前的原始梯度方向”
- `fisher_step_scale` 调的是控制器
- `fisher_arrow_length` 调的是显示
- `cartesian=false` 是旧的角度控制
- `cartesian=true` 是新的笛卡尔速度场控制

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

### 参数解释

- `--pointcloud`
  输入点云路径，通常是 `.ply` 或 `.pcd`
- `--point_scale 0.001`
  点云缩放比例。像 `Kettle.ply` 这类毫米单位点云通常需要乘 `0.001`
- `--ground`
  是否添加地面
- `--ground_size 1.0`
  地面边长，单位是世界坐标
- `--coord_frame`
  是否添加坐标轴，方便判断方向
- `--output_dir`
  输出目录，保存 RGB、depth 和 `c2w`

### 这一步在测什么

- 点云是否能正确加载
- `c2w -> Open3D render` 是否方向正确
- 深度尺度是否合理

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

### 参数解释

- `--input_dir`
  Phase 1 的输出目录，里面需要有 `view_*_rgb.png / *_depth.npy / *_c2w.npy`
- `--config`
  OmniMap 配置文件，建议使用仿真专用的 `config/sim_rtabmap_config.yaml`
- `--output`
  Phase 2 输出目录，保存 3DGS、在线可视化等结果
- `--vis_gui`
  是否打开 OmniMap GUI
- `--terminate`
  是否在重放结束后调用 `omni.terminate()` 输出最终结果

### 这一步在测什么

- 仿真 RGBD 是否能被 `OMNI.track(...)` 正确消费
- `keyframes` 和 `gaussians` 是否开始增长

### 期望结果

- 控制台连续打印 `[OmniMapRunner] idx=...`
- `initialized=True`
- `keyframes` / `gaussians` 不再长期为 0
- 输出目录里能看到 `3dgs_final.ply`、`online_vis/` 等产物

## Phase 3：单步 Fisher 调试

这个脚本只做一件事：先 warm-up OmniMap，然后计算一次 `next pose`。

### 最小命令

```bash
python3 sim/test_phase3_motion_policy.py \
  --input_dir sim/sim_outputs/phase1_kettle_scaled \
  --config config/sim_rtabmap_config.yaml \
  --output sim/sim_outputs/phase3
```

### 参数解释

- `--input_dir`
  Phase 1 输出目录
- `--config`
  OmniMap 仿真配置
- `--output`
  Phase 3 输出目录，保存 `phase3_next_c2w.npy` 和 `phase3_motion_result.json`

### 带 GUI 的单窗口命令

```bash
python3 sim/test_phase3_motion_policy.py \
  --input_dir sim/sim_outputs/phase1_kettle_scaled \
  --config config/sim_rtabmap_config.yaml \
  --output sim/sim_outputs/phase3_gui \
  --vis_gui \
  --show_fisher_arrows \
  --fisher_arrow_length 0.07 \
  --hold_gui_sec 5
```

### 参数解释

- `--vis_gui`
  打开 Fisher / OmniMap GUI
- `--show_fisher_arrows`
  在热力图上叠加速度场箭头
- `--fisher_arrow_length 0.07`
  设置箭头长度
- `--hold_gui_sec 5`
  由于 Phase 3 是一次性脚本，结束前额外停留 5 秒，避免看起来像“闪退”

### 双窗口命令

```bash
python3 sim/test_phase3_motion_policy.py \
  --input_dir sim/sim_outputs/phase1_kettle_scaled \
  --config config/sim_rtabmap_config.yaml \
  --output sim/sim_outputs/phase3_split \
  --vis_gui \
  --show_fisher_arrows \
  --fisher_window_mode split \
  --hold_gui_sec 5
```

### 参数解释

- `--fisher_window_mode split`
  把热力图和速度场拆成两个窗口
- `--hold_gui_sec 5`
  保持窗口 5 秒，方便观察

split 模式下两个窗口都会显示：

- 观测到的 TSDF 点云
- 相机轨迹
- 当前相机姿态

区别只在叠加层：

- A 窗口显示彩色 Fisher 信息场
- B 窗口显示按梯度大小着色的球形速度场，以及同色箭头

### 更密的场与箭头

```bash
python3 sim/test_phase3_motion_policy.py \
  --input_dir sim/sim_outputs/phase1_kettle_scaled \
  --config config/sim_rtabmap_config.yaml \
  --output sim/sim_outputs/phase3_dense \
  --vis_gui \
  --show_fisher_arrows \
  --fisher_window_mode split \
  --fisher_num_samples 128 \
  --fisher_num_dense_points 8192 \
  --hold_gui_sec 5
```

### 参数解释

- `--fisher_num_samples 128`
  增加半球采样点数量，让速度场箭头更密
- `--fisher_num_dense_points 8192`
  增加热力图插值点数量，让 Fisher 彩色场更细

### 更保守的控制步长

```bash
python3 sim/test_phase3_motion_policy.py \
  --input_dir sim/sim_outputs/phase1_kettle_scaled \
  --config config/sim_rtabmap_config.yaml \
  --output sim/sim_outputs/phase3_small_step \
  --fisher_step_scale 0.01
```

### 参数解释

- `--fisher_step_scale 0.01`
  降低控制缩放，让单步 `next pose` 更保守

### 单步笛卡尔速度场控制

```bash
python3 sim/test_phase3_motion_policy.py \
  --input_dir sim/sim_outputs/phase1_kettle_scaled \
  --config config/sim_rtabmap_config.yaml \
  --output sim/sim_outputs/phase3_cartesian \
  --cartesian \
  --dt 0.1 \
  --radial_gain 2.0
```

### 参数解释

- `--cartesian`
  启用笛卡尔速度场控制
- `--dt 0.1`
  用 0.1 秒做一次速度积分
- `--radial_gain 2.0`
  按当前半径误差计算径向回正速度

### 期望结果

- 输出目录中生成：
  - `phase3_next_c2w.npy`
  - `phase3_motion_result.json`
- 控制台出现：
  - `[FisherMotionPolicy] ...`
  - `[Phase3] Saved next pose ...`
- 如果开启 GUI：
  - `combined` 模式下，热力图和箭头叠在一起
  - `split` 模式下，两个窗口都带 TSDF 点云和相机上下文

重点看 [phase3_motion_result.json](/home/fuyx/lanzc/omnimap/sim/sim_outputs/phase3/phase3_motion_result.json)：

- `grad_theta_raw`
- `grad_phi_raw`
- `grad_norm_raw`
- `delta_theta_applied`
- `delta_phi_applied`
- `step_scale_theta`
- `step_scale_phi`
- `clipped_theta`
- `clipped_phi`
- `fallback_used`
- `current_theta`
- `current_phi`
- `next_theta`
- `next_phi`
- `controller_mode`
- `reference_radius`
- `current_radius`
- `radial_error`
- `vt_world`
- `vn_world`
- `velocity_world`
- `next_position`

## Phase 4：完整闭环主循环

这个脚本会持续执行：

`render -> track -> Fisher policy -> next pose -> log`

### 最小闭环命令

```bash
python3 sim_fisher_closed_loop.py \
  --pcd_path replica/log_pcd_0/对比结果/RoboSeg_geo/Kettle.ply \
  --point_scale 0.001 \
  --config config/sim_rtabmap_config.yaml \
  --num_steps 20 \
  --save_dir sim/sim_outputs/phase4
```

### 参数解释

- `--pcd_path`
  输入点云路径
- `--point_scale 0.001`
  点云缩放比例
- `--config`
  仿真专用 OmniMap 配置
- `--num_steps 20`
  闭环推进步数
- `--save_dir`
  闭环输出目录

### 单窗口 GUI 调试命令

```bash
python3 sim_fisher_closed_loop.py \
  --pcd_path replica/log_pcd_0/对比结果/RoboSeg_geo/Kettle.ply \
  --point_scale 0.001 \
  --config config/sim_rtabmap_config.yaml \
  --num_steps 20 \
  --save_dir sim/sim_outputs/phase4_gui \
  --vis_gui \
  --show_fisher_arrows \
  --step_delay_sec 0.1 \
  --hold_gui_sec 2.0
```

### 参数解释

- `--vis_gui`
  打开 GUI
- `--show_fisher_arrows`
  显示速度场箭头
- `--step_delay_sec 0.1`
  每一步结束后暂停 0.1 秒，方便观察刷新过程
- `--hold_gui_sec 2.0`
  所有步骤结束后再停留 2 秒

### 双窗口 GUI 调试命令

```bash
python3 sim_fisher_closed_loop.py \
  --pcd_path replica/log_pcd_0/对比结果/RoboSeg_geo/Kettle.ply \
  --point_scale 0.001 \
  --config config/sim_rtabmap_config.yaml \
  --num_steps 20 \
  --save_dir sim/sim_outputs/phase4_split \
  --vis_gui \
  --show_fisher_arrows \
  --fisher_window_mode split \
  --step_delay_sec 0.1 \
  --hold_gui_sec 2.0
```

### 参数解释

- `--fisher_window_mode split`
  把热力图和速度场拆成两个窗口
- `--step_delay_sec 0.1`
  让你能看清每次更新
- `--hold_gui_sec 2.0`
  闭环结束后保持窗口 2 秒

### 更密的热力图和速度场

```bash
python3 sim_fisher_closed_loop.py \
  --pcd_path replica/log_pcd_0/对比结果/RoboSeg_geo/Kettle.ply \
  --point_scale 0.001 \
  --config config/sim_rtabmap_config.yaml \
  --num_steps 20 \
  --save_dir sim/sim_outputs/phase4_dense \
  --vis_gui \
  --show_fisher_arrows \
  --fisher_window_mode split \
  --fisher_num_samples 128 \
  --fisher_num_dense_points 8192 \
  --step_delay_sec 0.1 \
  --hold_gui_sec 2.0
```

### 参数解释

- `--fisher_num_samples 128`
  提高半球采样点密度，增加速度场箭头数量
- `--fisher_num_dense_points 8192`
  提高 Fisher 彩色场密度

### 保存每一步渲染结果

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

### 参数解释

- `--save_frames`
  把每一步的 `rgb / depth / c2w` 都保存下来
- `--num_steps 10`
  这里用 10 步做较快的可视检查

### 更激进的控制步长

```bash
python3 sim_fisher_closed_loop.py \
  --pcd_path replica/log_pcd_0/对比结果/RoboSeg_geo/Kettle.ply \
  --point_scale 0.001 \
  --config config/sim_rtabmap_config.yaml \
  --num_steps 20 \
  --save_dir sim/sim_outputs/phase4_fast \
  --fisher_step_scale 0.06
```

### 参数解释

- `--fisher_step_scale 0.06`
  增大控制缩放，让相机运动更激进

### 笛卡尔闭环命令

```bash
python3 sim_fisher_closed_loop.py \
  --pcd_path replica/log_pcd_0/对比结果/RoboSeg_geo/Kettle.ply \
  --point_scale 0.001 \
  --config config/sim_rtabmap_config.yaml \
  --num_steps 20 \
  --save_dir sim/sim_outputs/phase4_cartesian \
  --cartesian \
  --dt 0.1 \
  --radial_gain 2.0
```

### 参数解释

- `--cartesian`
  启用笛卡尔速度场控制
- `--dt 0.1`
  每一步按 0.1 秒做 3D 速度积分
- `--radial_gain 2.0`
  用径向误差做回正，尽量维持在初始参考球面附近

### 期望结果

- 控制台连续出现：
  - `[OmniMapRunner] idx=...`
  - `[FisherMotionPolicy] ...`
  - `[ClosedLoop] step=...`
- 输出目录至少包含：
  - `loop_log.jsonl`
  - `trajectory_c2w_last.npy`
- 如果开启 `--save_frames`，还会有每一步的 RGBD 和 `c2w`
- 如果开启 GUI：
  - `combined` 模式下，在一个窗口里看热力图 + 箭头
  - `split` 模式下，在两个窗口里分别看热力图和速度场，但都带 TSDF 点云和相机上下文

重点看 [loop_log.jsonl](/home/fuyx/lanzc/omnimap/sim/sim_outputs/phase4/loop_log.jsonl)：

- `grad_theta_raw`
- `grad_phi_raw`
- `grad_norm_raw`
- `delta_theta_applied`
- `delta_phi_applied`
- `step_scale_theta`
- `step_scale_phi`
- `clipped_theta`
- `clipped_phi`
- `fallback_used`
- `current_theta`
- `current_phi`
- `next_theta`
- `next_phi`
- `controller_mode`
- `reference_radius`
- `current_radius`
- `radial_error`
- `vt_world`
- `vn_world`
- `velocity_world`
- `next_position`
- `num_keyframes`
- `num_gaussians`

## 调参指南

现象：箭头方向看着对，但相机步子太大  
处理：先降 `--fisher_step_scale`

现象：热力图有变化，但相机几乎不动  
处理：提高 `--fisher_step_scale`

现象：速度场箭头太稀  
处理：提高 `--fisher_num_samples`

现象：Fisher 彩色场不够细  
处理：提高 `--fisher_num_dense_points`

现象：箭头太长太乱  
处理：先减小 `--fisher_arrow_length`

现象：日志里总是 `fallback_used=true`  
处理：查看 `grad_norm_raw` 和 `grad_norm_epsilon`

现象：`clipped_theta` 或 `clipped_phi` 长期为真  
处理：说明控制量总被裁剪，优先降低 `--fisher_step_scale`

现象：只改箭头长度，轨迹也变了  
处理：这不符合设计预期。正常情况下 `--fisher_arrow_length` 只能影响显示，不能影响 next pose

## 建议验收顺序

1. 先跑 Phase 1，确认尺度和朝向正确
2. 再跑 Phase 2，确认 OmniMap 后端真的在长图
3. 再跑 Phase 3，确认原始梯度、控制量、next pose 三层量都清楚
4. 最后跑 Phase 4，确认闭环多步推进时热力图、速度场和轨迹一致

不要一上来就闭环。基础链路没过时直接盯 Fisher 轨迹，错误会叠在一起，很难区分到底是场的问题、控制的问题，还是输入的问题。
