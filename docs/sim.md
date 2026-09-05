# sim 模块使用说明

`sim/` 是点云驱动的闭环仿真模块，用于验证 Fisher 主动视角规划。

## 推荐入口（简化 CLI）

日常运行请使用：

```bash
python3 sim/main.py ...
```

`sim/main.py` 只保留两种模式：

- 默认：无 GUI（仍会计算 Fisher、打印日志、保存产物）
- `--vis_gui`：开启 GUI，默认分离热力图窗口和速度场窗口

控制逻辑固定为：

- 笛卡尔控制（Cartesian）
- 角速度控制始终开启

## 常用参数（仅这 9 个）

`sim/main.py` 仅暴露以下常用参数：

1. `--pcd_path`
2. `--save_dir`
3. `--fisher_step_scale`
4. `--radial_gain`
5. `--angular_gain`
6. `--grad_eps`
7. `--dt`
8. `--linear_vel_max`
9. `--angular_speed_max`

## 最小命令

默认无 GUI：

```bash
python3 sim/main.py \
  --pcd_path replica/log_pcd_0/对比结果/RoboSeg_geo/Kettle.ply \
  --save_dir sim/sim_outputs/phase4_main \
  --fisher_step_scale 1e-4 \
  --radial_gain 0.2 \
  --angular_gain 2.0 \
  --grad_eps 0.01 \
  --dt 0.3 \
  --linear_vel_max 0.05 \
  --angular_speed_max 0.05
```

开启 GUI（split 双窗口）：

```bash
python3 sim/main.py \
  --pcd_path replica/log_pcd_0/对比结果/RoboSeg_geo/Kettle.ply \
  --save_dir sim/sim_outputs/phase4_main_gui \
  --fisher_step_scale 1 \
  --radial_gain 0.2 \
  --angular_gain 1.0 \
  --grad_eps 0.01 \
  --dt 0.1 \
  --linear_vel_max 0.05 \
  --angular_speed_max 0.5 \
  --vis_gui
```

## 输出产物

每次运行都会输出并保存：

- `loop_log.jsonl`
- `loop_debug.csv`
- `trajectory_c2w_last.npy`
- `nbv_vis/*`（Fisher 相关导出）

## 高级入口（保留所有功能）

高级调参与实验入口保留在：

```bash
python3 sim/sim_fisher_closed_loop.py ...
```

该入口仍保留完整高级参数（如 headless / 采样密度 / 可视化细节开关），用于研究场景与深度调试。

全接口 demo（覆盖高级入口全部常用开关）：

```bash
python3 sim/sim_fisher_closed_loop.py \
  --pcd_path replica/log_pcd_0/对比结果/RoboSeg_geo/Kettle.ply \
  --config config/sim_rtabmap_config.yaml \
  --save_dir sim/sim_outputs/phase4_full_api_demo \
  --width 640 --height 480 \
  --fx 525.0 --fy 525.0 --cx 319.5 --cy 239.5 \
  --point_scale 0.001 \
  --scene room_0 \
  --num_steps 50 \
  --init_theta 0.0 --init_phi 0.35 --radius_scale 1.5 \
  --fisher_step_scale 1e-4 \
  --cartesian \
  --dt 0.1 \
  --radial_gain 0.2 \
  --linear_vel_max 0.5 \
  --angular_gain 2.0 \
  --angular_speed_max 1.0 \
  --enable_angular \
  --grad_eps 0.01 \
  --spherical_speed_min 1e-4 \
  --max_delta_theta 0.20 --max_delta_phi 0.15 \
  --fisher_arrow_length 0.07 \
  --show_fisher_heatmap \
  --show_fisher_arrows \
  --fisher_window_mode split \
  --fisher_num_samples 128 \
  --fisher_num_dense_points 1024 \
  --fisher_idw_power 2.0 \
  --fisher_display_radius_scale 0.92 \
  --fisher_arrow_radius_scale 0.92 \
  --vis_gui \
  --step_delay_sec 0.1 \
  --hold_gui_sec 2.0 \
  --log_profile debug \
  --log_every 5 \
  --save_frames
```
