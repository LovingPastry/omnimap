# sim 模块说明

这个目录用于承载 **OmniMap 仿真闭环** 的新增代码。

当前目标不是接真机，也不是重写 OmniMap，而是先把下面这条纯仿真链路跑通：

1. 加载静态场景（点云 + 可选地面）
2. 在 Open3D 中从给定位姿渲染 RGBD
3. 后续把 RGBD 喂给 OmniMap 主流程建图
4. 再接 Fisher 半球热力图 / 速度场 / 相机推进闭环

---

## 当前目录结构

```text
sim/
├── README.md
├── TODO.md
├── __init__.py
├── assets.py
├── scene_simulator.py
├── test_phase1_scene_simulator.py
├── utils_pose.py
└── motion_policy.py
```

另有一个仓库根目录下的占位主入口：

```text
sim_fisher_closed_loop.py
```

---

## 当前各模块功能

### 1. `sim/TODO.md`

作用：
- 仿真闭环的分阶段实施指南
- 规定每个阶段的目标、接口、验收标准和风险点

当前开发节奏应严格按 `TODO.md` 推进，不要跳阶段。

---

### 2. `sim/assets.py`

当前已实现：
- `create_ground_plane(...)`
  - 创建一个薄盒子形式的地面
  - 支持设置尺寸、顶部高度、厚度、颜色
- `create_coordinate_frame(...)`
  - 创建 Open3D 坐标轴用于调试场景方向

用途：
- 给 `SceneSimulator` 提供可复用的基础几何体 helper
- 避免把地面/坐标轴构造硬编码在主类里

---

### 3. `sim/scene_simulator.py`

当前已实现的核心类：
- `SceneSimulator`
- `RenderResult`

#### `SceneSimulator` 当前支持的能力

- 初始化 Open3D `OffscreenRenderer`
- 加载 `.ply` / `.pcd` 点云
- 清理非法点（NaN / Inf）
- 可选 voxel downsample
- 自动补默认颜色
- 缓存：
  - `scene_center`
  - `aabb`
- 添加地面
- 添加坐标轴
- 设置 pinhole 相机内参
- 从给定 `c2w` 位姿渲染：
  - RGB (`uint8`, `H x W x 3`)
  - Depth (`float32`, `H x W`, 单位米)

#### 关键约定

对外接口使用：
- `c2w`（camera-to-world）

内部会自动转换成：
- `w2c`（world-to-camera）

这个约定必须保持，后续 Phase 2/3 会依赖它。

---

### 4. `sim/test_phase1_scene_simulator.py`

这是 **Phase 1 的最小测试脚本**。

用途：
- 独立验证 `SceneSimulator` 是否能正常完成离屏 RGBD 渲染
- 在接入 OmniMap 之前，先确认：
  - 点云能加载
  - 地面能显示
  - 改变位姿后 RGBD 响应正常

脚本会：
1. 加载点云
2. 设置相机内参
3. 可选添加地面和坐标轴
4. 自动生成 3 个测试视角
5. 渲染并保存以下调试产物：
   - RGB 图
   - depth `.npy`
   - depth 可视化图
   - 每个视角对应的 `c2w`

---

### 5. `sim/utils_pose.py`

当前状态：
- 还未实现

后续用途：
- 统一管理 `c2w / w2c / posevec` 的转换
- 服务于 Phase 2（仿真 RGBD -> OmniMap）

这是后续高优先级模块之一，因为 pose 语义一旦错，后面所有图都可能“看着能跑，实际上全错”。

---

### 6. `sim/motion_policy.py`

当前状态：
- 还未实现

后续用途：
- 基于当前 Fisher 场，推进下一时刻球面位姿
- 服务于 Phase 3 闭环运动控制

---

### 7. `sim_fisher_closed_loop.py`

当前状态：
- 仍是占位骨架

后续用途：
- 作为完整仿真闭环主入口
- 串起：
  - SceneSimulator
  - OmniMapRunner
  - FisherMotionPolicy
  - 可视化与日志

---

## Phase 1 测试脚本使用方法

### 前置条件

你需要进入仓库建议环境（尤其要保证 `open3d` 可导入）：

```bash
cd /home/sumi/experience/omnimap
source source_env.sh
python3 -c "import open3d as o3d; print(o3d.__version__)"
```

如果这里都过不了，先不要跑 Phase 1 测试脚本。

---

### 基本用法

```bash
python3 sim/test_phase1_scene_simulator.py \
  --pointcloud /path/to/scene.ply \
  --ground \
  --coord_frame \
  --output_dir sim_outputs/phase1
```

---

### 常用参数

#### 必填

- `--pointcloud`
  - 点云路径
  - 支持 `.ply` / `.pcd`

#### 常用可选

- `--output_dir`
  - 输出目录
  - 默认：`sim_outputs/phase1`

- `--width --height`
  - 渲染分辨率
  - 默认：`640 x 480`

- `--fx --fy --cx --cy`
  - 相机内参
  - 默认值适合做基础测试

- `--voxel_size`
  - 点云体素降采样大小
  - 大点云建议加上，避免渲染压力过高

- `--ground`
  - 添加地面

- `--ground_size`
  - 地面尺寸

- `--ground_z`
  - 地面顶部高度

- `--coord_frame`
  - 添加坐标轴，帮助判断方向对不对

- `--radius_scale`
  - 自动生成测试相机位置时使用的场景尺度倍数

---

### 运行结果说明

脚本每次运行会输出 3 组视角的结果，保存到 `output_dir` 下。

每组视角会生成：

- `view_00_rgb.png`
- `view_00_rgb_depth.npy`
- `view_00_rgb_depth_vis.png`
- `view_00_c2w.npy`

第二、第三组同理。

说明：
- `*_rgb.png`：彩色渲染结果
- `*_depth.npy`：原始深度数组（单位米）
- `*_depth_vis.png`：深度可视化图
- `*_c2w.npy`：用于该次渲染的相机位姿

---

## Phase 1 验收时你应该重点看什么

### 1. RGB 是否正常

检查：
- 点云有没有显示出来
- 地面有没有显示出来
- 三个视角是不是明显不同

### 2. Depth 是否正常

检查：
- 近处是不是更浅，远处是不是更深
- 是否存在整图全黑/全零
- 是否有大量 NaN / Inf

### 3. 相机位姿响应是否正常

检查：
- 同一视角重复渲染应一致
- 改变相机位置后，图像内容应按预期变化

如果这里都不稳定，就不要继续做 Phase 2。

---

## 当前开发建议

严格按下面顺序走：

1. 跑通 `test_phase1_scene_simulator.py`
2. 确认 Open3D 离屏渲染环境稳定
3. 检查 3 组输出图与 depth 是否正常
4. 再开始实现 `sim/utils_pose.py`
5. 再开始接 `OmniMapRunner`

不要跳过 Phase 1 验收直接做后面。那不是快，是返工预备役。

---

## 当前已知限制

1. 当前 `SceneSimulator` 只负责静态场景 RGBD 渲染
2. 还没有接 OmniMap 主流程
3. 还没有 pose 转换模块
4. 还没有 Fisher 驱动位姿推进模块
5. 还没有完整闭环入口实现

这些都属于后续阶段，不是当前脚本的责任。

---

## 一句话总结

当前 `sim/` 目录已经具备：
- **Phase 1 的核心渲染器**
- **Phase 1 的最小测试脚本**
- **后续闭环开发的分阶段施工指南**

现在最重要的不是继续堆代码，而是先把 **Phase 1 测试跑通并验收**。
