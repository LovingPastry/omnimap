# 仿真闭环实现 TODO（Fisher 球形速度场 + OmniMap 建图）

> 目标：
> 在**不依赖 ROS、不接真机、不改下游伺服**的前提下，构建一个纯仿真闭环：
>
> 1. 加载静态点云模型 + 地面
> 2. 从给定相机位姿在 Open3D 中渲染 RGBD
> 3. 将 RGBD 与位姿喂给 OmniMap 主流程进行建图
> 4. 调用现有 Fisher 半球场逻辑，实时更新热力图与速度场
> 5. 基于当前 Fisher 速度场推进下一时刻相机位姿
> 6. 循环执行，观察轨迹、建图和 Fisher 场的动态变化

---

## 0. 总体设计原则

### 0.1 这次到底要复用什么

必须复用：
- `OMNI.track(...)`
- `GSBackEnd.process_track_data(...)`
- `GSBackEnd.update_fisher_hemisphere_pc(...)`
- `gaussian/renderer/nbv/*` 里的 Fisher 半球场与速度场逻辑
- `FisherVisualizer`
- `HemisphereCamera`

尽量不要改：
- Fisher 计算主体逻辑
- TSDF / GS 后端主建图逻辑
- 现有真实系统相关接口

只新增/补充：
- 仿真场景渲染器
- pose 格式转换工具
- 闭环主循环
- 从当前 Fisher 场推进下一步位姿的策略层

### 0.2 当前闭环的职责边界

本仿真系统拆成四层：

1. **Scene Simulator**
   - 管环境（点云、地面、Open3D 场景）
   - 输入相机位姿，输出 RGBD

2. **OmniMap Runner**
   - 负责把 RGBD 和 pose 封装成 OmniMap 所需格式
   - 调用 `OMNI.track(...)`

3. **Fisher Motion Policy**
   - 从当前 OmniMap/GS 状态得到 Fisher 半球场
   - 从当前相机对应的球坐标状态出发，计算下一步位姿

4. **Main Loop**
   - 串起渲染、建图、Fisher 场更新、位姿推进、可视化

### 0.3 这版先不做什么

本轮禁止扩需求：
- 不接 ROS
- 不发 topic
- 不接真机控制器
- 不做物理引擎
- 不做碰撞检测
- 不做复杂路径优化
- 不做 coverage/baseline 逻辑
- 不做多线程优化

重点只有一件事：**让 Fisher 驱动的球面运动闭环先跑起来，并可视化观察规律。**

---

# Phase 1：搭建仿真场景与 RGBD 渲染链路

## 1.1 交付目标

给定：
- 点云模型路径
- 地面配置
- 相机内参
- 一个相机位姿 `c2w`

输出：
- RGB 图像
- Depth 图像
- 输出尺寸、颜色格式、深度单位稳定可控

这是整个系统的输入源。这个阶段不打通，后面全是空转。

---

## 1.2 建议新增文件

- `sim/scene_simulator.py`
- `sim/assets.py`（可选，用来管理地面、坐标轴、辅助几何体）

---

## 1.3 需要实现的类与接口

### 类：`SceneSimulator`

建议接口：

```python
class SceneSimulator:
    def __init__(self, width: int, height: int):
        ...

    def load_pointcloud(self, path: str, voxel_size: float | None = None):
        ...

    def add_ground(self, size=4.0, z=0.0, color=(0.5, 0.5, 0.5)):
        ...

    def set_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        ...

    def render(self, c2w: np.ndarray):
        ...
```

### `render(...)` 的输入输出要求

输入：
- `c2w`: `4x4` numpy 数组，表示 camera-to-world

输出：
- `rgb`: `H x W x 3`, `uint8`, RGB 顺序
- `depth`: `H x W`, `float32`, 单位建议为 **米**

注意：
- 后续喂 OmniMap 时可再根据 `depth_scale` 转换
- 不要在 simulator 层过早转成 `uint16`

---

## 1.4 场景侧实现要求

### 点云加载要求

必须支持：
- `.ply`
- `.pcd`

建议处理步骤：
1. 读取点云
2. 若无颜色，赋默认灰色
3. 去除 NaN / Inf
4. 必要时 voxel downsample
5. 计算包围盒、中心、尺度信息

需要额外输出/缓存：
- `scene_center`
- `aabb`
- `pcd`

### 地面实现要求

地面至少满足：
- 能在 RGBD 中稳定出现
- 位置可控
- 尺寸足够大，不要一移动相机就拍不到

推荐实现：
- 用 `TriangleMesh.create_box(...)` 做一个薄平板
- 厚度设得很小，例如 `0.01`
- 颜色用中灰，避免过亮/过暗

不要一上来搞复杂地形。那不是当前任务。

---

## 1.5 Open3D 渲染要求

优先使用：
- `open3d.visualization.rendering.OffscreenRenderer`

你要确认的关键点：
1. 当前 Open3D 版本是否支持离屏渲染
2. 渲染出来的 depth 单位是什么
3. 相机外参矩阵方向是否和 OmniMap 保持一致

### 最小验证任务

在本阶段单独写一个最小 demo，验证下面三件事：
- 固定相机位姿能出 RGB
- 同一位姿多次渲染一致
- 改位姿后，画面和深度变化方向正确

### 必须保存的调试产物

每次测试至少保存：
- `debug_rgb.png`
- `debug_depth.npy`
- `debug_depth_vis.png`

目的：
- 防止你眼睛只看彩色图，忽略 depth 已经错了

---

## 1.6 本阶段验收标准

达到以下条件才算 Phase 1 结束：

- 能从指定 `c2w` 渲染 RGB
- 能从指定 `c2w` 渲染 depth
- 地面出现在图里
- 改变相机位姿后，RGBD 响应正确
- 输出尺寸和数据类型稳定
- 有至少 3 组位姿测试图保存到本地

---

## 1.7 本阶段高风险点

### 风险 1：Open3D 离屏渲染环境问题

现象：
- 黑屏
- 只有 RGB 没有 depth
- EGL / GL 上下文失败

对策：
- Phase 1 必须独立验证，不要和 OmniMap 主循环绑一起查错

### 风险 2：相机坐标系不一致

现象：
- 图能出，但方向反了
- 深度正常，建图发散

对策：
- 先用简单坐标轴/立方体测试外参方向
- 必须明确 Open3D 的 camera extrinsic 到底需要 `w2c` 还是 `c2w`

---

# Phase 2：打通仿真 RGBD -> OmniMap 主流程

## 2.1 交付目标

把 Phase 1 渲染出的仿真 RGBD，直接喂给：
- `OMNI.track(...)`

要求：
- 不通过 ROS
- 不通过 message filter
- 不通过 topic
- 直接 Python 函数调用

---

## 2.2 建议新增文件

- `sim/omnimap_runner.py`
- `sim/pose_utils.py`

---

## 2.3 需要实现的类与接口

### 类：`OmniMapRunner`

建议接口：

```python
class OmniMapRunner:
    def __init__(self, args, config):
        self.omni = OMNI(args, config)

    def step(self, idx, rgb, depth_m, c2w, intrinsics_vec):
        ...
```

### `step(...)` 输入要求

输入：
- `idx`: 当前帧编号
- `rgb`: `H x W x 3`, RGB, uint8
- `depth_m`: `H x W`, float32, 单位米
- `c2w`: `4x4` numpy
- `intrinsics_vec`: `[fx, fy, cx, cy]`

输出：
- 无需复杂返回值，第一版只要求建图能推进
- 但建议返回当前 `viewpoint` 或 `gs_backend` 状态引用

---

## 2.4 pose 格式转换是本阶段核心难点

当前 OmniMap 里同时使用两套 pose 表示：

1. `pose_tensor`
- 是 `w2c`
- 形式是 `[tx, ty, tz, qx, qy, qz, qw]`

2. `pose_44`
- 是 `c2w`
- 形式是 `4x4 matrix`

### 必须实现的工具函数

放在 `sim/pose_utils.py`：

```python
def c2w_to_w2c(c2w):
    ...

def w2c_to_posevec(w2c):
    ...


def c2w_to_posevec(c2w):
    ...


def posevec_to_se3(pose):
    ...
```

### 要求

每个转换函数都要：
- 写明输入输出语义
- 单独做单元验证
- 不允许靠猜

### 必须做的验证

验证链路：

```python
c2w -> w2c -> posevec -> SE3 -> matrix
```

要求：
- 最终结果与原矩阵一致或数值接近

---

## 2.5 喂给 `OMNI.track(...)` 的数据要求

### RGB 数据

从：
- `H x W x 3`

转成：
- `torch.Tensor`
- `3 x H x W`
- 再 `image[None]`

### Depth 数据

从：
- `H x W`, 单位米

转成：
- `torch.Tensor`
- `depth[None]`

### Intrinsics

按当前工程风格：
- `[fx, fy, cx, cy]`

### Pose

- `pose_tensor[None]`
- `pose_44_tensor[None]`

---

## 2.6 本阶段建议执行顺序

1. 用固定同一帧 RGBD 连续喂 1 次，保证不崩
2. 再喂 2~3 帧轻微变化的 pose
3. 再喂 10 帧小轨迹
4. 最后打开 `vis_gui=True` 看 Fisher 半球是否出现

不要一上来就 100 帧循环。先保证单步正确。

---

## 2.7 建议加的调试输出

每次 `step(...)` 至少打印/保存：
- `idx`
- 当前相机位置
- 当前 depth 的 min/max
- 当前是否成功初始化 `scene_center`
- 当前 keyframe 数量
- 当前 gaussian 数量

如果不打印这些，你后面会完全失明。

---

## 2.8 本阶段验收标准

达到以下条件才算结束：

- 能连续输入至少 10 帧仿真 RGBD
- OmniMap 主流程不报错
- TSDF / GS 后端状态有更新
- `scene_center` 能正常初始化
- Fisher 半球热力图在 Open3D 中能显示

---

## 2.9 本阶段高风险点

### 风险 1：depth 单位错误

现象：
- 图像正常
- 建图尺度严重错误
- 热力图位置飘

对策：
- 统一把 simulator 输出设为米
- 在进入 OmniMap 前唯一一次做 `depth_scale` 约定检查

### 风险 2：pose 方向错

现象：
- 画面看着对，建图却发散
- 相机轨迹方向与场景不一致

对策：
- 单独把当前 pose 可视化成 frustum，跟场景对照

---

# Phase 3：打通 Fisher 速度场 -> 下一时刻位姿推进

## 3.1 交付目标

相机不再使用手工给定轨迹，而是：
- 根据当前建图状态
- 计算当前相机的 Fisher 梯度/速度方向
- 推出下一时刻的位姿

这是仿真闭环的核心阶段。

---

## 3.2 建议新增文件

- `sim/motion_policy.py`

---

## 3.3 需要实现的类与接口

### 类：`FisherMotionPolicy`

建议接口：

```python
class FisherMotionPolicy:
    def __init__(self, step_gain_theta, step_gain_phi):
        ...

    def next_pose(self, gs_backend, current_viewpoint, idx):
        ...
```

### 输入

- `gs_backend`: 当前 `self.omni.gs`
- `current_viewpoint`: 当前视角对应的 `Camera` / `HemisphereCamera`
- `idx`: 当前帧号

### 输出

- `next_c2w`: 下一时刻位姿
- 可选附加：
  - 当前梯度值
  - 当前 `(theta, phi)`
  - 下一步 `(theta, phi)`

---

## 3.4 本阶段推荐推进策略

### 第一版必须采用：基于当前相机的角度梯度推进

不要先走“取离当前最近采样点的箭头方向”那种绕路做法。

当前已有最直接的抓手是：
- `HemisphereCamera.from_camera(...)`
- `compute_view_gradient(...)`

### 推荐公式

令当前相机在球面参数为：
- `theta_t`
- `phi_t`

从当前 Fisher evaluator 得到：
- `dF/dtheta`
- `dF/dphi`

然后更新：

```python
theta_next = theta_t + gain_theta * dF_dtheta
phi_next   = clamp(phi_t + gain_phi * dF_dphi, 0, pi/2)
```

注意：
- `theta` 要做 wrap 到 `[0, 2pi)`
- `phi` 要 clamp 到 `[0, pi/2]`
- 半径先固定，不做动态变化

### 为什么先固定半径

因为本轮目标是“球形速度场”。
一旦半径也变成变量，你会把问题从 2D 球面流形优化升级成 3D 位姿优化，复杂度没必要暴涨。

---

## 3.5 当前位姿到球坐标的桥接

你需要做：
- 从当前 `viewpoint` 生成 `HemisphereCamera`
- 用该对象获得当前 `(theta, phi)`
- 更新角度后，再反求 `c2w`

### 必须实现的辅助能力

如果 `HemisphereCamera` 已能完成大多数转换，就复用它。
如果不够，则在 `pose_utils.py` 补以下函数：

```python
def camera_to_hemisphere(camera, center):
    ...


def hemisphere_to_c2w(hemi_cam):
    ...
```

---

## 3.6 如何得到当前用于推进的 viewpoint

建议不要依赖 `next_viewpoint`。
原因：
- 当前仓库这条链没闭环赋值
- 你这版主循环自己就是闭环控制器

正确做法：
- 使用当前输入给 `OMNI.track(...)` 的相机位姿所对应的 viewpoint
- 或从当前帧构造出的 `HemisphereCamera` 直接继续推进

### 最稳妥的办法

主循环自己维护一个状态变量：
- `current_c2w`
- `current_theta`
- `current_phi`

而不是把控制权交给不存在的 `next_viewpoint`

---

## 3.7 为减少重复计算，建议做一个小增强

### 推荐修改 `GSBackEnd.update_fisher_hemisphere_pc(...)`

在里面缓存：

```python
self.last_field_result = field_result
```

### 这样做的价值

主循环后续可以直接读：
- `sample_vals`
- `sample_vel_dirs`
- `base_hemi`
- `debug_stats`

而不用在同一帧里额外重复算一遍 Fisher 场。

这不是功能性改动，是去冗余。建议做。

---

## 3.8 本阶段验收标准

达到以下条件才算结束：

- 相机能在球面上自动推进
- 推进后仍然保持朝向球心
- 每一帧都能重新渲染 RGBD
- 每一帧都能重新喂给 OmniMap
- Fisher 热力图随时间更新
- 速度场方向与相机运动方向大体一致

---

## 3.9 本阶段高风险点

### 风险 1：步长太大，轨迹抖动

现象：
- 视角跳来跳去
- 热力图变化剧烈
- 建图无法稳定累积

对策：
- 先从很小的 `gain_theta / gain_phi` 开始
- 每步位移要小

### 风险 2：梯度太小，几乎不动

现象：
- 相机卡住
- 每一帧位置几乎不变

对策：
- 记录梯度范数
- 小于阈值时，可采用 fallback：
  - 保持上一方向微动
  - 或做极小随机扰动

### 风险 3：靠近半球极点数值不稳定

现象：
- `phi` 接近 `pi/2` 时方向变怪

对策：
- 严格 clamp `phi`
- 靠近极点时限制每步最大增量

---

# Phase 4：主循环整合、实时显示与实验记录

## 4.1 交付目标

把前面三阶段合成一个真正可跑的闭环实验入口。

建议主入口文件：
- `sim_fisher_closed_loop.py`

---

## 4.2 主循环必须完成的事情

每个 time step 必须严格执行：

1. 使用当前 `c2w` 从 simulator 渲染 `rgb/depth`
2. 将 `rgb/depth/c2w` 喂给 OmniMap
3. 更新 Fisher 半球热力图与速度场
4. 读取当前 Fisher 场 / 梯度信息
5. 计算下一时刻 `c2w`
6. 更新轨迹显示
7. 保存必要日志

建议伪代码：

```python
current_c2w = init_c2w
for idx in range(num_steps):
    rgb, depth = simulator.render(current_c2w)
    runner.step(idx, rgb, depth, current_c2w, intrinsics)
    field_result = runner.omni.gs.last_field_result
    current_c2w = motion_policy.next_pose(...)
```

---

## 4.3 建议主脚本参数

第一版至少支持：
- `--pcd_path`
- `--config`
- `--width`
- `--height`
- `--fx --fy --cx --cy`
- `--num_steps`
- `--init_pose`
- `--hemi_radius`
- `--gain_theta`
- `--gain_phi`
- `--vis_gui`
- `--save_dir`

如无必要，不要在第一版加太多参数。

---

## 4.4 必须记录的实验数据

每一帧至少保存：
- `idx`
- `c2w`
- `theta`
- `phi`
- `camera_center`
- `gradient_theta`
- `gradient_phi`
- `fisher_current_score`
- `num_keyframes`
- `num_gaussians`

建议保存成：
- `jsonl`
- 或 `csv`

同时建议保存：
- 轨迹图
- 若干关键帧截图
- 热力图导出

---

## 4.5 可视化要求

Open3D 窗口里至少稳定显示：
- 地图点云 / GS 点云
- 当前相机位置
- 相机轨迹
- Fisher 半球热力图
- 速度场箭头

额外建议：
- 用不同颜色区分当前相机与历史轨迹
- 定期保存 screenshot，便于回看实验

---

## 4.6 本阶段验收标准

达到以下条件才算完成首版闭环：

- 能连续跑完指定步数，例如 30~100 步
- 不依赖 ROS
- 每步都有 RGBD 渲染
- 每步都能推进 OmniMap 建图
- 每步都更新 Fisher 热力图和速度场
- 相机轨迹可视化连续
- 有实验日志和关键帧输出

---

## 4.7 本阶段高风险点

### 风险 1：主循环太慢

现象：
- 一帧几秒
- GUI 卡顿

对策：
- 第一版先接受低帧率
- 先保证逻辑闭环正确，再谈性能优化

### 风险 2：可视化和算法状态不同步

现象：
- 画出来的是上一帧的场
- 当前相机和热力图对不上

对策：
- 所有更新顺序固定，不要乱插渲染调用
- 每帧只允许一种 authoritative state

---

# 最终建议的文件组织

建议最终至少有这些文件：

```text
sim/
├── TODO.md
├── scene_simulator.py
├── omnimap_runner.py
├── motion_policy.py
├── pose_utils.py
└── debug_tools.py        # 可选

sim_fisher_closed_loop.py
```

---

# 推荐的开发顺序（必须按这个来）

## Step 1
先单独完成 `SceneSimulator.render(...)`

验收：
- 能稳定输出 RGBD

## Step 2
完成 `pose_utils.py`

验收：
- 所有 pose 转换自洽

## Step 3
完成 `OmniMapRunner.step(...)`

验收：
- 能喂 10 帧仿真数据进 OmniMap

## Step 4
在 `GSBackEnd` 增加 `last_field_result` 缓存（推荐）

验收：
- 每帧可直接读到最近一次 Fisher 场结果

## Step 5
完成 `FisherMotionPolicy.next_pose(...)`

验收：
- 能根据当前梯度推进球面位姿

## Step 6
完成 `sim_fisher_closed_loop.py`

验收：
- 能完整跑闭环

---

# 首版完成后的下一步（不是本轮）

首版跑通后，再考虑：
- 梯度平滑
- 自适应步长
- 多初始位姿实验
- 轨迹质量评估
- 和真机实验做对比

本轮不要做这些。先闭环，后增强。

---

# 一句话施工要求

这份 TODO 的唯一目标不是“列任务”，而是强制你按正确颗粒度施工：

- **先打通渲染输入源**
- **再打通 OmniMap 主链**
- **再接 Fisher 推进**
- **最后做完整闭环**

谁先后顺序搞反，谁就会在错误的层上 debug 一整天。
