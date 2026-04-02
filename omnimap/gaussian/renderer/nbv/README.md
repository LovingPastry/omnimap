# Fisher / NBV 模块说明

这个目录放的是 3DGS 的 Next-Best-View（NBV）相关逻辑，当前主要包含：

- `legacy_fisher.py`：基础 Fisher / 信息势能场评估器，包含半球采样、视角打分、中心差分梯度、半球场构建等主体逻辑
- `diag_fisher.py`：在 `legacy_fisher.py` 基础上派生出的不同统计形式实现（`DiagFisherEvaluator` / `LogFisherEvaluator` / `LogSquareFisherEvaluator`）
- `hemisphere_field.py`：半球采样、插值、颜色映射
- `visualization.py`：Open3D 可视化与导出
- `interfaces.py`：共享数据结构与接口协议
- `debug.py`：调试信息拼装

## 当前真实调用链

当前仓库运行时，`gs_backend.py` 中实际使用的是：

```python
from gaussian.renderer.nbv.diag_fisher import LogFisherEvaluator as FisherEvaluator
```

也就是说：

- **运行入口类**：`diag_fisher.py::LogFisherEvaluator`
- **主体实现来源**：`LogFisherEvaluator` 继承自 `legacy_fisher.py::LegacyFisherEvaluator`
- **可视化入口**：`visualization.py::FisherVisualizer`

因此，文档中提到 `legacy_fisher.py` 时，表示的是“主体逻辑所在文件”；
而在追踪真实运行链路时，应以 `LogFisherEvaluator -> LegacyFisherEvaluator` 这条继承链为准。

## 当前信息场可视化链路

调用主链路：

1. `GSBackEnd.update_fisher_hemisphere_pc(...)`
2. `self.fisher_eval.build_hemisphere_field(...)`
3. `self.fisher_visualizer.apply_field_result(...)`

其中：

- `build_hemisphere_field(...)` 负责“算什么”
- `apply_field_result(...)` 负责“怎么画”

所以：

- 计算开关应放在 evaluator 侧，避免不必要的中心差分开销
- 显示开关应放在 visualizer 侧，保证渲染入口统一

## 新增：速度场可视化

速度场基于 Fisher 信息场在各采样点的局部梯度。

### 计算方式

在当前真实调用链中，运行时实例是 `diag_fisher.py` 里的 `LogFisherEvaluator`，但速度场相关主体逻辑实现在其父类 `legacy_fisher.py` 中。

也就是说，在 `legacy_fisher.py` 中：

- 每个半球采样点先计算该点的 Fisher 分数
- 当 `enable_velocity_field=True` 时，再调用仓库已有的中心差分函数：

```python
compute_view_gradient(...)
```

得到：

- `dF/dtheta`
- `dF/dphi`

随后将角度空间梯度转换到半球曲面上的 3D 切向方向，得到每个采样点对应的单位速度方向 `sample_vel_dirs`

### 显示方式

在 `visualization.py` 中：

- 当 `show_velocity_field=True` 且 `field_result.sample_vel_dirs` 不为空时
- 在每个采样点绘制一个红色小箭头
- 箭头起点为采样点
- 所有箭头长度一致
- 当前不做插值

## 配置项

### 1. 是否计算速度场

```python
config["enable_velocity_field"]
```

- 类型：`bool`
- 默认：`False`
- 作用：是否在采样点上额外执行中心差分并计算速度场方向
- 建议：只有需要看速度场时才打开，因为中心差分会增加计算量

### 2. 是否显示速度场箭头

```python
config["show_velocity_field"]
```

- 类型：`bool`
- 默认：`False`
- 作用：是否在 Open3D 窗口中渲染速度场箭头
- 注意：仅打开这个开关而未打开 `enable_velocity_field` 时，不会有箭头

### 3. 箭头长度

```python
config["velocity_arrow_length"]
```

- 类型：`float`
- 默认：`0.07`
- 约定：
  - 当 `<= 1.0` 时，按半球半径比例解释
  - 当 `> 1.0` 时，按世界坐标绝对长度解释

### 4. 调试日志

```python
config["velocity_debug_log"]
```

- 类型：`bool`
- 默认：`False`
- 作用：是否打印速度场中心差分过程中的 Fisher 值与梯度日志

## 为什么这么放开关

这次的开关拆成两层：

### evaluator 侧：`enable_velocity_field`

放在 `legacy_fisher.build_hemisphere_field(...)` 最合适，因为：

- 速度场需要中心差分，代价明显高于纯颜色场可视化
- 关掉时应该直接不算，而不是算完再丢弃
- evaluator 本来就是场数据生成入口

### visualizer 侧：`show_velocity_field`

放在 `visualization.apply_field_result(...)` 最合适，因为：

- 这里已经是 Fisher 半球渲染统一入口
- Fisher 点云和箭头应在同一帧一起更新/清理
- 避免在 `gs_backend.py` 里散落额外 Open3D 渲染逻辑

## 当前涉及的关键文件

### `diag_fisher.py` + `legacy_fisher.py`

当前真实调用方式是：

- `gs_backend.py` 实例化 `diag_fisher.py::LogFisherEvaluator`
- `LogFisherEvaluator` 继承 `legacy_fisher.py::LegacyFisherEvaluator`

因此关键逻辑分层如下：

- `diag_fisher.py`
  - 决定当前使用哪种统计形式（当前默认是 `LogFisherEvaluator`）
- `legacy_fisher.py`
  - `compute_view_gradient(...)`：中心差分计算梯度
  - `build_hemisphere_field(...)`：
    - 生成 `sample_vals`
    - 在开关打开时生成 `sample_vel_dirs`

### `interfaces.py`

`HemisphereFieldResult` 中增加：

```python
sample_vel_dirs: Optional[torch.Tensor] = None
```

用于把 evaluator 计算出的速度方向传给可视化层。

### `visualization.py`

增加速度场箭头渲染逻辑：

- `_build_arrow_mesh(...)`：构造单个 Open3D 箭头
- `apply_field_result(...)`：
  - 统一控制 Fisher 半球点云
  - 统一控制速度场箭头的添加、移除、刷新

### `gs_backend.py`

仍然通过：

```python
self.fisher_visualizer.apply_field_result(field_result)
```

触发渲染。

这是当前最合适的箭头调用位置，因为它和 Fisher 半球点云更新严格同步。

## 推荐使用方式

若想开启速度场显示，建议同时设置：

```python
enable_velocity_field = True
show_velocity_field = True
velocity_arrow_length = 0.07
```

如果只是调试信息场颜色，不想承担额外中心差分开销：

```python
enable_velocity_field = False
show_velocity_field = False
```

## 备注

当前速度场是“采样点级别”的直接可视化：

- 不做插值
- 不生成稠密向量场
- 不对箭头长度做幅值编码

也就是说，当前箭头只表达“方向”，不表达“梯度大小”强弱。
