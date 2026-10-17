# Tools

离线工具脚本，与主建图流程分开。脚本自行把仓库根加入 `sys.path`，
并把 `outputs/`、`config/` 锚定到仓库根，因此从任意工作目录调用都可以。

## tsdf_integrate.py

由 3DGS 渲染出的深度与彩色图做 TSDF 融合，生成彩色三角网格。

```bash
python tools/tsdf_integrate.py --dataset replica --scene room_0
python tools/tsdf_integrate.py --dataset scannet --scene scene0000_00
```

输入：`outputs/{scene}/renders/tsdfdepth_after_opt/` 与 `tsdfrgb_after_opt/`，
由 `demo.py` 结束时的 `OMNI.terminate()` → `gs.eval_rendering(iteration="after_opt")` 产出。
所以必须先跑完 `demo.py` 再跑这个脚本。

输出：`outputs/{scene}/tsdf_mesh_w{weight}.ply`，每个 `--weight` 一份。

与在线 `omnimap/tsdf_backend.py` 的区别：后者在建图过程中用传感器深度增量融合
（voxel 0.03，输出 `tsdf_mesh.ply`）；本脚本是离线后处理，用优化后的渲染深度、
更细的体素（默认 0.01），用于最终高质量网格。两者不重复。
