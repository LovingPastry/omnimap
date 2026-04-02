import os  # nopep8

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys  # nopep8

sys.path.append(os.path.join(os.path.dirname(__file__), "omnimap"))  # nopep8
import time
import torch
import cv2
import re
import os
import argparse
import numpy as np
import lietorch
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

from omnimap.util.utils import load_config
from tqdm import tqdm, trange
from torch.multiprocessing import Process, Queue
from omni import OMNI
from natsort import natsorted
from scipy.spatial.transform import Rotation as R


class SceneSimulator:
    """
    职责：
        - 加载静态点云
        - 按需加载地面
        - 管理 Open3D 场景
        - 从给定位姿渲染 RGBD

    这里要解决的事情：
        - 点云导入
        - 地面构造
        - 用 Open3D 的离屏渲染得到 color/depth
    """

    def __init__(self, pointcloud_path, ground=True, width=640, height=480): ...
    def set_camera_intrinsics(self, fx, fy, cx, cy): ...
    def render(self, c2w):
        # return rgb, depth
        ...


class OmniMapRunner:
    """
    职责：
        - 初始化 OMNI
        - 把仿真出来的 RGBD 喂进去
        - 维持 tstamp / intrinsics / pose 格式兼容

    这里的核心不是创新，是适配数据格式
    """

    def __init__(self, args, config):
        self.omni = OMNI(args, config)

    def step(self, idx, rgb, depth, w2c_pose_vec, c2w_44, intrinsics):
        self.omni.track(...)


class FisherMotionPolicy:
    """
    职责：
        - 从当前 viewpoint 出发
        - 调用现有半球场逻辑
        - 取速度方向
        - 更新相机球坐标位置

    这里的关键是：
        - 你不需要 next_viewpoint
        - 你直接在主循环里拿当前 viewpoint
        - 然后自己调用：`build_hemisphere_field(...)` 或复用 `update_fisher_hemisphere_pc(...)` 之后缓存结果
        - 再从 sample_vel_dirs / 梯度里选下一步
    """

    def __init__(self, step_theta, step_phi): ...
    def compute_next_pose(self, gs_backend, current_viewpoint, idx):
        # 返回下一时刻 c2w
        ...


def main():
    """主函数
    职责：
        - 管状态
        - 管时间步
        - 串起仿真、建图、规划、显示

    伪代码：
    ```
    pose = init_pose
    for idx in range(T):
        rgb, depth = simulator.render(pose.c2w)
        omni_runner.step(idx, rgb, depth, pose.w2c_vec, pose.c2w_44, intrinsics)
        pose = motion_policy.compute_next_pose(
            gs_backend=omni_runner.omni.gs,
            current_viewpoint=...,
            idx=idx
        )
    ```
    """
    pass
