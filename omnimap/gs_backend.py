import random
import time
import math
import copy
import numpy as np
import torch
import cv2
import torch.multiprocessing as mp
from tqdm import trange, tqdm
from munch import munchify
from lietorch import SE3, SO3
import open3d as o3d
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from util.utils import Log, clone_obj
from util.vis_utils import (
    draw_camera,
    create_camera_trajectory_line,
    update_camera_trajectory,
)
from gaussian.renderer import render
from gaussian.utils.loss_utils import l1_loss, ssim
from gaussian.scene.gaussian_model import GaussianModel
from gaussian.renderer.fisher_grandient import NextBestViewOptimizer
from gaussian.utils.graphics_utils import getProjectionMatrix2
from gaussian.utils.sh_utils import SH2RGB
from gaussian.utils.mapping_utils import (
    to_se3_vec,
    get_loss_normal,
    get_loss_mapping_rgbd,
    get_loss_depth_normal,
    SE3_exp,
)
from gaussian.utils.camera_utils import Camera, HemisphereCamera
from gaussian.utils.eval_utils import (
    eval_rendering,
    eval_rendering_kf,
    eval_fast,
    eval_rendering_all,
    set_all_camera_deblur,
    eval_rendering_blur,
)

# from gaussian.gui import gui_utils, slam_gui
import warnings

warnings.filterwarnings("ignore")
from visual_module import timeit
import math
from typing import Tuple, Dict, Optional, List
from tsdf_backend import TSDFBackEnd


class GSBackEnd(mp.Process):
    def __init__(
        self, config, tsdfs: TSDFBackEnd, save_dir: str, vis_gui: bool = False
    ):
        super().__init__()
        self.config = config

        self.iteration_count = 0
        # only need save keyframe viewpoint
        self.allviewpoints = []
        self.keyviewpoints = []
        self.keyframe_stamps = []
        self.initialized = False
        self.save_dir = save_dir
        self.tsdfs = tsdfs

        self.opt_params = munchify(config["opt_params"])

        self.gaussians = GaussianModel(sh_degree=0, config=self.config)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(self.opt_params)
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.no_key_count = 0.0
        self.cameras_extent = 6.0
        self.set_hyperparams()
        self.vis_gui = vis_gui

        # for multi-view
        self.key_camera_centers = []
        self.key_center_rays = []
        self.key_graph = {}

        self.sence_center = None
        self.fisher_hemi_geometry = None
        self.fisher_frame0_exported = False
        self._last_gs_points = None
        self._last_gs_colors = None
        self._last_fisher_points = None
        self._last_fisher_colors = None

    def set_gui(self):
        # OpenCV window name
        self.window_name = "omnimap - Visualization"
        self.images = [
            np.zeros((self.vis_h, self.vis_w, 3), dtype=np.uint8)
        ] * 5  # Placeholder for 4 images
        # self.texts = ["", "", "", ""]  # Placeholder for 4 text lines
        # Initialize OpenCV window (First time)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Display 4 images and 4 text lines in the OpenCV window
        self.edge = 50
        self.img_display = np.zeros(
            ((self.edge * 4 + self.vis_h * 2), (self.edge * 4 + self.vis_w * 3), 3),
            dtype=np.uint8,
        )  # Create a blank canvas
        if self.vis_w >= 600:
            self.font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf", 50
            )
        else:
            self.font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf", 42
            )

        self.o3d_window = o3d.visualization.VisualizerWithKeyCallback()
        self.o3d_window.create_window(
            window_name="3DGS Point Viewer", width=860, height=540
        )
        self.gs_pc_geometries, self.cam_lines, self.cam_plan = None, None, None
        self.cam_traj = create_camera_trajectory_line()
        self.fisher_hemi_geometry = None

    def add_camera(self, pose, size=0.1):
        if self.cam_lines is not None:
            self.o3d_window.remove_geometry(self.cam_lines)
            self.o3d_window.remove_geometry(self.cam_plane)
        self.cam_lines, self.cam_plane = draw_camera(pose)
        self.o3d_window.add_geometry(self.cam_lines)
        self.o3d_window.add_geometry(self.cam_plane)

    def update_gs_pc(self):
        if self.cam_traj is not None:
            self.o3d_window.remove_geometry(self.cam_traj)
        self.o3d_window.add_geometry(self.cam_traj)
        if self.gs_pc_geometries is not None:
            self.o3d_window.remove_geometry(
                self.gs_pc_geometries, reset_bounding_box=False
            )

        opacity = self.gaussians.get_opacity.detach().squeeze().cpu().numpy()
        mask = opacity > 0.3
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(
            self.gaussians.get_xyz.detach().cpu().numpy()[mask]
        )
        rgbs = SH2RGB(self.gaussians.get_features.detach()).squeeze().cpu().numpy()
        pc.colors = o3d.utility.Vector3dVector(rgbs[mask])
        self._last_gs_points = np.asarray(pc.points).copy()
        self._last_gs_colors = np.asarray(pc.colors).copy()
        if self.config["scene"] == "room_0":
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(-np.inf, -np.inf, -np.inf), max_bound=(np.inf, np.inf, 0.7)
            )
            pc = pc.crop(bbox)
        # debug
        # o3d.visualization.draw_geometries([pc])
        self.gs_pc_geometries = pc
        self.o3d_window.add_geometry(self.gs_pc_geometries)
        self.o3d_window.poll_events()
        self.o3d_window.update_renderer()

    def update_images(self, tstamp, hz):
        # reset the text area
        self.img_display = np.zeros(
            ((self.edge * 4 + self.vis_h * 2), (self.edge * 4 + self.vis_w * 3), 3),
            dtype=np.uint8,
        )
        mask_img, ins_img, ins_num = self.tsdfs.get_vis_imgs(self.images[4])
        # Arrange 4 images in a 2x2 grid
        self.img_display[
            self.edge * 2 : (self.edge * 2 + self.vis_h),
            self.edge : (self.edge + self.vis_w),
        ] = self.images[0]  # Top-left
        self.img_display[
            self.edge * 2 : (self.edge * 2 + self.vis_h),
            (2 * self.edge + self.vis_w) : (2 * self.edge + self.vis_w * 2),
        ] = self.images[2]  # Top-middle
        self.img_display[
            self.edge * 2 : (self.edge * 2 + self.vis_h),
            (3 * self.edge + self.vis_w * 2) : (3 * self.edge + self.vis_w * 3),
        ] = mask_img  # Top-right
        self.img_display[
            (self.edge * 3 + self.vis_h) : (self.edge * 3 + self.vis_h * 2),
            self.edge : (self.edge + self.vis_w),
        ] = self.images[1]  # Bottom-left
        self.img_display[
            (self.edge * 3 + self.vis_h) : (self.edge * 3 + self.vis_h * 2),
            (2 * self.edge + self.vis_w) : (2 * self.edge + self.vis_w * 2),
        ] = self.images[3]  # Bottom-middle
        self.img_display[
            (self.edge * 3 + self.vis_h) : (self.edge * 3 + self.vis_h * 2),
            (3 * self.edge + self.vis_w * 2) : (3 * self.edge + self.vis_w * 3),
        ] = ins_img  # Bottom-right

        gaussian_count = len(self.gaussians.get_xyz)
        kf_len = len(self.keyframe_stamps)
        formatted_text = (
            f" Frame: {tstamp:4d}    Gaussians: {gaussian_count:6d}    KFs: {kf_len:3d}"
        )

        img_pil = Image.fromarray(self.img_display)
        draw = ImageDraw.Draw(img_pil)
        draw.text((400, 20), formatted_text, font=self.font, fill=(255, 255, 255))
        self.img_display = np.array(img_pil)
        img_display = cv2.resize(
            self.img_display,
            (
                int(self.img_display.shape[1] * 0.5),
                int(self.img_display.shape[0] * 0.5),
            ),
        )
        image_save_dir = f"{self.save_dir}/online_vis/"
        os.makedirs(image_save_dir, exist_ok=True)
        if tstamp % self.config["instance"]["instance_skip"] == 0:
            cv2.imwrite(f"{image_save_dir}/{tstamp}.jpg", img_display)
        cv2.imshow(self.window_name, img_display)
        cv2.resizeWindow(self.window_name, img_display.shape[1], img_display.shape[0])
        cv2.moveWindow(self.window_name, 875, 50)
        cv2.waitKey(1)  # Refresh window

    def set_hyperparams(self):
        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.max_keyframe_skip = self.config["Training"]["max_keyframe_skip"]
        self.window_size = self.config["Training"]["window_size"]
        self.frame_itr = self.config["Training"]["frame_itr"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.use_omni_normal = self.config["Training"]["use_omni_normal"]
        self.normal_weight = self.config["Training"]["normal_weight"]
        self.use_post_refine = self.config["opt_params"]["post_refine"]
        if self.use_post_refine:
            self.post_itr = self.config["opt_params"]["post_itr"]
        self.wait_latest_keyframe = False
        self.deblur = self.config["Training"]["deblur"]
        self.camera_optimizer = None

    def post_refine(self):
        Log("Starting post refinement", tag="GaussianSplatting")
        if self.config["Training"]["compensate_exposure"]:
            opt_params = []
            for view in self.keyviewpoints:
                # for view in self.allviewpoints:
                # print(view.uid)
                opt_params.append(
                    {
                        "params": [view.exposure_a],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": "exposure_a_{}".format(view.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [view.exposure_b],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": "exposure_b_{}".format(view.uid),
                    }
                )
            self.exposure_optimizers = torch.optim.Adam(opt_params)

        for iteration in (pbar := trange(1, self.post_itr + 1)):
            loss = 0.0
            use_indices = torch.randperm(len(self.keyframe_stamps))[: self.window_size]
            viewpoints = [self.keyviewpoints[random_id] for random_id in use_indices]
            # use_indices = torch.randperm(len(self.allviewpoints))[:self.window_size]
            # viewpoints = [self.allviewpoints[random_id] for random_id in use_indices]
            for viewpoint in viewpoints:
                render_pkg = render(viewpoint, self.gaussians, self.background)
                image, depth = render_pkg["render"], render_pkg["depth"]
                image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
                loss += get_loss_mapping_rgbd(
                    self.config, image, depth, viewpoint, self.deblur
                )
                if self.use_omni_normal:
                    loss += self.normal_weight * get_loss_normal(depth, viewpoint)
                else:
                    loss += self.normal_weight * get_loss_depth_normal(depth, viewpoint)

            loss.backward()

            with torch.no_grad():
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                lr = self.gaussians.update_learning_rate(iteration)
                if self.deblur:
                    self.camera_optimizer.step()
                    self.camera_optimizer.zero_grad(set_to_none=True)
                if self.config["Training"]["compensate_exposure"]:
                    self.exposure_optimizers.step()
                    self.exposure_optimizers.zero_grad(set_to_none=True)
            pbar.set_description(
                f"Global GS Refinement lr {lr:.3E} loss {loss.item():.3f}"
            )

    @timeit
    def finalize(self):
        if self.use_post_refine:
            self.post_refine()
        self.gaussians.save_ply(f"{self.save_dir}/3dgs_final.ply")
        return

    @torch.no_grad()
    @timeit
    def eval_fast(self, gtimages, traj, depth_scale=1000.0):
        self.cam_params = set_all_camera_deblur(
            gtimages, self.keyframe_stamps, self.keyviewpoints, self.save_dir
        )
        eval_fast(
            gtimages,
            traj,
            self.gaussians,
            self.background,
            self.projection_matrix,
            self.K,
            self.cam_params,
        )
        eval_rendering_kf(self.keyviewpoints, self.gaussians, self.background)

    @torch.no_grad()
    @timeit
    def eval_rendering(self, gtimages, gtdepths, traj, depth_scale=1000.0):
        eval_rendering(
            gtimages,
            gtdepths,
            traj,
            self.gaussians,
            self.save_dir,
            self.background,
            self.projection_matrix,
            self.K,
            self.tsdfs,
            iteration="after_opt",
            depth_scale=depth_scale,
            cam_params=self.cam_params,
        )
        eval_rendering_kf(self.keyviewpoints, self.gaussians, self.background)

    def reset(self):
        self.iteration_count = 0
        self.current_window = []
        self.initialized = False

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(viewpoint, self.gaussians, self.background)
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping_rgbd(self.config, image, depth, viewpoint)
            if self.use_omni_normal:
                loss_init += self.normal_weight * get_loss_normal(depth, viewpoint)
            else:
                loss_init += self.normal_weight * get_loss_depth_normal(
                    depth, viewpoint
                )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset:
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        Log("Initialized map")
        return render_pkg

    def map(self, viewpoints, iters, current_id, is_keyframe, corr_index=None):
        """
        3dgs training
        """

        for iter in range(iters):
            self.iteration_count += 1
            loss_for_gs = 0
            if corr_index is not None:
                # all_view_rgb = []
                all_view_depth = []
                all_view_points = []
            for view_id, viewpoint in enumerate(viewpoints):
                render_pkg = render(viewpoint, self.gaussians, self.background)
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["n_touched"],
                )
                loss_this = get_loss_mapping_rgbd(
                    self.config, image, depth, viewpoint, self.deblur
                )
                if self.use_omni_normal:
                    loss_this += self.normal_weight * get_loss_normal(depth, viewpoint)
                else:
                    if corr_index is not None:
                        normal_loss, points = get_loss_depth_normal(
                            depth,
                            viewpoint,
                            current_id=current_id,
                            corr=(corr_index is not None),
                        )
                    else:
                        normal_loss = get_loss_depth_normal(
                            depth,
                            viewpoint,
                            current_id=current_id,
                            corr=(corr_index is not None),
                        )
                    loss_this += self.normal_weight * normal_loss
                if corr_index is not None:
                    # all_view_rgb.append(image)
                    all_view_depth.append(depth)
                    all_view_points.append(points)
                loss_for_gs += loss_this
                if self.vis_gui:
                    if iter == iters - 1 and view_id == 0:
                        pre_image = image.permute(1, 2, 0).clone().detach()
                        self.images[1] = (
                            torch.clamp(pre_image, 0, 1).cpu().numpy()[:, :, ::-1] * 255
                        )
                        depth_vis = depth[0].clone().detach().cpu().numpy()
                        min_depth, max_depth = 0.1, 5.0
                        depth_vis = np.clip(depth_vis, 0.1, 5.0)
                        depth_norm = (
                            (depth_vis - min_depth) / (max_depth - min_depth)
                        ) * 255
                        depth_norm = depth_norm.astype(np.uint8)
                        self.images[3] = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                        self.images[4] = depth[0].clone().detach()
            loss_for_gs.backward()

            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                # --- 新增：周期性 densify/prune + reset_opacity（像 initialize_map 那样）---
                if self.iteration_count % self.gaussian_update_every == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        None,
                    )

                if self.iteration_count % self.gaussian_reset == 0:
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                # loss_for_blur.backward()
                # with torch.no_grad():
                if self.deblur:
                    self.camera_optimizer.step()
                    self.camera_optimizer.zero_grad(set_to_none=True)

    def process_track_data(self, packet, hz):
        """处理跟踪数据的核心函数，负责将每一帧数据集成到3D高斯地图中

        Args:
            packet: 包含当前帧所有数据的字典
            hz: 当前处理频率，用于自适应参数调整
        """
        # 1. 初始化投影矩阵（仅在首次调用时执行）
        if not hasattr(self, "projection_matrix"):
            H, W = packet["images"].shape[-2:]
            self.K = K = list(packet["intrinsics"]) + [W, H]
            self.projection_matrix = (
                getProjectionMatrix2(
                    znear=0.01, zfar=100.0, fx=K[0], fy=K[1], cx=K[2], cy=K[3], W=W, H=H
                )
                .transpose(0, 1)
                .cuda()
            )
        # 2. 从数据包中提取当前帧信息
        w2c = SE3(packet["poses"]).matrix().cuda()
        tstamp = packet["tstamp"]
        idx = int(tstamp)

        # 3. 创建当前帧的相机视点对象
        viewpoint = Camera.init_from_tracking(
            packet["images"] / 255.0,
            packet["depths"],
            w2c,
            idx,
            self.projection_matrix,
            self.K,
            tstamp,
            normal=packet["normals"],
            bg=self.background,
        )

        # 4. 初始化阶段处理（仅在首次调用时执行）
        if not self.initialized:
            self.reset()
            new_points, new_coplors, is_keyframe = self.tsdfs.initializing_check()

            # 将TSDF中的几何信息转换为3D高斯点（使用较小的初始尺度）
            self.gaussians.extend_from_tsdfs(
                new_points, new_coplors, self.tsdfs.voxel_size / 5
            )
            # initialize map for a large amount of iterations
            self.initialize_map(0, viewpoint)
            self.initialized = True
            if self.vis_gui:
                self.vis_h, self.vis_w = (
                    packet["images"].shape[1],
                    packet["images"].shape[2],
                )
                self.set_gui()
        # new image needs initialize for viewpoint and gs
        else:
            # 非初始化阶段，检查是否需要添加新的几何点
            new_points, new_coplors, is_keyframe = self.tsdfs.initializing_check()
            # 将TSDF中的新几何点转换为3D高斯点（使用较大的尺度）
            self.gaussians.extend_from_tsdfs(
                new_points, new_coplors, self.tsdfs.voxel_size / 2
            )

        # # --- 新增：只在 keyframe 才扩展新高斯（初始化阶段已做过）---
        # if self.initialized and is_keyframe:
        #     if new_points is not None and hasattr(new_points, "numel") and new_points.numel() > 0:
        #         self.gaussians.extend_from_tsdfs(new_points, new_coplors, self.tsdfs.voxel_size/2)

        # 5. 关键帧判断逻辑
        # 5.1 如果设置了等待最新关键帧标志，强制当前帧为关键帧
        if self.wait_latest_keyframe:
            is_keyframe = True
            self.wait_latest_keyframe = False

        # 5.2 非关键帧计数器递增
        if not is_keyframe:
            self.no_key_count += 1
            # 如果连续非关键帧数超过阈值，强制设为关键帧
        if self.no_key_count >= self.max_keyframe_skip:
            is_keyframe = True

        # 5.3 如果与上一个关键帧间隔太小，不设为关键帧
        if len(self.keyframe_stamps) > 0 and (idx - self.keyframe_stamps[-1]) < 3:
            is_keyframe = False
        # self.allviewpoints.append(viewpoint)
        if is_keyframe:
            self.keyframe_stamps.append(idx)
            self.keyviewpoints.append(viewpoint)
            self.no_key_count = 0
            # tell the tsdf to reset
            self.tsdfs.reset_unregistered()
        # 7. 运动模糊处理（可选功能）
        if self.deblur:
            # 7.1 首次初始化优化器
            if self.camera_optimizer is None:
                opt_params = []
                # 添加当前帧权重参数
                opt_params.append(
                    {
                        "params": [viewpoint.weight_this],
                        "lr": self.config["opt_params"]["deblur_weight"],
                        "name": "weight_this_{}".format(viewpoint.uid),
                    }
                )
                # 添加模糊权重参数
                opt_params.append(
                    {
                        "params": [viewpoint.weight_blur],
                        "lr": self.config["opt_params"]["deblur_weight"],
                        "name": "weight_blur_{}".format(viewpoint.uid),
                    }
                )
                # 添加模糊X方向平移参数
                opt_params.append(
                    {
                        "params": [viewpoint.blur_tran_x],
                        "lr": self.config["opt_params"]["deblur_trans"],
                        "name": "blur_tran_x_{}".format(viewpoint.uid),
                    }
                )
                # 添加模糊Y方向平移参数
                opt_params.append(
                    {
                        "params": [viewpoint.blur_tran_y],
                        "lr": self.config["opt_params"]["deblur_trans"],
                        "name": "blur_tran_y_{}".format(viewpoint.uid),
                    }
                )
                # 创建Adam优化器
                self.camera_optimizer = torch.optim.Adam(opt_params)
            else:
                # 7.2 为现有优化器添加新参数
                new_params = []
                new_params.append(
                    {
                        "params": [viewpoint.weight_this],
                        "lr": self.config["opt_params"]["deblur_weight"],
                        "name": "weight_this_{}".format(viewpoint.uid),
                    }
                )
                new_params.append(
                    {
                        "params": [viewpoint.weight_blur],
                        "lr": self.config["opt_params"]["deblur_weight"],
                        "name": "weight_blur_{}".format(viewpoint.uid),
                    }
                )
                new_params.append(
                    {
                        "params": [viewpoint.blur_tran_x],
                        "lr": self.config["opt_params"]["deblur_trans"],
                        "name": "blur_tran_x_{}".format(viewpoint.uid),
                    }
                )
                new_params.append(
                    {
                        "params": [viewpoint.blur_tran_y],
                        "lr": self.config["opt_params"]["deblur_trans"],
                        "name": "blur_tran_y_{}".format(viewpoint.uid),
                    }
                )
                for param_group in new_params:
                    self.camera_optimizer.add_param_group(param_group)

        # 8. 构建优化窗口：随机选择历史关键帧 + 当前帧
        use_indices = torch.randperm(len(self.keyframe_stamps))[
            : self.window_size
        ]  # 随机选择指定数量的关键帧索引
        viewpoints = [viewpoint] + [
            self.keyviewpoints[random_id] for random_id in use_indices
        ]  # 构建优化视点列表

        # 9. 执行地图优化：使用选定的视点窗口优化3D高斯参数
        self.map(viewpoints, self.frame_itr, idx, is_keyframe)

        # 10. 视角规划
        if self.sence_center is None:
            self.sence_center = self.tsdfs.get_pointcloud_center()
            if self.sence_center is not None:
                Log(
                    f"Scene center initialized at {self.sence_center.cpu().numpy()}",
                    tag="NextBestView",
                )

        if self.sence_center is not None:
            hemisphere_viewpoint = HemisphereCamera.from_camera(
                viewpoint, self.sence_center
            )
            torch.cuda.synchronize()

            history_hessian = self.cal_history_hessian(
                keyframe_viewpoints=self.keyviewpoints
            )
            fisher = self.cal_fisher_info(hemisphere_viewpoint, history_hessian)
            Log(
                f"history_hessian at frame {idx}: {history_hessian}"
                f"fisher at frame {idx}: {fisher}",
                tag="NextBestView",
            )
            self.get_fisher_grad(hemisphere_viewpoint, history_hessian)
            torch.cuda.synchronize()

        if self.vis_gui:
            gt_image = packet["images"].permute(1, 2, 0).clone()
            self.images[0] = torch.clamp(gt_image, 0, 255).cpu().numpy()[:, :, ::-1]
            depth = packet["depths"].clone().cpu().numpy()
            min_depth, max_depth = 0.1, 5.0
            depth = np.clip(depth, min_depth, max_depth)
            depth_norm = ((depth - min_depth) / (max_depth - min_depth)) * 255
            depth_norm = depth_norm.astype(np.uint8)
            self.images[2] = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            self.update_images(tstamp, hz)
            pose = np.linalg.inv(w2c.cpu().numpy())
            self.cam_traj = update_camera_trajectory(self.cam_traj, [pose[:3, 3]])
            if idx % 10 == 0:
                self.add_camera(pose)
                self.update_gs_pc()
                self.update_fisher_hemisphere_pc(
                    viewpoint=viewpoint,
                    idx=idx,
                    num_samples=32,
                    num_dense_points=2048,
                    power=2.0,
                )

        if not self.vis_gui and idx == 0:
            self.update_fisher_hemisphere_pc(
                viewpoint=viewpoint,
                idx=idx,
                num_samples=32,
                num_dense_points=2048,
                power=2.0,
            )

    @timeit
    def gs_instance(self, vis=False):
        """assiocate the instance id to gs"""
        gs_xyz = self.gaussians.get_xyz.clone().detach()
        ids, colors = self.tsdfs.get_instance_ids(gs_xyz)
        # save gaussian id
        torch.save(ids, f"{self.save_dir}/gs_id.pt")
        bgr_reversed = torch.flip(colors, dims=[1])
        self.gaussians.set_instance_coloor(bgr_reversed)
        # debug gaussians vis
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(gs_xyz.cpu().numpy())
        pc.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
        if vis:
            o3d.visualization.draw_geometries([pc])
        o3d.io.write_point_cloud(f"{self.save_dir}/instance_gs.ply", pc)

    def cal_history_hessian(
        self, keyframe_viewpoints: List[Camera] = None
    ) -> torch.Tensor:
        """累计历史视角 Fisher 对角近似: H_train = Σ cur_H"""
        if keyframe_viewpoints is None:
            keyframe_viewpoints = self.keyviewpoints

        # 保持与 gaussians.cal_cur_hessian 的参数选择一致（xyz + opacity）
        params = [self.gaussians.capture()[1], self.gaussians.capture()[6]]
        history_hessian = torch.zeros(
            sum(p.numel() for p in params),
            device=params[0].device,
            dtype=params[0].dtype,
        )

        for keyframe in keyframe_viewpoints:
            keyframe_hessian = self.gaussians.cal_cur_hessian(
                cam=keyframe, return_per_point=False
            )
            history_hessian += keyframe_hessian

        return history_hessian

    def cal_fisher_info(self, cam: Camera, history_hessian: torch.Tensor):
        cur_hessian = self.gaussians.cal_cur_hessian(cam=cam, return_per_point=False)

        reg_lambda = float(self.config.get("fisher_reg_lambda", 0.1))
        i_train = torch.reciprocal(history_hessian + reg_lambda)
        fisher = torch.sum(cur_hessian * i_train)
        return fisher

    @timeit
    def get_fisher_grad(
        self, viewpoint: HemisphereCamera, history_hessian: torch.Tensor, eps=0.01
    ):
        """
        使用有限差分近似 Fisher 信息量对 theta/phi 的梯度。
        对 phi 采用边界策略：靠近 [0, pi/2] 边界时自动退化为单边差分，避免 clamp 导致伪零梯度。
        返回: torch.Tensor([dF/dtheta, dF/dphi])
        不会修改传入的 viewpoint
        """
        base_theta = float(viewpoint.theta.item())
        base_phi = float(viewpoint.phi.item())
        device = viewpoint.theta.device

        def fisher_at(theta_val, phi_val):
            temp_cam = copy.deepcopy(viewpoint)
            temp_cam.set_angles(theta_val, phi_val)
            fisher = self.cal_fisher_info(temp_cam, history_hessian)
            Log(
                f"Fisher at (theta={theta_val:.3f}, phi={phi_val:.3f}): {fisher.item():.6f}",
                tag="NextBestView",
            )
            return fisher

        f_theta_plus = fisher_at(base_theta + eps, base_phi)
        f_theta_minus = fisher_at(base_theta - eps, base_phi)
        dF_dtheta = (f_theta_plus - f_theta_minus) / (2.0 * eps)

        phi_min, phi_max = 0.0, math.pi / 2.0
        phi_step_plus = min(eps, max(phi_max - base_phi, 0.0))
        phi_step_minus = min(eps, max(base_phi - phi_min, 0.0))

        if phi_step_plus > 0.0 and phi_step_minus > 0.0:
            f_phi_plus = fisher_at(base_theta, base_phi + phi_step_plus)
            f_phi_minus = fisher_at(base_theta, base_phi - phi_step_minus)
            dF_dphi = (f_phi_plus - f_phi_minus) / (phi_step_plus + phi_step_minus)
        elif phi_step_plus > 0.0:
            f_phi_base = fisher_at(base_theta, base_phi)
            f_phi_plus = fisher_at(base_theta, base_phi + phi_step_plus)
            dF_dphi = (f_phi_plus - f_phi_base) / phi_step_plus
        elif phi_step_minus > 0.0:
            f_phi_base = fisher_at(base_theta, base_phi)
            f_phi_minus = fisher_at(base_theta, base_phi - phi_step_minus)
            dF_dphi = (f_phi_base - f_phi_minus) / phi_step_minus
        else:
            dF_dphi = torch.zeros_like(dF_dtheta)

        Log(
            f"Fisher gradient: dF/dtheta={dF_dtheta:.6f}, dF/dphi={dF_dphi:.6f}",
            tag="NextBestView",
        )

        return torch.stack([dF_dtheta, dF_dphi]).to(device)

    def _fibonacci_hemisphere_dirs(self, n: int, device: torch.device):
        # 均匀采样 Z-up 上半球单位方向 (z >= 0)
        i = torch.arange(n, device=device, dtype=torch.float32) + 0.5
        z = i / n  # [0, 1]
        golden = (1.0 + 5.0**0.5) / 2.0
        az = (2.0 * math.pi * i / golden) % (2.0 * math.pi)
        r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        x = r * torch.cos(az)
        y = r * torch.sin(az)
        dirs = torch.stack([x, y, z], dim=-1)  # [n, 3]
        return dirs

    def _scalarize_fisher(self, fisher_tensor):
        f = torch.as_tensor(fisher_tensor).detach().float()
        f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        return float(f.mean().item())

    def _idw_on_sphere(self, sample_dirs, sample_vals, query_dirs, power=2.0):
        dots = torch.clamp(query_dirs @ sample_dirs.T, -1.0, 1.0)
        d = 1.0 - dots
        wgt = 1.0 / (torch.pow(d, power) + 1e-6)
        wgt = wgt / (wgt.sum(dim=1, keepdim=True) + 1e-8)
        return (wgt * sample_vals[None, :]).sum(dim=1)

    def _fisher_values_to_colors(self, fisher_vals: torch.Tensor):
        fisher_vals = fisher_vals.detach().float()
        lo = torch.quantile(fisher_vals, 0.05)
        hi = torch.quantile(fisher_vals, 0.95)
        denom = torch.clamp(hi - lo, min=1e-8)
        fisher_norm = torch.clamp((fisher_vals - lo) / denom, 0.0, 1.0)
        fisher_u8 = (fisher_norm * 255.0).to(torch.uint8).cpu().numpy()
        fisher_img = fisher_u8.reshape(-1, 1)
        bgr = cv2.applyColorMap(fisher_img, cv2.COLORMAP_TURBO).reshape(-1, 3)
        rgb = bgr[:, ::-1].astype(np.float32) / 255.0
        return rgb, fisher_norm

    def _compute_history_hessian_for_fisher(self):
        if len(self.keyviewpoints) > 0:
            return self.cal_history_hessian(keyframe_viewpoints=self.keyviewpoints)
        params = [self.gaussians.capture()[1], self.gaussians.capture()[6]]
        return torch.zeros(
            sum(p.numel() for p in params),
            device=params[0].device,
            dtype=params[0].dtype,
        )

    def update_fisher_hemisphere_pc(
        self,
        viewpoint,
        idx: int,
        num_samples: int = 32,
        num_dense_points: int = 2048,
        power: float = 2.0,
    ):
        if self.sence_center is None:
            self.sence_center = self.tsdfs.get_pointcloud_center()
        if self.sence_center is None:
            Log(
                "Skip Fisher hemisphere visualization: scene center is unavailable.",
                tag="NextBestView",
            )
            return

        base_hemi = HemisphereCamera.from_camera(viewpoint, self.sence_center)
        history_hessian = self._compute_history_hessian_for_fisher()

        sample_dirs = self._fibonacci_hemisphere_dirs(num_samples, self.sence_center.device)
        fisher_vals = []
        for k in range(num_samples):
            d = sample_dirs[k]
            theta = torch.atan2(d[1], d[0])
            phi = torch.asin(torch.clamp(d[2], 0.0, 1.0))
            hc = copy.deepcopy(base_hemi)
            hc.set_angles(theta=float(theta.item()), phi=float(phi.item()))
            fisher_vals.append(
                self._scalarize_fisher(self.cal_fisher_info(hc, history_hessian))
            )

        sample_vals = torch.tensor(
            fisher_vals, device=self.sence_center.device, dtype=torch.float32
        )
        dense_dirs = self._fibonacci_hemisphere_dirs(
            num_dense_points, self.sence_center.device
        )
        dense_vals = self._idw_on_sphere(
            sample_dirs, sample_vals, dense_dirs, power=power
        )
        dense_colors, _ = self._fisher_values_to_colors(dense_vals)

        center = self.sence_center.detach().float()
        radius = float(base_hemi.radius)
        dense_points = center[None, :] + radius * dense_dirs

        hemi_pc = o3d.geometry.PointCloud()
        hemi_pc.points = o3d.utility.Vector3dVector(dense_points.detach().cpu().numpy())
        hemi_pc.colors = o3d.utility.Vector3dVector(dense_colors)

        if self.vis_gui and hasattr(self, "o3d_window") and self.o3d_window is not None:
            if self.fisher_hemi_geometry is not None:
                self.o3d_window.remove_geometry(
                    self.fisher_hemi_geometry, reset_bounding_box=False
                )
            self.fisher_hemi_geometry = hemi_pc
            self.o3d_window.add_geometry(
                self.fisher_hemi_geometry, reset_bounding_box=False
            )
            self.o3d_window.poll_events()
            self.o3d_window.update_renderer()

        self._last_fisher_points = np.asarray(hemi_pc.points).copy()
        self._last_fisher_colors = np.asarray(hemi_pc.colors).copy()

        if self._last_gs_points is None or self._last_gs_colors is None:
            opacity = self.gaussians.get_opacity.detach().squeeze().cpu().numpy()
            mask = opacity > 0.3
            gs_points = self.gaussians.get_xyz.detach().cpu().numpy()[mask]
            gs_colors = SH2RGB(self.gaussians.get_features.detach()).squeeze().cpu().numpy()[mask]
            if self.config.get("scene") == "room_0":
                z_mask = gs_points[:, 2] <= 0.7
                gs_points = gs_points[z_mask]
                gs_colors = gs_colors[z_mask]
            self._last_gs_points = gs_points
            self._last_gs_colors = gs_colors

        Log(
            (
                f"Fisher hemisphere updated at frame {idx}: "
                f"samples={num_samples}, dense={num_dense_points}, "
                f"min={sample_vals.min().item():.6f}, max={sample_vals.max().item():.6f}"
            ),
            tag="NextBestView",
        )

        self.export_frame0_fisher_artifacts_if_needed(idx)

    def export_frame0_fisher_artifacts_if_needed(self, idx: int):
        if idx != 0 or self.fisher_frame0_exported:
            return
        if self._last_gs_points is None or self._last_fisher_points is None:
            return

        out_dir = os.path.join(self.save_dir, "nbv_vis")
        os.makedirs(out_dir, exist_ok=True)

        merged_points = np.vstack([self._last_gs_points, self._last_fisher_points])
        merged_colors = np.vstack([self._last_gs_colors, self._last_fisher_colors])

        merged_pc = o3d.geometry.PointCloud()
        merged_pc.points = o3d.utility.Vector3dVector(merged_points)
        merged_pc.colors = o3d.utility.Vector3dVector(merged_colors)

        ply_path = os.path.join(out_dir, "frame0_gs_plus_fisher_hemi.ply")
        png_path = os.path.join(out_dir, "frame0_gs_plus_fisher_hemi.png")

        o3d.io.write_point_cloud(ply_path, merged_pc)
        if self.vis_gui and hasattr(self, "o3d_window") and self.o3d_window is not None:
            self.o3d_window.capture_screen_image(png_path, do_render=True)
            save_msg = f"Saved frame-0 Fisher artifacts: {ply_path}, {png_path}"
        else:
            save_msg = f"Saved frame-0 Fisher artifacts (without screenshot): {ply_path}"

        self.fisher_frame0_exported = True
        Log(
            save_msg,
            tag="NextBestView",
        )
