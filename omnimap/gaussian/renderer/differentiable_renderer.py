import torch
import torch.nn as nn
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from typing import Tuple, Dict, Optional
import math
from gaussian.scene.gaussian_model import GaussianModel
from gaussian.utils.camera_utils import Camera

class DifferentiableRenderer:
    """
    封装 diff-gaussian-rasterization 并提供梯度追踪
    """
    def __init__():
        pass
        
    def render(self, 
               viewpoint_camera: Camera,
               pc: GaussianModel, 
               bg_color: torch.Tensor,
               scaling_modifier: float = 1.0,
               render_mode: str = "rgb") -> Dict[str, torch.Tensor]:
        """
        渲染场景（完全可微分），接口与 gaussian.renderer.render() 一致
        
        Args:
            viewpoint_camera: Camera 对象（包含内外参和光栅化器）
            pc: GaussianModel 高斯模型
            bg_color: 背景颜色 [3]
            scaling_modifier: 尺度修正系数
            render_mode: "rgb" 或 "instance"
            
        Returns:
            dict with keys: 'render', 'depth', 'viewspace_points', 'visibility_filter', 'radii', 'n_touched'
        """
        # 1. 准备屏幕空间点（用于梯度追踪和 densification）
        screenspace_points = torch.zeros_like(
            pc.get_xyz, 
            dtype=pc.get_xyz.dtype, 
            requires_grad=True, 
            device="cuda"
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        # 2. 获取光栅化器（复用 Camera 对象的配置）
        rasterizer = viewpoint_camera.rasterizer
        
        # 3. 准备输入数据
        means3D = pc.get_xyz
        means2D = screenspace_points
        scales = pc.get_scaling
        opacity = pc.get_opacity
        rotations = pc.get_rotation
        cov3D_precomp = None
        colors_precomp = None
        
        # 4. 根据渲染模式选择特征
        if render_mode == "instance":
            shs = pc.get_instance_features  # 实例分割颜色
        else:
            shs = pc.get_features  # 正常RGB颜色
        
        # 5. 调用光栅化器
        rendered_image, radii, rendered_expected_depth, n_touched = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )
        
        # 6. 返回与原函数相同的格式
        return {
            "render": rendered_image,
            "depth": rendered_expected_depth,
            "viewspace_points": means2D,
            "visibility_filter": radii > 0,
            "radii": radii,
            "n_touched": n_touched
        }

    def render_with_normals(self, 
                           viewpoint_camera: Camera,
                           pc: GaussianModel,
                           bg_color: torch.Tensor = torch.tensor([0.0, 0.0, 0.0], device="cuda"),
                           scaling_modifier: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        扩展功能：同时渲染RGB、深度和法向
        
        Args:
            viewpoint_camera: Camera 对象
            pc: GaussianModel
            bg_color: 背景颜色
            scaling_modifier: 尺度修正
            
        Returns:
            包含 'render', 'depth', 'normal' 等键的字典
        """
        # 基础渲染
        result = self.render(viewpoint_camera, pc, bg_color, scaling_modifier)
        
        # 额外渲染法向
        normal_image = self._render_normals(viewpoint_camera, pc)
        result["normal"] = normal_image
        
        return result
    
    def render_distance(self, 
                        viewpoint_camera: Camera,
                        pc: GaussianModel,
                        bg_color: torch.Tensor = torch.tensor([0.0, 0.0, 0.0], device="cuda"),
                        scaling_modifier: float = 1.0) -> Dict[str, torch.Tensor]:
        rendered_ins = self.render(viewpoint_camera, pc, bg_color, scaling_modifier, render_mode="instance")
        return {"render": rendered_ins["render"]}
    

    def _render_normals(self, viewpoint_camera: Camera, pc: GaussianModel) -> torch.Tensor:
        """
        渲染法向图
        """
        colors_precomp = self._compute_normals(pc,viewpoint_camera.world_view_transform)
        
        rasterizer = viewpoint_camera.rasterizer
        normal_image, _, _, _ = rasterizer(
            means3D=pc.get_xyz,
            means2D=torch.zeros_like(pc.get_xyz[:, :2]),
            shs=None,
            colors_precomp=colors_precomp,
            opacities=pc.get_opacity,
            scales=pc.get_scaling,
            rotations=pc.get_rotation,
            cov3D_precomp=None,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )
        return normal_image
    
    def _compute_normals(self, gaussians: GaussianModel, 
                        camera_pose: torch.Tensor) -> torch.Tensor:
        """
        从协方差矩阵的主方向估计法向量
        """
        cov = gaussians.get_covariance_matrices()  # [N, 3, 3]
        
        # 特征值分解（主方向 = 最小特征值对应的特征向量）
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        normals_world = eigenvectors[:, :, 0]  # [N, 3] 最小特征向量
        
        # 转换到相机坐标系
        R_cam = camera_pose[:3, :3]
        normals_cam = (R_cam @ normals_world.T).T
        
        # 归一化并映射到 [0, 1]
        normals_cam = normals_cam / (torch.norm(normals_cam, dim=1, keepdim=True) + 1e-7)
        normals_rgb = (normals_cam + 1) / 2  # [-1, 1] -> [0, 1]
        
        return normals_rgb
    
    
    def _get_projection_matrix(self, viewmat: torch.Tensor) -> torch.Tensor:
        """
        构建投影矩阵（OpenGL 风格）
        """
        near, far = 0.01, 100.0
        P = torch.zeros(4, 4, device=viewmat.device)
        P[0, 0] = 2 * self.fx / self.W
        P[1, 1] = 2 * self.fy / self.H
        P[0, 2] = 2 * (self.cx / self.W) - 1
        P[1, 2] = 2 * (self.cy / self.H) - 1
        P[2, 2] = -(far + near) / (far - near)
        P[2, 3] = -2 * far * near / (far - near)
        P[3, 2] = -1
        
        return (viewmat @ P).T  # 转置以匹配光栅化器的约定
    
    def _compute_depth_colors(self, means3D: torch.Tensor, 
                             camera_pose: torch.Tensor) -> torch.Tensor:
        """
        计算每个高斯的深度（相机坐标系 Z 轴）
        """
        # 转换到相机坐标系
        ones = torch.ones(means3D.shape[0], 1, device=means3D.device)
        means_homo = torch.cat([means3D, ones], dim=1)  # [N, 4]
        means_cam = (camera_pose @ means_homo.T).T  # [N, 4]
        
        depth = means_cam[:, 2:3]  # [N, 1]
        
        # 归一化到 [0, 1]（假设深度在 [0, 10] 米）
        depth_norm = torch.clamp(depth / 10.0, 0, 1)
        
        # 编码为 RGB（灰度）
        return depth_norm.repeat(1, 3)
    

    

