import torch
import torch.nn as nn
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from typing import Tuple, Dict, Optional
import math
from gaussian.scene.gaussian_model import GaussianModel
from gaussian.utils.camera_utils import Camera
from gaussian.renderer.differentiable_renderer import DifferentiableRenderer

class FisherInformationComputer:
    """
    计算基于 3DGS 的 Fisher Information
    """
    def __init__(self, renderer: DifferentiableRenderer):
        self.renderer = renderer if renderer is not None else DifferentiableRenderer()
        
    def compute_fisher_trace(self, 
                            pc: GaussianModel,
                            viewpoint_camera: Camera,
                            use_weights: bool = True,
                            sparse_mode: bool = True) -> torch.Tensor:
        """
        计算 tr(J^T W J) - Fisher Information 的迹
        
        Args:
            pc: 高斯地图
            viewpoint_camera: Camera 对象（包含位姿）
            use_weights: 是否使用像素权重 W
            sparse_mode: 是否使用稀疏优化（仅计算可见高斯）
            
        Returns:
            fisher_trace: 标量 Fisher Information
        """
        # 1. 渲染图像（保留计算图）
        bg_color = torch.zeros(3, device="cuda")
        render_result = self.renderer.render_with_normals(
            viewpoint_camera, pc, bg_color
        )
        
        image = render_result['render']  # [3, H, W]
        depth = render_result['depth']    # [1, H, W]
        normal = render_result['normal']  # [3, H, W]
        
        # 2. 计算像素权重
        if use_weights:
            weights = self.compute_pixel_weights(
                viewpoint_camera, depth, normal, pc
            )  # [H, W]
        else:
            H, W = image.shape[1], image.shape[2]
            weights = torch.ones(H, W, device="cuda")
        
        # 3. 展平
        image_flat = image.reshape(3, -1)  # [3, HW]
        weights_flat = weights.flatten()   # [HW]
        
        # 4. 计算 Fisher Information
        if sparse_mode:
            fisher = self._compute_fisher_sparse(
                image_flat, weights_flat, pc, viewpoint_camera
            )
        else:
            fisher = self._compute_fisher_dense(
                image_flat, weights_flat, pc
            )
        
        return fisher
    

    def _compute_fisher_sparse(self,
                              image_flat: torch.Tensor,
                              weights: torch.Tensor,
                              pc: GaussianModel,
                              viewpoint_camera: Camera) -> torch.Tensor:
        """
        稀疏计算：只计算可见高斯的贡献
        """
        fisher_sum = 0.0
        
        # 获取可见高斯 ID
        visible_mask = self._get_visible_gaussians(pc, viewpoint_camera)
        visible_indices = torch.where(visible_mask)[0]
        
        # 只对可见高斯计算梯度
        param_groups = [
            ('xyz', pc._xyz[visible_mask]),
            ('scaling', pc._scaling[visible_mask]),
            ('rotation', pc._rotation[visible_mask]),
            ('opacity', pc._opacity[visible_mask])
        ]
        
        for name, param in param_groups:
            if not param.requires_grad:
                continue
                
            # 对每个参数分量计算梯度范数
            for i in range(param.numel()):
                # 计算 ∂I/∂θ_i
                grad_i = torch.autograd.grad(
                    outputs=image_flat.sum(),
                    inputs=param.flatten(),
                    retain_graph=True,
                    create_graph=False
                )[0]
                
                # 加权梯度范数
                grad_i_reshaped = grad_i.reshape(3, -1)
                weighted_grad = grad_i_reshaped * weights.unsqueeze(0).sqrt()
                fisher_sum += torch.sum(weighted_grad ** 2)
        
        return fisher_sum
    
    def _compute_fisher_dense(self,
                             image_flat: torch.Tensor,
                             weights: torch.Tensor,
                             pc: GaussianModel) -> torch.Tensor:
        """
        密集计算：构造完整雅可比矩阵（仅用于验证）
        """
        all_params = pc.get_all_parameters()  # [14N]
        
        jacobian_list = []
        for i in range(all_params.numel()):
            grad_i = torch.autograd.grad(
                outputs=image_flat,
                inputs=all_params,
                grad_outputs=torch.eye(image_flat.numel(), device=all_params.device)[i],
                retain_graph=True,
                create_graph=False
            )[0]
            jacobian_list.append(grad_i)
        
        J = torch.stack(jacobian_list, dim=1)  # [3*HW, 14N]
        W_diag = weights.repeat(3)
        J_weighted = J * W_diag.unsqueeze(1).sqrt()
        fisher_matrix = J_weighted.T @ J_weighted
        
        return torch.trace(fisher_matrix)
    def _get_visible_gaussians(self,
                               pc: GaussianModel,
                               viewpoint_camera: Camera) -> torch.Tensor:
        """
        返回在当前视角下可见的高斯球掩码
        """
        # 转换到相机坐标系
        means_world = pc.get_xyz  # [N, 3]
        ones = torch.ones(means_world.shape[0], 1, device="cuda")
        means_homo = torch.cat([means_world, ones], dim=1)  # [N, 4]
        
        w2c = viewpoint_camera.world_view_transform
        means_cam = (w2c @ means_homo.T).T  # [N, 4]
        
        # 深度检查
        valid_depth = (means_cam[:, 2] > 0.1) & (means_cam[:, 2] < 10.0)
        
        # 视野范围检查
        fx = viewpoint_camera.fx
        fy = viewpoint_camera.fy
        W = viewpoint_camera.image_width
        H = viewpoint_camera.image_height
        
        # 投影到像素坐标
        x_proj = means_cam[:, 0] / (means_cam[:, 2] + 1e-7) * fx + W / 2
        y_proj = means_cam[:, 1] / (means_cam[:, 2] + 1e-7) * fy + H / 2
        
        valid_x = (x_proj >= 0) & (x_proj < W)
        valid_y = (y_proj >= 0) & (y_proj < H)
        
        visible = valid_depth & valid_x & valid_y
        
        return visible
    
    def compute_pixel_weights(self,
                             viewpoint_camera: Camera,
                             depth: torch.Tensor,
                             normal: torch.Tensor,
                             pc: GaussianModel) -> torch.Tensor:
        """
        计算像素权重 w = (1/σ²) · cos(θ) · edge_weight
        """
        H, W = depth.shape[1], depth.shape[2]
        device = depth.device
        
        # 1. 基础权重
        sigma_rgb = 0.01
        base_weight = 1.0 / (sigma_rgb ** 2)
        
        # 2. 入射角权重
        rays = self._compute_ray_directions(viewpoint_camera, H, W)  # [H, W, 3]
        normal_perm = normal.permute(1, 2, 0)  # [H, W, 3]
        cos_angle = torch.abs(torch.sum(rays * normal_perm, dim=-1))
        cos_angle = torch.clamp(cos_angle, 0.1, 1.0)
        
        # 3. 深度梯度权重
        depth_grad_x = torch.abs(depth[:, :, 1:] - depth[:, :, :-1])
        depth_grad_y = torch.abs(depth[:, 1:, :] - depth[:, :-1, :])
        
        depth_grad_x = torch.cat([depth_grad_x, depth_grad_x[:, :, -1:]], dim=2)
        depth_grad_y = torch.cat([depth_grad_y, depth_grad_y[:, -1:, :]], dim=1)
        
        depth_grad = torch.sqrt(depth_grad_x[0]**2 + depth_grad_y[0]**2)
        tau = 0.1
        edge_weight = torch.exp(-depth_grad**2 / (2 * tau**2))
        
        # 4. 组合权重
        weights = base_weight * cos_angle * edge_weight
        
        return weights
    
    def _compute_ray_directions(self, 
                               viewpoint_camera: Camera,
                               H: int, W: int) -> torch.Tensor:
        """
        计算每个像素的视线方向（相机坐标系）
        """
        device = "cuda"
        fx, fy = viewpoint_camera.fx, viewpoint_camera.fy
        cx, cy = viewpoint_camera.cx, viewpoint_camera.cy
        
        i, j = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        
        x = (j - cx) / fx
        y = (i - cy) / fy
        
        rays = torch.stack([x, y, torch.ones_like(x)], dim=-1)
        rays = rays / torch.norm(rays, dim=-1, keepdim=True)
        
        return rays
    
    
class OptimizedFisherComputer(FisherInformationComputer):
    """
    使用稀疏性优化的版本
    """
    def compute_fisher_trace_fast(self,
                                  gaussians: GaussianModel,
                                  camera_pose: torch.Tensor) -> torch.Tensor:
        """
        只计算可见高斯球的贡献
        """
        # 1. 快速渲染获取可见高斯 ID
        with torch.no_grad():
            visible_ids = self._get_visible_gaussians(gaussians, camera_pose)
        
        # 2. 只对可见高斯计算梯度
        fisher_sum = 0.0
        
        for gid in visible_ids:
            # 提取该高斯的参数
            params = {
                'mean': gaussians.means[gid],
                'scale': gaussians.scales[gid],
                'quat': gaussians.quats[gid],
                'color': gaussians.colors[gid],
                'opacity': gaussians.opacities[gid]
            }
            
            # 逐参数计算贡献
            for key, param in params.items():
                for i in range(param.numel()):
                    # 计算 ∂I/∂θ_{gid,i}
                    grad_i = self._compute_single_gradient(
                        gaussians, camera_pose, gid, key, i
                    )
                    
                    fisher_sum += torch.sum(grad_i ** 2)
        
        return fisher_sum
    
    def _get_visible_gaussians(self,
                               gaussians: GaussianModel,
                               camera_pose: torch.Tensor) -> list:
        """
        返回在当前视角下可见的高斯球 ID
        """
        # 简化版：基于视锥体剔除
        means_cam = self._transform_to_camera(gaussians.means, camera_pose)
        
        # 深度检查
        valid_depth = (means_cam[:, 2] > 0.1) & (means_cam[:, 2] < 10.0)
        
        # 视野范围检查（简化）
        valid_x = torch.abs(means_cam[:, 0]) < means_cam[:, 2]
        valid_y = torch.abs(means_cam[:, 1]) < means_cam[:, 2]
        
        visible = valid_depth & valid_x & valid_y
        
        return torch.where(visible)[0].tolist()
    
    def _transform_to_camera(self,
                            points: torch.Tensor,
                            camera_pose: torch.Tensor) -> torch.Tensor:
        """
        将 3D 点从世界坐标系转换到相机坐标系
        """
        ones = torch.ones(points.shape[0], 1, device=points.device)
        points_homo = torch.cat([points, ones], dim=1)  # [N, 4]
        points_cam = (camera_pose @ points_homo.T).T  # [N, 4]
        return points_cam[:, :3]