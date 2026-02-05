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
    
class NextBestViewSelector:
    """
    使用 Fisher Information 选择最优视角
    """
    def __init__(self, 
                 fisher_computer: FisherInformationComputer,
                 pc: GaussianModel):
        self.fisher_computer = fisher_computer
        self.pc = pc
        
    def select_nbv(self,
                   current_camera: Camera,
                   num_candidates: int = 50,
                   num_optimize: int = 5,
                   search_radius: float = 2.0) -> Tuple[Camera, float]:
        """
        选择下一个最佳视角
        
        Args:
            current_camera: 当前 Camera 对象
            num_candidates: 采样候选点数量
            num_optimize: 精细优化的候选数
            search_radius: 搜索半径（米）
            
        Returns:
            best_camera: 最优 Camera 对象
            best_fisher: 对应的 Fisher Information
        """
        device = "cuda"
        
        # 1. 生成候选视角（Camera 对象列表）
        candidates = self._sample_candidates(
            current_camera, num_candidates, search_radius
        )
        
        # 2. 粗筛选
        rough_scores = []
        print(f"Evaluating {num_candidates} candidates...")
        
        for i, cam in enumerate(candidates):
            with torch.no_grad():
                fisher = self.fisher_computer.compute_fisher_trace(
                    self.pc, cam, sparse_mode=True
                )
            rough_scores.append(fisher.item())
            
            if (i+1) % 10 == 0:
                print(f"  {i+1}/{num_candidates} done")
        
        # 3. 选择 top-K
        top_k_idx = np.argsort(rough_scores)[-num_optimize:]
        print(f"\nTop-{num_optimize} candidates: {[rough_scores[i] for i in top_k_idx]}")
        
        # 4. 精细优化
        best_camera = None
        best_fisher = -float('inf')
        
        for idx in top_k_idx:
            cam_init = candidates[idx]
            cam_opt, fisher_opt = self._optimize_camera(cam_init, max_iter=20)
            
            if fisher_opt > best_fisher:
                best_fisher = fisher_opt
                best_camera = cam_opt
        
        print(f"\nBest Fisher Information: {best_fisher:.4f}")
        return best_camera, best_fisher
    

    def _sample_candidates(self,
                          center_camera: Camera,
                          num_samples: int,
                          radius: float) -> list:
        """
        在当前相机周围采样候选视角（返回 Camera 对象）
        """
        device = "cuda"
        candidates = []
        
        center_pos = center_camera.camera_center  # [3]
        
        for _ in range(num_samples):
            # 随机位置
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            r = np.random.uniform(0.5 * radius, radius)
            
            offset = torch.tensor([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ], device=device)
            
            new_pos = center_pos + offset
            target = center_pos + torch.randn(3, device=device) * 0.5
            
            # 构建位姿矩阵
            c2w = self._look_at(new_pos, target)  # camera-to-world
            w2c = torch.inverse(c2w)  # world-to-camera
            
            # 创建新的 Camera 对象
            new_camera = Camera.init_from_gui(
                uid=0,
                T=w2c,
                FoVx=center_camera.FoVx,
                FoVy=center_camera.FoVy,
                fx=center_camera.fx,
                fy=center_camera.fy,
                cx=center_camera.cx,
                cy=center_camera.cy,
                H=center_camera.image_height,
                W=center_camera.image_width
            )
            new_camera.set_GSRasterization()
            
            candidates.append(new_camera)
        
        return candidates
    
    def _look_at(self, eye: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        构建 look-at 变换矩阵（world-to-camera）
        """
        device = eye.device
        
        # Z 轴（相机朝向）
        z = target - eye
        z = z / (torch.norm(z) + 1e-7)
        
        # X 轴（右方向）
        up = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
        x = torch.cross(up, z)
        x = x / (torch.norm(x) + 1e-7)
        
        # Y 轴（向上）
        y = torch.cross(z, x)
        
        # 构建变换矩阵
        pose = torch.eye(4, device=device)
        pose[:3, 0] = x
        pose[:3, 1] = y
        pose[:3, 2] = z
        pose[:3, 3] = eye
        
        # world-to-camera（需要求逆）
        return torch.inverse(pose)
    
    def _optimize_pose(self,
                      camera_init: Camera,
                      max_iter: int = 30,
                      lr: float = 0.01) -> Tuple[torch.Tensor, float]:
        """
        在 SE(3) 上优化位姿以最大化 Fisher Information
        """
        xi = torch.zeros(6, device="cuda", requires_grad=True)
        optimizer = torch.optim.Adam([xi], lr=lr)
        
        best_fisher = -float('inf')
        best_camera = camera_init
        
        for iteration in range(max_iter):
            optimizer.zero_grad()
            
            # 从李代数恢复位姿
            delta_T = self._exp_se3(xi)
            new_w2c = delta_T @ camera_init.world_view_transform
            
            # 创建新 Camera（避免修改原对象）
            new_camera = Camera.init_from_gui(
                uid=0,
                T=new_w2c,
                FoVx=camera_init.FoVx,
                FoVy=camera_init.FoVy,
                fx=camera_init.fx,
                fy=camera_init.fy,
                cx=camera_init.cx,
                cy=camera_init.cy,
                H=camera_init.image_height,
                W=camera_init.image_width
            )
            new_camera.set_GSRasterization()
            
            # 计算 Fisher Information
            fisher = self.fisher_computer.compute_fisher_trace(
                self.pc, new_camera, sparse_mode=True
            )
            
            # 梯度上升
            loss = -fisher
            loss.backward()
            optimizer.step()
            
            if fisher.item() > best_fisher:
                best_fisher = fisher.item()
                best_camera = new_camera
        
        return best_camera, best_fisher
    
    def _exp_se3(self, xi: torch.Tensor) -> torch.Tensor:
        """
        指数映射：se(3) -> SE(3)
        
        xi = [ρ, φ] ∈ R^6
        其中 ρ ∈ R^3 是平移部分，φ ∈ R^3 是旋转部分
        """
        device = xi.device
        rho = xi[:3]  # 平移
        phi = xi[3:]  # 旋转（轴角）
        
        # 旋转部分（Rodrigues 公式）
        theta = torch.norm(phi)
        
        if theta < 1e-6:
            R = torch.eye(3, device=device)
        else:
            k = phi / theta
            K = self._skew_symmetric(k)
            R = torch.eye(3, device=device) + torch.sin(theta) * K + \
                (1 - torch.cos(theta)) * (K @ K)
        
        # 平移部分（带旋转修正）
        if theta < 1e-6:
            V = torch.eye(3, device=device)
        else:
            k = phi / theta
            K = self._skew_symmetric(k)
            V = torch.eye(3, device=device) + \
                (1 - torch.cos(theta)) / theta * K + \
                (theta - torch.sin(theta)) / theta * (K @ K)
        
        t = V @ rho
        
        # 组装 SE(3)
        T = torch.eye(4, device=device)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T
    
    def _skew_symmetric(self, v: torch.Tensor) -> torch.Tensor:
        """
        向量 v 的反对称矩阵 [v]_×
        """
        device = v.device
        return torch.tensor([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ], device=device)


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