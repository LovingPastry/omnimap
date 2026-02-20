import torch
import torch.nn as nn
from typing import Tuple, Dict
from gaussian.renderer import render
from gaussian.scene.gaussian_model import GaussianModel
from gaussian.utils.camera_utils import Camera


class FisherGradientComputer:
    """
    计算 Fisher Information 对视角的梯度（用于梯度上升优化）
    """
    def __init__(self, pc: GaussianModel):
        """
        Args:
            pc: 高斯模型（包含场景几何）
        """
        self.pc = pc
        
    def compute_fisher_and_gradient(self, 
                                    camera: Camera,
                                    bg_color: torch.Tensor,
                                    sparse_mode: bool = True
                                    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 Fisher Information 和对相机位姿的梯度
        
        Args:
            camera: Camera 对象（包含可优化的位姿参数）
            bg_color: 背景颜色 [3]
            sparse_mode: 是否只计算可见高斯
            
        Returns:
            fisher: 标量 Fisher Information
            grad_pose: 对位姿参数的梯度 dict{'R': [3, 3], 'T': [3]}
        """
        # 确保相机位姿参数可求导
        if not camera.R.requires_grad:
            camera.R.requires_grad = True
        if not camera.T.requires_grad:
            camera.T.requires_grad = True
        
        # 1. 渲染图像
        render_result = render(camera, self.pc, bg_color)
        
        image = render_result['render']  # [3, H, W]
        depth = render_result['depth']   # [1, H, W]
        normal = render_result['normal'] # [3, H, W]
        
        # 2. 计算像素权重
        weights = self._compute_pixel_weights(camera, depth, normal)  # [H, W]
        
        # 3. 计算 Fisher Information
        if sparse_mode:
            fisher = self._compute_fisher_sparse(
                image, weights, render_result['visibility_filter']
            )
        else:
            fisher = self._compute_fisher_dense(image, weights)
        
        # 4. 对相机位姿求导
        if camera.R.grad is not None:
            camera.R.grad.zero_()
        if camera.T.grad is not None:
            camera.T.grad.zero_()
        
        grads = torch.autograd.grad(
            outputs=fisher,
            inputs=[camera.R, camera.T],
            create_graph=False,
            retain_graph=False
        )
        
        grad_R = grads[0].detach() if grads[0] is not None else torch.zeros_like(camera.R)
        grad_T = grads[1].detach() if grads[1] is not None else torch.zeros_like(camera.T)
        
        return fisher.detach(), {'R': grad_R, 'T': grad_T}
    
    def _compute_fisher_sparse(self,
                              image: torch.Tensor,
                              weights: torch.Tensor,
                              visible_mask: torch.Tensor) -> torch.Tensor:
        """
        稀疏计算：只计算可见高斯的贡献
        
        Args:
            image: [3, H, W] 渲染图像
            weights: [H, W] 像素权重
            visible_mask: [N] 可见高斯的布尔掩码（来自 radii > 0）
        """
        H, W = image.shape[1], image.shape[2]
        image_flat = image.reshape(3, -1)  # [3, HW]
        weights_flat = weights.flatten()    # [HW]
        
        # 只对可见高斯的参数计算梯度
        xyz_visible = self.pc._xyz[visible_mask]
        scaling_visible = self.pc._scaling[visible_mask]
        rotation_visible = self.pc._rotation[visible_mask]
        opacity_visible = self.pc._opacity[visible_mask]
        
        # 组合参数（需要梯度）
        params_list = [
            xyz_visible.flatten(),
            scaling_visible.flatten(),
            rotation_visible.flatten(),
            opacity_visible.flatten()
        ]
        all_params_visible = torch.cat(params_list).requires_grad_(True)
        
        # 计算加权梯度范数之和
        fisher_sum = 0.0
        
        # 对每个像素计算雅可比
        for pixel_idx in range(image_flat.shape[1]):
            # 计算 ∂I_p/∂θ (对所有参数)
            pixel_grads = torch.autograd.grad(
                outputs=image_flat[:, pixel_idx].sum(),
                inputs=all_params_visible,
                retain_graph=True,
                create_graph=True,
                allow_unused=True
            )[0]
            
            if pixel_grads is not None:
                # 加权梯度范数: w_p · ||∂I_p/∂θ||²
                weighted_norm = weights_flat[pixel_idx] * torch.sum(pixel_grads ** 2)
                fisher_sum = fisher_sum + weighted_norm
        
        return fisher_sum
    
    def _compute_fisher_dense(self,
                             image: torch.Tensor,
                             weights: torch.Tensor) -> torch.Tensor:
        """
        密集计算：所有高斯（用于验证）
        """
        image_flat = image.reshape(3, -1)  # [3, HW]
        weights_flat = weights.flatten()    # [HW]
        
        # 使用现有的 get_all_parameters 方法
        all_params = self.pc.get_all_parameters().requires_grad_(True)
        fisher_sum = 0.0
        # 计算雅可比矩阵的加权 Frobenius 范数
        for pixel_idx in range(image_flat.shape[1]):
            pixel_grads = torch.autograd.grad(
                outputs=image_flat[:, pixel_idx].sum(),
                inputs=all_params,
                retain_graph=True,
                create_graph=True,
                allow_unused=True
            )[0]
            
            if pixel_grads is not None:
                weighted_norm = weights_flat[pixel_idx] * torch.sum(pixel_grads ** 2)
                fisher_sum = fisher_sum + weighted_norm
        return fisher_sum
    
    def compute_fisher_gradient_implicit(self, 
                                        camera: Camera,
                                        bg_color: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        不显式计算 Fisher 矩阵，直接计算梯度
        
        核心技巧：
        1. 让渲染过程对 camera.R, camera.T 可微
        2. 计算 ∂(Σ w_p · ||∂I_p/∂θ||²)/∂d
        3. PyTorch 自动追踪嵌套梯度
        """
        # 确保位姿可导
        camera.R.requires_grad_(True)
        camera.T.requires_grad_(True)
        
        # 1. 渲染（保留计算图）
        from gaussian.renderer import render
        render_result = render(camera, self.pc, bg_color)
        
        image = render_result['render']  # [3, H, W]
        depth = render_result['depth']
        normal = render_result['normal']
        
        # 2. 计算权重
        weights = self._compute_pixel_weights(camera, depth, normal)  # [H, W]
        weights_flat = weights.flatten()
        
        # 3. 对每个像素计算 ||∂I_p/∂θ||²（这里 θ 是高斯参数）
        image_flat = image.reshape(3, -1)  # [3, HW]
        
        # 获取高斯参数（需要梯度）
        theta = self.pc.get_all_parameters()
        theta.requires_grad_(True)
        
        fisher_sum = torch.tensor(0.0, device="cuda", requires_grad=True)
        
        for pixel_idx in range(image_flat.shape[1]):
            # 计算 ∂I_p/∂θ
            grad_theta = torch.autograd.grad(
                outputs=image_flat[:, pixel_idx].sum(),
                inputs=theta,
                retain_graph=True,
                create_graph=True  # ⚠️ 关键：保留计算图以便二阶导数
            )[0]
            
            if grad_theta is not None:
                # ||∂I_p/∂θ||²
                grad_norm_sq = torch.sum(grad_theta ** 2)
                
                # 加权累加（这个计算图连接到 camera.R 和 camera.T）
                fisher_sum = fisher_sum + weights_flat[pixel_idx] * grad_norm_sq
        
        # 4. 对相机位姿求导（PyTorch 自动处理嵌套梯度）
        grad_R, grad_T = torch.autograd.grad(
            outputs=fisher_sum,
            inputs=[camera.R, camera.T],
            create_graph=False
        )
        
        return {
            'R': grad_R.detach(),
            'T': grad_T.detach(),
            'fisher': fisher_sum.detach()
        }
    
    
    
    def _compute_pixel_weights(self,
                              camera: Camera,
                              depth: torch.Tensor,
                              normal: torch.Tensor) -> torch.Tensor:
        """
        计算像素权重 w = (1/σ²) · cos(θ) · edge_weight
        """
        H, W = depth.shape[1], depth.shape[2]
        device = depth.device
        
        # 1. 基础权重（测量不确定性）
        sigma_rgb = 0.01
        base_weight = 1.0 / (sigma_rgb ** 2)
        
        # 2. 入射角权重
        rays = self._compute_ray_directions(camera, H, W)  # [H, W, 3]
        normal_perm = normal.permute(1, 2, 0)  # [H, W, 3]
        
        cos_angle = torch.abs(torch.sum(rays * normal_perm, dim=-1))
        cos_angle = torch.clamp(cos_angle, 0.1, 1.0)
        
        # 3. 深度梯度权重（边缘检测）
        depth = depth.squeeze(0)  # [H, W]
        depth_grad_x = torch.abs(depth[:, 1:] - depth[:, :-1])
        depth_grad_y = torch.abs(depth[1:, :] - depth[:-1, :])
        
        # Padding
        depth_grad_x = torch.cat([depth_grad_x, depth_grad_x[:, -1:]], dim=1)
        depth_grad_y = torch.cat([depth_grad_y, depth_grad_y[-1:, :]], dim=0)
        
        depth_grad = torch.sqrt(depth_grad_x**2 + depth_grad_y**2)
        tau = 0.1
        edge_weight = torch.exp(-depth_grad**2 / (2 * tau**2))
        
        # 4. 组合权重
        weights = base_weight * cos_angle * edge_weight
        
        return weights
    
    def _compute_ray_directions(self, 
                               camera: Camera,
                               H: int, W: int) -> torch.Tensor:
        """
        计算每个像素的视线方向（相机坐标系）
        """
        device = "cuda"
        fx, fy = camera.fx, camera.fy
        cx, cy = camera.cx, camera.cy
        
        i, j = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        
        # 像素坐标转归一化坐标
        x = (j - cx) / fx
        y = (i - cy) / fy
        
        # 视线方向 [x, y, 1] 归一化
        rays = torch.stack([x, y, torch.ones_like(x)], dim=-1)
        rays = rays / torch.norm(rays, dim=-1, keepdim=True)
        
        return rays


class NextBestViewOptimizer:
    """
    使用梯度上升法优化视角以最大化 Fisher Information
    """
    def __init__(self, pc: GaussianModel):
        """
        Args:
            pc: 高斯模型
        """
        self.gradient_computer = FisherGradientComputer(pc)
        self.pc = pc
        
    def optimize_view(self,
                     camera_init: Camera,
                     bg_color: torch.Tensor = torch.zeros(3, device="cuda"),
                     max_iter: int = 50,
                     lr_rotation: float = 0.01,
                     lr_translation: float = 0.05,
                     verbose: bool = True) -> Tuple[Camera, float]:
        """
        从初始视角开始，使用梯度上升优化到最大 Fisher Information
        
        Args:
            camera_init: 初始相机位姿
            bg_color: 背景颜色
            max_iter: 最大迭代次数
            lr_rotation: 旋转梯度的学习率
            lr_translation: 平移梯度的学习率
            verbose: 是否打印优化过程
            
        Returns:
            best_camera: 最优相机
            best_fisher: 最优 Fisher 值
        """
        # 创建可优化的相机（深拷贝）
        camera = Camera.init_from_gui(
            uid=camera_init.uid,
            T=camera_init.world_view_transform.T.inverse(),
            FoVx=camera_init.FoVx,
            FoVy=camera_init.FoVy,
            fx=camera_init.fx,
            fy=camera_init.fy,
            cx=camera_init.cx,
            cy=camera_init.cy,
            H=camera_init.image_height,
            W=camera_init.image_width
        )
        camera.set_GSRasterization()
        
        # 使用李代数参数化位姿（避免直接优化旋转矩阵）
        # se(3) 参数：[ω_x, ω_y, ω_z, v_x, v_y, v_z]
        xi = nn.Parameter(torch.zeros(6, device="cuda"))
        optimizer = torch.optim.Adam([xi], lr=1.0)
        
        best_fisher = -float('inf')
        best_xi = xi.clone()
        
        if verbose:
            print(f"Starting NBV optimization from initial camera pose...")
            print(f"  Rotation LR: {lr_rotation}, Translation LR: {lr_translation}")
        
        for iteration in range(max_iter):
            optimizer.zero_grad()
            
            # 1. 从李代数恢复位姿
            delta_T = self._exp_se3(xi)  # [4, 4]
            new_w2c = delta_T @ camera_init.world_view_transform
            
            # 2. 更新相机位姿
            camera.R = new_w2c[:3, :3].clone()
            camera.T = new_w2c[:3, 3].clone()
            camera.R.requires_grad = True
            camera.T.requires_grad = True
            camera.set_GSRasterization()  # 更新光栅化器
            
            # 3. 计算 Fisher Information 和梯度
            fisher, grad_dict = self.gradient_computer.compute_fisher_and_gradient(
                camera, bg_color, sparse_mode=True
            )
            
            # 4. 将梯度投影回李代数空间
            grad_R = grad_dict['R']  # [3, 3]
            grad_T = grad_dict['T']  # [3]
            
            # 提取反对称部分（旋转梯度）
            grad_omega = self._vee(grad_R - grad_R.T) / 2  # [3]
            grad_v = grad_T  # [3]
            
            grad_xi = torch.cat([grad_omega, grad_v])  # [6]
            
            # 5. 手动梯度上升（最大化 Fisher）
            with torch.no_grad():
                xi[:3] += lr_rotation * grad_xi[:3]      # 旋转部分
                xi[3:] += lr_translation * grad_xi[3:]   # 平移部分
            
            # 6. 记录最优值
            if fisher.item() > best_fisher:
                best_fisher = fisher.item()
                best_xi = xi.clone()
            
            # 7. 打印进度
            if verbose and (iteration + 1) % 10 == 0:
                print(f"  Iter {iteration+1}/{max_iter}: Fisher = {fisher.item():.4f}, "
                      f"Best = {best_fisher:.4f}")
        
        # 8. 恢复最优位姿
        delta_T_best = self._exp_se3(best_xi)
        final_w2c = delta_T_best @ camera_init.world_view_transform
        
        camera_best = Camera.init_from_gui(
            uid=camera_init.uid,
            T=final_w2c.T.inverse(),
            FoVx=camera_init.FoVx,
            FoVy=camera_init.FoVy,
            fx=camera_init.fx,
            fy=camera_init.fy,
            cx=camera_init.cx,
            cy=camera_init.cy,
            H=camera_init.image_height,
            W=camera_init.image_width
        )
        camera_best.set_GSRasterization()
        
        if verbose:
            print(f"\nOptimization finished!")
            print(f"  Initial Fisher: {fisher.item():.4f}")
            print(f"  Final Fisher:   {best_fisher:.4f}")
            print(f"  Improvement:    {best_fisher - fisher.item():.4f}")
        
        return camera_best, best_fisher
    
    def _exp_se3(self, xi: torch.Tensor) -> torch.Tensor:
        """
        李代数指数映射：se(3) -> SE(3)
        
        Args:
            xi: [6] 李代数参数 [ω, v]
        
        Returns:
            T: [4, 4] SE(3) 变换矩阵
        """
        device = xi.device
        omega = xi[:3]  # 旋转部分 so(3)
        v = xi[3:]      # 平移部分
        
        theta = torch.norm(omega)
        
        # 旋转部分
        if theta < 1e-6:
            R = torch.eye(3, device=device)
        else:
            k = omega / theta  # 单位方向
            K = self._skew(k)
            R = torch.eye(3, device=device) + \
                torch.sin(theta) * K + \
                (1 - torch.cos(theta)) * (K @ K)
        
        # 平移部分
        if theta < 1e-6:
            V = torch.eye(3, device=device)
        else:
            k = omega / theta
            K = self._skew(k)
            V = torch.eye(3, device=device) + \
                (1 - torch.cos(theta)) / theta * K + \
                (theta - torch.sin(theta)) / theta * (K @ K)
        
        t = V @ v
        
        # 构建 SE(3) 矩阵
        T = torch.eye(4, device=device)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T
    
    def _skew(self, v: torch.Tensor) -> torch.Tensor:
        """
        向量到反对称矩阵：R³ -> so(3)
        """
        device = v.device
        return torch.tensor([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ], device=device, dtype=v.dtype)
    
    def _vee(self, S: torch.Tensor) -> torch.Tensor:
        """
        反对称矩阵到向量：so(3) -> R³
        """
        return torch.stack([S[2, 1], S[0, 2], S[1, 0]])


# ============ 使用示例 ============
if __name__ == "__main__":
    # 初始化
    pc = GaussianModel(sh_degree=0)
    optimizer = NextBestViewOptimizer(pc)
    
    # 当前相机
    current_camera = Camera.init_from_tracking(...)
    
    # 梯度上升优化
    best_camera, best_fisher = optimizer.optimize_view(
        camera_init=current_camera,
        max_iter=50,
        lr_rotation=0.01,
        lr_translation=0.05,
        verbose=True
    )
    
    print(f"Optimized camera position: {best_camera.camera_center}")
    print(f"Fisher Information: {best_fisher:.4f}")