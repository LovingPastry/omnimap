import torch
import torch.nn as nn
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from typing import Tuple, Dict, Optional
import math
from gaussian.scene.gaussian_model import GaussianModel
from gaussian.utils.camera_utils import Camera
from gaussian.renderer.fisher_information import FisherInformationComputer

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
            # 随机位置（Z-up 角度语义）
            # theta: 绕 z 轴方位角 [0, 2pi)
            # phi: 相对 XY 平面的仰角 [0, pi/2]（仅采样上半球）
            theta = np.random.uniform(0.0, 2.0 * np.pi)
            phi = np.random.uniform(0.0, 0.5 * np.pi)
            r = np.random.uniform(0.5 * radius, radius)
            
            offset = torch.tensor([
                r * np.cos(phi) * np.cos(theta),
                r * np.cos(phi) * np.sin(theta),
                r * np.sin(phi),
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