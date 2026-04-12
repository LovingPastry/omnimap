import os
import torch
import numpy as np
from lietorch import SE3
import cv2

from util.utils import load_config
from gs_backend import GSBackEnd
from tsdf_backend import TSDFBackEnd
import time
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
from torchvision import transforms
from visual_module import timeit


class OMNI:
    def __init__(self, args, config, device="cuda:0"):
        super(OMNI, self).__init__()
        # self.load_weights(args.weights)
        self.config = config
        self.args = args
        self.device = device
        self.images = {}
        self.depths = {}
        self.poses = {}
        self.intrinsics = None
        self.depth_scale = args.depth_scale
        self._last_track_pose_44 = None
        self._last_track_tstamp = None
        self._headless_boundary_snapshot_enabled = (
            (not bool(self.args.vis_gui))
            and (str(getattr(self.args, "output", "None")) != "None")
            and bool(self.config.get("headless_save_fisher_boundary_snapshots", True))
        )
        self._headless_first_snapshot_done = False

        self.omni_normal = None
        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
        # tsdf-fusion
        self.tsdf_fusion = TSDFBackEnd(config, self.args.output, self.args.vis_gui)
        
        # 3dgs
        self.gs = GSBackEnd(config, self.tsdf_fusion, self.args.output, self.args.vis_gui)
        
        self.last_time = time.time()
        self.iteration_count = 0
        self.hz = 0

    def _latest_viewpoint_for_snapshot(self):
        latest_viewpoint = getattr(self.gs, "viewpoint", None)
        if latest_viewpoint is None:
            keyviews = getattr(self.gs, "keyviewpoints", None)
            if keyviews:
                latest_viewpoint = keyviews[-1]
        return latest_viewpoint

    def _export_boundary_fisher_snapshot(self, *, tag: str, idx: int) -> bool:
        if not self._headless_boundary_snapshot_enabled:
            return False
        if self._last_track_pose_44 is None:
            return False
        export_fn = getattr(self.gs, "export_fisher_snapshot", None)
        if export_fn is None:
            return False
        latest_viewpoint = self._latest_viewpoint_for_snapshot()
        if latest_viewpoint is None:
            return False
        try:
            export_fn(
                viewpoint=latest_viewpoint,
                pose=np.asarray(self._last_track_pose_44, dtype=np.float64),
                idx=int(idx),
                tag=str(tag),
            )
            return True
        except Exception:
            return False

    
    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def prior_extractor(self, im_tensor):
        input_size = im_tensor.shape[-2:]
        trans_totensor = transforms.Compose([transforms.Resize((512, 512), antialias=True)])
        im_tensor = trans_totensor(im_tensor).cuda()
        normal = self.omni_normal(im_tensor) * 2.0 - 1.0
        normal = F.interpolate(normal, input_size, mode='bicubic')
        normal = normal.float().squeeze()
        return normal

    def sharpness(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # normalized_laplacian_var = laplacian_var / (gray.size)
        return laplacian_var

    def track(self, tstamp, image, depth, pose, progress_bar, intrinsics=None, is_last=False, pose_44=None, update_rate=1):
        """ main thread - update map """
        
        if self.intrinsics is None:
            self.intrinsics = intrinsics[0]

        self.tsdf_fusion.integrate(image[0].permute(1,2,0).clone(), depth[0].clone(), intrinsics[0].clone(), pose_44[0].clone(), tstamp)
        
        with torch.no_grad():
            self.images[tstamp] = image
            self.depths[tstamp] = depth
            self.poses[tstamp] = SE3(pose[0]).matrix().to("cuda")
        
        if self.config["Training"]["use_omni_normal"]:
            # normalize images
            inputs = image[None].to("cuda") / 255.0
            inputs = inputs.sub_(self.MEAN).div_(self.STDV)
            normal = self.prior_extractor(inputs[0])
        
        data = { 
                'tstamp':   tstamp,
                'poses':    pose[0],
                'images':   image[0],
                'depths':   depth[0],
                'intrinsics':   intrinsics[0],
                'normals': normal if self.config["Training"]["use_omni_normal"] else None
                }
        
        self.gs.process_track_data(data, self.hz)

        if pose_44 is not None:
            pose_44_np = (
                pose_44[0].detach().cpu().numpy()
                if isinstance(pose_44, torch.Tensor)
                else np.asarray(pose_44[0])
            )
            self._last_track_pose_44 = np.asarray(pose_44_np, dtype=np.float64)
            self._last_track_tstamp = int(tstamp)
            if (
                self._headless_boundary_snapshot_enabled
                and not self._headless_first_snapshot_done
            ):
                self._headless_first_snapshot_done = self._export_boundary_fisher_snapshot(
                    tag="first",
                    idx=self._last_track_tstamp,
                )
            
        if tstamp % update_rate == 0:
            loss_dict = {
                "KFs": f"{len(self.gs.keyviewpoints)}",
                "Points": f"{len(self.gs.gaussians.get_xyz)}"
            }
            progress_bar.set_postfix(loss_dict)
            progress_bar.update(update_rate)
            
            
        self.iteration_count += 1
        current_time = time.time()
        time_diff = current_time - self.last_time
        
        if time_diff >= 1.0:
            self.hz = self.iteration_count / time_diff
            self.last_time = current_time
            self.iteration_count = 0
        
    
    @timeit
    def terminate(self):
        if self._headless_boundary_snapshot_enabled and self._last_track_tstamp is not None:
            self._export_boundary_fisher_snapshot(
                tag="last",
                idx=int(self._last_track_tstamp),
            )
        self.gs.eval_fast(self.images, self.poses,  depth_scale=self.depth_scale)
        self.gs.finalize()
        self.gs.gs_instance()
        self.gs.eval_rendering(self.images, self.depths, self.poses,  depth_scale=self.depth_scale)
        self.tsdf_fusion.finalize()
        return 
        
