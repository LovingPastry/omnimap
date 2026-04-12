import torch
import numpy as np
import cv2
import os,sys

import time
import functools

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[{func.__qualname__}] 耗时: {end - start:.6f} s")
        return result
    return wrapper


class VisualModule:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda")

    def load_models(self):
        raise NotImplementedError
        
    def erode_mask(self, mask, kernel_size=5):
        """ 对掩码进行腐蚀操作 """
        import torch.nn.functional as F
        kernel = torch.ones(1, 1, kernel_size, kernel_size).to(mask.device)
        eroded_mask = F.conv2d(mask.unsqueeze(0).unsqueeze(0).float(), kernel, padding=kernel_size//2)
        eroded_mask = eroded_mask.squeeze(0).squeeze(0) >= kernel_size*kernel_size
        return eroded_mask
    
    def detect(self, img, vis_gui=False):
        raise NotImplementedError

class YOLOVisualModule(VisualModule):
    """ 加载视觉模型：YOLO-World, TAP, SBERT """
    def __init__(self, config, save_dir, vis_gui):
        super().__init__(config, save_dir, vis_gui)
        self.load_models()
        
    def load_models(self):
        from mmengine.config import Config
        from mmdet.apis import init_detector
        from mmdet.datasets import Compose
        from mmdet.datasets.pipelines import get_test_pipeline_cfg
        from tokenize_anything import model_registry
        from sentence_transformers import SentenceTransformer
        import spacy
        
        # yolo-world model
        config = self.config['path']['yolo_config']
        cfg = Config.fromfile(config)
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(config))[0])
        checkpoint = self.config['path']['yolo_cp'] 
        self.yolo_world = init_detector(cfg, checkpoint=checkpoint, device="cuda")
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
        test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
        self.yolo_world_test_pipeline = Compose(test_pipeline_cfg)
        with open("pretrained_models/yolo_labels.txt") as f:
            lines = f.readlines()
        self.yolo_texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
        self.yolo_world.reparameterize(self.yolo_texts)
        self.yolo_score = 0.1
        self.yolo_max_dets = 100

        # tap model 
        model_type = "tap_vit_l"
        checkpoint = self.config['path']['tap_cp1']
        concept_weights = self.config['path']['tap_cp2']
        self.nlp = spacy.load("en_core_web_sm")
        self.tap_model = model_registry[model_type](checkpoint=checkpoint)
        self.tap_model.concept_projector.reset_weights(concept_weights)
        self.tap_model.text_decoder.reset_cache(max_batch_size=1000)
        
        # SBERT model
        self.sbert_model = SentenceTransformer(self.config['path']['sbert_cp'])
        
    def detect(self, img, vis_gui=False):
        """
        执行目标检测和分割
        
        Args:
            img: RGB图像，形状为(H, W, 3)
            vis_gui: 是否可视化
            
        Returns:
            final_masks: 最终的实例掩码列表
            caption_fts: 实例的语义特征
            last_mask_image: 可视化的掩码图像（如果vis_gui为True）
        """
        from tokenize_anything.utils.image_utils import im_rescale, im_vstack
        import torch
        
        last_mask_image = None
        
        '''[1] yolo-world - 目标检测'''        
        data_info = dict(img=img, img_id=0, texts=self.yolo_texts)
        data_info = self.yolo_world_test_pipeline(data_info)
        data_batch = dict(inputs=data_info['inputs'].unsqueeze(0), data_samples=[data_info['data_samples']])
        with torch.no_grad():
            output = self.yolo_world.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        
        # 分数阈值过滤
        pred_instances = pred_instances[pred_instances.scores.float() > self.yolo_score]
        
        # 最大检测数限制
        if len(pred_instances.scores) > self.yolo_max_dets:
            indices = pred_instances.scores.float().topk(self.yolo_max_dets)[1]
            pred_instances = pred_instances[indices]
        
        # 获取边界框
        min_rects = pred_instances['bboxes']
        min_rects = torch.unique(min_rects, dim=0).cpu().numpy()  # min_rects是最终检测结果
        
        # 如果没有检测到物体，返回空
        if len(min_rects) == 0:
            return [], [], last_mask_image
        
        '''[2] TAP - 实例分割'''        
        img_list, img_scales = im_rescale(img, scales=[1024], max_size=1024)
        input_size, original_size = img_list[0].shape, img.shape[:2]
        img_batch = im_vstack(img_list, fill_value=self.tap_model.pixel_mean_value, size=(1024, 1024))
        
        # 获取TAP模型输入
        inputs = self.tap_model.get_inputs({"img": img_batch})
        inputs.update(self.tap_model.get_features(inputs))
        
        # 准备边界框点
        batch_points = np.zeros((len(min_rects), 2, 3), dtype=np.float32)
        batch_points[:, 0, 0] = min_rects[:, 0]  # min x
        batch_points[:, 0, 1] = min_rects[:, 1]  # min y
        batch_points[:, 0, 2] = 2
        batch_points[:, 1, 0] = min_rects[:, 2]  # max x
        batch_points[:, 1, 1] = min_rects[:, 3]  # max y
        batch_points[:, 1, 2] = 3 
        
        inputs["points"] = batch_points
        inputs["points"][:, :, :2] *= np.array(img_scales, dtype="float32")
        
        # 执行TAP模型推理
        outputs = self.tap_model.get_outputs(inputs)
        iou_score, mask_pred = outputs["iou_pred"], outputs["mask_pred"]
        iou_score[:, 1:] -= 1000.0  # 惩罚松散点的分数
        mask_index = torch.arange(iou_score.shape[0]), iou_score.argmax(1)
        
        iou_scores, masks = iou_score[mask_index], mask_pred[mask_index]
        masks = self.tap_model.upscale_masks(masks[:, None], img_batch.shape[1:-1])
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = self.tap_model.upscale_masks(masks, original_size).gt(0).squeeze(1)
        
        # 按掩码面积排序
        mask_areas = torch.tensor([mask.sum().item() for mask in masks])
        sorted_indices = torch.argsort(mask_areas, descending=True)
        sorted_masks = masks[sorted_indices]
        mask_id = torch.zeros(sorted_masks[0].shape)
        
        # 处理重叠掩码
        ok_area_mask = []
        final_masks = []
        for i, mask in enumerate(sorted_masks):
            mask_id[mask] = i+1
        
        for new_id in range(len(sorted_masks)):
            new_mask = mask_id == new_id+1
            new_mask = self.erode_mask(new_mask)
            if torch.sum(new_mask) < 100:
                continue
            final_masks.append(new_mask)
            ok_area_mask.append(new_id)
        
        if not ok_area_mask:
            return [], [], last_mask_image
            
        ok_area_mask = torch.tensor(np.stack(ok_area_mask)).long()
        final_masks = torch.stack(final_masks).cuda()
        
        # 可视化掩码
        if vis_gui:
            mask_image = np.ones(img.shape)*255*0.2
            for i in range(len(final_masks)):
                mask = final_masks[i].cpu().numpy()  
                color = np.random.random(3)*255
                mask_colored = np.stack([mask * color[0], mask * color[1], mask * color[2]], axis=-1)  
                mask_image = np.maximum(mask_image, mask_colored)  
            last_mask_image = mask_image
        
        '''[3] SBERT - 语义特征提取'''        
        sem_tokens = outputs["sem_tokens"][mask_index].unsqueeze_(1)
        captions = self.tap_model.generate_text(sem_tokens)
        captions = captions[sorted_indices][ok_area_mask]
        
        # 处理文本描述
        new_captions = []
        for sentence in captions:
            doc = self.nlp(str(sentence))
            subject = ""
            for npp in doc.noun_chunks:
                if sentence.startswith(str(npp)):
                    subject = str(npp)
                    break
            if not subject:
                subject = sentence
            new_captions.append(subject)
        
        # 提取语义特征
        caption_fts = self.sbert_model.encode(
            new_captions,
            convert_to_tensor=True,
            device="cuda",
            show_progress_bar=False,
        ).detach()
        caption_fts = caption_fts / caption_fts.norm(dim=-1, keepdim=True)
        
        return final_masks, caption_fts, last_mask_image
