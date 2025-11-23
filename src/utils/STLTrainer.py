# ============================================================================
# Single Task Learning Trainer
# Gwade Steve
# MTTL and Application to Malaria Detection
# April 2025
# ============================================================================
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import nms
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_recall_fscore_support, classification_report
from scipy.stats import pearsonr
import time
from tqdm import tqdm
from DataUtils import FlexibleMalariaDataset, set_seeds, create_class_balanced_sampler
from Losses import *
from model import create_model_for_task, create_multitask_model, validate_model_setup
from experiment_manager import ExperimentManager
from Components import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
from DataUtils import set_seeds, seed_worker


def compute_detection_metrics(predictions, targets, iou_thresholds=[0.5]):
    # Prepare torchmetrics format
    preds = []
    gts = []
    for pred, target in zip(predictions, targets):
        preds.append({
            "boxes": pred if len(pred) > 0 else torch.zeros((0, 4)),
            "scores": torch.ones(len(pred)) if len(pred) > 0 else torch.zeros((0,)),
            "labels": torch.zeros(len(pred), dtype=torch.int64) if len(pred) > 0 else torch.zeros((0,), dtype=torch.int64)
        })
        gts.append({
            "boxes": target if len(target) > 0 else torch.zeros((0, 4)),
            "labels": torch.zeros(len(target), dtype=torch.int64) if len(target) > 0 else torch.zeros((0,), dtype=torch.int64)
        })

    # Compute mAP for each threshold
    map_metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=iou_thresholds)
    map_metric.update(preds, gts)
    result = map_metric.compute()
    metrics = {}
    for idx, thr in enumerate(iou_thresholds):
        key = f"mAP_{thr:.2f}"
        metrics[key] = result["map"].cpu().numpy()[idx] if "map" in result else 0.0

    # tp/fp/fn/ious
    metrics['tp'] = 0
    metrics['fp'] = 0
    metrics['fn'] = 0
    metrics['ious'] = []
    for pred, target in zip(predictions, targets):
        if len(target) == 0 and len(pred) == 0:
            continue
        elif len(target) == 0:
            metrics['fp'] += len(pred)
            continue
        elif len(pred) == 0:
            metrics['fn'] += len(target)
            continue
        ious = compute_bbox_iou(pred, target)
        max_ious = ious.max(dim=1)[0] if len(ious) > 0 else torch.tensor([])
        tp = (max_ious >= iou_thresholds[0]).sum().item()
        fp = len(pred) - tp
        fn = len(target) - tp
        metrics['tp'] += tp
        metrics['fp'] += fp
        metrics['fn'] += fn
        metrics['ious'].extend(max_ious.tolist())

    return metrics

def compute_bbox_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes"""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return torch.tensor([])
    
    # Convert center format to corner format
    def center_to_corner(boxes):
        x_center, y_center, width, height = boxes.unbind(-1)
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    boxes1_corner = center_to_corner(boxes1)
    boxes2_corner = center_to_corner(boxes2)
    
    # Compute intersection
    x1 = torch.max(boxes1_corner[:, None, 0], boxes2_corner[None, :, 0])
    y1 = torch.max(boxes1_corner[:, None, 1], boxes2_corner[None, :, 1])
    x2 = torch.min(boxes1_corner[:, None, 2], boxes2_corner[None, :, 2])
    y2 = torch.min(boxes1_corner[:, None, 3], boxes2_corner[None, :, 3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Compute union
    area1 = (boxes1_corner[:, 2] - boxes1_corner[:, 0]) * (boxes1_corner[:, 3] - boxes1_corner[:, 1])
    area2 = (boxes2_corner[:, 2] - boxes2_corner[:, 0]) * (boxes2_corner[:, 3] - boxes2_corner[:, 1])
    union = area1[:, None] + area2[None, :] - intersection
    
    return intersection / (union + 1e-6)

def compute_localization_metrics(pred_heatmaps, target_heatmaps):
    """
    Computes regression and similarity metrics for heatmaps.
    pred_heatmaps and target_heatmaps should be tensors of shape [B, 1, H, W].
    """
    if pred_heatmaps is None or target_heatmaps is None:
        return {'mse': float('inf'), 'mae': float('inf'), 'dice': 0.0, 'correlation': 0.0}
    
    # Ensure shapes match for calculation
    if pred_heatmaps.shape[-2:] != target_heatmaps.shape[-2:]:
        target_heatmaps = F.interpolate(target_heatmaps, size=pred_heatmaps.shape[-2:], mode='bilinear', align_corners=False)

    pred_flat = pred_heatmaps.view(-1)
    target_flat = target_heatmaps.view(-1)
    
    mse = F.mse_loss(pred_heatmaps, target_heatmaps).item()
    mae = F.l1_loss(pred_heatmaps, target_heatmaps).item()
    
    # Dice score for similarity
    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
    
    # Correlation
    pred_np = pred_flat.detach().cpu().numpy()
    target_np = target_flat.detach().cpu().numpy()
    correlation = 0.0
    if len(np.unique(pred_np)) > 1 and len(np.unique(target_np)) > 1:
        correlation, _ = pearsonr(pred_np, target_np)
        if np.isnan(correlation):
            correlation = 0.0
    
    return {
        'mse': mse,
        'mae': mae,
        'dice': dice.item(),
        'correlation': correlation
    }

def compute_segmentation_metrics(pred_logits, target_mask):
    """
    Compute segmentation metrics (Dice, IoU).
    """
    if pred_logits is None or target_mask is None:
        return {'dice': 0.0, 'iou': 0.0}
    
    # Upscale the low-resolution prediction
    pred_logits_upscaled = F.interpolate(
        pred_logits, 
        size=target_mask.shape[-2:], 
        mode='bilinear', 
        align_corners=False
    )
    
    pred_probs = torch.sigmoid(pred_logits_upscaled).cpu()
    target_mask = target_mask.cpu()

    pred_flat = pred_probs.view(-1)
    target_flat = target_mask.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)

    intersection_binary = ((pred_flat > 0.5) & (target_flat > 0.5)).sum().float()
    union_binary = ((pred_flat > 0.5) | (target_flat > 0.5)).sum().float()
    iou = intersection_binary / (union_binary + 1e-6)
    
    return {'dice': dice.item(), 'iou': iou.item()}

def compute_severity_metrics(pred_logits, target_class):
    if pred_logits is None or target_class is None:
        return {'accuracy': 0.0}
    
    # index of the highest logit value (pred class)
    pred_class = torch.argmax(pred_logits, dim=1)
    
    # Compare the predicted classes to the true classes
    correct_predictions = (pred_class == target_class)
    
    # Calculate accuracy by (number of correct predictions) / (total number of predictions)
    accuracy = correct_predictions.float().mean().item()
    
    return {'accuracy': accuracy}

def create_task_specific_collate_fn(task):
    """Create a task-specific collate function"""
    
    def task_collate_fn(batch):
        # Stack images
        images = torch.stack([item['image'] for item in batch])
        
        batch_data = {'image': images}
        
        if task == 'detection':
            batch_data.update({
                'bboxes': [item['bboxes'] for item in batch],
                'labels': [item['labels'] for item in batch],
                'num_objects': [item['num_objects'] for item in batch],
                'image_id': [item['image_id'] for item in batch]
            })
        
        elif task == 'regression':
            batch_data.update({
                'parasitemia_score': torch.stack([item['parasitemia_score'] for item in batch]),
                'image_id': [item['image_id'] for item in batch]
            })
        
        elif task == 'localization':
            batch_data.update({
                'heatmap': torch.stack([item['heatmap'] for item in batch]),
                'image_id': [item['image_id'] for item in batch]
            })
            
        elif task == 'segmentation':
            batch_data.update({
                'mask': torch.stack([item['mask'] for item in batch]),
                'image_id': [item['image_id'] for item in batch]
            })
        
        elif task == 'severity':
            batch_data.update({
                'image_id': [item['image_id'] for item in batch],
                'severity_class': torch.tensor([item['severity_class'] for item in batch], dtype=torch.long),
                'severity_label': [item['severity_label'] for item in batch]
            })
            
        return batch_data
    
    return task_collate_fn

#STL Trainer
class STLTrainer:
    def __init__(self, task, config, train_samples, val_samples, manager: ExperimentManager):
        self.task = task
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.manager = manager if manager else None
        self.model_save_path = self.manager.get_model_path() if manager else f'best_{self.task}_model.pth'
        self.num_classes = config.get('num_classes_detection', 3)
        print(f"Using device {self.device}")
        
        task_mapping = {
            'regression': 'parasitemia', 
            'localization': 'heatmap', 
            'heatmap': 'heatmap', 
            'detection': 'detection', 
            'segmentation': 'segmentation', 
            'severity': 'severity',
            'cell_classif': 'cell_classif',
            'roi_classif': 'roi_classif'
            }

        self.model_task = task_mapping.get(task, task)

        # The data and collate function for 'roi_classif' is the same as 'detection'
        if self.task == 'cell_classif':
            collate_fn = None 
        else:
            # When the task is 'roi_classif', we use the 'detection' collate function.
            collate_fn_task = 'detection' if self.task == 'roi_classif' else self.task
            if collate_fn_task == 'heatmap':
                collate_fn_task = 'localization'
            collate_fn = create_task_specific_collate_fn(collate_fn_task)
            
        self.model = create_model_for_task(config, task).to(device)
        
        dataset_task_mode = 'detection' if self.task == 'roi_classif' else self.task
        if dataset_task_mode == 'heatmap':
            dataset_task_mode = 'localization'
        self.train_dataset = FlexibleMalariaDataset(train_samples, task_mode=dataset_task_mode, augment=True)
        self.val_dataset = FlexibleMalariaDataset(val_samples, task_mode=dataset_task_mode, augment=False)
        
        g = torch.Generator()
        g.manual_seed(config.get('seed', 12))
        
        sampler = None
        if self.task in ['detection', 'roi_classif'] and config.get('use_sampler', False):
            print("Attempting to create a class-balanced sampler...")
            #num_sampler_classes = self.config.get('num_classes_detection', 2)
            NUM_DATASET_CLASSES = 3
            sampler = create_class_balanced_sampler(
                train_samples, 
                priority_class=0, # Priority to inf
                num_classes=NUM_DATASET_CLASSES
            )
            
            if sampler:
                print(f"Sampler was correctly created for task '{self.task}'.")
            else:
                print(f"Warning: Sampler creation failed for task '{self.task}'.")
        else:
            print(f"Sampler not used for task '{self.task}' (not a detection task or not enabled in config).")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
            sampler=sampler,           
            shuffle=(sampler is None)  
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True, 
            collate_fn=collate_fn, 
            worker_init_fn=seed_worker, 
            generator=g
        )

        #self.optimizer = optim.AdamW(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        print("\nSetting up optimizer with parameter groups...")
        
        head_params = [p for p in self.model.task_heads.parameters() if p.requires_grad]
        adapter_params = [p for n, p in self.model.backbone.named_parameters() if 'adapters' in n and p.requires_grad]
        backbone_params = [p for n, p in self.model.backbone.named_parameters() if 'adapters' not in n and p.requires_grad]

        # main LR for new components (head and adapters)
        lr_main = config['learning_rate']
        lr_backbone = config.get('backbone_learning_rate', lr_main / 10.0)

        param_groups = [
            {'params': head_params, 'lr': lr_main, 'weight_decay': config['weight_decay']},
            {'params': adapter_params, 'lr': lr_main, 'weight_decay': config['weight_decay']},
        ]
        print(f"Head and Adapter params LR: {lr_main}")

        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': lr_backbone, 'weight_decay': 0.0})
            print(f"Fine-tuned Backbone params LR: {lr_backbone}")
        
        self.optimizer = optim.AdamW(param_groups)
        
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        print(f"Automatic Mixed Precision (AMP) {'enabled' if self.use_amp else 'disabled'}.")
        
        self.warmup_epochs = config.get('warmup_epochs', 5)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=(len(self.train_loader) * (config['num_epochs'] - self.warmup_epochs)), eta_min=config['learning_rate'] * 0.01)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.001, total_iters=len(self.train_loader) * self.warmup_epochs)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[len(self.train_loader) * self.warmup_epochs])
        
        self.history = defaultdict(list)
        self.map_metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5]).to(self.device)
        
        print(f"STL Trainer initialized for {task.upper()} task")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
    def _filter_batch_for_2_class(self, batch):
        if self.task not in ['detection', 'roi_classif', 'cell_classif']:
            return batch

        original_labels_list = batch.get('labels', batch.get('bbox_labels'))
        original_bboxes_list = batch['bboxes']
        
        new_labels_list = []
        new_bboxes_list = []

        for i in range(len(original_labels_list)):
            labels_tensor = original_labels_list[i]
            bboxes_tensor = original_bboxes_list[i]
            mask = (labels_tensor != 2)
            
            new_labels_list.append(labels_tensor[mask])
            new_bboxes_list.append(bboxes_tensor[mask])
            
        batch['labels'] = new_labels_list
        batch['bbox_labels'] = new_labels_list
        batch['bboxes'] = new_bboxes_list
        
        return batch
    
    def _filter_batch_for_detection(self, batch):
        if self.task == 'cell_classif':
            num_classes = self.config.get('num_classes_classif', 3)
            if num_classes >= 3:
                return batch 

            labels_tensor = batch['label']
            if num_classes == 1:
                mask = (labels_tensor == 0) # Keep only 'Infected'
            elif num_classes == 2:
                mask = (labels_tensor != 2) # Keep 'Infected' and 'Healthy'
            
            # Filter based on maskl
            for key in batch:
                batch[key] = batch[key][mask]
            return batch

        # detection/roi_classif batches
        elif self.task in ['detection', 'roi_classif']:
            num_classes = self.config.get('num_classes_detection', 3)
            if num_classes >= 3:
                return batch 

            original_labels_list = batch.get('labels', batch.get('bbox_labels'))
            original_bboxes_list = batch['bboxes']
            
            new_labels_list = []
            new_bboxes_list = []

            for i in range(len(original_labels_list)):
                labels_tensor = original_labels_list[i]
                bboxes_tensor = original_bboxes_list[i]

                if num_classes == 1:
                    mask = (labels_tensor == 0)
                elif num_classes == 2:
                    mask = (labels_tensor != 2)
                
                new_labels_list.append(labels_tensor[mask])
                new_bboxes_list.append(bboxes_tensor[mask])
                
            batch['labels'] = new_labels_list
            batch['bbox_labels'] = new_labels_list
            batch['bboxes'] = new_bboxes_list
            return batch
        
        else:
            return batch
    
    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(self.train_loader, desc=f"Training {self.task}"):
            batch = self._filter_batch_for_detection(batch)

            self.optimizer.zero_grad()
            images = batch['image'].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                if self.task == 'roi_classif':
                    outputs = self.model(images, targets=batch)
                    if 'roi_classif' not in outputs:
                        continue
                    loss, _, _ = self.model.compute_loss(outputs, targets=None)
                
                elif self.task == 'cell_classif':
                    labels = batch['label'].to(self.device)
                    outputs = self.model(images)
                    loss, _, _ = self.model.compute_loss(outputs, {'label': labels})
                
                elif self.task == 'detection' and self.config.get('detection_head') == 'RCNN':
                    targets = self._prepare_rcnn_detection_targets(batch)
                    if not any(len(t['labels']) > 0 for t in targets):
                        print("Skipping batch with no valid targets")
                        continue
                    loss_dict = self.model(images, targets=targets)['detection']
                    loss = sum(l for l in loss_dict.values())
                else:
                    outputs = self.model(images)
                    targets = self._prepare_targets(batch)
                    loss, _, _ = self.model.compute_loss(outputs, targets)

            if isinstance(loss, torch.Tensor):
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                epoch_loss += loss.item()
        
        if len(self.train_loader) > 0:
            avg_loss = epoch_loss / len(self.train_loader)
        else:
            avg_loss = 0.0
            
        return avg_loss, {}

    def _prepare_targets(self, batch):
        targets = {}
        if self.task == 'detection':
            if self.config.get('detection_head') == 'RCNN':
                targets['detection'] = self._prepare_rcnn_detection_targets(batch)
            else:
                targets['detection'] = self._prepare_detection_targets(batch)
        elif self.task == 'cell_classif':
            targets['label'] = batch['label'].to(self.device)
        elif self.task == 'heatmap' or self.task == 'localization':
            targets['heatmap'] = batch['heatmap'].to(self.device)
        elif self.task == 'regression':
            targets['parasitemia'] = batch['parasitemia_score'].to(self.device)
        elif self.task == 'segmentation':
            targets['segmentation'] = batch['mask'].unsqueeze(1).to(self.device)
        elif self.task == 'severity':
            targets['severity_class'] = batch['severity_class'].to(self.device)
        return targets
    
    def validate_epoch(self):
        self.model.eval()
        epoch_loss = 0.0
        if self.task == 'detection': self.map_metric.reset()
        all_preds, all_labels = [], []
        epoch_metrics = defaultdict(float)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validating {self.task}"):
                batch = self._filter_batch_for_detection(batch)
                if not batch.get('image', torch.empty(0)).numel(): continue

                images = batch['image'].to(self.device)
                
                loss = torch.tensor(0.0, device=self.device) # loss as a tensor

                if self.task == 'roi_classif':
                    outputs = self.model(images, targets=batch)
                    if 'roi_classif' in outputs:
                        loss, _, _ = self.model.compute_loss(outputs, targets=None)
                        preds = torch.argmax(outputs['roi_classif'], dim=1)
                        labels = outputs['roi_classif_labels']
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                elif self.task == 'detection' and self.config.get('detection_head') == 'RCNN':
                    predictions = self.model(images, targets=None)
                    
                    self.model.train()
                    rcnn_targets = self._prepare_rcnn_detection_targets(batch)
                    if any(len(t['labels']) > 0 for t in rcnn_targets):
                        loss_dict = self.model(images, targets=rcnn_targets)['detection']
                        loss_values = [v for v in loss_dict.values() if isinstance(v, torch.Tensor)]
                        if loss_values:
                            loss = sum(loss_values)
                    self.model.eval()
                    
                    self.map_metric.update(predictions['detection'], rcnn_targets)
                
                else: # cell_classif, segmentation, severity, heatmap, ...
                    outputs = self.model(images)
                    targets_dict = self._prepare_targets(batch)
                    loss, _, _ = self.model.compute_loss(outputs, targets_dict)
                    
                    if self.task in ['cell_classif', 'severity']:
                        task_key = self.task
                        labels = targets_dict.get('label' if task_key == 'cell_classif' else 'severity_class')
                        if labels is not None:
                            preds = torch.argmax(outputs[task_key], dim=1)
                            all_preds.extend(preds.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())
                    elif self.task == 'segmentation':
                        metrics_seg = compute_segmentation_metrics(outputs['segmentation'], targets_dict['segmentation'])
                        epoch_metrics['dice'] += metrics_seg.get('dice', 0) * len(images)
                        epoch_metrics['iou'] += metrics_seg.get('iou', 0) * len(images)
                    elif self.task == 'heatmap' or self.task == 'localization':
                        metrics_loc = compute_localization_metrics(outputs['heatmap'], targets_dict['heatmap'])
                        epoch_metrics['dice'] += metrics_loc.get('dice', 0) * len(images)
                        epoch_metrics['mse'] += metrics_loc.get('mse', 0) * len(images)
                
                if torch.isfinite(loss):
                    epoch_loss += loss.item()

        # aggregation 
        avg_loss = epoch_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        
        final_metrics = {}
        if self.task == 'detection':
            result = self.map_metric.compute()
            final_metrics['mAP_50'] = result.get("map_50", torch.tensor(0.0)).item()

        elif self.task in ['cell_classif', 'severity', 'roi_classif']:
            if all_labels:
                report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
                final_metrics['accuracy'] = report.get('accuracy', 0.0)
                final_metrics['f1_macro'] = report.get('macro avg', {}).get('f1-score', 0.0)
            else:
                final_metrics['accuracy'] = 0.0
                final_metrics['f1_macro'] = 0.0
        
        elif self.task in ['segmentation', 'heatmap', 'localization']:
            num_samples = len(self.val_dataset)
            for key in epoch_metrics:
                final_metrics[key] = epoch_metrics[key] / num_samples if num_samples > 0 else 0
                
        return avg_loss, final_metrics
    
    def _get_yolo_preds_and_targets(self, outputs, batch, conf_thresh=0.25, nms_thresh=0.45):
        detection_outputs = outputs['detection']
        batch_size = detection_outputs[0].shape[0]
        grid_sizes = self.config.get('grid_sizes', [64, 32, 16])
        
        all_boxes_batch = [[] for _ in range(batch_size)]
        all_scores_batch = [[] for _ in range(batch_size)]

        for i, scale_output in enumerate(detection_outputs):
            grid_size = grid_sizes[i]
            objectness_prob = scale_output[..., 0]
            bbox_coords = scale_output[..., 1 + self.config['num_classes_detection']:]
            
            grid_y, grid_x = torch.meshgrid(torch.arange(grid_size, device=self.device), torch.arange(grid_size, device=self.device), indexing='ij')
            
            pred_x_offset, pred_y_offset, pred_w_scale, pred_h_scale = bbox_coords.unbind(-1)

            decoded_x = (grid_x + pred_x_offset) / grid_size
            decoded_y = (grid_y + pred_y_offset) / grid_size
            decoded_w = pred_w_scale / grid_size
            decoded_h = pred_h_scale / grid_size
            
            decoded_boxes_center_wh = torch.stack([decoded_x, decoded_y, decoded_w, decoded_h], dim=-1)

            for b in range(batch_size):
                conf_mask = objectness_prob[b] > conf_thresh
                if conf_mask.sum() > 0:
                    all_boxes_batch[b].append(decoded_boxes_center_wh[b][conf_mask])
                    all_scores_batch[b].append(objectness_prob[b][conf_mask])

        predictions_for_metric = []
        for b in range(batch_size):
            if len(all_boxes_batch[b]) > 0:
                boxes_to_process = torch.cat(all_boxes_batch[b], dim=0)
                scores_to_process = torch.cat(all_scores_batch[b], dim=0)
                
                x_center, y_center, w, h = boxes_to_process.unbind(1)
                x1, y1, x2, y2 = x_center - w / 2, y_center - h / 2, x_center + w / 2, y_center + h / 2
                boxes_for_nms = torch.stack([x1, y1, x2, y2], dim=1)
                
                keep_indices = nms(boxes_for_nms, scores_to_process, nms_thresh)
                
                predictions_for_metric.append({
                    "boxes": boxes_for_nms[keep_indices],
                    "scores": scores_to_process[keep_indices],
                    "labels": torch.zeros(len(keep_indices), dtype=torch.int64, device=self.device)
                })
            else:
                predictions_for_metric.append({
                    "boxes": torch.empty(0, 4, device=self.device),
                    "scores": torch.empty(0, device=self.device),
                    "labels": torch.empty(0, dtype=torch.int64, device=self.device)
                })

        targets_for_metric = []
        for i in range(len(batch['image'])):
            boxes_center_wh = batch['bboxes'][i]
            
            if len(boxes_center_wh) > 0:
                x_center, y_center, w, h = boxes_center_wh.unbind(1)
                x1, y1, x2, y2 = x_center - w / 2, y_center - h / 2, x_center + w / 2, y_center + h / 2
                target_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                
                targets_for_metric.append({
                    "boxes": target_boxes_xyxy.to(self.device),
                    "labels": torch.zeros(len(target_boxes_xyxy), dtype=torch.int64, device=self.device)
                })
            else:
                targets_for_metric.append({
                    "boxes": torch.empty(0, 4, device=self.device),
                    "labels": torch.empty(0, dtype=torch.int64, device=self.device)
                })
                
        return predictions_for_metric, targets_for_metric

    def _compute_detection_metrics(self, predictions, targets):
        self.map_metric.update(predictions, targets)
        result = self.map_metric.compute()
        map_50 = result["map_50"].item() if "map_50" in result and not torch.isnan(result["map_50"]) else 0.0
        return {"mAP_50": map_50}
    
    def _prepare_rcnn_detection_targets(self, batch):
        targets = []
        img_h, img_w = self.config.get('image_size', (512, 512))
        num_det_classes = self.config.get('num_classes_detection', 1)
        
        for i in range(len(batch.get('image',[]))):
            boxes_yolo = batch['bboxes'][i]
            if len(boxes_yolo) > 0:
                cx, cy, w, h = boxes_yolo.unbind(1)
                x1, y1, x2, y2 = (cx-w/2)*img_w, (cy-h/2)*img_h, (cx+w/2)*img_w, (cy+h/2)*img_h
                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                
                if num_det_classes == 1:
                    labels = torch.ones(len(boxes_yolo), dtype=torch.int64)
                else:
                    original_labels = batch['labels'][i] 
                    labels = original_labels.long() + 1
                
                targets.append({"boxes": boxes_xyxy.to(self.device), "labels": labels.to(self.device)})
            else:
                targets.append({"boxes": torch.empty((0, 4), device=self.device), "labels": torch.empty(0, dtype=torch.int64, device=self.device)})
        return targets
    
    def _prepare_detection_targets(self, batch):
        """
        Encodes ground truth boxes into the grid-space format required by the YOLO loss.
        Instead of Only parasitized cells (label==0), all cells are mapped under one label and are used for detection.
        """
        targets = {'objectness': [], 'classes': [], 'bboxes': []}
        actual_batch_size = len(batch['image'])
        grid_sizes = self.config.get('grid_sizes', [64, 32, 16])
        num_classes = self.config['num_classes_detection']

        for scale_idx, grid_size in enumerate(grid_sizes):
            obj_target = torch.zeros(actual_batch_size, grid_size, grid_size, device=self.device)
            cls_target = torch.zeros(actual_batch_size, grid_size, grid_size, dtype=torch.long, device=self.device)
            bbox_target = torch.zeros(actual_batch_size, grid_size, grid_size, 4, device=self.device)

            for b in range(actual_batch_size):
                bboxes = batch['bboxes'][b]
                labels = batch['labels'][b]
                if len(bboxes) == 0: continue

                # we filter only inf
                #mask = (labels == 0)
                #bboxes = bboxes[mask]
                #labels = labels[mask]
                if len(bboxes) == 0: continue

                assigned_cells_in_image = set()
                for box, label in zip(bboxes, labels):
                    x_img, y_img, w_img, h_img = box
                    grid_x_float = x_img * grid_size
                    grid_y_float = y_img * grid_size
                    grid_x = int(grid_x_float)
                    grid_y = int(grid_y_float)
                    cell_id = (grid_y, grid_x)
                    if cell_id in assigned_cells_in_image:
                        continue
                    assigned_cells_in_image.add(cell_id)
                    grid_x = min(grid_x, grid_size - 1)
                    grid_y = min(grid_y, grid_size - 1)
                    target_x = grid_x_float - grid_x
                    target_y = grid_y_float - grid_y
                    target_w = w_img * grid_size
                    target_h = h_img * grid_size
                    obj_target[b, grid_y, grid_x] = 1.0
                    cls_target[b, grid_y, grid_x] = 0  # 0 for single label
                    bbox_target[b, grid_y, grid_x, :] = torch.tensor(
                        [target_x, target_y, target_w, target_h], device=self.device
                    )

            targets['objectness'].append(obj_target)
            targets['classes'].append(cls_target)
            targets['bboxes'].append(bbox_target)
        return targets
    
    def train(self, num_epochs):
        best_metric = -1.0 
        patience_counter = 0
        max_patience = self.config.get('patience', 15)
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            train_loss, _ = self.train_epoch()
            val_loss, val_metrics = self.validate_epoch()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            for k, v in val_metrics.items():
                self.history[f'val_{k}'].append(v)
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            current_metric = 0
            is_better = False
            
            if self.task == 'detection':
                metric_name = 'mAP_50'
                current_metric = val_metrics.get(metric_name, 0)
                print(f"Val - {metric_name}: {current_metric:.4f}")
                if current_metric > best_metric:
                    is_better = True
            
            elif self.task in ['cell_classif', 'severity', 'roi_classif']:
                metric_name = 'f1_macro'
                current_metric = val_metrics.get(metric_name, 0)
                print(f"Val - Accuracy: {val_metrics.get('accuracy', 0):.4f} | F1-Macro: {current_metric:.4f}")
                if current_metric > best_metric:
                    is_better = True
            
            elif self.task in ['segmentation', 'heatmap', 'localization']:
                metric_name = 'dice'
                current_metric = val_metrics.get(metric_name, 0)
                if self.task == 'segmentation':
                    print(f"Val - Dice Score: {current_metric:.4f} | IoU: {val_metrics.get('iou', 0):.4f}")
                else: 
                    print(f"Val - Dice Score: {current_metric:.4f} | MSE: {val_metrics.get('mse', 0):.4f}")
                
                if current_metric > best_metric:
                    is_better = True
            
            else: 
                metric_name = 'Val Loss'
                current_metric = val_loss
                if best_metric == -1.0: 
                    best_metric = float('inf') 
                
                if current_metric < best_metric:
                    is_better = True

            if is_better:
                best_metric = current_metric
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"Saved best model to {self.model_save_path} ({metric_name}: {best_metric:.4f})")
            else:
                patience_counter += 1
                if patience_counter != 0: 
                    print(f"Patience Counter: {patience_counter}/{max_patience}")
                if patience_counter >= max_patience:
                    print("Early stopping triggered")
                    break
        
        print("\nLoading best model for final evaluation...")
        if os.path.exists(self.model_save_path):
            self.model.load_state_dict(torch.load(self.model_save_path))
        print("Training completed.")
        
    def plot_training_history(self):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
        
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_title(f'{self.task.upper()} - Loss Curve')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.6)

        metric_keys = [k for k in self.history.keys() if k.startswith('val_') and k != 'val_loss']
        if metric_keys:
            for key in metric_keys:
                metric_name = key.replace('val_', '').upper()
                axes[1].plot(self.history[key], label=f'Validation {metric_name}')
            axes[1].set_title(f'{self.task.upper()} - Validation Metrics')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Score')
            axes[1].legend()
            axes[1].grid(True, linestyle='--', alpha=0.6)
        else:
            axes[1].axis('off')

        plt.tight_layout()
        plt.show()
        return fig
        
    def predict(self, samples):
        """
        Runs inference on a list of samples and returns predictions and ground truth
        """
        self.model.eval()
        temp_dataset = FlexibleMalariaDataset(samples, task_mode=self.task, augment=False)
        if self.task == 'cell_classif':
            collate_fn = None 
        else:
            collate_fn = create_task_specific_collate_fn(self.task)
        temp_loader = DataLoader(
            temp_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True, 
            collate_fn=collate_fn
        )
        
        all_predictions = []
        all_ground_truth = []

        with torch.no_grad():
            for batch in tqdm(temp_loader, desc=f"Generating predictions for '{self.task}'"):
                images = batch['image'].to(self.device)
                outputs = self.model(images)
                
                if self.task == 'cell_classif':
                    probs = F.softmax(outputs['cell_classif'], dim=1)
                    all_predictions.extend(list(torch.unbind(probs.cpu(), dim=0)))
                    all_ground_truth.extend(batch['label'].tolist())
                    
                elif self.task == 'detection':
                    if self.config.get('detection_head') == 'RCNN':
                        predictions_batch = outputs['detection']
                        targets_batch = self._prepare_rcnn_detection_targets(batch)
                    else: 
                        predictions_batch, targets_batch = self._get_yolo_preds_and_targets(outputs, batch)
                    
                    all_predictions.extend([{k: v.cpu() for k, v in p.items()} for p in predictions_batch])
                    all_ground_truth.extend([{k: v.cpu() for k, v in t.items()} for t in targets_batch])

                elif self.task == 'segmentation':
                    preds_batch = F.interpolate(outputs['segmentation'], size=images.shape[-2:], mode='bilinear', align_corners=False)
                    all_predictions.extend(list(torch.unbind(torch.sigmoid(preds_batch).cpu(), dim=0)))
                    all_ground_truth.extend(list(torch.unbind(batch['mask'], dim=0)))

                elif self.task == 'severity':
                    all_predictions.extend(list(torch.unbind(F.softmax(outputs['severity'], dim=1).cpu(), dim=0)))
                    all_ground_truth.extend(batch['severity_class'].tolist())
                    
                # ...
                
        return all_predictions, all_ground_truth      