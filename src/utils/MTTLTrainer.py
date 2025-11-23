# ============================================================================
# MultiTask Learning Trainer
# Gwade Steve
# MTTL and Application to Malaria Detection
# April 2025
# ============================================================================
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
import time
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from scipy.stats import pearsonr
import torch.nn.functional as F
from experiment_manager import ExperimentManager
from sklearn.metrics import classification_report
import itertools

from DataUtils import FlexibleMalariaDataset, seed_worker, create_class_balanced_sampler
from model import create_multitask_model
from Evaluator import compute_segmentation_metrics 

def compute_localization_metrics(pred_heatmaps, target_heatmaps):
    """
    Computes regression and similarity metrics for heatmaps.
    pred_heatmaps and target_heatmaps should be tensors of shape [B, 1, H, W].
    """
    if pred_heatmaps is None or target_heatmaps is None:
        return {'mse': float('inf'), 'mae': float('inf'), 'dice': 0.0, 'correlation': 0.0}
    
    # Ensure shapes match
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
        size=target_mask.shape[-2:], # Target shape is [B, 512, 512]
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

# MTTLTrainer
class MTTLTrainer:
    def __init__(self, config, train_samples, val_samples, manager: ExperimentManager):
        self.config = config
        self.manager = manager
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_path = self.manager.get_model_path()
        self.num_classes = config.get('num_classes_detection', 3)
        
        print(f"Using device: {self.device}")
        
        self.model = create_multitask_model(config, tasks=config['active_tasks']).to(self.device)
        
        self.train_dataset_mttl = FlexibleMalariaDataset(train_samples, task_mode='multi_task', augment=True)
        self.val_dataset = FlexibleMalariaDataset(val_samples, task_mode='multi_task', augment=False)

        g = torch.Generator(); g.manual_seed(config.get('seed', 12))
        
        sampler = None
        if 'detection' in config['active_tasks'] and config.get('use_sampler', False):
            print("Attempting to create a class-balanced sampler for MTTL...")
            NUM_DATASET_CLASSES = config.get('num_classes_detection', 3)
            sampler = create_class_balanced_sampler(
                train_samples, 
                priority_class=0, # prioritize 'Infected'
                num_classes=3
            )

            if sampler:
                print(f"Sampler correctly created for MTTL, configured for {NUM_DATASET_CLASSES} dataset classes.")
            else:
                print(f"Warning: Sampler creation failed for MTTL.")
        else:
            print(f"Sampler not used for MTTL (detection not an active task or not enabled in config).")

        self.train_loader_mttl = DataLoader(
            self.train_dataset_mttl,
            batch_size=config['batch_size'],
            num_workers=0,
            pin_memory=True,
            collate_fn=self.mttl_collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
            sampler=sampler,
            shuffle=(sampler is None)
        )

        classif_batch_size = config.get('classif_batch_size', 512)

        self.train_loader_classif = None
        if 'cell_classif' in config['active_tasks'] and classif_batch_size > 0:
            print(f"Creating dedicated data loader for classifier boosting (batch size: {classif_batch_size})...")
            train_dataset_classif = FlexibleMalariaDataset(train_samples, task_mode='cell_classif', augment=True)
            self.train_loader_classif = DataLoader(
                train_dataset_classif, 
                batch_size=classif_batch_size, 
                shuffle=True, num_workers=0, pin_memory=True,
                worker_init_fn=seed_worker, generator=g
            )

        self.val_loader = DataLoader(
            self.val_dataset, batch_size=config.get('val_batch_size', config['batch_size']), shuffle=False,
            num_workers=0, pin_memory=True, collate_fn=self.mttl_collate_fn,
            worker_init_fn=seed_worker, generator=g
        )
        
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.grad_accumulation_steps = config.get('grad_accumulation_steps', 1)
        
        print("\nSetting up optimizer with parameter groups...")
        lr_main = config['learning_rate']
        lr_backbone = config.get('backbone_learning_rate', lr_main / 10.0)
        
        loss_params = [p for p in self.model.loss_function.parameters() if p.requires_grad]
        
        param_groups = [
            {'params': self.model.task_heads.parameters(), 'lr': lr_main},
            {'params': [p for n, p in self.model.backbone.named_parameters() if 'adapters' in n], 'lr': lr_main},
            {'params': [p for n, p in self.model.backbone.named_parameters() if 'adapters' not in n and p.requires_grad], 'lr': lr_backbone}
        ]
        
        if loss_params:
            param_groups.append({'params': loss_params, 'lr': lr_main})
        self.optimizer = optim.AdamW(param_groups, weight_decay=config['weight_decay'])
        
        self.warmup_epochs = config.get('warmup_epochs', 10)
        num_optimizer_steps_per_epoch = len(self.train_loader_mttl) // self.grad_accumulation_steps
        total_optimizer_steps = config['num_epochs'] * num_optimizer_steps_per_epoch
        num_warmup_steps = self.warmup_epochs * num_optimizer_steps_per_epoch

        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_optimizer_steps - num_warmup_steps, eta_min=config['learning_rate'] * 0.01)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=num_warmup_steps)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[num_warmup_steps])
        
        self.epoch_history = defaultdict(list)
        self.step_history = defaultdict(list)
        self.map_metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5, 0.75]).to(self.device)
        print(f"MTTL Trainer initialized for tasks: {config['active_tasks']}")
        
    def _prepare_task_targets(self, batch):
        task_targets = {}
        img_h, img_w = self.config.get('image_size', (512, 512))
        
        if 'heatmap' in self.config['active_tasks'] or 'localization' in self.config['active_tasks']:
            task_targets['heatmap'] = batch['heatmap'].to(self.device)

        # Prepare targets for RoI Classifier N class, N>2
        if 'roi_classif' in self.config['active_tasks']:
            # Read the number of classes the classifier head was configured with
            num_classif_classes = self.config.get('num_classes_classif', 3)

            list_of_gt_boxes_all = []
            labels_to_cat_all = []
            for i in range(len(batch['image'])):
                bboxes_yolo_all = batch['bboxes'][i]
                labels_all = batch['bbox_labels'][i]

                # Filter out any labels that are outside the range of what the classifier can handle.
                if len(labels_all) > 0:
                    valid_mask = labels_all < num_classif_classes
                    bboxes_yolo_all = bboxes_yolo_all[valid_mask]
                    labels_all = labels_all[valid_mask]
                
                if len(bboxes_yolo_all) > 0 and len(labels_all) > 0:
                    cx, cy, w, h = bboxes_yolo_all.unbind(1)
                    x1, y1, x2, y2 = (cx-w/2)*img_w, (cy-h/2)*img_h, (cx+w/2)*img_w, (cy+h/2)*img_h
                    list_of_gt_boxes_all.append(torch.stack([x1, y1, x2, y2], dim=1).to(self.device))
                    labels_to_cat_all.append(labels_all.to(self.device))
            
            if labels_to_cat_all:
                task_targets['roi_classif'] = {
                    'boxes': list_of_gt_boxes_all,
                    'labels': torch.cat(labels_to_cat_all).long()
                }

        # Prepare targets for Detector (both 1-class and N-class) 
        if 'detection' in self.config['active_tasks']:
            det_targets_list = []
            num_det_classes = self.config.get('num_classes_detection', 1)
            
            for i in range(len(batch['image'])):
                bboxes_yolo = batch['bboxes'][i]
                labels_yolo = batch['bbox_labels'][i]

                # If single-class detection, filter to only keep 'infected' (label 0)
                if num_det_classes == 1 and len(labels_yolo) > 0:
                    infected_mask = (labels_yolo == 0)
                    bboxes_yolo = bboxes_yolo[infected_mask]
                
                if len(bboxes_yolo) > 0:
                    cx, cy, w, h = bboxes_yolo.unbind(1)
                    x1, y1, x2, y2 = (cx-w/2)*img_w, (cy-h/2)*img_h, (cx+w/2)*img_w, (cy+h/2)*img_h
                    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                    
                    if num_det_classes == 1:
                        labels = torch.ones(len(boxes_xyxy), dtype=torch.int64)
                    else:
                        labels = labels_yolo.long() + 1
                    
                    det_targets_list.append({"boxes": boxes_xyxy.to(self.device), "labels": labels.to(self.device)})
                else:
                    det_targets_list.append({"boxes": torch.empty((0, 4), device=self.device), "labels": torch.empty(0, dtype=torch.int64, device=self.device)})
            
            task_targets['detection'] = det_targets_list
        
        if 'segmentation' in self.config['active_tasks']:
            task_targets['segmentation'] = batch['mask'].unsqueeze(1).to(self.device)

        return task_targets

    @staticmethod
    def mttl_collate_fn(batch):
        keys = batch[0].keys()
        collated_batch = {key: [item[key] for item in batch] for key in keys}
        for key, value_list in collated_batch.items():
            if key in ['image', 'mask', 'heatmap','localization'] and isinstance(value_list[0], torch.Tensor):
                collated_batch[key] = torch.stack(value_list)
            elif isinstance(value_list[0], (int, float)):
                 collated_batch[key] = torch.tensor(value_list)
        return collated_batch

    def _filter_batch_for_2_class(self, batch):
        tasks_that_need_filtering = {'detection', 'roi_classif', 'cell_classif'}
        if not tasks_that_need_filtering.intersection(self.config['active_tasks']):
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
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        classif_iterator = iter(itertools.cycle(self.train_loader_classif)) if self.train_loader_classif and 'cell_classif' in self.config['active_tasks'] else None
        
        pbar = tqdm(enumerate(self.train_loader_mttl), total=len(self.train_loader_mttl), desc="Training MTTL")
        
        self.optimizer.zero_grad()

        for i, mttl_batch in pbar:
            if self.num_classes == 2:
                mttl_batch = self._filter_batch_for_2_class(mttl_batch)
            loss_inputs = {}
            loss_targets = {}
            multiscale_features_cache = None

            active_full_image_tasks = [task for task in ['detection', 'segmentation', 'severity', 'roi_classif', 'heatmap', 'localization'] if task in self.config['active_tasks']]
            
            if active_full_image_tasks:
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    images_full = mttl_batch['image'].to(self.device)
                    multiscale_features_cache = self.model.backbone(images_full, task=None, return_multiscale=True)
                    fpn_features = None
                    if self.model.fpn:
                        fpn_input = [multiscale_features_cache[f'c{i}'] for i in range(2, 5)]
                        fpn_features = self.model.fpn(fpn_input)
                        
                    # Get all ground truth targets
                    task_gts = self._prepare_task_targets(mttl_batch)
                    
                    if 'detection' in active_full_image_tasks and fpn_features: 
                        #det_targets = self.model._prepare_rcnn_detection_targets(mttl_batch)
                        det_targets = task_gts.get('detection')
                        det_loss_dict = self.model.task_heads['detection'](fpn_features, images_full, det_targets)
                        det_loss = sum(l for l in det_loss_dict.values())
                        loss_inputs['detection_loss'] = det_loss
                        
                    if 'heatmap' in active_full_image_tasks:
                        heatmap_logits = self.model.task_heads['heatmap'](multiscale_features_cache)
                        loss_inputs['heatmap'] = heatmap_logits
                        loss_targets['heatmap'] = task_gts.get('heatmap')

                    if 'segmentation' in active_full_image_tasks:
                        seg_logits = self.model.task_heads['segmentation'](multiscale_features_cache)
                        loss_inputs['segmentation'] = seg_logits
                        loss_targets['segmentation'] = task_gts.get('segmentation')
                    
                    if 'severity' in active_full_image_tasks:
                        global_features = F.adaptive_avg_pool2d(multiscale_features_cache['c4'], (1, 1)).flatten(1)
                        sev_logits = self.model.task_heads['severity'](global_features)
                        loss_inputs['severity'] = sev_logits
                        loss_targets['severity_class'] = mttl_batch['severity_class'].to(self.device)
                        
                    if 'roi_classif' in self.config['active_tasks'] and 'roi_classif' in task_gts:
                        roi_gts = task_gts['roi_classif']
                        pooled_features = self.model.roi_classifier_pooler(fpn_features, roi_gts['boxes'], [(images_full.shape[-2], images_full.shape[-1])] * len(images_full))
                        feature_vectors = pooled_features.flatten(start_dim=1)
                        roi_logits = self.model.task_heads['cell_classif'](feature_vectors)
                        loss_inputs['roi_classif'] = roi_logits
                        loss_inputs['roi_classif_labels'] = roi_gts['labels']

            if 'cell_classif' in self.config['active_tasks'] and classif_iterator:
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    classif_batch = next(classif_iterator)
                    images_patch = classif_batch['image'].to(self.device)
                    labels = classif_batch['label'].to(self.device)
                    
                    feature_vectors = self.model.backbone(images_patch, task='cell_classif', return_multiscale=False)
                    classif_logits = self.model.task_heads['cell_classif'](feature_vectors)
                    
                    loss_inputs['cell_classif'] = classif_logits
                    loss_targets['label'] = labels

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Grab the 'uncertainty_info' dictionary
                combined_loss, loss_breakdown, uncertainty_info = self.model.loss_function(loss_inputs, loss_targets)
                
                if 'roi_classif' in loss_inputs: 
                    loss_breakdown['roi_classif_raw'] = self.model.loss_function.enhanced_cell_classif_loss(loss_inputs['roi_classif'], loss_inputs['roi_classif_labels']).item()
                
                # Add the sigma values to the dictionary for display and history tracking
                for task, u_info in uncertainty_info.items():
                    loss_breakdown[f'sigma_{task}'] = u_info['sigma']
                    self.step_history[f'step_sigma_{task}'].append(u_info['sigma'])
                
                if 'detection_loss' in loss_inputs: loss_breakdown['detection_raw'] = loss_inputs['detection_loss'].item()
                if 'cell_classif' in loss_inputs: loss_breakdown['classif_raw'] = self.model.loss_function.enhanced_cell_classif_loss(loss_inputs['cell_classif'], loss_targets['label']).item()
                if 'segmentation' in loss_inputs: loss_breakdown['segmentation_raw'] = self.model.loss_function.enhanced_segmentation_loss(loss_inputs['segmentation'], loss_targets['segmentation']).item()
                if 'severity' in loss_inputs: loss_breakdown['severity_raw'] = self.model.loss_function.enhanced_severity_loss(loss_inputs['severity'], loss_targets['severity_class']).item()
                if 'heatmap' in loss_inputs:
                    heatmap_targets = loss_targets.get('heatmap')
                    if heatmap_targets is not None:
                        loss_breakdown['heatmap_raw'] = self.model.loss_function.enhanced_heatmap_loss(
                            loss_inputs['heatmap'], heatmap_targets
                        ).item()
                
            self.scaler.scale(combined_loss / self.grad_accumulation_steps).backward()

            if (i + 1) % self.grad_accumulation_steps == 0 or (i + 1) == len(self.train_loader_mttl):
                
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                is_grad_finite = all(
                    torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None
                )
                if is_grad_finite:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    print(f"\n[CRITICAL TRAINING WARNING] Detected inf/NaN gradients at step {i}. Skipping optimizer update to prevent model corruption.")
                    self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                if self.scheduler:
                    self.scheduler.step()
            
            total_loss += combined_loss.item()
            pbar.set_postfix({k.replace('_weighted','_w').replace('_raw','').replace('sigma_','s_'): f"{v:.2f}" for k, v in loss_breakdown.items()})
        
        return total_loss / len(self.train_loader_mttl) if len(self.train_loader_mttl) > 0 else 0
    
    def validate_epoch(self):
        self.model.eval()
        all_task_outputs, all_task_targets = defaultdict(list), defaultdict(list)
        total_val_loss = 0.0
        pbar = tqdm(self.val_loader, desc="Validating MTTL")
        
        with torch.no_grad():
            for batch in pbar:
                if self.config.get('num_classes_detection', 3) == 2:
                    batch = self._filter_batch_for_2_class(batch)
                
                # Perform the forward pass and prepare ground truths
                outputs = self.model.forward_mttl(batch, is_train=False)
                task_gts = self._prepare_task_targets(batch)

                # Prepare targets dictionary for the loss function
                targets_for_loss = {}
                if 'segmentation' in self.config['active_tasks'] and 'segmentation' in task_gts:
                    targets_for_loss['segmentation'] = task_gts.get('segmentation')
                    
                if 'heatmap' in self.config['active_tasks'] and 'heatmap' in task_gts:
                    targets_for_loss['heatmap'] = task_gts.get('heatmap')
                
                if 'severity' in self.config['active_tasks'] and 'severity_class' in batch:
                    targets_for_loss['severity_class'] = batch['severity_class'].to(self.device)
                    
                if 'cell_classif' in self.config['active_tasks'] and 'cell_classif_labels' in outputs:
                    targets_for_loss['label'] = outputs['cell_classif_labels']
                    
                if 'roi_classif' in self.config['active_tasks'] and 'roi_classif' in task_gts:
                    outputs['roi_classif_labels'] = task_gts['roi_classif']['labels']

                # Calculate the validation loss using all available outputs
                loss, _, _ = self.model.compute_loss(outputs, targets_for_loss)
                if torch.isfinite(loss):
                    total_val_loss += loss.item()
                
                # Aggregate all
                for task in self.config['active_tasks']:
                    if task in outputs:
                        if task == 'detection':
                            all_task_outputs['detection'].extend(outputs['detection'])
                            all_task_targets['detection'].extend(task_gts.get('detection', []))
                            
                        elif task == 'heatmap' or task == 'localization':
                            all_task_outputs['heatmap'].append(outputs['heatmap'].cpu())
                            all_task_targets['heatmap'].append(batch['heatmap'])
                        
                        elif task == 'roi_classif' and 'roi_classif' in outputs:
                            all_task_outputs['roi_classif'].append(outputs['roi_classif'].cpu())
                            all_task_targets['roi_classif'].append(outputs['roi_classif_labels'].cpu())

                        elif task == 'segmentation':
                            all_task_outputs['segmentation'].append(outputs['segmentation'].cpu())
                            all_task_targets['segmentation'].append(batch['mask'])

                        elif task == 'severity':
                            all_task_outputs['severity'].append(outputs['severity'].cpu())
                            all_task_targets['severity'].append(batch['severity_class'])

                        elif task == 'cell_classif' and 'cell_classif' in outputs:
                            all_task_outputs['cell_classif'].append(outputs['cell_classif'].cpu())
                            all_task_targets['cell_classif'].append(outputs['cell_classif_labels'].cpu())


        avg_val_loss = total_val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        final_metrics = {}
        for task in self.config['active_tasks']:
            if not all_task_outputs[task]: continue
            
            preds_batches, gts_batches = all_task_outputs[task], all_task_targets[task]
            
            if task == 'detection':
                self.map_metric.reset()
                self.map_metric.update(preds_batches, gts_batches)
                det_metrics = self.map_metric.compute()
                final_metrics['detection'] = {
                    'map_50': det_metrics.get('map_50', torch.tensor(-1.0)).item(),
                    'map_75': det_metrics.get('map_75', torch.tensor(-1.0)).item(),
                    'mar_100': det_metrics.get('mar_100', torch.tensor(-1.0)).item()
                }
            
            elif task in ['cell_classif', 'severity', 'roi_classif']:
                preds_full = torch.cat(preds_batches)
                gts_full = torch.cat(gts_batches).numpy()
                preds_np = torch.argmax(preds_full, dim=1).numpy()
                report = classification_report(gts_full, preds_np, output_dict=True, zero_division=0)
                final_metrics[task] = {
                    'accuracy': report.get('accuracy', 0.0),
                    'f1_macro': report.get('macro avg', {}).get('f1-score', 0.0)
                }
                
            elif task == 'heatmap' or task == 'localization':
                preds_tensor = torch.cat(preds_batches)
                gts_tensor = torch.cat(gts_batches)
                final_metrics['heatmap'] = compute_localization_metrics(preds_tensor, gts_tensor)
            
            elif task == 'segmentation':
                preds_tensor = torch.cat(preds_batches)
                gts_tensor = torch.cat(gts_batches)
                final_metrics['segmentation'] = compute_segmentation_metrics(preds_tensor, gts_tensor)
                
        return avg_val_loss, final_metrics
    
    def train(self):
        num_epochs = self.config['num_epochs']
        patience = self.config.get('patience', 15)
        primary_tasks_weights = self.config.get('primary_tasks_weights', {})
        
        best_task_metrics = defaultdict(lambda: -1.0)
        best_composite_score = -1.0
        patience_counter = 0

        task_primary_metrics = {
            'detection': 'map_50',
            'cell_classif': 'f1_macro',
            'segmentation': 'dice',
            'roi_classif': 'f1_macro',
            'severity': 'f1_macro',
            'heatmap': 'dice',
            'localization': 'dice'
        }
        
        task_to_head_map = {
            'detection': 'detection',
            'segmentation': 'segmentation',
            'severity': 'severity',
            'cell_classif': 'cell_classif',
            'roi_classif': 'cell_classif',
            'heatmap': 'heatmap',
            'localization': 'heatmap'
        }

        for epoch in range(num_epochs):
            start_time = time.time()
            train_loss = self.train_epoch()
            val_loss, val_metrics_all_tasks = self.validate_epoch()
            epoch_time = time.time() - start_time
            
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) ---")
            print(f"  Avg Train Loss: {train_loss:.4f} | Avg Val Loss: {val_loss:.4f}")
            
            self.epoch_history['epoch_train_loss'].append(train_loss)
            self.epoch_history['epoch_val_loss'].append(val_loss)
            composite_score = 0.0
            
            for task, metrics in val_metrics_all_tasks.items():
                print(f"  Validation - {task.upper()}:")
                for metric_name, value in metrics.items():
                    print(f"    - {metric_name}: {value:.4f}")
                    self.epoch_history[f'val_{task}_{metric_name}'].append(value)
                
                weight = primary_tasks_weights.get(task, 0.0)
                primary_metric_key = task_primary_metrics.get(task)
                if primary_metric_key and primary_metric_key in metrics:
                    composite_score += metrics[primary_metric_key] * weight

                if primary_metric_key:
                    current_metric_val = metrics.get(primary_metric_key, -1.0)
                    if current_metric_val > best_task_metrics[task]:
                        best_task_metrics[task] = current_metric_val
                        
                        head_key_to_save = task_to_head_map.get(task)
                        
                        if head_key_to_save and head_key_to_save in self.model.task_heads:
                            head_save_path = os.path.join(self.manager.current_experiment_dir, f'best_{task}_head.pth')
                            torch.save(self.model.task_heads[head_key_to_save].state_dict(), head_save_path)
                            print(f">>> New best for task '{task}'! Head saved to '{head_save_path}' ({primary_metric_key}: {current_metric_val:.4f})")

            print(f"\nComposite Validation Score: {composite_score:.4f} (Best: {best_composite_score:.4f})")
            
            if composite_score > best_composite_score:
                print(f">>> New best for COMPOSITE score! Saving full model to '{self.model_save_path}'...")
                best_composite_score = composite_score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_save_path)
            else:
                patience_counter += 1
                print(f"No improvement in composite score. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
        
        print("\n--- Training Complete ---")
        print("Loading best model (based on composite score) for final evaluation...")
        if os.path.exists(self.model_save_path):
            self.model.load_state_dict(torch.load(self.model_save_path))
        print("Best task-specific heads have also been saved individually.")

    def plot_training_history(self):
        active_tasks = self.config['active_tasks']
        num_tasks = len(active_tasks)
        
        num_plots = 2 + num_tasks
        num_rows = (num_plots + 1) // 2
        
        fig, axes = plt.subplots(num_rows, 2, figsize=(20, 6 * num_rows), constrained_layout=True)
        axes = axes.flatten()
        
        fig.suptitle('MTTL Training Analysis', fontsize=20, y=1.03)

        ax_loss = axes[0]
        ax_loss.plot(self.epoch_history['epoch_train_loss'], label='Avg Train Loss', color='royalblue', lw=2)
        ax_loss.plot(self.epoch_history['epoch_val_loss'], label='Avg Val Loss', color='darkorange', lw=2)
        ax_loss.set_title('Overall Model Loss', fontsize=14)
        ax_loss.set_ylabel('Loss'); ax_loss.set_xlabel('Epoch'); ax_loss.legend(); ax_loss.grid(True)
        
        ax_sigma = axes[1]
        for task in active_tasks:
            key = f'step_sigma_{task}'
            if key in self.step_history and self.step_history[key]:
                
                s = pd.Series(self.step_history[key]).rolling(window=max(1, len(self.train_loader_mttl)//10)).mean()
                x_axis = np.linspace(0, len(self.epoch_history['epoch_train_loss']), len(s))
                ax_sigma.plot(x_axis, s, label=f'Sigma ({task})')
        ax_sigma.set_title('Task Uncertainty (Sigma)', fontsize=14)
        ax_sigma.set_ylabel('Value'); ax_sigma.set_xlabel('Epoch'); ax_sigma.legend(); ax_sigma.grid(True)
        ax_sigma.set_yscale('log')
        
        plot_idx = 2
        for task in active_tasks:
            ax_metric = axes[plot_idx]
            ax_metric.set_title(f'Validation Metrics: {task.upper()}', fontsize=14)
            ax_metric.set_ylabel('Score'); ax_metric.set_xlabel('Epoch'); ax_metric.grid(True)
            
            has_data = False
            for metric in self.epoch_history.keys():
                if metric.startswith(f'val_{task}_'):
                    metric_name = metric.replace(f'val_{task}_', '')
                    ax_metric.plot(self.epoch_history[metric], 'o-', label=f'{metric_name.upper()}', markersize=4)
                    has_data = True
            
            if has_data:
                ax_metric.legend()
                plot_idx += 1
            else:
                ax_metric.set_visible(False) 
                
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
            
        return fig