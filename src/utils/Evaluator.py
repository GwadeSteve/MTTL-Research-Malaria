# ====================================================================
# SYSTEM EVALUATOR FOR STL/MTTL
# Gwade Steve
# MTTL and Application to Malaria Detection
# April 2025
# ====================================================================
import os
from sklearn import metrics
import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_recall_fscore_support
from scipy.stats import pearsonr
import json
from experiment_manager import ExperimentManager
from DataUtils import FlexibleMalariaDataset, crop_cell_from_image, seed_worker
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

    map_metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=iou_thresholds)
    map_metric.update(preds, gts)
    result = map_metric.compute()
    metrics = {}
    for idx, thr in enumerate(iou_thresholds):
        key = f"mAP_{thr:.2f}"
        metrics[key] = result["map"].cpu().numpy()[idx] if "map" in result else 0.0

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
    if pred_heatmaps is None or target_heatmaps is None:
        return {
            'mse': float('inf'), 
            'mae': float('inf'), 
            'dice': 0.0, 
            'iou': 0.0, 
            'correlation': 0.0
        }

    pred_heatmaps = pred_heatmaps.float()
    target_heatmaps = target_heatmaps.float()
    
    pred_flat = pred_heatmaps.view(-1)
    target_flat = target_heatmaps.view(-1)
    
    # Checking on if perfect negatives are well handled
    target_sum = target_flat.sum().item()
    pred_sum = pred_flat.sum().item()
    target_max = target_flat.max().item()
    pred_max = pred_flat.max().item()
    
    print(f"\n--- Heatmap Metric Debug ---")
    print(f"Target Sum: {target_sum:.8f} | Target Max: {target_max:.8f}")
    print(f"Pred Sum: {pred_sum:.8f} | Pred Max: {pred_max:.8f}")

    target_is_empty = target_sum < 1e-6
    pred_is_empty = pred_sum < 1e-6
    
    print(f"Is Target Empty? -> {target_is_empty}")
    print(f"Is Pred Empty?   -> {pred_is_empty}")

    if target_is_empty and pred_is_empty:
        print("True Negative case triggered. Returning perfect scores.")
        return {'mse': 0.0, 'mae': 0.0, 'dice': 1.0, 'iou': 1.0, 'correlation': 1.0}

    # calculate metrics    
    mse = nn.MSELoss()(pred_heatmaps, target_heatmaps).item()
    mae = nn.L1Loss()(pred_heatmaps, target_heatmaps).item()
    
    intersection = (pred_flat * target_flat).sum()
    union_sum = pred_flat.sum() + target_flat.sum()
    dice = (2.0 * intersection) / (union_sum + 1e-6)
    
    intersection_binary = ((pred_flat > 0.5) & (target_flat > 0.5)).sum().float()
    union_binary = ((pred_flat > 0.5) | (target_flat > 0.5)).sum().float()
    iou = intersection_binary / (union_binary + 1e-6)
    
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
        'iou': iou.item(), 
        'correlation': correlation
    }

def compute_segmentation_metrics(pred_logits, target_mask):
    if pred_logits is None or target_mask is None:
        return {'dice': 0.0, 'iou': 0.0}
    
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
    
    pred_class = torch.argmax(pred_logits, dim=1)
    correct_predictions = (pred_class == target_class)
    accuracy = correct_predictions.float().mean().item()
    
    return {'accuracy': accuracy}

def create_task_specific_collate_fn(task):
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

class AdvancedEvaluator:
    
    @staticmethod
    def mttl_collate_fn(batch):
        keys = batch[0].keys()
        collated_batch = {key: [item[key] for item in batch] for key in keys}
        for key, value_list in collated_batch.items():
            if key in ['image', 'mask', 'heatmap'] and isinstance(value_list[0], torch.Tensor):
                collated_batch[key] = torch.stack(value_list)
            elif isinstance(value_list[0], (int, float)):
                 collated_batch[key] = torch.tensor(value_list)
        return collated_batch
    
    def __init__(self, run_directory: str, test_samples, cell_classifier_path: str = None):
        print(f"Initializing evaluator for experiment: {run_directory}")
        if not os.path.isdir(run_directory):
            raise FileNotFoundError(f"Experiment directory not found: {run_directory}")
        
        self.run_directory = run_directory
        self.test_samples = test_samples
        self.manager = ExperimentManager()
        self.manager.current_experiment_dir = run_directory
        
        with open(os.path.join(run_directory, 'config.json'), 'r') as f:
            self.config = json.load(f)
        
        self.task = self.config['active_tasks'][0]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        from model import create_model_for_task, create_multitask_model
        if self.config.get('mode') == 'MTTL':
            self.model = create_multitask_model(self.config, tasks=self.config['active_tasks']).to(self.device)
        else:
            self.model = create_model_for_task(self.config, self.task).to(self.device)
        model_path = os.path.join(run_directory, 'best_model.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Loaded primary model for task '{self.task}' from {run_directory}")

        self.cell_classifier_path = cell_classifier_path
        self.classifier_model = None
        if self.cell_classifier_path:
            print(f"Loading secondary cell classifier from: {self.cell_classifier_path}")
            classifier_run_dir = os.path.dirname(self.cell_classifier_path)
            with open(os.path.join(classifier_run_dir, 'config.json'), 'r') as f:
                classifier_config = json.load(f)
            
            self.classifier_model = create_model_for_task(classifier_config, 'cell_classif').to(self.device)
            self.classifier_model.load_state_dict(torch.load(self.cell_classifier_path, map_location=self.device))
            self.classifier_model.eval()
            print("Cell classifier model loaded successfully.")
            
    def _filter_batch_for_2_class(self, batch):
        if self.task not in ['detection', 'roi_classif']:
            return batch
        original_labels_list = batch.get('labels', batch.get('bbox_labels'))
        original_bboxes_list = batch['bboxes']
        new_labels_list, new_bboxes_list = [], []
        for i in range(len(original_labels_list)):
            labels_tensor, bboxes_tensor = original_labels_list[i], original_bboxes_list[i]
            mask = (labels_tensor != 2)
            new_labels_list.append(labels_tensor[mask])
            new_bboxes_list.append(bboxes_tensor[mask])
        batch['labels'] = new_labels_list
        batch['bbox_labels'] = new_labels_list
        batch['bboxes'] = new_bboxes_list
        return batch

    def run(self, iou_thresh=0.5, conf_thresh=0.5, conf_thresh_vis=0.75, focus_infected=False):
        if self.config.get('mode', 'MTTL') != 'STL':
            print("Warning: .run() called on a non-STL model. Defaulting to .run_mttl()")
            return self.run_mttl(iou_thresh, conf_thresh, conf_thresh_vis)
        is_full_system_eval = False
        run_mode_title = f"EVALUATION FOR '{self.task.upper()}'"
        if self.task == 'detection' and self.classifier_model:
            run_mode_title = "FULL SYSTEM (DETECTOR + CLASSIFIER) EVALUATION"
            is_full_system_eval = True

        print(f"\n{'='*60}\n{run_mode_title}")
        print(f"Using IoU Threshold: {iou_thresh} | Confidence Threshold: {conf_thresh}\n{'='*60}")

        returned_data = self._get_predictions()
        if self.task == 'cell_classif':
            predictions, ground_truth, patches = returned_data
        else:
            predictions, ground_truth = returned_data
            patches = None 
        
        final_metrics = {}
        if self.task == 'detection' and self.classifier_model:
            final_metrics = self._evaluate_full_system(predictions, conf_thresh, conf_thresh_vis)
        elif self.task == 'detection':
            final_metrics = self._evaluate_detection(predictions, ground_truth, iou_thresh, conf_thresh, conf_thresh_vis, focus_infected=focus_infected)
        elif self.task in ['severity', 'cell_classif', 'roi_classif']:
            final_metrics = self._evaluate_classification(predictions, ground_truth, patches)
        elif self.task == 'segmentation':
            final_metrics = self._evaluate_segmentation(predictions, ground_truth)
        elif self.task == 'heatmap' or self.task == 'localization':
            final_metrics = self._evaluate_heatmap(predictions, ground_truth)
        
        print("\nFinal Metrics Summary")
        print(json.dumps(final_metrics, indent=2, default=float))
        
        if self.manager.current_experiment_dir:
            if is_full_system_eval:
                final_report = {
                    'system_performance': final_metrics,
                    'evaluation_mode': 'combined_detector_classifier_system',
                    'components': {}
                }

                try:
                    with open(os.path.join(self.run_directory, 'test_results.json'), 'r') as f:
                        detector_results = json.load(f)
                    with open(os.path.join(self.run_directory, 'metadata.json'), 'r') as f:
                        detector_metadata = json.load(f)
                    final_report['components']['detector'] = {
                        'metadata': detector_metadata,
                        'performance': detector_results
                    }
                except FileNotFoundError:
                    final_report['components']['detector'] = {'error': 'Could not load detector results or metadata.'}

                try:
                    classifier_run_dir = os.path.dirname(self.cell_classifier_path)
                    with open(os.path.join(classifier_run_dir, 'test_results.json'), 'r') as f:
                        classifier_results = json.load(f)
                    with open(os.path.join(classifier_run_dir, 'metadata.json'), 'r') as f:
                        classifier_metadata = json.load(f)
                    final_report['components']['classifier'] = {
                        'metadata': classifier_metadata,
                        'performance': classifier_results
                    }
                except FileNotFoundError:
                    final_report['components']['classifier'] = {'error': 'Could not load classifier results or metadata.'}
                
                output_filename = 'combined_results.json'
                metrics_to_save = final_report

            else:
                output_filename = 'test_results.json'
                metrics_to_save = final_metrics

            self.manager.log_final_results(metrics_to_save, filename=output_filename)
            print(f"\nEvaluation complete. Results logged to '{output_filename}' in {self.run_directory}")
        
        return final_metrics

    def run_mttl(self, iou_thresh=0.5, conf_thresh=0.5, conf_thresh_vis=0.75, focus_infected=False, use_roi_classifier=False):
        if self.config.get('mode', 'STL') != 'MTTL':
            print("Warning: .run_mttl() called on a non-MTTL model. Defaulting to standard .run()")
            return self.run(iou_thresh, conf_thresh, conf_thresh_vis)

        print(f"\n{'='*60}\nCOMPREHENSIVE MTTL EVALUATION")
        print(f"Directory: {self.run_directory}\n{'='*60}")

        active_tasks = self.config.get('active_tasks', [])
        full_report = {}

        main_model_path = os.path.join(self.run_directory, 'best_model.pth')
        if not os.path.exists(main_model_path):
            raise FileNotFoundError(f"Main model file 'best_model.pth' not found in {self.run_directory}")
        
        task_to_head_map = {
            'detection': 'detection',
            'segmentation': 'segmentation',
            'severity': 'severity',
            'cell_classif': 'cell_classif',
            'roi_classif': 'cell_classif',
            'heatmap': 'heatmap',
            'localization': 'heatmap'
        }
        
        # Evaluating det using roi classif
        if use_roi_classifier and 'detection' in active_tasks and 'roi_classif' in active_tasks:
            print("\n" + "="*60)
            print("EVALUATING COMBINED DETECTOR + ROI CLASSIFIER SYSTEM")
            print("="*60)
            self.model.load_state_dict(torch.load(main_model_path, map_location=self.device))
            self.model.eval()
            
            # Use a special method to get the combined system predictions
            system_predictions, ground_truth = self._get_combined_system_predictions(conf_thresh=conf_thresh)
            
            # Evaluate these new predictions as a standard detection task
            system_metrics = self._evaluate_detection(
                system_predictions, ground_truth, 
                iou_thresh, conf_thresh, conf_thresh_vis, 
                focus_infected=focus_infected, is_combined_system=True
            )
            full_report['combined_system'] = system_metrics

        for task in active_tasks:
            if use_roi_classifier and task == 'detection':
                print(f"\n--- Skipping standard '{task.upper()}' evaluation (covered by combined system evaluation) ---")
                continue
            print(f"\n--- Evaluating Task: {task.upper()} ---")
            
            self.model.load_state_dict(torch.load(main_model_path, map_location=self.device))
            
            best_head_path = os.path.join(self.run_directory, f'best_{task}_head.pth')
            if os.path.exists(best_head_path):
                print(f"Loading best head from: {best_head_path}")
                head_state_dict = torch.load(best_head_path, map_location=self.device)
                head_key_to_load = task_to_head_map.get(task)
                if head_key_to_load and head_key_to_load in self.model.task_heads:
                    self.model.task_heads[head_key_to_load].load_state_dict(head_state_dict)
                else:
                    print(f"Warning: Could not find a corresponding head key for task '{task}' in the model.")
            else:
                print(f"Warning: Best head for '{task}' not found. Using head from the main composite model.")
            
            self.model.eval()

            # temporary switch to get GTs and preds
            original_task = self.task
            self.task = task
            returned_data = self._get_predictions()
            self.task = original_task # Reset

            if task == 'cell_classif':
                predictions, ground_truth, patches = returned_data
            else:
                predictions, ground_truth = returned_data
                patches = None 
            
            # evaluate and store metrics
            task_metrics = {}
            if task == 'detection':
                task_metrics = self._evaluate_detection(predictions, ground_truth, iou_thresh, conf_thresh, conf_thresh_vis, focus_infected=focus_infected)
            elif task in ['severity', 'cell_classif', 'roi_classif']:
                task_metrics = self._evaluate_classification(predictions, ground_truth, patches)
            elif task == 'segmentation':
                task_metrics = self._evaluate_segmentation(predictions, ground_truth)
            elif task == 'heatmap' or task == 'localization':
                task_metrics = self._evaluate_heatmap(predictions, ground_truth)
            
            full_report[task] = task_metrics

        print("\n\n" + "="*60)
        print("MTTL FINAL REPORT SUMMARY")
        print("="*60)
        print(json.dumps(full_report, indent=4, default=float))

        if self.manager.current_experiment_dir:
            self.manager.log_final_results(full_report, filename="mttl_test_results.json")
            print(f"\nEvaluation complete. Full report logged to 'mttl_test_results.json' in {self.run_directory}")
        
        return full_report
    
    def _get_combined_system_predictions(self, conf_thresh=0.5):
        # det data format
        temp_dataset = FlexibleMalariaDataset(self.test_samples, task_mode='detection', augment=False)
        collate_fn = create_task_specific_collate_fn('detection')
        temp_loader = DataLoader(temp_dataset, batch_size=self.config['batch_size'], collate_fn=collate_fn , shuffle=False, num_workers=0)
        
        final_predictions = []
        final_ground_truth = []
        img_h, img_w = self.config.get('image_size', (512, 512))
        num_classes = self.config.get('num_classes_detection', 3)

        with torch.no_grad():
            for batch in tqdm(temp_loader, desc="Running combined system inference"):
                if num_classes == 2:
                    batch = self._filter_batch_for_2_class(batch)
                images = batch['image'].to(self.device)
                
                # forward pass backbone adn fpn once
                multiscale_features = self.model.backbone(images, task=None, return_multiscale=True)
                fpn_input = [multiscale_features[f'c{i}'] for i in range(2, 5)]
                fpn_features = self.model.fpn(fpn_input)

                # first detection proposals
                detection_outputs = self.model.task_heads['detection'](fpn_features, images, targets=None)

                # for each image in the batch, classify the detected boxes
                for i, det_output in enumerate(detection_outputs):
                    conf_mask = det_output['scores'] >= conf_thresh
                    boxes_to_classify = [det_output['boxes'][conf_mask]]
                    
                    if boxes_to_classify[0].shape[0] == 0:
                        final_predictions.append({'boxes': torch.empty(0, 4), 'scores': torch.empty(0), 'labels': torch.empty(0, dtype=torch.long)})
                        continue

                    # pool featres for the detected boxes
                    single_image_fpn_features = {k: v[i:i+1] for k, v in fpn_features.items()}
                    pooled_features = self.model.roi_classifier_pooler(single_image_fpn_features, boxes_to_classify, [(img_h, img_w)])

                    # classify the pooled features
                    feature_vectors = pooled_features.flatten(start_dim=1)
                    class_logits = self.model.task_heads['cell_classif'](feature_vectors)
                    class_probs = F.softmax(class_logits, dim=1)
                    
                    # confs from classifier
                    final_scores, final_labels_indices = torch.max(class_probs, dim=1)
                    final_labels = final_labels_indices + 1 # Convert to 1-indexed

                    final_predictions.append({
                        'boxes': boxes_to_classify[0].cpu(),
                        'scores': final_scores.cpu(),
                        'labels': final_labels.cpu()
                    })

                # Prepare GTs just like in _get_predictions
                gt_batch = self._prepare_rcnn_detection_targets(batch)
                final_ground_truth.extend([{k: v.cpu() for k, v in t.items()} for t in gt_batch])
        
        return final_predictions, final_ground_truth

    def _get_predictions(self):
        
        if self.config.get('mode') == 'MTTL':
            dataset_task_mode = 'multi_task'
            collate_fn = self.mttl_collate_fn 
        else: 
            dataset_task_mode = self.task
            if self.task == 'roi_classif': dataset_task_mode = 'detection'
            if self.task == 'heatmap': dataset_task_mode = 'localization'
            collate_fn = None if self.task == 'cell_classif' else create_task_specific_collate_fn(dataset_task_mode)
            
        temp_dataset = FlexibleMalariaDataset(self.test_samples, task_mode=dataset_task_mode, augment=False)
    
        g = torch.Generator()
        g.manual_seed(self.config.get('seed', 12))
        temp_loader = DataLoader(
            temp_dataset, batch_size=self.config['batch_size'], shuffle=False, 
            num_workers=0, pin_memory=True, collate_fn=collate_fn, worker_init_fn=seed_worker, generator=g
        )
        
        num_classes = self.config.get('num_classes_detection', 3)
        
        all_preds, all_gt = [], []
        all_patches = [] 
        total_inference_time = 0
        num_batches = 0
        num_images = 0
        with torch.no_grad():
            for batch in tqdm(temp_loader, desc=f"Generating predictions for '{self.task}'"):
                if num_classes == 2:
                    batch = self._filter_batch_for_2_class(batch)
                start_time = time.time()
                images = batch['image'].to(self.device)
                if self.config.get('mode') == 'MTTL':
                    outputs = self.model.forward_mttl(batch, is_train=False)
                else:
                    outputs = self.model(images, targets=batch if self.task == 'roi_classif' else None)
                
                end_time = time.time()
                num_images += images.shape[0]

                if self.task == 'cell_classif':
                    for img_tensor in images:
                        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img_np = (img_np * std) + mean
                        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                        all_patches.append(img_np)
                    all_preds.extend(list(torch.unbind(F.softmax(outputs['cell_classif'], dim=1).cpu(), dim=0)))
                    all_gt.extend(batch['label'].tolist())
                
                elif self.task == 'roi_classif':
                    # The model's output already contains the predictions and corresponding labels
                    probs = F.softmax(outputs['roi_classif'], dim=1)
                    all_preds.extend(list(torch.unbind(probs.cpu(), dim=0)))
                    all_gt.extend(outputs['roi_classif_labels'].cpu().tolist())
                
                elif self.task == 'heatmap' or self.task == 'localization':
                    preds_batch = outputs['heatmap']
                    all_preds.extend(list(torch.unbind(preds_batch.cpu(), dim=0)))
                    all_gt.extend(list(torch.unbind(batch['heatmap'], dim=0)))
                elif self.task == 'detection':
                    total_inference_time += (end_time - start_time)
                    num_batches += images.shape[0]
                    preds_batch = outputs['detection']
                    gt_batch = self._prepare_rcnn_detection_targets(batch)
                    all_preds.extend([{k: v.cpu() for k, v in p.items()} for p in preds_batch])
                    all_gt.extend([{k: v.cpu() for k, v in t.items()} for t in gt_batch])
                elif self.task == 'severity':
                    all_preds.extend(list(torch.unbind(F.softmax(outputs['severity'], dim=1).cpu(), dim=0)))
                    all_gt.extend(batch['severity_class'].tolist())
                elif self.task == 'segmentation':
                    preds_batch_logits = outputs['segmentation']
                    all_preds.extend(list(torch.unbind(preds_batch_logits.cpu(), dim=0)))
                    all_gt.extend(list(torch.unbind(batch['mask'], dim=0)))
               
        if self.task in ['detection', 'roi_classif']:
            self.avg_inference_time_ms = (total_inference_time / num_images) * 1000    
        if self.task == 'cell_classif':
            return all_preds, all_gt, all_patches
        return all_preds, all_gt

    def _evaluate_heatmap(self, predictions, targets):
        preds_tensor = torch.stack(predictions)
        gt_tensor = torch.stack(targets)
        metrics = compute_localization_metrics(preds_tensor, gt_tensor)
        
        print("\n--- Heatmap Localization Metrics ---")
        print(f"Mean Squared Error (MSE): {metrics.get('mse', 0):.4f}")
        print(f"Mean Absolute Error (MAE): {metrics.get('mae', 0):.4f}")
        print(f"Dice Score (Similarity): {metrics.get('dice', 0):.4f}")
        print(f"Spatial Correlation: {metrics.get('correlation', 0):.4f}")
        
        self._visualize_heatmap(predictions, targets)
        
        return metrics
    
    def _visualize_heatmap(self, predictions, targets):
        num_images_to_show = min(len(self.test_samples), 3)
        if num_images_to_show == 0:
            return

        fig, axes = plt.subplots(num_images_to_show, 4, figsize=(20, num_images_to_show * 5), dpi=100)
        if num_images_to_show == 1:
            axes = axes.reshape(1, -1)
            
        if num_images_to_show == 2:
            axes = np.array([axes])

        for i in range(num_images_to_show):
            original_image = self.test_samples[i]['image']
            gt_heatmap_low_res = targets[i].squeeze()
            pred_probs_low_res = predictions[i].squeeze()
            
            target_size = (original_image.shape[0], original_image.shape[1])

            # upsample both
            gt_heatmap_upsampled = F.interpolate(
                gt_heatmap_low_res.unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode='bilinear', align_corners=False
            ).squeeze().numpy()

            pred_heatmap_upsampled = F.interpolate(
                pred_probs_low_res.unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode='bilinear', align_corners=False
            ).squeeze().numpy()
            
            is_negative_sample = gt_heatmap_low_res.max().item() < 0.1 
            fig_title = f"Image ID: {self.test_samples[i]['image_id']}"
            if is_negative_sample:
                fig_title += " (Ground Truth: Negative)"
            axes[i, 0].set_title(fig_title, loc='left', fontsize=14, pad=20, x=-0.1, y=1.1)
            
            # plot
            axes[i, 0].imshow(original_image)
            axes[i, 0].set_title("Original Image")
            axes[i, 0].set_axis_off()

            # GT
            axes[i, 1].imshow(gt_heatmap_upsampled, cmap='hot', vmin=0, vmax=1)
            axes[i, 1].set_title("Ground Truth (Upsampled)")
            axes[i, 1].set_axis_off()

            # PRED
            axes[i, 2].imshow(pred_heatmap_upsampled, cmap='hot', vmin=0, vmax=1)
            axes[i, 2].set_title("Predicted Heatmap (Upsampled)")
            axes[i, 2].set_axis_off()

            # Overlay
            axes[i, 3].imshow(original_image)
            axes[i, 3].imshow(pred_heatmap_upsampled, cmap='hot', alpha=0.6, vmin=0, vmax=1)
            axes[i, 3].set_title("Prediction Overlay")
            axes[i, 3].set_axis_off()

        plt.tight_layout()
        self.manager.save_plot(fig, "heatmap_visualizations")
        plt.show()

    def _evaluate_full_system(self, detection_predictions, conf_thresh, conf_thresh_vis):
        print("\n--- Evaluating Full Detector + Classifier System ---")
        
        final_preds_for_metric = []
        final_gt_for_metric = []
        patch_transform = A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])
        test_samples_map = {sample['image_id']: sample for sample in self.test_samples}
        
        total_processing_time = 0
        total_images = len(detection_predictions)

        for i, pred_dict in enumerate(tqdm(detection_predictions, desc="Classifying detected cells")):
            start_time = time.time()
            original_image_np = self.test_samples[i]['image']
            conf_mask = pred_dict['scores'] >= conf_thresh
            boxes_xyxy = pred_dict['boxes'][conf_mask]
            
            if len(boxes_xyxy) == 0:
                final_preds_for_metric.append({'boxes': torch.empty(0, 4), 'scores': torch.empty(0), 'labels': torch.empty(0, dtype=torch.long)})
                end_time = time.time()
                total_processing_time += (end_time - start_time)
                continue

            cell_patches = [crop_cell_from_image(original_image_np, box.numpy()) for box in boxes_xyxy]
            valid_indices = [idx for idx, p in enumerate(cell_patches) if p is not None]
            
            if not valid_indices:
                final_preds_for_metric.append({'boxes': torch.empty(0, 4), 'scores': torch.empty(0), 'labels': torch.empty(0, dtype=torch.long)})
                end_time = time.time()
                total_processing_time += (end_time - start_time)
                continue

            valid_patches = [cell_patches[idx] for idx in valid_indices]
            valid_boxes = boxes_xyxy[valid_indices]
            patch_tensors = torch.stack([patch_transform(image=p)['image'] for p in valid_patches]).to(self.device)
            
            with torch.no_grad():
                class_logits = self.classifier_model(patch_tensors)['cell_classif']
                class_probs = F.softmax(class_logits, dim=1)
            
            pred_scores, pred_labels_indices = torch.max(class_probs, dim=1)
            pred_labels = pred_labels_indices + 1
            
            final_preds_for_metric.append({'boxes': valid_boxes.cpu(), 'scores': pred_scores.cpu(), 'labels': pred_labels.cpu()})
            end_time = time.time()
            total_processing_time += (end_time - start_time)

        for sample in self.test_samples:
            gt_bboxes_yolo = torch.from_numpy(sample['detection']['bboxes']).float()
            gt_labels = torch.from_numpy(sample['detection']['labels']).long() + 1
            img_h, img_w, _ = sample['image'].shape
            cx, cy, w, h = gt_bboxes_yolo.unbind(1)
            x1, y1 = (cx - w / 2) * img_w, (cy - h / 2) * img_h
            x2, y2 = (cx + w / 2) * img_w, (cy + h / 2) * img_h
            final_gt_for_metric.append({'boxes': torch.stack([x1, y1, x2, y2], dim=1), 'labels': gt_labels})
            
        print("\n--- Final 3-Class System Performance (mAP) ---")
        map_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
        map_metric.update(final_preds_for_metric, final_gt_for_metric)
        final_results = map_metric.compute()

        avg_classification_time_ms = (total_processing_time / total_images) * 1000
        detector_time_ms = getattr(self, 'avg_inference_time_ms', 0)
        total_system_time_ms = detector_time_ms + avg_classification_time_ms
        
        metrics = {
            'mAP_50': final_results['map_50'].item(), 'mAP_75': final_results['map_75'].item(),
            'mAP_per_class_50': final_results['map_per_class'].tolist(),
            'mAR_100_per_class_50': final_results['mar_100_per_class'].tolist(),
            'avg_detector_time_ms': detector_time_ms,
            'avg_classifier_stage_time_ms': avg_classification_time_ms,
            'avg_total_system_time_ms': total_system_time_ms
        }
        
        self._visualize_detection(final_preds_for_metric, final_gt_for_metric, conf_thresh_vis, is_combined=True)
        print(f"\n--- System Inference Speed Breakdown ---")
        print(f"Detector Stage:     {detector_time_ms:.2f} ms")
        print(f"Classifier Stage:   {avg_classification_time_ms:.2f} ms")
        print(f"------------------------------------")
        print(f"Total End-to-End Time: {total_system_time_ms:.2f} ms per image")
        
        return metrics

    def _evaluate_classification(self, pred_probs, true_labels, patches=None):
        if self.task == 'severity':
            class_names = ['negative', 'low', 'moderate', 'high']
        else: 
            num_classes = self.config.get('num_classes_classif', 3)
            full_class_list = ['Infected', 'Healthy', 'WBC']
            class_names = full_class_list[:num_classes]
        true_labels = np.array(true_labels)
        pred_probs = torch.stack(pred_probs).numpy()
        pred_labels = np.argmax(pred_probs, axis=1)
        possible_labels = list(range(len(class_names)))
        
        print("\n--- Classification Report ---")
        print(classification_report(true_labels, pred_labels, target_names=class_names, labels=possible_labels, zero_division=0))
        report = classification_report(true_labels, pred_labels, target_names=class_names, labels=possible_labels, output_dict=True, zero_division=0)
        
        metrics = {'accuracy': report['accuracy'], 'f1_macro': report['macro avg']['f1-score']}
        for name in class_names:
            metrics[f"f1_{name.lower().replace(' ', '_')}"] = report.get(name, {}).get('f1-score', 0)
        
        self._visualize_classification_plots(true_labels, pred_probs, class_names, patches)
        return metrics
    
    def _evaluate_detection(self, predictions, targets, iou_thresh, conf_thresh, conf_thresh_vis, focus_infected=True, is_combined_system=False):
        num_classes = self.config.get('num_classes_detection', 1)
        is_multiclass = num_classes > 1
        full_class_list = ['Infected', 'Healthy', 'WBC']
        class_names = full_class_list[:num_classes] if is_multiclass else ['Infected']
        print(f"\n--- Evaluating in {'Multi-Class' if is_multiclass else 'Single-Class'} detection mode ---")
        
        metrics = {}
        
        title_prefix = "Combined System" if is_combined_system else "Detector-Only"

        targets_for_eval = []
        for t in targets:
            gt_boxes = t['boxes']
            gt_labels = t['labels']

            if not is_multiclass and len(gt_labels) > 0:
                infected_mask = (gt_labels == 1)
                gt_boxes = gt_boxes[infected_mask]
                gt_labels = gt_labels[infected_mask]
            
            targets_for_eval.append({'boxes': gt_boxes, 'labels': gt_labels})

        targets_for_metric = []
        for t in targets_for_eval:
            t_copy = t.copy()
            t_copy["labels"] = t_copy["labels"] - 1 
            targets_for_metric.append(t_copy)
            
        preds_for_metric = []
        for p in predictions:
            p_copy = p.copy()
            p_copy["labels"] = p_copy["labels"] - 1
            preds_for_metric.append(p_copy)
        
        map_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=is_multiclass, iou_thresholds=[0.5, 0.75])
        map_metric.update(preds_for_metric, targets_for_metric)
        map_results = map_metric.compute()
        
        metrics.update({'mAP_50': map_results['map_50'].item(), 'mAP_75': map_results['map_75'].item()})
        print(f"\n--- Overall Performance (mAP) ({title_prefix}) ---")
        print(f"Overall mAP@50: {metrics['mAP_50']:.4f}, Overall mAP@75: {metrics['mAP_75']:.4f}")
        
        optimal_metrics = self._analyze_and_plot_thresholds(predictions, targets_for_eval, iou_thresh)
        metrics.update(optimal_metrics)
        optimal_conf = optimal_metrics.get('optimal_confidence_threshold', conf_thresh)
        
        class_stats = {i: {'tp': 0, 'fp': 0, 'fn': 0, 'count_errors': [], 'total_gt_count': 0} for i in range(num_classes)}

        for pred_dict, target_dict in zip(preds_for_metric, targets_for_metric):
            conf_mask = pred_dict['scores'] >= optimal_conf
            
            for class_idx in range(num_classes):
                pred_labels_c = pred_dict['labels'][conf_mask]
                target_labels_c = target_dict['labels']
                pred_count = (pred_labels_c == class_idx).sum().item()
                gt_count = (target_labels_c == class_idx).sum().item()
                class_stats[class_idx]['count_errors'].append(abs(pred_count - gt_count))
                class_stats[class_idx]['total_gt_count'] += gt_count

                pred_mask = (pred_dict['labels'][conf_mask] == class_idx)
                target_mask = (target_dict['labels'] == class_idx)
                pred_boxes_c = pred_dict['boxes'][conf_mask][pred_mask]
                target_boxes_c = target_dict['boxes'][target_mask]

                if len(target_boxes_c) == 0:
                    class_stats[class_idx]['fp'] += len(pred_boxes_c)
                    continue
                if len(pred_boxes_c) == 0:
                    class_stats[class_idx]['fn'] += len(target_boxes_c)
                    continue

                iou_matrix = box_iou(pred_boxes_c, target_boxes_c)
                if iou_matrix.numel() > 0:
                    matched_gts = torch.zeros(len(target_boxes_c), dtype=torch.bool)
                    max_iou_per_pred, max_idx_per_pred = iou_matrix.max(dim=1)
                    
                    tp_c = 0
                    for pred_idx, iou in enumerate(max_iou_per_pred):
                        if iou >= iou_thresh:
                            gt_idx = max_idx_per_pred[pred_idx]
                            if not matched_gts[gt_idx]:
                                tp_c += 1
                                matched_gts[gt_idx] = True
                    
                    class_stats[class_idx]['tp'] += tp_c
                    class_stats[class_idx]['fp'] += len(pred_boxes_c) - tp_c
                    class_stats[class_idx]['fn'] += len(target_boxes_c) - tp_c

        # aggregate all
        print(f"\n--- Detailed Metrics & Counts (at Optimal Conf>{optimal_conf:.2f}, IoU>{iou_thresh}) ---")
        for i, name in enumerate(class_names):
            stats = class_stats[i]
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            
            mae = np.mean(stats['count_errors']) if stats['count_errors'] else 0.0
            total_error = np.sum(stats['count_errors'])
            total_gt = stats['total_gt_count']
            count_accuracy = (1 - (total_error / total_gt)) * 100 if total_gt > 0 else 100.0
            
            if is_multiclass:
                metrics.setdefault('f1_per_class', []).append(f1)
                metrics.setdefault('recall_per_class', []).append(recall)
                metrics.setdefault('precision_per_class', []).append(precision)
                metrics.setdefault('count_mae_per_class', []).append(mae)
                metrics.setdefault('count_accuracy_per_class_percent', []).append(count_accuracy)
            else:
                metrics['optimal_precision'] = precision
                metrics['optimal_recall'] = recall
                metrics['optimal_f1_score'] = f1
                metrics['count_mae'] = mae
                metrics['count_accuracy_percent'] = count_accuracy

            print(f"- {name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} | Count MAE={mae:.2f}, Count Acc={count_accuracy:.1f}%")
        
        if is_multiclass:
            map_per_class = map_results.get('map_per_class', torch.tensor([-1.0]))
            if map_per_class.numel() == num_classes:
                print("\n--- mAP@50 per Class ---")
                metrics['mAP_per_class_50'] = map_per_class.tolist()
                for i, name in enumerate(class_names):
                    print(f"- {name}: {map_per_class[i]:.4f}")
        
        # visualization
        self._visualize_detection(predictions, targets_for_eval, optimal_conf, focus_infected=focus_infected)
        
        return metrics
   
    def _visualize_detection(self, predictions, targets, conf_thresh, is_combined=False, focus_infected=False):
        num_images_to_show = min(len(self.test_samples), 3)
        num_classes = self.config.get('num_classes_detection', 1)
        print(f"Visualizing {num_images_to_show} images. Focus on Infected: {focus_infected}")
        is_multiclass = num_classes > 1
        
        PRED_CLASS_MAP = {
            1: ('Inf', 'red'), 
            2: ('Healthy', 'dodgerblue'), 
            3: ('WBC', 'darkviolet')
        }
        
        GT_CLASS_MAP = {
            1: ('Inf', 'lime'),     
            2: ('Healthy', 'cyan'), 
            3: ('WBC', 'magenta')
        }

        fig, axes = plt.subplots(num_images_to_show, 1, figsize=(12, num_images_to_show * 10), constrained_layout=True)
        if num_images_to_show == 1: axes = [axes]

        for i in range(num_images_to_show):
            ax = axes[i]
            ax.imshow(self.test_samples[i]['image'])
            ax.set_axis_off()
            
            pred_for_img = predictions[i]
            target_for_img = targets[i]
            
            # Dynamic title with counts
            title_str = f"Image ID: {self.test_samples[i]['image_id']} (Conf > {conf_thresh:.2f})\n"
            if is_multiclass:
                gt_labels = target_for_img['labels']
                gt_counts = {k: (gt_labels == k).sum().item() for k in PRED_CLASS_MAP.keys() if k <= num_classes}
                
                conf_mask = pred_for_img['scores'] >= conf_thresh
                pred_labels = pred_for_img['labels'][conf_mask]
                pred_counts = {k: (pred_labels == k).sum().item() for k in PRED_CLASS_MAP.keys() if k <= num_classes}
                
                title_str += "GT Counts: " + ", ".join([f"{PRED_CLASS_MAP[k][0][:3]}={v}" for k, v in gt_counts.items()])
                title_str += " | Pred Counts: " + ", ".join([f"{PRED_CLASS_MAP[k][0][:3]}={v}" for k, v in pred_counts.items()])
            else: 
                num_gt = len(target_for_img['boxes'])
                num_pred = (pred_for_img['scores'] >= conf_thresh).sum().item()
                title_str += f"GT: {num_gt} | Pred: {num_pred}"
            
            ax.set_title(title_str, fontsize=12, pad=10)

            # Draw Ground Truth boxes with color-coding and optional focus
            for box, label in zip(target_for_img['boxes'], target_for_img['labels']):
                if focus_infected and label.item() != 1:
                    continue
                    
                class_info = GT_CLASS_MAP.get(label.item(), ('Unknown', 'yellow'))
                _, color = class_info
                
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.5, edgecolor=color, facecolor='none', linestyle='--')
                ax.add_patch(rect)
                
            # Again
            conf_mask = pred_for_img['scores'] >= conf_thresh
            for box, score, label in zip(pred_for_img['boxes'][conf_mask], pred_for_img['scores'][conf_mask], pred_for_img['labels'][conf_mask]):
                if focus_infected and label.item() != 1:
                    continue
                
                class_info = PRED_CLASS_MAP.get(label.item(), ('Unknown', 'gray'))
                class_name, color = class_info
                
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f"{class_name} {score:.2f}", color='white', fontsize=9, bbox=dict(facecolor=color, alpha=0.7))
        
        # Build legend
        legend_elements = []
        if focus_infected:
            legend_elements.append(patches.Patch(edgecolor=GT_CLASS_MAP[1][1], facecolor='none', linestyle='--', label='GT Infected'))
            legend_elements.append(patches.Patch(color=PRED_CLASS_MAP[1][1], label='Pred Infected'))
        else:
            for label_id, info in GT_CLASS_MAP.items():
                if label_id <= num_classes:
                    legend_elements.append(patches.Patch(edgecolor=info[1], facecolor='none', linestyle='--', label=f'GT {info[0]}'))
            for label_id, info in PRED_CLASS_MAP.items():
                if label_id <= num_classes:
                    legend_elements.append(patches.Patch(color=info[1], label=f'Pred {info[0]}'))
        
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=len(legend_elements), fontsize=12)
        self.manager.save_plot(fig, f"test_visualizations_conf_{conf_thresh:.2f}_focus_{focus_infected}")
        plt.show()
        
    def _analyze_and_plot_thresholds(self, predictions, targets, iou_thresh):
        from scipy.optimize import linear_sum_assignment
        
        print(f"\n--- Confidence Threshold Analysis (at IoU={iou_thresh}) ---")
        print("Pre-computing IoU matrices for all images...")
        
        iou_matrices_all = []
        iou_matrices_inf = []
        
        for pred_dict, target_dict in tqdm(zip(predictions, targets), total=len(predictions), desc="Pre-computing IoU"):
            pred_boxes = pred_dict['boxes']
            target_boxes = target_dict['boxes']
            pred_labels = pred_dict['labels']
            target_labels = target_dict['labels']
            
            if len(pred_boxes) > 0 and len(target_boxes) > 0:
                iou_mat = box_iou(pred_boxes, target_boxes)  # (num_preds, num_gts)
            else:
                iou_mat = torch.empty((len(pred_boxes), len(target_boxes)))
            
            iou_matrices_all.append((iou_mat, pred_labels, target_labels, pred_dict['scores']))
            
            pred_inf_mask = (pred_labels == 1)
            target_inf_mask = (target_labels == 1)
            
            if pred_inf_mask.sum() > 0 and target_inf_mask.sum() > 0:
                iou_mat_inf = box_iou(
                    pred_boxes[pred_inf_mask], 
                    target_boxes[target_inf_mask]
                )
            else:
                iou_mat_inf = torch.empty((pred_inf_mask.sum().item(), target_inf_mask.sum().item()))
            
            iou_matrices_inf.append((
                iou_mat_inf, 
                pred_inf_mask, 
                target_inf_mask, 
                pred_dict['scores'],
                pred_boxes.shape[0],
                target_boxes.shape[0]
            ))
        
        print(f"Pre-computed {len(iou_matrices_all)} IoU matrices")
        
        thresholds = np.arange(0.1, 0.96, 0.05)
        results = []
        results_infected = []

        for conf_thresh in tqdm(thresholds, desc="Analyzing thresholds"):
            tp, fp, fn = 0, 0, 0
            tp_inf, fp_inf, fn_inf = 0, 0, 0
            
            for iou_data_all, iou_data_inf in zip(iou_matrices_all, iou_matrices_inf):
                iou_mat, pred_labels, target_labels, pred_scores = iou_data_all
                
                conf_mask = pred_scores >= conf_thresh
                num_preds_filt = conf_mask.sum().item()
                num_targets = iou_mat.shape[1]
                
                if num_targets == 0:
                    fp += num_preds_filt
                elif num_preds_filt == 0:
                    fn += num_targets
                else:
                    iou_mat_filt = iou_mat[conf_mask, :]
                    cost_matrix = (1.0 - iou_mat_filt).cpu().numpy()
                    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
                    
                    tp_img = 0
                    for p_idx, g_idx in zip(pred_indices, gt_indices):
                        if iou_mat_filt[p_idx, g_idx].item() >= iou_thresh:
                            tp_img += 1
                    
                    tp += tp_img
                    fp += num_preds_filt - tp_img
                    fn += num_targets - tp_img
                
                iou_mat_inf, pred_inf_mask, target_inf_mask, pred_scores_all, total_preds, total_targets = iou_data_inf
                
                conf_mask_inf = pred_scores_all >= conf_thresh
                conf_mask_inf[~pred_inf_mask] = False
                
                num_preds_inf_filt = conf_mask_inf.sum().item()
                num_targets_inf = target_inf_mask.sum().item()
                
                if num_targets_inf == 0:
                    fp_inf += num_preds_inf_filt
                elif num_preds_inf_filt == 0:
                    fn_inf += num_targets_inf
                else:
                    valid_pred_indices = torch.where(conf_mask_inf)[0]
                    infected_pred_indices = torch.where(pred_inf_mask)[0]
                    
                    idx_mapping = {old_idx.item(): new_idx 
                                for new_idx, old_idx in enumerate(infected_pred_indices)}
                    iou_rows_to_use = [idx_mapping[i.item()] for i in valid_pred_indices 
                                    if i.item() in idx_mapping]
                    
                    if len(iou_rows_to_use) > 0:
                        iou_mat_inf_filt = iou_mat_inf[iou_rows_to_use, :]
                        
                        cost_matrix_inf = (1.0 - iou_mat_inf_filt).cpu().numpy()
                        pred_indices_inf, gt_indices_inf = linear_sum_assignment(cost_matrix_inf)
                        
                        tp_inf_img = 0
                        for p_idx, g_idx in zip(pred_indices_inf, gt_indices_inf):
                            if iou_mat_inf_filt[p_idx, g_idx].item() >= iou_thresh:
                                tp_inf_img += 1
                        
                        tp_inf += tp_inf_img
                        fp_inf += len(iou_rows_to_use) - tp_inf_img
                        fn_inf += num_targets_inf - tp_inf_img

            # Compute metrics for this threshold
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            results.append({'threshold': conf_thresh, 'precision': precision, 'recall': recall, 'f1_score': f1})
            
            precision_inf = tp_inf / (tp_inf + fp_inf + 1e-6)
            recall_inf = tp_inf / (tp_inf + fn_inf + 1e-6)
            f1_inf = 2 * (precision_inf * recall_inf) / (precision_inf + recall_inf + 1e-6)
            results_infected.append({
                'threshold': conf_thresh, 
                'f1_score_infected': f1_inf, 
                'precision_infected': precision_inf, 
                'recall_infected': recall_inf
            })

        # results
        df = pd.DataFrame(results)
        best_idx = df['f1_score'].idxmax()
        best_row = df.loc[best_idx]
        
        df_infected = pd.DataFrame(results_infected)
        best_idx_infected = df_infected['f1_score_infected'].idxmax()
        best_row_infected = df_infected.loc[best_idx_infected]

        print("\n--- Optimal Performance Found (Overall) ---")
        print(f"Optimal Confidence Threshold (for Overall F1): {best_row['threshold']:.2f}")
        print(f"- Precision: {best_row['precision']:.4f}")
        print(f"- Recall:    {best_row['recall']:.4f}")
        print(f"- F1-Score:  {best_row['f1_score']:.4f}")

        print("\n--- Optimal Performance Found (Infected Class Only) ---")
        print(f"Optimal Confidence Threshold (for Infected F1): {best_row_infected['threshold']:.2f}")
        print(f"- Precision: {best_row_infected['precision_infected']:.4f}")
        print(f"- Recall: {best_row_infected['recall_infected']:.4f}")
        print(f"- F1-Score: {best_row_infected['f1_score_infected']:.4f}")

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi=100)
        axes[0].plot(df['threshold'], df['precision'], 'b-o', label='Precision (Overall)', markersize=4)
        axes[0].plot(df['threshold'], df['recall'], 'g-s', label='Recall (Overall)', markersize=4)
        axes[0].plot(df['threshold'], df['f1_score'], 'r-^', label='F1-Score (Overall)', markersize=4, linewidth=2)
        axes[0].plot(df_infected['threshold'], df_infected['f1_score_infected'], color='purple', linestyle='--', marker='x', label='F1-Score (Infected)', markersize=5)
        axes[0].axvline(x=best_row['threshold'], color='k', linestyle='--', label=f'Optimal Threshold (Overall) ({best_row["threshold"]:.2f})')
        axes[0].set_title('Metrics vs. Confidence Threshold', fontsize=14)
        axes[0].set_xlabel('Confidence Threshold', fontsize=12)
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[0].set_ylim(0, 1.05)

        axes[1].plot(df['recall'], df['precision'], 'm-o', label='Precision-Recall Curve (Overall)', markersize=4)
        axes[1].scatter(best_row['recall'], best_row['precision'], marker='*', s=200, color='red', zorder=5, label='Best Overall F1-Score Point')
        axes[1].set_title('Precision-Recall Curve', fontsize=14)
        axes[1].set_xlabel('Recall', fontsize=12)
        axes[1].set_ylabel('Precision', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[1].set_xlim(0, 1.05)
        axes[1].set_ylim(0, 1.05)
        
        for i, row in df.iterrows():
            if i % 2 == 0:
                axes[1].text(row['recall']+0.01, row['precision']-0.01, f"{row['threshold']:.2f}", fontsize=8, alpha=0.7)

        plt.tight_layout()
        self.manager.save_plot(fig, "threshold_analysis")
        plt.show()

        return {
            'optimal_confidence_threshold': best_row['threshold'],
            'optimal_precision': best_row['precision'],
            'optimal_recall': best_row['recall'],
            'optimal_f1_score': best_row['f1_score'],
            'optimal_confidence_threshold_infected': best_row_infected['threshold']
        }
    
    def _evaluate_severity(self, pred_probs, true_labels):
        true_labels = np.array(true_labels)
        pred_probs = torch.stack(pred_probs).numpy()
        pred_labels = np.argmax(pred_probs, axis=1)
        class_names = ['negative', 'low', 'moderate', 'high']
        
        print("\n--- Classification Report ---")
        print(classification_report(true_labels, pred_labels, target_names=class_names))
        report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)
        
        metrics = {
            'accuracy': report['accuracy'],
            'f1_macro': report['macro avg']['f1-score'],
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall']
        }
        
        self._visualize_severity_cm(true_labels, pred_labels, class_names)
        self._visualize_severity_roc(true_labels, pred_probs, class_names)
        return metrics

    def _evaluate_segmentation(self, predictions, targets):
        preds_tensor = torch.stack(predictions)
        gt_tensor = torch.stack(targets)
        metrics = compute_segmentation_metrics(preds_tensor, gt_tensor)
        print("\n--- Segmentation Metrics ---")
        print(f"Dice Score: {metrics['dice']:.4f}, IoU (Jaccard): {metrics['iou']:.4f}")
        self._visualize_segmentation(predictions, targets)
        return metrics

    def _visualize_classification_plots(self, y_true, y_prob, class_names, patches=None):
        fig_cm, ax_cm = plt.subplots(figsize=(8, 8), dpi=100)
        cm = confusion_matrix(y_true, np.argmax(y_prob, axis=1))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, ax=ax_cm, values_format='d')
        ax_cm.set_title(f"{self.task.replace('_', ' ').title()} Confusion Matrix")
        self.manager.save_plot(fig_cm, "confusion_matrix")
        plt.show()
        
        fig_roc, ax_roc = plt.subplots(figsize=(10, 8), dpi=100)
        
        if len(class_names) == 2:
            y_prob_positive_class = y_prob[:, 1]
            
            fpr, tpr, _ = roc_curve(y_true, y_prob_positive_class)
            roc_auc = auc(fpr, tpr)
            
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
            
        else:
            y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
            colors = plt.cm.get_cmap('viridis', len(class_names))
            
            for i in range(len(class_names)):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, color=colors(i), lw=2, label=f'ROC curve of class {class_names[i]} (area = {roc_auc:0.2f})')

        ax_roc.plot([0, 1], 'k--', lw=2)
        ax_roc.set_xlim([0.0, 1.0]); ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate'); ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f"{self.task.replace('_', ' ').title()} ROC Curve"); ax_roc.legend(loc="lower right"); ax_roc.grid(True)
        self.manager.save_plot(fig_roc, "roc_curve")
        plt.show()
        
        if self.task == 'cell_classif' and patches is not None and len(patches) > 0:
            y_pred = np.argmax(y_prob, axis=1)
            
            import random
            num_samples = min(len(patches), 9)
            indices_to_show = random.sample(range(len(patches)), num_samples)
            
            fig_samples, axes = plt.subplots(3, 3, figsize=(10, 10))
            fig_samples.suptitle('Random Sample of Cell Classification Results', fontsize=16)
            
            for i, ax in enumerate(axes.flat):
                if i < len(indices_to_show):
                    idx = indices_to_show[i]
                    ax.imshow(patches[idx])
                    gt_label = class_names[y_true[idx]]
                    pred_label = class_names[y_pred[idx]]
                    is_correct = (gt_label == pred_label)
                    title_color = 'green' if is_correct else 'red'
                    ax.set_title(f"GT: {gt_label}\nPred: {pred_label}", color=title_color)
                ax.axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            self.manager.save_plot(fig_samples, "sample_classifications")
            plt.show()
    
    def _visualize_confidence_analysis(self, tp_scores, fp_scores, iou_thresh):
        print("\n--- Confidence Score Analysis ---")
        if tp_scores:
            tp_mean, tp_std = np.mean(tp_scores), np.std(tp_scores)
            print(f"True Positives : n={len(tp_scores)}, Mean={tp_mean:.3f}, Std={tp_std:.3f}")
            print(f"                 95% Conf. Interval: [{max(0, tp_mean - 1.96 * tp_std):.3f}, {min(1, tp_mean + 1.96 * tp_std):.3f}]")
        else:
            print("True Positives : n=0")

        if fp_scores:
            fp_mean, fp_std = np.mean(fp_scores), np.std(fp_scores)
            print(f"False Positives: n={len(fp_scores)}, Mean={fp_mean:.3f}, Std={fp_std:.3f}")
            print(f"                 95% Conf. Interval: [{max(0, fp_mean - 1.96 * fp_std):.3f}, {min(1, fp_mean + 1.96 * fp_std):.3f}]")
        else:
            print("False Positives: n=0")

        # plotting
        data = []
        for score in tp_scores: data.append({'Confidence Score': score, 'Prediction Type': 'True Positive'})
        for score in fp_scores: data.append({'Confidence Score': score, 'Prediction Type': 'False Positive'})
        if not data:
            print("No data to plot for confidence analysis.")
            return
        df = pd.DataFrame(data)

        # Create the 1x2 Subplot Figure 
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=100)
        fig.suptitle(f'Confidence Score Analysis (IoU Threshold > {iou_thresh})', fontsize=16)

        # Plot 1: Overlaid Histograms (KDE) 
        sns.histplot(data=df, x='Confidence Score', hue='Prediction Type',
                     bins=50, kde=True, stat='density', common_norm=False,
                     palette={'True Positive': 'forestgreen', 'False Positive': 'orangered'},
                     ax=axes[0])
        axes[0].set_title('Score Density Distribution')
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Density')
        axes[0].grid(True, linestyle='--', alpha=0.6)

        sns.violinplot(data=df, x='Confidence Score', y='Prediction Type', ax=axes[1],
                       orient='h', palette={'True Positive': 'forestgreen', 'False Positive': 'orangered'},
                       inner=None, linewidth=1.5)
        sns.stripplot(data=df, x='Confidence Score', y='Prediction Type', ax=axes[1],
                      orient='h', color='black', alpha=0.2, jitter=0.25, size=2.5)
        axes[1].set_title('Score Distribution and Individual Points')
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_ylabel('')
        axes[1].set_xlim(-0.05, 1.05)
        axes[1].grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self.manager.save_plot(fig, f"confidence_analysis_iou_{iou_thresh}")
        plt.show()

    def _visualize_severity_cm(self, y_true, y_pred, class_names):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
        ax.set_title("Severity Classification Confusion Matrix")
        self.manager.save_plot(fig, "confusion_matrix")
        plt.show()

    def _visualize_severity_roc(self, y_true, y_prob, class_names):
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        colors = ['blue', 'red', 'green', 'purple']
        for i, color in zip(range(len(class_names)), colors):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve of class {class_names[i]} (area = {roc_auc:0.2f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('Multi-class Receiver Operating Characteristic (ROC)'); ax.legend(loc="lower right"); ax.grid(True)
        self.manager.save_plot(fig, "roc_curve")
        plt.show()

    def _visualize_segmentation(self, predictions, targets):
        num_images_to_show = min(len(self.test_samples), 3)
        fig, axes = plt.subplots(num_images_to_show, 3, figsize=(15, num_images_to_show * 5), dpi=100)
        if num_images_to_show == 1: axes = np.array([axes])
        for i in range(num_images_to_show):
            axes[i, 0].imshow(self.test_samples[i]['image']); axes[i, 0].set_title("Original Image"); axes[i, 0].set_axis_off()
            axes[i, 1].imshow(targets[i].squeeze(), cmap='gray'); axes[i, 1].set_title("Ground Truth Mask"); axes[i, 1].set_axis_off()
            axes[i, 2].imshow(predictions[i].squeeze(), cmap='gray', vmin=0, vmax=1); axes[i, 2].set_title("Predicted Mask"); axes[i, 2].set_axis_off()
        plt.tight_layout()
        self.manager.save_plot(fig, "segmentation_visualizations")
        plt.show()
        
    def _prepare_rcnn_detection_targets(self, batch):
        targets = []
        img_h, img_w = self.config.get('image_size', (512, 512))
        num_det_classes = self.config.get('num_classes_detection', 1)

        for i in range(len(batch.get('image', []))):
            bboxes_yolo = batch['bboxes'][i]
            
            labels_yolo = batch.get('bbox_labels', [[]]*len(batch['image']))[i]
            if not isinstance(labels_yolo, (list, np.ndarray, torch.Tensor)) or len(labels_yolo) == 0:
                 labels_yolo = batch.get('labels', [[]]*len(batch['image']))[i]
                 
            if num_det_classes == 1 and len(labels_yolo) > 0:
                mask = (labels_yolo == 0) 
                bboxes_yolo = bboxes_yolo[mask]
                labels_yolo = labels_yolo[mask]
            elif num_det_classes == 2 and len(labels_yolo) > 0:
                mask = (labels_yolo != 2) 
                bboxes_yolo = bboxes_yolo[mask]
                labels_yolo = labels_yolo[mask]

            if len(bboxes_yolo) > 0:
                cx, cy, w, h = bboxes_yolo.unbind(1)
                x1, y1, x2, y2 = (cx-w/2)*img_w, (cy-h/2)*img_h, (cx+w/2)*img_w, (cy+h/2)*img_h
                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                
                if num_det_classes == 1:
                    labels = torch.ones(len(boxes_xyxy), dtype=torch.int64)
                else:
                    labels = labels_yolo.long() + 1
                
                targets.append({"boxes": boxes_xyxy.cpu(), "labels": labels.cpu()})
            else:
                targets.append({"boxes": torch.empty((0, 4)), "labels": torch.empty(0, dtype=torch.int64)})
        return targets    
        
        