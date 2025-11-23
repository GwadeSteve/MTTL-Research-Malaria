# ============================================================================
# Orchestrator for our Components in MTTL
# Gwade Steve
# MTTL and Application to Malaria Detection
# April 2025
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
import numpy as np
import traceback
from Losses import FlexibleMalariaLoss
from Components import ( 
        BackboneNetwork, EfficientRCNNHead, FeaturePyramidNetwork, CellClassifierHead, 
        YOLODetectionHead, ParasitemiaHead, LocalizationHeatmapHead, 
        SegmentationHead, SeverityHead )
from torchvision.ops import MultiScaleRoIAlign
from DataUtils import set_seeds

# Main MalariaModel class
class MalariaModel(nn.Module):
    def __init__(self, config, mode='MTTL', active_tasks=None, backbone_unfreeze_layers=None):
        super(MalariaModel, self).__init__()
        if active_tasks:
            canonical_map = {'localization': 'heatmap', 'heatmap': 'heatmap'}
            active_tasks = [canonical_map.get(task, task) for task in active_tasks]
        self.config = config
        self.mode = mode.upper()
        self.detection_head_type = config.get('detection_head', 'RCNN')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fpn = None
        self.roi_classifier_pooler = None

        # Task validation
        all_tasks = ['detection', 'parasitemia', 'localization', 'heatmap', 'segmentation', 'severity', 'cell_classif', 'roi_classif']
        if active_tasks is None:
            self.active_tasks = all_tasks if self.mode == 'MTTL' else [all_tasks[0]]
        else:
            self.active_tasks = active_tasks

        if self.mode == 'STL' and len(self.active_tasks) > 1:
            raise ValueError("STL mode supports only one task at a time.")
        if self.mode == 'MTTL' and len(self.active_tasks) < 2:
            print(f"Warning: MTTL mode typically uses >1 task, but configured for {len(self.active_tasks)}.")
        
        # Default adapter configuration
        if 'adapter_config' not in config:
            config['adapter_config'] = {
                'detection_rank': 16, 'detection_alpha': 32, 'detection_dropout': 0.1,
                'parasitemia_rank': 8, 'parasitemia_alpha': 16, 'parasitemia_dropout': 0.05,
                'heatmap_rank': 8, 'heatmap_alpha': 16, 'heatmap_dropout': 0.05,
                'segmentation_rank': 8, 'segmentation_alpha': 16, 'segmentation_dropout': 0.05,
                'severity_rank': 8, 'severity_alpha': 16, 'severity_dropout': 0.05
            }

        # Create active task adapter config and ensure specific keys for each task
        active_adapter_config = {'tasks': self.active_tasks}
        for task in self.active_tasks:
            for key in ['rank', 'alpha', 'dropout']:
                config_key = f'{task}_{key}'
                default_val = 16 if key == 'rank' else (32 if key == 'alpha' else 0.1)
                active_adapter_config[config_key] = config['adapter_config'].get(config_key, default_val)

        # backbone with integrated adapters
        self.backbone = BackboneNetwork(
            architecture=config['backbone_arch'],
            pretrained=True,
            freeze_backbone=True,
            adapter_config=active_adapter_config,
            unfreeze_layers=backbone_unfreeze_layers,
            mttl_mode=True
        )
        
        if 'detection' in self.active_tasks or 'roi_classif' in self.active_tasks:
            if self.config['backbone_arch'] == 'resnet50':
                fpn_in_channels = [512, 1024, 2048] 
            else: 
                fpn_in_channels = [128, 256, 512]  

            self.fpn = FeaturePyramidNetwork(
                in_channels_list=fpn_in_channels, 
                out_channels=64
            ).to(self.device)
            print("Shared FPN module initialized.")

        # If RoI classification is active, we will create its dedicated pooler
        if 'roi_classif' in self.active_tasks:
            # This pooler will take features from the FPN (named '0', '1', '2')
            self.roi_classifier_pooler = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2'], 
                output_size=7, 
                sampling_ratio=2
            ).to(self.device)
            print("MultiScaleRoIAlign pooler initialized for RoI classification task.")

        # task heads
        self.task_heads = nn.ModuleDict()

        if 'detection' in self.active_tasks:
            num_det_classes = config.get('num_classes_detection', 1) 
            if self.detection_head_type == 'RCNN':
                print("Using EfficientRCNNHead for detection.")
                self.task_heads['detection'] = EfficientRCNNHead(
                    num_classes=num_det_classes,
                    fpn_dim=64
                )
            else: 
                print("Using YOLODetectionHead for detection.")
                self.task_heads['detection'] = YOLODetectionHead(
                    feature_dims=self.backbone.get_multiscale_dims(),
                    num_classes=num_det_classes,
                    grid_sizes=config.get('grid_sizes', [64, 32, 16])
                )
                
        if 'cell_classif' in self.active_tasks or 'roi_classif' in self.active_tasks:
            if 'cell_classif' not in self.task_heads:
                if 'roi_classif' in self.active_tasks:
                    classifier_in_features = 64 * 7 * 7
                else:
                    classifier_in_features = self.backbone.get_feature_dim()
                
                self.task_heads['cell_classif'] = CellClassifierHead(
                    in_features=classifier_in_features, 
                    num_classes=config.get('num_classes_classif', 3),
                    representation_size=256
                )

        if 'parasitemia' in self.active_tasks:
            self.task_heads['parasitemia'] = ParasitemiaHead(
                feature_dim=self.backbone.get_feature_dim(),
                bottleneck_dim=config.get('adapter_bottleneck_dim', 64),
            )

        if 'heatmap' in self.active_tasks:
            self.task_heads['heatmap'] = LocalizationHeatmapHead(
                feature_dims=self.backbone.get_multiscale_dims(),
                heatmap_size=config.get('heatmap_size', (64, 64))
            )
            
        if 'segmentation' in self.active_tasks:
            self.task_heads['segmentation'] = SegmentationHead(
                feature_dims=self.backbone.get_multiscale_dims(),
                out_size=config.get('segmentation_size', (64, 64))
            )
        if 'severity' in self.active_tasks:
            self.task_heads['severity'] = SeverityHead(
                feature_dim=self.backbone.get_feature_dim(),
                num_classes=config.get('severity_num_classes', 4),
                bottleneck_dim=config.get('severity_bottleneck_dim', 64)
            )

        # loss function
        loss_cfg = config.get('loss_config', {})
        loss_cfg['tasks'] = self.active_tasks
        
        num_classes_dict = {
            'detection': config.get('num_classes_detection', 1),
            'cell_classif': config.get('num_classes_classif', 3),
            'severity': config.get('num_classes_severity', 4)
        }
        
        loss_cfg['use_uncertainty'] = (self.mode == 'MTTL')

        self.loss_function = FlexibleMalariaLoss(
            num_classes_dict=num_classes_dict,
            loss_config=loss_cfg,
            device=self.device
        )

        print("\nModel Parameter Breakdown")
        breakdown = self.get_parameter_breakdown()
        total_trainable = breakdown.get('total_trainable', 0)
        total_params = breakdown.get('total_parameters', 0)
        print(f"Total Trainable Parameters: {total_trainable:,} ({breakdown.get('trainable_percentage', 'N/A')})")
        print(f"Total Parameters: {total_params:,}")
        print("-" * 20)
        for component, stats in breakdown.items():
            if isinstance(stats, dict) and 'trainable' in stats:
                print(f"{component.replace('_', ' ').title()}:")
                print(f"- Trainable: {stats['trainable']:,}")
                if 'trainable_core' in stats:
                    print(f"- Core: {stats['trainable_core']:,}")
                    print(f"- Adapters: {stats['trainable_adapters']:,}")
        print("---------------------------------\n")
        print("MalariaModel initialized successfully.")

    def _calculate_model_stats(self):
        """Calculate model statistics"""
        backbone_total = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

        head_total = sum(p.numel() for p in self.task_heads.parameters())
        head_trainable = sum(p.numel() for p in self.task_heads.parameters() if p.requires_grad)

        loss_total = sum(p.numel() for p in self.loss_function.parameters())
        loss_trainable = sum(p.numel() for p in self.loss_function.parameters() if p.requires_grad)

        model_total = backbone_total + head_total + loss_total
        model_trainable = backbone_trainable + head_trainable + loss_trainable

        self.model_stats = {
            'backbone_with_adapters': {'total': f'{backbone_total:,}', 'trainable': f'{backbone_trainable:,}'},
            'task_heads': {'total': f'{head_total:,}', 'trainable': f'{head_trainable:,}'},
            'loss_function': {'total': f'{loss_total:,}', 'trainable': f'{loss_trainable:,}'},
            'model_total': {'total': f'{model_total:,}', 'trainable': f'{model_trainable:,}'},
            'training_efficiency_ratio': f'{(model_trainable / model_total * 100):.2f}%' if model_total > 0 else 'N/A'
        }

    def get_parameter_breakdown(self):
        breakdown = {}

        # Backbone parameters
        backbone_total = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        adapter_params = sum(p.numel() for n, p in self.backbone.named_parameters() if 'adapter' in n and p.requires_grad)
        backbone_core_trainable = backbone_trainable - adapter_params

        breakdown['backbone'] = {
            'total': backbone_total,
            'trainable': backbone_trainable,
            'trainable_core': backbone_core_trainable,
            'trainable_adapters': adapter_params
        }

        # Task heads (only the ones active)
        head_params = {}
        for task_name in self.active_tasks:
            if task_name in self.task_heads:
                head = self.task_heads[task_name]
                total = sum(p.numel() for p in head.parameters())
                trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
                head_params[f"{task_name}_head"] = {'total': total, 'trainable': trainable}
            else:
                head_params[f"{task_name}_head"] = {'total': 0, 'trainable': 0}
        breakdown['task_heads'] = head_params

        # Loss function parameters if nay esp in mttl mode
        loss_total = sum(p.numel() for p in self.loss_function.parameters())
        loss_trainable = sum(p.numel() for p in self.loss_function.parameters() if p.requires_grad)
        breakdown['loss_function'] = {'total': loss_total, 'trainable': loss_trainable}

        # Overall stats
        total_params = sum(p.numel() for p in self.parameters())
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        breakdown['total_parameters'] = total_params
        breakdown['total_trainable'] = total_trainable
        breakdown['trainable_percentage'] = f"{(total_trainable / total_params * 100):.2f}%" if total_params > 0 else "N/A"

        return breakdown
        
    # prepare targets for R-CNN detection head
    def _prepare_rcnn_detection_targets(self, batch):
        targets = []
        img_h, img_w = self.config.get('image_size', (512, 512))
        num_det_classes = self.config.get('num_classes_detection', 1)

        for i in range(len(batch.get('image', []))):
            boxes_yolo = batch['bboxes'][i]
            if len(boxes_yolo) > 0:
                cx, cy, w, h = boxes_yolo.unbind(1)
                x1, y1, x2, y2 = (cx-w/2)*img_w, (cy-h/2)*img_h, (cx+w/2)*img_w, (cy+h/2)*img_h
                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                
                if num_det_classes == 1:
                    labels = torch.ones(len(boxes_yolo), dtype=torch.int64)
                else:
                    original_labels = batch['bbox_labels'][i]
                    labels = original_labels.long() + 1
                
                targets.append({"boxes": boxes_xyxy.to(self.device), "labels": labels.to(self.device)})
            else:
                targets.append({"boxes": torch.empty((0, 4), device=self.device), "labels": torch.empty(0, dtype=torch.int64, device=self.device)})
        return targets    
    
    # MTTL forward pass
    def forward_mttl(self, batch, is_train=False):
        images = batch['image'].to(self.device)
        outputs = {}
        img_h, img_w = self.config.get('image_size', (512, 512))
        multiscale_features = self.backbone(images, task=None, return_multiscale=True)
        
        # FPN features computed and cached 
        fpn_features = None
        if self.fpn:
            fpn_input = [multiscale_features[f'c{i}'] for i in range(2, 5)]
            fpn_features = self.fpn(fpn_input)

        if 'detection' in self.task_heads:
            det_targets = self._prepare_rcnn_detection_targets(batch) if is_train else None
            outputs['detection'] = self.task_heads['detection'](fpn_features, images, det_targets)
            
        if 'heatmap' in self.task_heads:
            outputs['heatmap'] = self.task_heads['heatmap'](multiscale_features)
        
        if 'segmentation' in self.task_heads:
            outputs['segmentation'] = self.task_heads['segmentation'](multiscale_features)
            
        if 'severity' in self.task_heads:
            global_features = F.adaptive_avg_pool2d(multiscale_features['c4'], (1, 1)).flatten(1)
            outputs['severity'] = self.task_heads['severity'](global_features)
            
        # Check if batch has detection annotations before trying to access 
        has_detection_annotations = 'bboxes' in batch and 'bbox_labels' in batch

        if 'roi_classif' in self.active_tasks and self.roi_classifier_pooler and fpn_features and has_detection_annotations:
            list_of_gt_boxes = []
            labels_to_cat = []
            
            # List to keep track of which image each box belongs to.
            box_to_image_id = [] 

            for i in range(len(images)):
                bboxes_yolo = batch['bboxes'][i]
                labels = batch.get('bbox_labels', batch.get('labels'))[i]
                
                if len(bboxes_yolo) > 0 and len(labels) > 0:
                    cx, cy, w, h = bboxes_yolo.unbind(1)
                    x1, y1, x2, y2 = (cx-w/2)*img_w, (cy-h/2)*img_h, (cx+w/2)*img_w, (cy+h/2)*img_h
                    list_of_gt_boxes.append(torch.stack([x1, y1, x2, y2], dim=1).to(self.device))
                    labels_to_cat.append(labels.to(self.device))
                    
                    # For each box in this image, we record its image index 'i'
                    box_to_image_id.append(torch.full((len(bboxes_yolo),), i, device=self.device))
            
            if labels_to_cat:
                # Concatenate all boxes and labels from the batch into single tensors
                all_boxes = torch.cat(list_of_gt_boxes)
                all_labels = torch.cat(labels_to_cat)
                all_box_to_image_id = torch.cat(box_to_image_id)

                # Filter the boxes and labels based on the classifier's config
                num_classif_classes = self.config.get('num_classes_classif', 3)
                valid_mask = all_labels < num_classif_classes
                
                final_boxes = all_boxes[valid_mask]
                final_labels = all_labels[valid_mask]
                final_box_to_image_id = all_box_to_image_id[valid_mask]

                # convert the boxes back to a list-of-tensors format for RoI Align
                boxes_per_image = []
                for i in range(len(images)):
                    boxes_per_image.append(final_boxes[final_box_to_image_id == i])

                # RoI pooling only on the valid boxes
                pooled_features = self.roi_classifier_pooler(fpn_features, boxes_per_image, [(img_h, img_w)] * len(images))
                
                # Check if pooling produced any features (it might be empty if all boxes were filtered)
                if pooled_features.shape[0] > 0:
                    feature_vectors = pooled_features.flatten(start_dim=1)
                    outputs['roi_classif'] = self.task_heads['cell_classif'](feature_vectors)
                    # The labels are the final, filtered labels
                    outputs['roi_classif_labels'] = final_labels.long()
        
        if 'cell_classif' in self.active_tasks and has_detection_annotations:
            list_of_gt_boxes = []
            labels_to_cat = []
            for i in range(len(images)):
                bboxes_yolo = batch['bboxes'][i]
                labels = batch['bbox_labels'][i]
                if len(bboxes_yolo) > 0:
                    cx, cy, w, h = bboxes_yolo.unbind(1)
                    x1, y1, x2, y2 = (cx-w/2)*img_w, (cy-h/2)*img_h, (cx+w/2)*img_w, (cy+h/2)*img_h
                    list_of_gt_boxes.append(torch.stack([x1, y1, x2, y2], dim=1).to(self.device))
                    if len(labels) > 0: labels_to_cat.append(labels.to(self.device))
            
            if labels_to_cat:
                feature_map_for_roi = multiscale_features['c4']
                pooled_features = roi_align(input=feature_map_for_roi, boxes=list_of_gt_boxes, output_size=(7, 7), spatial_scale=feature_map_for_roi.shape[-1]/img_w)
                feature_vectors = F.adaptive_avg_pool2d(pooled_features, (1, 1)).flatten(start_dim=1)
                outputs['cell_classif'] = self.task_heads['cell_classif'](feature_vectors)
                outputs['cell_classif_labels'] = torch.cat(labels_to_cat).long()
                
        return outputs
    
    # stl forward pass
    def forward(self, x, targets=None, return_features=False):
        # We check dims
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"Expected input [B, 3, H, W], got {x.shape}")

        tasks_to_process = self.active_tasks
        if not tasks_to_process:
             raise ValueError("No active tasks specified for the model.")
        
        task_name = tasks_to_process[0] 
        outputs = {}
        
        # We first Handle patch-based classification separately since the data flow is different
        if task_name == 'cell_classif':
            feature_vector = self.backbone(x, task='cell_classif', return_multiscale=False)
            outputs['cell_classif'] = self.task_heads['cell_classif'](feature_vector)
            return outputs

        # Then our Main forward pass for all full-image tasks 
        multiscale_features = self.backbone(x, task=task_name, return_multiscale=True)
        
        fpn_features = None
        if self.fpn and (task_name in ['detection', 'roi_classif']):
            fpn_input = [multiscale_features[f'c{i}'] for i in range(2, 5)]
            fpn_features = self.fpn(fpn_input)

        if task_name == 'roi_classif':
            if self.roi_classifier_pooler and fpn_features and targets is not None:
                list_of_gt_boxes, labels_to_cat = [], []
                img_h, img_w = x.shape[-2:]
                for i in range(len(x)):
                    # we get labels from the batch dictionary
                    batch_labels = targets.get('labels', targets.get('bbox_labels'))
                    if batch_labels is None: continue
                    
                    bboxes_yolo, labels = targets['bboxes'][i], batch_labels[i]

                    if len(bboxes_yolo) > 0 and len(labels) > 0:
                        cx, cy, w, h = bboxes_yolo.unbind(1)
                        x1, y1, x2, y2 = (cx-w/2)*img_w, (cy-h/2)*img_h, (cx+w/2)*img_w, (cy+h/2)*img_h
                        list_of_gt_boxes.append(torch.stack([x1, y1, x2, y2], dim=1).to(self.device))
                        labels_to_cat.append(labels.to(self.device))
                
                if labels_to_cat:
                    pooled_features = self.roi_classifier_pooler(fpn_features, list_of_gt_boxes, [(img_h, img_w)] * len(x))
                    feature_vectors = pooled_features.flatten(start_dim=1)
                    outputs['roi_classif'] = self.task_heads['cell_classif'](feature_vectors)
                    outputs['roi_classif_labels'] = torch.cat(labels_to_cat).long()
        
        elif task_name == 'detection' and self.detection_head_type == 'RCNN':
            if fpn_features:
                detection_targets = targets if self.training else None
                outputs['detection'] = self.task_heads['detection'](fpn_features, x, detection_targets)
        
        elif task_name in ['parasitemia', 'severity']:
            global_features = F.adaptive_avg_pool2d(multiscale_features['c4'], (1, 1)).flatten(1)
            outputs[task_name] = self.task_heads[task_name](global_features)

        elif task_name in ['segmentation', 'heatmap']:
             outputs[task_name] = self.task_heads[task_name](multiscale_features)
        
        if return_features: 
            return outputs, {'multiscale': multiscale_features, 'fpn': fpn_features}
            
        return outputs
    
    # compute loss method
    def compute_loss(self, outputs, targets=None):
        if self.detection_head_type == 'RCNN' and 'detection' in self.active_tasks:
            if 'detection' in outputs and isinstance(outputs.get('detection'), dict):
                loss_dict = outputs['detection']
                
                if self.mode == 'MTTL':
                    outputs['detection_loss'] = sum(loss if isinstance(loss, torch.Tensor) else torch.tensor(loss, device=self.device, dtype=torch.float32)
                    for loss in loss_dict.values())
                else: 
                    total_loss = sum(loss for loss in loss_dict.values())
                    loss_breakdown = {k: v.item() for k, v in loss_dict.items()}
                    return total_loss, loss_breakdown, {}
                
        return self.loss_function(outputs, targets)

    def get_model_info(self):
        info = {
            'mode': self.mode,
            'active_tasks': self.active_tasks,
            'model_stats': self.model_stats,
            'backbone_info': self.backbone.get_feature_info(),
            'loss_info': {
                'type': 'uncertainty_weighted' if self.mode == 'MTTL' else 'raw_loss',
            },
            'task_head_info': {}
        }
        for task, head in self.task_heads.items():
            info['task_head_info'][task] = head.get_output_info()
        return info

    def switch_to_single_task(self, task_name):
        if task_name not in ['detection', 'parasitemia', 'heatmap']:
            raise ValueError(f"Invalid task '{task_name}'")
        self.mode = 'STL'
        self.active_tasks = [task_name]
        loss_config = self.loss_function.config.copy()
        loss_config['tasks'] = [task_name]
        loss_config['use_uncertainty'] = False
        self.loss_function = FlexibleMalariaLoss(
            num_classes=self.config['num_classes'],
            loss_config=loss_config
        )
        print(f"Model switched to STL mode for task: {task_name}")
        return self

    def get_evaluation_metrics(self, task_name):
        if task_name == 'detection':
            return {
                'primary_metrics': ['mAP', 'mAP_50', 'mAP_75'],
                'secondary_metrics': ['precision', 'recall', 'f1_score'],
                'class_wise': True,
                'iou_thresholds': [0.5, 0.75, 0.9],
                'confidence_threshold': 0.5
            }
        elif task_name == 'parasitemia':
            return {
                'primary_metrics': ['mae', 'rmse', 'r2'],
                'secondary_metrics': ['mape', 'correlation'],
                'clinical_metrics': ['clinical_accuracy', 'severity_classification'],
                'range_analysis': True
            }
        elif task_name == 'heatmap':
            return {
                'primary_metrics': ['dice_score', 'iou', 'pixel_accuracy'],
                'secondary_metrics': ['precision', 'recall', 'f1_score'],
                'spatial_metrics': ['center_distance', 'boundary_accuracy'],
                'attention_metrics': ['top_k_accuracy', 'localization_error']
            }
        else:
            raise ValueError(f"Unknown task: {task_name}")

def create_model_for_task(config, task_name):
    """Create a model configured for single task evaluation"""
    task_mapping = {
        'regression': 'parasitemia',
        'localization': 'heatmap',
        'detection': 'detection',
        'segmentation': 'segmentation',
        'severity': 'severity'
    }
    internal_task = task_mapping.get(task_name, task_name)
    return MalariaModel(
        config=config,
        mode='STL',
        active_tasks=[internal_task],
        backbone_unfreeze_layers=config.get('backbone_unfreeze_layers', None)
    )

def create_multitask_model(config, tasks=None):
    """Create a model configured for multi-task learning"""
    if tasks is None:
        tasks = ['detection', 'segmentation', 'severity', 'cell_classif']
    return MalariaModel(
        config=config,
        mode='MTTL',
        active_tasks=tasks,
        backbone_unfreeze_layers=config.get('backbone_unfreeze_layers', None)
    )

# Check if the model passes work for the given setup
def validate_model_setup(model, config):
    task = config.get('active_tasks', ['detection'])[0]
    device = next(model.parameters()).device
    print(f"Running Validation for Task: '{task}'")

    if task == 'cell_classif':
        # Patch-based classifier needs small images and batch > 1 for BatchNorm
        validation_batch_size = 2 
        input_shape = (3, 64, 64)
        print(f"Validation using cell patch input: batch_size={validation_batch_size}, shape={input_shape}")
    
    elif task == 'roi_classif':
        # RoI-based classifier needs full images and batch > 1 for the head's BatchNorm
        validation_batch_size = 2
        input_shape = (3, 512, 512)
        print(f"Validation using full image input: batch_size={validation_batch_size}, shape={input_shape}")
        
    else: # For detection, segmentation, severity, ...
        # A batch size of 1 is ok for these
        validation_batch_size = 1
        input_shape = (3, 512, 512)
        print(f"Validation using full image input: batch_size={validation_batch_size}, shape={input_shape}")

    # dummy input tensors with the correct batch dims
    test_input_eval = torch.randn(1, *input_shape).to(device) 
    test_input_train = torch.randn(validation_batch_size, *input_shape).to(device)

    try:
        # Eval Mode Pass
        model.eval()
        with torch.no_grad():
            # For eval, targets are not strictly needed for most tasks forward pass
            outputs = model(test_input_eval, targets=None)

        validation_info = {'status': 'passed', 'output_tasks': list(outputs.keys()), 'output_shapes': {}}
        for t, output in outputs.items():
            if isinstance(output, list) and output:
                shapes = [o.get('boxes', o).shape for o in output if isinstance(o, dict) or hasattr(o, 'shape')]
                validation_info['output_shapes'][f"{t}_eval"] = str(shapes)
            elif hasattr(output, 'shape'):
                validation_info['output_shapes'][f"{t}_eval"] = str(output.shape)
        
        # train Mode Pass 
        model.train()
        dummy_targets = None
        
        # dummy targets for each task that needs them.
        if task == 'detection' and hasattr(model, 'detection_head_type') and model.detection_head_type == 'RCNN':
            dummy_targets = [{"boxes": torch.tensor([[10,10,50,50]], dtype=torch.float32).to(device), 
                              "labels": torch.ones(1, dtype=torch.long).to(device)}] * validation_batch_size
        
        elif task == 'roi_classif':
            # dummy batch dictionary that mimics the dataloader output
            dummy_targets = {
                'bboxes': [torch.tensor([[0.5, 0.5, 0.1, 0.1]]) for _ in range(validation_batch_size)],
                'labels': [torch.tensor([0]) for _ in range(validation_batch_size)]
            }
        
        # targets format for each task forward pass
        outputs_train = model(test_input_train, targets=dummy_targets)
        
        #  for rcnn, we check loss dict
        for t, output in outputs_train.items():
            if isinstance(output, dict) and 'loss_classifier' in output: 
                validation_info['output_shapes'][f"{t}_train_loss"] = {k: f"{v.item():.4f}" for k,v in output.items()}

        model.eval() # back to eval
        return True, validation_info

    except Exception as e:
        import traceback
        return False, {'status': 'failed', 'error': str(e), 'traceback': traceback.format_exc()}

def compute_corrected_localization_metrics(predictions, targets):
    """localization metrics for attention/heatmap evaluation"""
    metrics = {}
    try:
        if predictions.dim() == 4:
            pred_sample = predictions[0, 0] if predictions.shape[1] == 1 else predictions[0]
            target_sample = targets[0, 0] if targets.shape[1] == 1 else targets[0]
        else:
            pred_sample = predictions.squeeze()
            target_sample = targets.squeeze()
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)
        mse = F.mse_loss(predictions, targets).item()
        mae = F.l1_loss(predictions, targets).item()
        pred_np = pred_flat.cpu().numpy()
        target_np = target_flat.cpu().numpy()
        if len(np.unique(target_np)) > 1 and len(np.unique(pred_np)) > 1:
            correlation = np.corrcoef(pred_np, target_np)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        def find_peaks_2d(heatmap, threshold=0.5):
            try:
                from scipy.ndimage import maximum_filter
                if heatmap.ndim > 2:
                    heatmap = heatmap.squeeze()
                peaks = (heatmap == maximum_filter(heatmap, size=5)) & (heatmap > threshold)
                return peaks
            except:
                return np.zeros_like(heatmap, dtype=bool)
        pred_peaks = find_peaks_2d(pred_sample.cpu().numpy())
        target_peaks = find_peaks_2d(target_sample.cpu().numpy())
        peak_intersection = (pred_peaks & target_peaks).sum()
        peak_union = (pred_peaks | target_peaks).sum()
        peak_iou = peak_intersection / (peak_union + 1e-8)
        k_values = [5, 10, 20]
        ranking_metrics = {}
        for k in k_values:
            try:
                _, pred_topk = torch.topk(pred_flat, min(k, len(pred_flat)))
                _, target_topk = torch.topk(target_flat, min(k, len(target_flat)))
                pred_set = set(pred_topk.cpu().numpy())
                target_set = set(target_topk.cpu().numpy())
                overlap = len(pred_set & target_set)
                ranking_metrics[f'top_{k}_accuracy'] = overlap / k
            except:
                ranking_metrics[f'top_{k}_accuracy'] = 0.0
        def compute_center_of_mass_safe(heatmap):
            try:
                if heatmap.dim() > 2:
                    heatmap = heatmap.squeeze()
                if heatmap.sum() == 0:
                    return None
                h, w = heatmap.shape
                y_range = torch.arange(h, dtype=torch.float32)
                x_range = torch.arange(w, dtype=torch.float32)
                y_coords, x_coords = torch.meshgrid(y_range, x_range, indexing='ij')
                total_mass = heatmap.sum()
                center_y = (y_coords * heatmap).sum() / total_mass
                center_x = (x_coords * heatmap).sum() / total_mass
                return center_y.item(), center_x.item()
            except:
                return None
        pred_center = compute_center_of_mass_safe(pred_sample)
        target_center = compute_center_of_mass_safe(target_sample)
        if pred_center is not None and target_center is not None:
            center_distance = np.sqrt((pred_center[0] - target_center[0])**2 +
                                      (pred_center[1] - target_center[1])**2)
            img_diagonal = np.sqrt(pred_sample.shape[-2]**2 + pred_sample.shape[-1]**2)
            normalized_center_error = center_distance / img_diagonal
        else:
            normalized_center_error = 1.0
        metrics.update({
            'localization_mse': mse,
            'localization_mae': mae,
            'spatial_correlation': correlation,
            'peak_iou': peak_iou,
            'center_distance_normalized': normalized_center_error,
            **{f'localization_{k}': v for k, v in ranking_metrics.items()}
        })
        overall_score = (
            (1 - normalized_center_error) * 0.3 +
            max(0, correlation) * 0.3 +
            peak_iou * 0.2 +
            ranking_metrics.get('top_10_accuracy', 0) * 0.2
        )
        metrics['overall_localization_score'] = overall_score
    except Exception as e:
        metrics = {
            'localization_mse': 0.0,
            'localization_mae': 0.0,
            'spatial_correlation': 0.0,
            'peak_iou': 0.0,
            'center_distance_normalized': 1.0,
            'overall_localization_score': 0.0,
            'localization_top_5_accuracy': 0.0,
            'localization_top_10_accuracy': 0.0,
            'localization_top_20_accuracy': 0.0
        }
    return metrics

