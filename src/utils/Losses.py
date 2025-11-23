# ============================================================================
# Loss master for our Components in MTTL
# Gwade Steve
# MTTL and Application to Malaria Detection
# April 2025
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from DataUtils import set_seeds

class FlexibleMalariaLoss(nn.Module):
    """
    Our Loss System for Malaria Detection:
    - DETECTION: RPN_cls (BCE) + RPN_reg (Smooth L1) + RoI_cls(Focal Loss) + RoI_reg (Smooth L1)
    - HEATMAP: Spatial Focal + Dice + Boundary Enhancement
    - CELL CLASSIF: Weighted CE loss
    - ROI CLASSIF: Weighted CE Loss
    - SEVERITY: Weighted CE Loss
    - MODE: Single Task (raw) OR Multi-Task (uncertainty weighted)
    """
    
    def __init__(self, num_classes_dict, device, loss_config=None):
        super(FlexibleMalariaLoss, self).__init__()
        
        self.num_classes_dict = num_classes_dict
        self.device = device
        
        # Default configuration
        if loss_config is None:
            self.config = {
                'tasks': ['detection', 'cell_classif', 'segmentation', 'severity', 'roi_classif', 'heatmap'],
                'use_uncertainty': True,
                'init_log_vars': {
                    'detection': -2.0,
                    'parasitemia': -0.3,
                    'heatmap': -1.0,
                    'segmentation': -1.0,
                    'severity': -1.0,
                    'cell_classif': -1.5,
                    'roi_classif': -1.5
                },
                
                'detection': {
                    'focal_alpha': 0.25,
                    'focal_gamma': 2.0,
                    'objectness_weight': 2.0,
                    'classification_weight': 1.0,
                    'bbox_weight': 1.0,
                    'hard_negative_ratio': 3.0,
                    'smooth_l1_beta': 0.1
                },
                
                'parasitemia': {
                    'loss_type': 'mae',
                    'huber_delta': 0.1,
                    'range_penalty': 0.1,
                    'clinical_thresholds': [0.0, 0.02, 0.1, 1.0],
                    'threshold_weights': [1.0, 2.0, 1.5, 1.0]
                },
                'cell_classif': {
                    'loss_type': 'cross_entropy',
                    'class_weights': None # [23, 1.0, 100] calculated weights from data analysis
                },
                'roi_classif': {
                    'loss_type': 'cross_entropy',
                    'class_weights': None # [23, 1.0, 100]
                },
                'heatmap': {
                    'loss_type': 'smooth_l1',
                    'smooth_l1_beta': 0.1
                },
                'segmentation': {
                    'dice_weight': 0.5, 
                    'bce_weight': 0.5, 
                    'dice_smooth': 1e-6
                },
                'severity': {
                    'loss_type': 'ce', 
                    'class_weights': None
                }
            }
            
        else:
            self.config = loss_config
            self._validate_config()
        
        # init uncertainty parameters for MTTL
        if self.config['use_uncertainty']:
            self.log_vars = nn.ParameterDict()
            for task in self.config['tasks']:
                init_val = self.config['init_log_vars'].get(task, -1.0)
                self.log_vars[task] = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
    
    def _validate_config(self):
        """Ensure all required configuration keys exist with defaults"""
        defaults = {
            'detection': {
                'focal_alpha': 0.25, 'focal_gamma': 2.0,
                'objectness_weight': 2.0, 'classification_weight': 1.0, 'bbox_weight': 1.0,
                'hard_negative_ratio': 3.0, 'smooth_l1_beta': 0.1
            },
            'parasitemia': {
                'loss_type': 'clinical_huber', 'huber_delta': 0.1, 'range_penalty': 0.1,
                'clinical_thresholds': [0.0, 0.02, 0.1, 1.0], 
                'threshold_weights': [1.0, 2.0, 1.5, 1.0]
            },
            'heatmap': {
                'focal_alpha': 0.8, 'focal_gamma': 3.0,
                'dice_weight': 0.4, 'focal_weight': 0.4, 'boundary_weight': 0.2,
                'dice_smooth': 1e-6
            }
        }
        
        for key, default_config in defaults.items():
            if key not in self.config:
                self.config[key] = default_config
            else:
                for subkey, default_val in default_config.items():
                    if subkey not in self.config[key]:
                        self.config[key][subkey] = default_val
    
    def parasitemia_loss_fcn(self, predictions, targets=None):
        if targets is None:
            # Throw error and stop
            raise ValueError("Targets must be provided for enhanced parasitemia loss.")

        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        
        config = self.config['parasitemia']
        
        # Ensure predictions and targets are on the same device
        if predictions.device != targets.device:
            targets = targets.to(predictions.device)
        
        if config['loss_type'] == 'clinical_huber':
            base_loss = F.huber_loss(predictions, targets, delta=config['huber_delta'], reduction='none')
            threshold_weights = self._compute_clinical_weights(targets, config)
            weighted_loss = base_loss * threshold_weights.unsqueeze(1)
            range_penalty = self._compute_range_penalty(predictions, config)
            total_loss = weighted_loss + range_penalty.unsqueeze(1)  # ← FIX: Ensure tensor
            return total_loss.mean()
        elif config['loss_type'] == 'mae':
            return F.l1_loss(predictions, targets, reduction='mean')
        elif config['loss_type'] == 'mse':
            return F.mse_loss(predictions, targets, reduction='mean')
        else:
            return F.huber_loss(predictions, targets, reduction='mean')
    
    def _compute_clinical_weights(self, targets, config):
        thresholds = torch.tensor(config['clinical_thresholds'], device=targets.device)
        weights_vals = torch.tensor(config['threshold_weights'], device=targets.device)
        
        weights = torch.ones_like(targets.squeeze())
        
        for i in range(len(thresholds) - 1):
            mask = (targets.squeeze() >= thresholds[i]) & (targets.squeeze() < thresholds[i+1])
            weights[mask] = weights_vals[i]
        
        return weights
    
    def _compute_range_penalty(self, predictions, config):
        below_zero = torch.clamp(-predictions, min=0)
        above_one = torch.clamp(predictions - 1, min=0)
        penalty = (below_zero + above_one) * config['range_penalty']
        return penalty.squeeze()
        
    def heatmap_loss_fcn(self, predictions, targets=None):
        """
        Focal Tversky + Boundary Loss
        """
        if targets is None:
            raise ValueError("Heatmap targets must be provided for loss calculation.")

        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        if targets.shape[-2:] != predictions.shape[-2:]:
            targets = F.interpolate(predictions, size=targets.shape[-2:],
                                  mode='bilinear', align_corners=False)

        pred_probs = predictions
        
        alpha = 0.7
        beta = 0.3
        gamma = 0.75
        smooth = 1e-5 # Increased smooth term

        tp = (pred_probs * targets).sum()
        fp = (pred_probs * (1 - targets)).sum()
        fn = ((1 - pred_probs) * targets).sum()

        # Promote to float32 for the division 
        denominator = tp.float() + alpha * fp.float() + beta * fn.float() + smooth
        tversky_index = (tp.float() + smooth) / denominator
        
        # Clamp the index to prevent negative input to pow() 
        tversky_index_clamped = torch.clamp(tversky_index, 0.0, 1.0)
        focal_tversky_loss = (1 - tversky_index_clamped).pow(gamma)

        boundary_loss = self._boundary_loss_fcn(pred_probs, targets)
        total_loss = 0.8 * focal_tversky_loss + 0.2 * boundary_loss

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"WARNING: NaN/Inf in heatmap_loss_fcn. Returning 0.")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return total_loss

    def _spatial_focal_loss(self, pred_logits, target, config):
        alpha, gamma = config['focal_alpha'], config['focal_gamma']
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        pred_probs = torch.sigmoid(pred_logits)
        p_t = pred_probs * target + (1 - pred_probs) * (1 - target)
        
        focal_weight = alpha * (1 - p_t).pow(gamma)
        return (focal_weight * bce_loss).mean()
    
    def _dice_loss(self, pred, target, config):
        smooth = config.get('dice_smooth', 1e-5)
        pred_flat, target_flat = pred.view(-1), target.view(-1)
        intersection = (pred_flat.float() * target_flat.float()).sum()
        denominator = pred_flat.float().sum() + target_flat.float().sum() + smooth
        dice_coeff = (2.0 * intersection + smooth) / denominator
        return 1 - dice_coeff

    def _boundary_loss_fcn(self, pred, target):
        # Sobel filters to detect edges
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)

        # Calculate gradients (edges) of the target and prediction
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        
        # We now take the boundary loss as the L1 distance between the target edges and predicted edges
        boundary_loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
        
        return boundary_loss
    
    def segmentation_loss_fcn(self, predictions, targets=None):
        """Segmentation loss: Dice + BCE"""
        if targets is None:
            raise ValueError("Segmentation targets must be provided.")
        config = self.config['segmentation']

        # Upsample predictions to match target size
        if predictions.shape[-2:] != targets.shape[-2:]:
            predictions = F.interpolate(
                predictions, 
                size=targets.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )

        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='mean')
        pred_probs = torch.sigmoid(predictions)
        dice_loss = self._dice_loss(pred_probs, targets, config)
        total_loss = config['dice_weight'] * dice_loss + config['bce_weight'] * bce_loss
        return total_loss
    
    def severity_loss_fcn(self, predictions, targets=None):
        """Severity classification loss CrossEntropy"""
        if targets is None:
            raise ValueError("Severity targets must be provided.")
        config = self.config['severity']
        if config['class_weights'] is not None:
            weights = torch.tensor(config['class_weights'], device=predictions.device)
        else:
            weights = None
        return F.cross_entropy(predictions, targets, weight=weights)
    
    def cell_classification_loss_fcn(self, predictions, targets=None):
        if targets is None:
            raise ValueError("Cell classification targets must be provided.")
        
        config = self.config.get('cell_classif', {})
        class_weights = config.get('class_weights', None)
        
        if class_weights is not None:
            weights = torch.tensor(class_weights, device=predictions.device, dtype=torch.float32)
        else:
            weights = None
            
        return F.cross_entropy(predictions, targets, weight=weights)
  
    def forward(self, outputs, targets=None):
        """
        Forward Pass: Single Task (raw) OR Multi-Task (uncertainty)
        
        Returns:
            total_loss: Combined weighted loss
            loss_breakdown: Detailed loss components  
            uncertainty_info: Uncertainty parameters (if MTTL)
        """
        individual_losses = {}
        loss_breakdown = {}
        uncertainty_info = {}
        
        if 'detection_loss' in outputs and 'detection' in self.config['tasks']:
            individual_losses['detection'] = outputs['detection_loss']
            
        if 'cell_classif' in outputs and 'cell_classif' in self.config['tasks']:
            classif_targets = targets.get('label') if targets else None 
            classif_loss = self.cell_classification_loss_fcn(outputs['cell_classif'], classif_targets)
            individual_losses['cell_classif'] = classif_loss
        
        if 'parasitemia' in outputs and 'parasitemia' in self.config['tasks']:
            parasitemia_targets = targets.get('parasitemia') if targets else None
            parasitemia_loss = self.parasitemia_loss_fcn(outputs['parasitemia'], parasitemia_targets)
            individual_losses['parasitemia'] = parasitemia_loss
        
        if 'heatmap' in outputs and 'heatmap' in self.config['tasks']:
            heatmap_targets = targets.get('heatmap') if targets else None
            if heatmap_targets is not None:
                heatmap_loss = self.heatmap_loss_fcn(outputs['heatmap'], heatmap_targets)
                individual_losses['heatmap'] = heatmap_loss
            
        if 'segmentation' in outputs and 'segmentation' in self.config['tasks']:
            seg_targets = targets.get('segmentation') if targets else None
            seg_loss = self.segmentation_loss_fcn(outputs['segmentation'], seg_targets)
            individual_losses['segmentation'] = seg_loss
            
        if 'severity' in outputs and 'severity' in self.config['tasks']:
            sev_targets = targets.get('severity_class') if targets else None
            sev_loss = self.severity_loss_fcn(outputs['severity'], sev_targets)
            individual_losses['severity'] = sev_loss
            
        if 'roi_classif' in outputs and 'roi_classif' in self.config['tasks']:
            roi_classif_targets = outputs.get('roi_classif_labels')
            if roi_classif_targets is not None:
                roi_classif_loss = self.cell_classification_loss_fcn(outputs['roi_classif'], roi_classif_targets)
                individual_losses['roi_classif'] = roi_classif_loss
        
        # Combine losses based on mode
        if self.config['use_uncertainty']:
            # MTTL Mode uses Uncertainty weighting
            # Formula: L = Σ[1/(2σᵢ²) × Lᵢ + log(σᵢ)]
            total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

            for task, raw_loss in individual_losses.items():
                if task in self.log_vars:
                    # Promote the raw loss to float32 for AMP stability
                    loss_val = raw_loss.float()
                    
                    # check for NaN/Inf in the raw task loss
                    if not torch.isfinite(loss_val):
                        print(f"\n[CRITICAL LOSS WARNING] Non-finite raw loss for task '{task}'. Skipping its contribution.")
                        loss_breakdown[f'{task}_raw'] = float('inf')
                        continue # Skip this task for this batch
                    
                    # Clamp the raw loss to prevent extreme values from destabilizing the uncertainty parameters
                    loss_val = torch.clamp(loss_val, max=10.0)
                    loss_breakdown[f'{task}_raw'] = loss_val.item()
                        
                    # Get the uncertainty parameter and clamp it to a safe range
                    log_var = torch.clamp(self.log_vars[task], min=-15.0, max=15.0)
                    
                    # Perform the uncertainty calculation
                    precision = torch.exp(-log_var)
                    weighted_loss = 0.5 * precision * loss_val + 0.5 * log_var
                    
                    # add to total_loss if the result is valid
                    if torch.isfinite(weighted_loss):
                        total_loss = total_loss + weighted_loss
                        loss_breakdown[f'{task}_weighted'] = weighted_loss.item()
                    else:
                        print(f"\n[CRITICAL LOSS WARNING] Weighted loss for '{task}' became non-finite. Skipping contribution.")
                        loss_breakdown[f'{task}_weighted'] = float('inf') 
                    
                    # Log diagnostics safely using the stable values.
                    sigma_squared = torch.exp(log_var)
                    uncertainty_info[task] = {
                        'log_var': log_var.item(),
                        'sigma_squared': sigma_squared.item(),
                        'sigma': torch.sqrt(sigma_squared).item(),
                        'precision': precision.item()
                    }
            
        else:
            # Single Task Mode: RAW loss only
            if len(individual_losses) == 1:
                task_name = list(individual_losses.keys())[0]
                total_loss = individual_losses[task_name]
                loss_breakdown[f'{task_name}_raw'] = total_loss.item()
                loss_breakdown['mode'] = 'single_task_enhanced'
            else:
                # Multi-task without uncertainty
                total_loss = sum(individual_losses.values())
                for task, loss_val in individual_losses.items():
                    loss_breakdown[f'{task}_raw'] = loss_val.item()
                loss_breakdown['mode'] = 'multi_task_enhanced'
        
        return total_loss, loss_breakdown, uncertainty_info