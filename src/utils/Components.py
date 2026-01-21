# ============================================================================
# Components for our MTTL Model ( Backbone, Adapters, Heads )
# Gwade Steve
# MTTL and Application to Malaria Detection
# April 2025
# ============================================================================
import sys
sys.path.append('..')
import math
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context('notebook')
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

from collections import OrderedDict
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from DataUtils import set_seeds

# Check the available device
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Lora Adapter for PEFT
class LoRAAdapter(nn.Module):
    """
    LoRA (Low-Rank Adaptation) for task-specific adaptation and PEFT
    """
    
    def __init__(self, input_dim, rank=16, alpha=32, dropout=0.1):
        super(LoRAAdapter, self).__init__()
        
        self.input_dim = input_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # W = W_0 + (alpha/rank) * B * A
        self.lora_A = nn.Linear(input_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, input_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # we init weights properly
        self._initialize_weights()
        
        # Calculate parameters
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _initialize_weights(self):
        """Init LoRA weights following best and paper practices"""
        # A: Kaiming uniform (like in the paper)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        # B: Zero initialization (starts as identity)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        """
        Forward pass -> x + scaling * B(A(x))
        """
        original_shape = x.shape
        
        # Handle input dimensions
        if x.dim() == 4:
            # Spatial features [B, C, H, W] -> [B*H*W, C]
            batch_size, channels, height, width = x.shape
            x_reshaped = x.view(batch_size, channels, -1).permute(0, 2, 1)  # [B, H*W, C]
            x_reshaped = x_reshaped.contiguous().view(-1, channels)  # [B*H*W, C]
        elif x.dim() == 2:
            # Global features [B, C]
            x_reshaped = x
        else:
            raise ValueError(f"Unsupported input dimension: {x.shape}")
        
        # LoRA transformation: scaling * B(A(x))
        lora_output = self.lora_A(self.dropout(x_reshaped))  # [*, rank]
        lora_output = self.lora_B(lora_output)  # [*, input_dim]
        adapted_x = x_reshaped + self.scaling * lora_output  # res connection
        
        # Reshape back to original format
        if len(original_shape) == 4:
            adapted_x = adapted_x.view(original_shape[0], -1, original_shape[1])  # [B, H*W, C]
            adapted_x = adapted_x.permute(0, 2, 1)  # [B, C, H*W]
            adapted_x = adapted_x.view(original_shape)  # [B, C, H, W]
        
        return adapted_x
    
    def get_adapter_info(self):
        """LoRA adapter information"""
        return {
            'input_dim': self.input_dim,
            'rank': self.rank,
            'alpha': self.alpha,
            'scaling': self.scaling,
            'total_params': self.total_params,
            'trainable_params': self.trainable_params,
            'compression_ratio': self.input_dim / self.rank,
            'parameter_efficiency': f"{(1 - self.total_params/(self.input_dim*self.input_dim))*100:.1f}%"
        }

# Resnet Block with integrated LoRA Adapters
class AdaptedResNetBlock(nn.Module):
    """ResNet block with integrated LoRA adapters for task-specific feature learning"""
    
    def __init__(self, original_block, adapter_config=None):
        super(AdaptedResNetBlock, self).__init__()
        
        self.original_block = original_block
        self.use_adapters = adapter_config is not None
        
        if self.use_adapters:
            # We start by robustly determining the output feature dimension of the block.
            # For both ResNet-18 (BasicBlock) and ResNet-50 (Bottleneck).
            if hasattr(original_block, 'conv3'):
                # This is a Bottleneck block (used in ResNet-50+).
                # The final output dimension comes from the last 1x1 conv layer.
                feat_dim = original_block.conv3.out_channels
            elif hasattr(original_block, 'conv2'):
                # This is a BasicBlock (used in ResNet-18/34).
                # The final output dimension comes from the second 3x3 conv layer.
                feat_dim = original_block.conv2.out_channels
            else:
                # Other blocks, we try to infer dimension
                raise TypeError(f"Unknown ResNet block type: {type(original_block)}")
                #feat_dim = next(original_block.parameters()).shape[0]
            
            # Task-specific adapters
            self.adapters = nn.ModuleDict()
            for task in adapter_config.get('tasks', []):
                self.adapters[task] = LoRAAdapter(
                    input_dim=feat_dim,
                    rank=adapter_config.get(f'{task}_rank', 8),
                    alpha=adapter_config.get(f'{task}_alpha', 16),
                    dropout=adapter_config.get(f'{task}_dropout', 0.05)
                )
    
    def forward(self, x, task=None):
        """Forward with optional task-specific adaptation"""
        # Original block forward
        x = self.original_block(x)
        
        # Use task-specific adaptation if enabaled
        if self.use_adapters and task is not None and task in self.adapters:
            x = self.adapters[task](x)
        
        return x

# Backbone network with full wrapping for adapters
class BackboneNetwork(nn.Module):
    """
    ResNet backbone with integrated task-specific LoRA adapters
    Adapters are injected INTO the backbone layers for task specialization
    """
    
    def __init__(self, architecture='resnet18', pretrained=True, freeze_backbone=True, adapter_config=None, unfreeze_layers=None, mttl_mode=False):
        super(BackboneNetwork, self).__init__()
        
        self.architecture = architecture
        self.freeze_backbone = freeze_backbone
        self.adapter_config = adapter_config
        self.unfreeze_layers = unfreeze_layers
        self.mttl_mode = mttl_mode
        
        # Load pretrained ResNet (based on the config resnet18 or resnet50, more architectures can be explored :) )
        if architecture == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = resnet18(weights=weights)
            self.feature_dims = [64, 128, 256, 512]
        elif architecture == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = resnet50(weights=weights)
            self.feature_dims = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Extract and wrap different stages with adapters ( wrappers )
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Wrap each block with adapters if specified
        self.layer1 = self._wrap_layer_with_adapters(resnet.layer1)  # 1/4 resolution -> C1
        self.layer2 = self._wrap_layer_with_adapters(resnet.layer2)  # 1/8 resolution -> C2
        self.layer3 = self._wrap_layer_with_adapters(resnet.layer3)  # 1/16 resolution -> C3
        self.layer4 = self._wrap_layer_with_adapters(resnet.layer4)  # 1/32 resolution -> C4
        
        self.avgpool = resnet.avgpool
        
        # freezing logic
        self._set_backbone_trainability()
        
        # Freeze backbone if specified (but keep adapters trainable)
        #if freeze_backbone:
        #    self._freeze_backbone_weights()
        
        # Calculate parameters
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        adapter_params = sum(p.numel() for n, p in self.named_parameters() if 'adapters' in n)
        
        print(f"BackboneNetwork ({architecture}) with integrated adapters initialized")
        print(f"Adapter config: {adapter_config}")
        print(f"Parameters: {self.total_params:,} total, {self.trainable_params:,} trainable")
        if adapter_config:
            print(f"Adapter parameters: {adapter_params:,} ({adapter_params/self.total_params*100:.1f}%)")
        
    def _set_backbone_trainability(self):
        # Freeze everything except adapters by default
        for name, param in self.named_parameters():
            if 'adapters' not in name:
                param.requires_grad = False

        # Unfreeze only specified blocks in each layer
        if self.mttl_mode and self.unfreeze_layers is not None:
            for layer_name, depth in self.unfreeze_layers.items():
                layer = getattr(self, layer_name, None)
                if layer is not None:
                    if depth == 'all':
                        # Unfreeze all blocks in this layer
                        for block in layer:
                            for param in block.original_block.parameters():
                                param.requires_grad = True
                    else:
                        # Unfreeze only the last n 'depth' blocks
                        for block in layer[-depth:]:
                            for param in block.original_block.parameters():
                                param.requires_grad = True

        print(f"Backbone weights frozen (except adapters), with selective unfreezing: {self.unfreeze_layers if self.mttl_mode else None}")
    
    def _wrap_layer_with_adapters(self, layer):
        """Wrap each block in a layer with adapter"""
        if self.adapter_config is None:
            return layer
        
        # Wrap each block with adapters
        adapted_blocks = nn.ModuleList()
        for block in layer:
            adapted_block = AdaptedResNetBlock(block, self.adapter_config)
            adapted_blocks.append(adapted_block)
        
        return adapted_blocks
    
    def _freeze_backbone_weights(self):
        """Freeze backbone weights but keep adapters trainable"""
        for name, param in self.named_parameters():
            if 'adapters' not in name:
                param.requires_grad = False
        
        print(f"Backbone weights frozen, adapters remain trainable")
    
    def forward(self, x, task=None, return_multiscale=False):
        """
        Forward pass with task-specific adaptation
        Args:
            x: Input images [B, 3, H, W]
            task: Task name for adapter selection ('detection', 'segmentation', 'heatmap', ... as i will be adding more tasks)
            return_multiscale: Whether to return multi-scale features
        """
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"Expected input shape [B, 3, H, W], got {x.shape}")
        
        # Initial convolution and poolinfg (18/50)
        x = self.conv1(x)      # [B, 64/256, 128, 128] since my input is 512x512 
        x = self.bn1(x)         
        x = self.relu(x)
        x = self.maxpool(x)    # [B, 64/256, 64, 64]
        
        # MS features with task-specific adaptation 18/50
        c1 = self._forward_layer(self.layer1, x, task)      # [B, 64/256, 64, 64]
        c2 = self._forward_layer(self.layer2, c1, task)     # [B, 128/512, 32, 32]
        c3 = self._forward_layer(self.layer3, c2, task)     # [B, 256/1024, 16, 16] 
        c4 = self._forward_layer(self.layer4, c3, task)     # [B, 512/2048, 8, 8]
        
        if return_multiscale:
            return {
                'c1': c1,  # High-resolution for heatmaps and small parasites
                'c2': c2,  # Medium-resolution for standard detection
                'c3': c3,  # Low-resolution for larger objects
                'c4': c4   # Global context features
            }
        else:
            # Global average pooling for single feature vector 
            global_features = self.avgpool(c4)  # [B, 512/2048, 1, 1]
            global_features = global_features.view(global_features.size(0), -1)  # [B, 512/2048]
            return global_features
    
    def _forward_layer(self, layer, x, task):
        """Forward through a layer with task-specific adaptation"""
        if self.adapter_config is None:
            return layer(x)
        
        # Forward through each adapted block
        for block in layer:
            x = block(x, task=task)
        return x
    
    def get_feature_dim(self):
        """Return the dimension of global features"""
        return self.feature_dims[-1]
    
    def get_multiscale_dims(self):
        """Return dimensions for multi-scale features"""
        return self.feature_dims
    
    def get_feature_info(self):
        """Return comprehensive feature information"""
        return {
            'architecture': self.architecture,
            'feature_dims': self.feature_dims,
            'total_params': self.total_params,
            'trainable_params': self.trainable_params,
            'frozen': self.freeze_backbone,
            'enhanced': True,
            'adapters_integrated': self.adapter_config is not None,
            'feature_resolutions': {
                'c1': '128x128 (1/4) for 512x512 input', 
                'c2': '64x64 (1/8) for 512x512 input', 
                'c3': '32x32 (1/16) for 512x512 input',
                'c4': '16x16 (1/32) for 512x512 input'
            },
            'use_cases': {
                'c1': 'Small parasites, fine details',
                'c2': 'Standard RBC detection',
                'c3': 'Larger cellular structures', 
                'c4': 'Global image context'
            }
        }

# Focal loss class for detection box head
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [N, C] logits
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        # red method
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# Feature Pyramid Network (FPN) modeule
class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network module.
    It takes multi-scale features from backbone and creates a feature pyramid.
    """
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        # Init weights
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, multiscale_features):
        # We pass the input features in order from lowest resolution to highest
        # from c4, c3, c2, ...
        laterals = [
            lat_conv(multiscale_features[i]) for i, lat_conv in enumerate(self.lateral_convs)
        ]

        fpn_outputs = []
        # Build top-down path (from smallest feature map up)
        for i in range(len(laterals) - 1, -1, -1):
            if i == len(laterals) - 1:
                p = laterals[i]
            else:
                upsampled = F.interpolate(fpn_outputs[-1], scale_factor=2, mode='nearest')
                p = laterals[i] + upsampled
            fpn_outputs.append(self.fpn_convs[i](p))
        
        # Reverse to get order from largest map to smallest (p2, p3, p4)
        fpn_outputs.reverse()
        
        # Create an OrderedDict for compatibility with torchvision's detection models
        out = OrderedDict()
        for i, fpn_feat in enumerate(fpn_outputs):
            out[str(i)] = fpn_feat # Keys are '0', '1', '2', ...
            
        return out

# Custom box head for our EfficientRCNNHead
class LightweightBoxHead(nn.Module):
    """
    Lightweight box head using depthwise separable convolutions
    to replace the default big 2 fc layers in Faster R-CNN
    """
    def __init__(self, in_channels, representation_size):
        super().__init__()
        mid_channels = in_channels * 4

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(in_channels * 7 * 7, representation_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu(x)
        return x

# Our custom version of Faster R-CNN head with Focal Loss and custom Box Head
class EfficientRCNNHead(nn.Module):
    def __init__(self, num_classes=1, fpn_dim=64, rpn_batch_size_per_image=256, rpn_positive_fraction=0.5, roi_batch_size_per_image=512, roi_positive_fraction=0.25): 
        super().__init__()
        
        # Our anchors and AR generated after Kmeans analysis on the annotation sizes and scales
        anchor_sizes = ((28,), (56,), (112,))
        aspect_ratios = ((0.75, 0.85, 0.91, 0.96, 1.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        
        rpn_head = RPNHead(fpn_dim, anchor_generator.num_anchors_per_location()[0])
        
        rpn_pre_nms_top_n = dict(training=4000, testing=2000)
        rpn_post_nms_top_n = dict(training=4000, testing=2000)

        self.rpn = RegionProposalNetwork(
            anchor_generator, rpn_head,
            fg_iou_thresh=0.7, bg_iou_thresh=0.3,
            batch_size_per_image=rpn_batch_size_per_image, 
            positive_fraction=rpn_positive_fraction,
            pre_nms_top_n=rpn_pre_nms_top_n,
            post_nms_top_n=rpn_post_nms_top_n,
            nms_thresh=0.7,
        )

        # Expects features named '0', '1', '2', ... that we get from our FPN 
        box_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2'], output_size=7, sampling_ratio=2)
        
        representation_size = 256
        
        # We use our custom lightweight box head
        box_head = LightweightBoxHead(in_channels=fpn_dim, representation_size=representation_size)

        # Custom predictor with num_classes + 1 (for background) 0 bg and the rest ... original + 1
        class Predictor(nn.Module):
            def __init__(self, rep_size, num_classes):
                super().__init__()
                self.cls_score = nn.Linear(rep_size, num_classes + 1)
                self.bbox_pred = nn.Linear(rep_size, (num_classes + 1) * 4)
            def forward(self, x):
                return self.cls_score(x), self.bbox_pred(x)

        box_predictor = Predictor(representation_size, num_classes)

        self.roi_heads = RoIHeads(
            box_roi_pool, box_head, box_predictor,
            fg_iou_thresh=0.5, bg_iou_thresh=0.5,
            batch_size_per_image=roi_batch_size_per_image, 
            positive_fraction=roi_positive_fraction,
            bbox_reg_weights=None,
            score_thresh=0.05, nms_thresh=0.5, detections_per_img=300,
        )
        
        # Changed loss to use our focal loss
        self.roi_heads.box_head.loss = FocalLoss(alpha=0.25, gamma=2.0)

        # No transform here, DataUtils handles that
        self.transform = GeneralizedRCNNTransform(
            min_size=512, max_size=512,
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0]
        )

        # Calculate parameters
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"EfficientRCNNHead: {num_classes} classes, expects pre-computed FPN features.")
        print(f"Total Params: {self.total_params:,}")
        print(f"Trainable Params: {self.trainable_params:,}")

    def forward(self, fpn_features, images, targets=None):
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)

        # The RPN and RoIHeads work directly with the FPN features
        proposals, proposal_losses = self.rpn(images, fpn_features, targets)
        detections, detector_losses = self.roi_heads(fpn_features, proposals, images.image_sizes, targets)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        return detections

    def get_output_info(self):
        return {
            'num_classes': self.roi_heads.box_predictor.cls_score.out_features - 1,
            'total_params': self.total_params,
            'trainable_params': self.trainable_params
        }

# Custom YOLODetectionHead 
class YOLODetectionHead(nn.Module):
    """
    Our YOLO-style detection head for small parasites with Focal Loss
    We use FPN-like multi-scale fusion + efficient depthwise convolutions + Focal Loss
    """
    
    def __init__(self, feature_dims, num_classes=3, grid_sizes=[32, 16, 8]): # Updated grid sizes for 512x512 input
        super(YOLODetectionHead, self).__init__()
        
        self.num_classes = num_classes
        self.grid_sizes = grid_sizes 
        self.feature_dims = feature_dims
        
        # FPN-like feature fusion 
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        # Lateral connections for FPN 
        fpn_dim = 64  # FPN feature dimension
        for feat_dim in feature_dims[-3:]:  # c2, c3, c4 ( order )
            self.lateral_convs.append(
                nn.Conv2d(feat_dim, fpn_dim, 1)  # 1x1 conv to reduce channels
            )
        
        # FPN output convolutions
        for _ in range(3):  # 3 scales
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, groups=fpn_dim),  # Depthwise
                    nn.Conv2d(fpn_dim, fpn_dim, 1),  # Pointwise
                    nn.BatchNorm2d(fpn_dim),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Detection heads for each scale with Focal Loss optimization
        self.yolo_heads = nn.ModuleList()
        
        for i in range(3):
            output_dim = 1 + num_classes + 4
            head = nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, groups=fpn_dim),
                nn.Conv2d(fpn_dim, 32, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, output_dim, 1)
            )
            self.yolo_heads.append(head)
                 
                
        self._initialize_weights()
        
        # Calculate parameters
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"YOLODetectionHead: {num_classes} classes ready")
        print(f"Grid sizes (512x512): {grid_sizes}")
        print(f"Total Params: {self.total_params:,}")
        print(f"Trainable Params: {self.trainable_params:,}")
    
    def _initialize_weights(self):
        """Init for Focal Loss"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    # Special init for classification bias (Focal Loss)
                    if module.out_channels == 1 + self.num_classes + 4:
                        # We init class prediction bias for rare class handling
                        prior_prob = 0.01  # Assumes 1% positive rate for parasites, seen from data analysis
                        bias_init = -math.log((1 - prior_prob) / prior_prob)
                        # Set bias for class predictions only
                        with torch.no_grad():
                            module.bias[1:1+self.num_classes].fill_(bias_init)
                    else:
                        nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, multiscale_features):
        """
        Forward pass for det head, Here we
        decode raw network logits into objectness probabilities and grid-space bbox coordinates.
        """
        
        # Extract features
        feature_keys = ['c2', 'c3', 'c4']
        features = [multiscale_features[key] for key in feature_keys]
        
        # FPN lateral connections
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        fpn_features = []
        # build featuremap from smallest to largest
        for i in range(len(laterals)):
            if i == 0:
                # The first feature is the processed top-level feature
                fpn_feat = laterals[-(i+1)] 
            else:
                # Subsequent/Next features are an addition of the upsampled previous FPN feature and the corresponding lateral connection from the backbone.
                upsampled = F.interpolate(fpn_features[-1], scale_factor=2, mode='nearest')
                fpn_feat = laterals[-(i+1)] + upsampled
            
            # a final conv block to refine the fused features
            fpn_feat = self.fpn_convs[len(laterals)-1-i](fpn_feat)
            fpn_features.append(fpn_feat)
        
        # Reverse to get [largest_map, medium_map, smallest_map] order
        fpn_features = fpn_features[::-1]
        
        # Detection heads with improved activations
        detections = []
        for i, fpn_feat in enumerate(fpn_features):
            # raw logits from pred head (for this scale), yolo_output shape: [B, num_classes + 5, H, W]
            yolo_output = self.yolo_heads[i](fpn_feat)
            
            # Reshape, new shape: [B, H, W, num_classes + 5]
            yolo_output = yolo_output.permute(0, 2, 3, 1).contiguous()
            
            # Separate the raw logits for each component of the prediction
            raw_objectness = yolo_output[..., 0:1]
            raw_class_logits = yolo_output[..., 1:1+self.num_classes]
            raw_bbox_logits = yolo_output[..., 1+self.num_classes:]

            # activation functions to decode the logits
            # Objectness: Sigmoid to get a probability [0, 1]
            objectness_prob = torch.sigmoid(raw_objectness)
            
            # Class scores: Softmax for multi-class probabilities
            class_probs = torch.softmax(raw_class_logits, dim=-1)

            # BBox coordinates: Sigmoid for xy offsets, Exp for wh scales
            bbox_xy_offset = torch.sigmoid(raw_bbox_logits[..., :2])
            
            # clamp for numerical stability
            bbox_wh_scale = torch.exp(raw_bbox_logits[..., 2:]).clamp(max=1000.0)
            bbox_coords = torch.cat([bbox_xy_offset, bbox_wh_scale], dim=-1)
            
            # Concatenate the decoded parts for scale output
            detection_output = torch.cat([objectness_prob, raw_class_logits, bbox_coords], dim=-1)
            
            detections.append(detection_output)
            
        return detections
    
    def get_output_info(self):
        """Return information about our yolo detection outputs"""
        return {
            'num_scales': len(self.grid_sizes),
            'grid_sizes': self.grid_sizes,
            'num_classes': self.num_classes,
            'output_format': 'objectness(1) + classes({}) + bbox_center_wh(4)'.format(self.num_classes),
            'total_outputs_per_cell': 1 + self.num_classes + 4,
            'total_params': self.total_params,
            'trainable_params': self.trainable_params
        }

# Parasitemia head
class ParasitemiaHead(nn.Module):
    """
    We go for a simple regression head with BatchNorm and output clamping for safety.
    The aim is toPredict a single value per sample (percentage, clamped to [0, 100] at inference).
    """
    def __init__(self, feature_dim, bottleneck_dim=64):
        super(ParasitemiaHead, self).__init__()
        self.feature_dim = feature_dim
        self.bottleneck_dim = bottleneck_dim

        self.layers = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim // 2, 1)  # Output: raw percentage
        )

        self._initialize_weights()

        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"ParasitemiaHead: {feature_dim} -> {bottleneck_dim} -> 1 (BatchNorm, clamped output)")
        print(f"Total params: {self.total_params:,}")
        print(f"Trainable params: {self.trainable_params:,}")

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, global_features, clamp_output=True):
        """
        Forward pass returns the parasitemia score (clamped to [0, 100] if clamp_output=True).
        Argzs:
            global_features: [B, feature_dim]
            clamp_output: bool, whether to clamp output to [0, 100]
        What we return:
            [B, 1] percentage (float, clamped if we need to)
        """
        if global_features.dim() != 2:
            raise ValueError(f"Expected 2D global features [B, D], got {global_features.shape}")
        output = self.layers(global_features)
        if clamp_output:
            output = torch.clamp(output, min=0.0, max=100.0)
        return output

    def get_output_info(self):
        return {
            'input_dim': self.feature_dim,
            'bottleneck_dim': self.bottleneck_dim,
            'output_dim': 1,
            'total_params': self.total_params,
            'trainable_params': self.trainable_params
        }

# LocalizationHeatmapHead 
class LocalizationHeatmapHead(nn.Module):
    """
    localization head with attention for parasite localization
    What we use is Multi-scale fusion + spatial attention + depthwise separable convs
    to generate high-res heatmaps indicating parasite centers.
    """
    
    def __init__(self, feature_dims, heatmap_size=(64, 64)):  # Output heatmap size
        super(LocalizationHeatmapHead, self).__init__()
        
        self.feature_dims = feature_dims
        self.heatmap_size = heatmap_size
        
        # MS feature fusion using c1 and c2
        unified_dim = 64
        
        # Fuse c1 and c2 features for high-resolution localization
        self.c1_proj = nn.Conv2d(feature_dims[0], unified_dim, 1)  # c1 projection
        self.c2_proj = nn.Conv2d(feature_dims[1], unified_dim, 1)  # c2 projection
        
        # Spatial attention module
        #self.spatial_attention = nn.Sequential(
        #    nn.Conv2d(unified_dim * 2, 64, 3, padding=1),
        #    nn.ReLU(inplace=True),
        #    nn.Conv2d(64, 1, 1),
        #    nn.Sigmoid()
        #)
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(unified_dim * 2, unified_dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(unified_dim // 2, 1, 1),
            #nn.Sigmoid()
        )
        
        def SeparableConvBlock(in_channels, out_channels):
            return nn.Sequential(
                # depthwise conv
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                # pointwise conv
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        fused_dim = unified_dim * 2 # 128 !! Note
        self.heatmap_head = nn.Sequential(
            SeparableConvBlock(fused_dim, 64),
            SeparableConvBlock(64, 32),
            nn.Conv2d(32, 1, kernel_size=1), # Final 1x1 conv to get single channel
            nn.Sigmoid()
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d(heatmap_size)        
        self._initialize_weights()
        
        # Calculate parameters
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"LocalizationHeatmapHead Ready")
        print(f"Output size: {heatmap_size}")
        print(f"Total Params: {self.total_params:,}")
        print(f"Trainable Params: {self.trainable_params:,}")
    
    def _initialize_weights(self):
        """Init localization head weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, multiscale_features):
        """
        Our forward with attention and multi-scale fusion
        Arguments:
            multiscale_features: Dict with keys ['c1', 'c2', ...] in ordder of resolution
        Returns:
            heatmap: [B, 1, H, W] localization heatmap
        """
        # c1 and c2 for better localization
        if 'c1' not in multiscale_features:
            raise ValueError("Missing 'c1' features for heatmap generation")
        
        c1 = multiscale_features['c1'] 
        
        # Handle both c1 only and c1+c2 
        if 'c2' in multiscale_features:
            # Multi-scale fusion
            c2 = multiscale_features['c2'] 
            
            # Project to uni dimension
            c1_proj = self.c1_proj(c1)  # [B, 128, H1, W1]
            c2_proj = self.c2_proj(c2)  # [B, 128, H2, W2]
            
            # Upsample c2 to match c1 size
            c2_upsampled = F.interpolate(c2_proj, size=c1_proj.shape[2:], mode='bilinear', align_corners=False)
            
            # Feature fusion
            fused_features = torch.cat([c1_proj, c2_upsampled], dim=1)  # [B, 256, H1, W1]
            
            # Apply spatial attention
            attention_map = self.spatial_attention(fused_features)  # [B, 1, H1, W1]
            attended_features = fused_features * attention_map
            
            # Generate heatmap
            heatmap = self.heatmap_head(attended_features)  # [B, 1, H1, W1]
        else:
            # else we just Use only c1 features
            c1_proj = self.c1_proj(c1)
            
            # deuplicate c1 for fusion (fallback)
            fused_features = torch.cat([c1_proj, c1_proj], dim=1)
            
            # Generate heatmap
            heatmap = self.heatmap_head(fused_features)
        
        # final output
        heatmap = self.adaptive_pool(heatmap)  # [B, 1, target_H, target_W]
        
        return heatmap
    
    def get_output_info(self):
        """Return information about heatmap outputs"""
        return {
            'input_dim': self.feature_dims[0],
            'output_size': self.heatmap_size,
            'output_channels': 1,
            'output_range': '0.0 to 1.0',
            'use_case': 'Spatial localization of parasites',
            'backward_compatible': True,
            'total_params': self.total_params,
            'trainable_params': self.trainable_params
        }

# SegmentationHead
class SegmentationHead(nn.Module):
    """
    Here we go for a lightweight, UNet-style decoder for binary cell segmentation. We
    Use depthwise separable convolutions for keeping params low.
    """
    def __init__(self, feature_dims, out_size=(64, 64)):
        super(SegmentationHead, self).__init__()

        def SeparableConvBlock(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        # Upsampling layers to decode feature maps
        #self.up_c4 = nn.ConvTranspose2d(feature_dims[3], 128, kernel_size=2, stride=2)
        #self.up_c3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        #self.up_c2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        # Fusion blocks using efficient separable convolutions
        #self.fuse3 = SeparableConvBlock(feature_dims[2] + 128, 128)
        #self.fuse2 = SeparableConvBlock(feature_dims[1] + 64, 64)
        #self.fuse1 = SeparableConvBlock(feature_dims[0] + 32, 32)

        # Final prediction layer outputs raw logits for stability
        #self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        
        # Just to test, I reduce upsampling layers
        self.up_c4 = nn.ConvTranspose2d(feature_dims[3], 64, kernel_size=2, stride=2)
        self.up_c3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_c2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)

        # Fusion blocks with reduced channels
        self.fuse3 = SeparableConvBlock(feature_dims[2] + 64, 64)
        self.fuse2 = SeparableConvBlock(feature_dims[1] + 32, 32)
        self.fuse1 = SeparableConvBlock(feature_dims[0] + 16, 16)

        # Final prediction layer
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d(out_size)
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"SegmentationHead Ready:\n")
        print(f"Total Params: {self.total_params:,}")
        print(f"Trainable Params: {self.trainable_params:,}")

    def forward(self, multiscale_features):
        c1, c2, c3, c4 = multiscale_features['c1'], multiscale_features['c2'], multiscale_features['c3'], multiscale_features['c4']
        
        x = self.up_c4(c4)
        x = self.fuse3(torch.cat([x, c3], dim=1))
        
        x = self.up_c3(x)
        x = self.fuse2(torch.cat([x, c2], dim=1))
        
        x = self.up_c2(x)
        x = self.fuse1(torch.cat([x, c1], dim=1))
        
        logits = self.final_conv(x)
        logits = self.adaptive_pool(logits)
        
        return logits # we return raw logits for BCEWithLogitsLoss

    def get_output_info(self):
        return {
            'total_params': self.total_params,
        }

# Severity Head
class SeverityHead(nn.Module):
    """
    A simple light MLP for 4-class severity classification.
    """
    def __init__(self, feature_dim, num_classes=4, bottleneck_dim=64):
        super(SeverityHead, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim),
            #nn.BatchNorm1d(bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(bottleneck_dim, num_classes)
        )
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"SeverityHead initialized for {num_classes} classes:")
        print(f"Total Params: {self.total_params:,}")
        print(f"Trainable Params: {self.trainable_params:,}")

    def forward(self, global_features):
        return self.layers(global_features)

    def get_output_info(self):
        return {
            'total_params': self.total_params,
        }

# Cell Classifier Head
class CellClassifierHead(nn.Module):
    """
    A MLP head for classifying cell feature vectors.
    It takes a feature vector from the backbone's GAP layer.
    """
    def __init__(self, in_features, num_classes=3, representation_size=256):
        super(CellClassifierHead, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(in_features, representation_size),
            nn.BatchNorm1d(representation_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            
            nn.Linear(representation_size, representation_size // 2),
            nn.BatchNorm1d(representation_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            
            nn.Linear(representation_size // 2, num_classes)
        )

        
        self.in_features = in_features 
        self.num_classes = num_classes
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"CellClassifierHead initialized for {num_classes} classes:")
        print(f"Input features: {in_features}, Representation size: {representation_size}")
        print(f"Total Params: {self.total_params:,}, Trainable: {self.trainable_params:,}")

    def forward(self, x):
        # `x` is the feature vector from the backbone, with shape [B, 2048]
        return self.layers(x)

    def get_output_info(self):
        return {
                'input_shape': f'[B, {self.in_features}]',
                'output_shape': f'[B, {self.num_classes}]',
                'total_params': self.total_params,
                'trainable_params': self.trainable_params}

# Task-specific adapters manager
class TaskSpecificAdapters(nn.Module):
    
    def __init__(self, backbone_dims, adapter_config):
        super(TaskSpecificAdapters, self).__init__()
        
        self.backbone_dims = backbone_dims
        self.config = adapter_config
        
        print(f"Adapter config: {adapter_config}")
        
        # We Still create standalone adapters only if specified
        self.adapters = nn.ModuleDict()
        
        # adapters for non-integrated backbone
        if adapter_config.get('standalone_mode', False):
            # Global feature adapters (for parasitemia task)
            global_dim = backbone_dims[-1]
            if 'parasitemia' in adapter_config['tasks']:
                self.adapters['parasitemia_global'] = LoRAAdapter(
                    input_dim=global_dim,
                    rank=adapter_config.get('global_rank', 32),
                    alpha=adapter_config.get('global_alpha', 64),
                    dropout=adapter_config.get('global_dropout', 0.1)
                )
                
            if 'severity' in adapter_config['tasks']:
                self.adapters['severity_global'] = LoRAAdapter(
                    input_dim=global_dim,
                    rank=adapter_config.get('severity_rank', 8),
                    alpha=adapter_config.get('severity_alpha', 16),
                    dropout=adapter_config.get('severity_dropout', 0.05)
                )
                
            if 'segmentation' in adapter_config['tasks']:
                # we Use c1/c2/c3/c4 for segmentation
                for i, scale in enumerate(['c1', 'c2', 'c3', 'c4']):
                    feat_dim = backbone_dims[i]
                    self.adapters[f'segmentation_{scale}'] = LoRAAdapter(
                        input_dim=feat_dim,
                        rank=adapter_config.get('segmentation_rank', 8),
                        alpha=adapter_config.get('segmentation_alpha', 16),
                        dropout=adapter_config.get('segmentation_dropout', 0.05)
                    )
            
            # Spatial feature adapters (for detection and heatmap)
            if 'detection' in adapter_config['tasks']:
                for i, scale in enumerate(['c2', 'c3', 'c4']):
                    feat_dim = backbone_dims[i+1]
                    self.adapters[f'detection_{scale}'] = LoRAAdapter(
                        input_dim=feat_dim,
                        rank=adapter_config.get('detection_rank', 16),
                        alpha=adapter_config.get('detection_alpha', 32),
                        dropout=adapter_config.get('detection_dropout', 0.1)
                    )
            
            if 'heatmap' in adapter_config['tasks']:
                c1_dim = backbone_dims[0]
                self.adapters['heatmap_c1'] = LoRAAdapter(
                    input_dim=c1_dim,
                    rank=adapter_config.get('heatmap_rank', 8),
                    alpha=adapter_config.get('heatmap_alpha', 16),
                    dropout=adapter_config.get('heatmap_dropout', 0.05)
                )
        
        # Calculate total parameters
        self.total_params = sum(p.numel() for p in self.parameters())
        if self.total_params > 0:
            print(f"Standalone adapter parameters: {self.total_params:,}")
        else:
            print("Using integrated backbone adapters (no standalone adapters)")
    
    def adapt_global_features(self, global_features, task):
        adapter_key = f'{task}_global'
        if adapter_key in self.adapters:
            return self.adapters[adapter_key](global_features)
        else:
            # Adapters handled in backbone return as-is
            return global_features
    
    def adapt_multiscale_features(self, multiscale_features, task):
        if not self.adapters:
            # Adapters handled in bb
            return multiscale_features
        
        # SA adapter mode
        adapted_features = {}
        for key, features in multiscale_features.items():
            adapter_key = f'{task}_{key}'
            if adapter_key in self.adapters:
                adapted_features[key] = self.adapters[adapter_key](features)
            else:
                adapted_features[key] = features
        
        return adapted_features
    
    def get_adapter_info(self):
        info = {
            'total_adapters': len(self.adapters), 
            'total_params': self.total_params,
            'supported_tasks': self.config['tasks'],
            'integration_mode': 'integrated' if self.total_params == 0 else 'standalone',
        }
        
        # Add individual adapter information
        for name, adapter in self.adapters.items():
            info[name] = adapter.get_adapter_info()
            
        return info