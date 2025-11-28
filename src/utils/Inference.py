# ====================================================================
# Making inference on new data
# Gwade Steve
# MTTL and Application to Malaria Detection
# April 2025
# ====================================================================

import os
import sys
import json
import argparse
import traceback
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from tqdm import tqdm

try:
    from model import create_model_for_task, create_multitask_model
    from DataUtils import MalariaPreprocessor
    from download_weights import download_models
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    sys.exit(1)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_MAP = {1: 'Infected', 2: 'Healthy', 3: 'WBC'}
COLOR_MAP = {'Infected': 'red', 'Healthy': 'blue', 'WBC': 'purple'}


def find_model_path(models_dir, mode, task, num_classes):
    """Locate model directory based on specifications."""
    mode_folder = "MTTL" if mode.upper() == "MTTL" else "STL"
    
    if task in ['detection', 'roi_classif']:
        path = Path(models_dir) / mode_folder / task / f"{num_classes}-Class"
    else:
        path = Path(models_dir) / mode_folder / task
    
    if not path.is_dir():
        raise FileNotFoundError(f"Model not found: {path}")
    
    return str(path)

def load_model(model_dir):
    """Load model and configuration from directory."""
    config_path = Path(model_dir) / 'config.json'
    with open(config_path) as f:
        config = json.load(f)
    
    active_tasks = config.get('active_tasks', [])
    
    if config.get('mode') == 'MTTL':
        model = create_multitask_model(config, tasks=active_tasks)
    else:
        model = create_model_for_task(config, active_tasks[0])
    
    model.to(DEVICE)
    model.load_state_dict(
        torch.load(Path(model_dir) / 'best_model.pth', map_location=DEVICE)
    )
    
    # Load task heads for MTTL
    if config.get('mode') == 'MTTL':
        for task_name in active_tasks:
            head_name = 'cell_classif' if task_name == 'roi_classif' else task_name
            head_path = Path(model_dir) / f'best_{task_name}_head.pth'
            if head_path.exists() and head_name in model.task_heads:
                model.task_heads[head_name].load_state_dict(
                    torch.load(head_path, map_location=DEVICE)
                )
    
    model.eval()
    return model, config

def prepare_image(image_path, size=(512, 512)):
    """Load and preprocess image for inference."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preprocessor = MalariaPreprocessor(target_size=size)
    img_resized = preprocessor.resize_image(img_rgb)
    img_clahe = preprocessor.clahe_normalization(img_resized)
    
    img_tensor = TF.to_tensor(img_clahe)
    img_normalized = TF.normalize(
        img_tensor, 
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    
    return img_normalized.unsqueeze(0).to(DEVICE), img_clahe

def run_inference(model, image_tensor):
    """Execute model inference."""
    with torch.no_grad():
        if model.config.get('mode') == 'MTTL':
            return model.forward_mttl({'image': image_tensor}, is_train=False)
        else:
            return model(image_tensor)

def analyze_detection(predictions, conf_threshold):
    """Extract detection statistics."""
    if 'detection' not in predictions:
        return None
    
    preds = predictions['detection'][0]
    scores = preds.get('scores', torch.tensor([]))
    labels = preds.get('labels', torch.tensor([]))
    
    counts = {"Infected": 0, "Healthy": 0, "WBC": 0}
    for label, score in zip(labels, scores):
        if score >= conf_threshold:
            class_name = CLASS_MAP.get(label.item())
            if class_name in counts:
                counts[class_name] += 1
    
    infected = counts["Infected"]
    healthy = counts["Healthy"]
    total_rbc = infected + healthy
    parasitemia = (infected / total_rbc * 100) if total_rbc > 0 else 0.0
    
    return {**counts, 'Parasitemia (%)': parasitemia}

def classify_boxes_batch(model, image_tensor, boxes_xyxy, img_h, img_w):
    """Efficiently classify multiple bounding boxes in one pass."""
    if len(boxes_xyxy) == 0:
        return np.array([])
    
    with torch.no_grad():
        # Extract features once
        multiscale_features = model.backbone(
            image_tensor, task='roi_classif', return_multiscale=True
        )
        fpn_input = [multiscale_features[f'c{i}'] for i in range(2, 5)]
        fpn_features = model.fpn(fpn_input)
        
        # Pool features for all boxes
        box_list = [torch.stack(boxes_xyxy)]
        pooled_features = model.roi_classifier_pooler(
            fpn_features, box_list, [(img_h, img_w)]
        )
        
        if pooled_features.shape[0] == 0:
            num_classes = model.config.get('num_classes_classif', 3)
            return np.full((len(boxes_xyxy), num_classes), 1.0 / num_classes)
        
        # Classify all boxes
        feature_vectors = pooled_features.flatten(start_dim=1)
        class_logits = model.task_heads['cell_classif'](feature_vectors)
        class_probs = F.softmax(class_logits, dim=1).cpu().numpy()
    
    return class_probs

def run_two_stage_roi(image_tensor, preprocessed_image, det_model_dir, 
                      roi_model_dir, conf_threshold, roi_num_classes):
    """Two-stage ROI classification: detection then classification."""
    img_h, img_w = preprocessed_image.shape[:2]
    
    # Stage 1: Detection
    det_model, _ = load_model(det_model_dir)
    det_outputs = run_inference(det_model, image_tensor)
    
    det_preds = det_outputs['detection'][0]
    scores = det_preds.get('scores', torch.tensor([]))
    boxes = det_preds.get('boxes', torch.tensor([]))
    det_labels = det_preds.get('labels', torch.tensor([]))
    
    mask = scores >= conf_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_det_labels = det_labels[mask]
    
    num_boxes = len(filtered_boxes)
    
    if num_boxes == 0:
        roi_class_names = ['Infected', 'Healthy', 'WBC'] if roi_num_classes == 3 else ['Infected', 'Healthy']
        return {
            'counts': {name: 0 for name in roi_class_names},
            'parasitemia': 0.0,
            'total_cells': 0,
            'details': [],
            'detection_boxes': filtered_boxes,
            'detection_scores': filtered_scores
        }
    
    # Stage 2: Classification
    roi_model, _ = load_model(roi_model_dir)
    roi_class_names = ['Infected', 'Healthy', 'WBC'] if roi_num_classes == 3 else ['Infected', 'Healthy']
    roi_counts = {name: 0 for name in roi_class_names}
    details = []
    
    boxes_list = [box for box in filtered_boxes]
    all_probs = classify_boxes_batch(roi_model, image_tensor, boxes_list, img_h, img_w)
    
    for idx, (box, det_score, det_label, probs) in enumerate(
        zip(filtered_boxes, filtered_scores, filtered_det_labels, all_probs)
    ):
        pred_class_idx = np.argmax(probs)
        pred_class_name = roi_class_names[pred_class_idx]
        pred_confidence = probs[pred_class_idx]
        
        roi_counts[pred_class_name] += 1
        
        details.append({
            'box_idx': idx,
            'box': box.cpu().numpy(),
            'det_label': CLASS_MAP.get(det_label.item(), 'Unknown'),
            'det_score': det_score.item(),
            'roi_pred_class': pred_class_name,
            'roi_confidence': pred_confidence,
            'roi_probs': probs
        })
    
    infected = roi_counts.get('Infected', 0)
    healthy = roi_counts.get('Healthy', 0)
    total_rbc = infected + healthy
    parasitemia = (infected / total_rbc * 100) if total_rbc > 0 else 0.0
    
    return {
        'counts': roi_counts,
        'parasitemia': parasitemia,
        'total_cells': num_boxes,
        'details': details,
        'detection_boxes': filtered_boxes,
        'detection_scores': filtered_scores
    }

def visualize_detection(ax, image, predictions, conf_threshold, focus=True):
    ax.imshow(image)
    
    preds = predictions[0] if isinstance(predictions, list) else predictions
    scores = preds.get('scores', [])
    boxes = preds.get('boxes', [])
    labels = preds.get('labels', [])
    
    for box, label, score in zip(boxes, labels, scores):
        if score >= conf_threshold:
            class_name = CLASS_MAP.get(label.item(), 'Unknown')
            
            if focus and class_name == 'Healthy':
                continue
            
            x1, y1, x2, y2 = box.cpu().numpy()
            color = COLOR_MAP.get(class_name, 'gray')
            
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, 
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            ax.text(
                x1, y1 - 5, f"{class_name} {score:.2f}",
                bbox=dict(facecolor=color, alpha=0.6),
                color='white', fontsize=8
            )
    
    ax.set_axis_off()

def visualize_segmentation(ax, image, predictions):
    """Overlay segmentation mask on image."""
    target_size = (image.shape[0], image.shape[1])
    pred_upsampled = F.interpolate(
        predictions.unsqueeze(0) if predictions.dim() == 3 else predictions,
        size=target_size,
        mode='bilinear',
        align_corners=False
    )
    
    mask = torch.sigmoid(pred_upsampled).squeeze().cpu().numpy() > 0.5
    
    ax.imshow(image)
    ax.imshow(mask, cmap='gray', alpha=0.5, vmin=0, vmax=1)
    ax.set_axis_off()

def visualize_heatmap(ax, image, predictions):
    """Overlay heatmap on image."""
    target_size = (image.shape[0], image.shape[1])
    pred_upsampled = F.interpolate(
        predictions.unsqueeze(0) if predictions.dim() == 3 else predictions,
        size=target_size,
        mode='bilinear',
        align_corners=False
    )
    
    heatmap = np.clip(pred_upsampled.squeeze().cpu().numpy(), 0, 1)
    
    ax.imshow(image)
    ax.imshow(heatmap, cmap='hot', alpha=0.6, vmin=0, vmax=1)
    ax.set_axis_off()

def visualize_roi_classification(ax, image, roi_results, roi_num_classes):
    """Draw classified bounding boxes."""
    ax.imshow(image)
    
    roi_colors = {'Infected': 'red', 'Healthy': 'blue', 'WBC': 'purple'}
    
    for detail in roi_results['details']:
        box = detail['box']
        roi_class = detail['roi_pred_class']
        roi_conf = detail['roi_confidence']
        
        x1, y1, x2, y2 = box
        color = roi_colors.get(roi_class, 'gray')
        
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        if roi_class in ['Infected', 'WBC']:
            ax.text(
                x1, y1 - 5, f"{roi_class} {roi_conf:.2f}",
                bbox=dict(facecolor=color, alpha=0.6),
                color='white', fontsize=8
            )
    
    ax.set_axis_off()

class InferenceEngine:
    """Inference engine for malaria cell analysis tasks."""
    
    def __init__(self, models_dir, mode='both', conf_threshold=0.5, 
                 det_num_classes_for_roi=3, focus=True):
        self.models_dir = models_dir
        self.mode = mode.lower()
        self.conf_threshold = conf_threshold
        self.det_num_classes_for_roi = det_num_classes_for_roi
        self.focus = focus
    
    def process_image(self, image_path, task, num_classes, eval_aux_tasks=False):
        """Process single image for specified task."""
        print(f"{task.upper()} Inference on Image (Focus: {'Infected only' if self.focus else 'All classes'})")
        print(f"\nProcessing: {Path(image_path).name}")
        
        try:
            image_tensor, preprocessed_image = prepare_image(image_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        
        results = {}
        summary_data = []
        
        # ROI classification
        if task == 'roi_classif':
            for model_mode in ['STL', 'MTTL']:
                if self.mode in [model_mode.lower(), 'both']:
                    try:
                        det_model_dir = find_model_path(
                            self.models_dir, model_mode, 'detection', 
                            self.det_num_classes_for_roi
                        )
                        roi_model_dir = find_model_path(
                            self.models_dir, model_mode, 'roi_classif', num_classes
                        )
                        
                        roi_results = run_two_stage_roi(
                            image_tensor, preprocessed_image, 
                            det_model_dir, roi_model_dir,
                            self.conf_threshold, num_classes
                        )
                        
                        results[model_mode] = {
                            'roi_results': roi_results,
                            'roi_num_classes': num_classes
                        }
                        
                        analysis = {
                            **roi_results['counts'],
                            'Total Cells': roi_results['total_cells']
                        }
                        if 'Infected' in roi_results['counts']:
                            analysis['Parasitemia (%)'] = roi_results['parasitemia']
                        
                        summary_data.append({'Model': model_mode, **analysis})
                        
                    except Exception as e:
                        print(f"Could not run {model_mode} ROI: {e}")
        
        # The other tasks
        else:
            for model_mode in ['STL', 'MTTL']:
                if self.mode in [model_mode.lower(), 'both']:
                    try:
                        model_dir = find_model_path(
                            self.models_dir, model_mode, task, num_classes
                        )
                        model, config = load_model(model_dir)
                        outputs = run_inference(model, image_tensor)
                        
                        results[model_mode] = {
                            'outputs': outputs,
                            'config': config
                        }
                        
                        if task == 'detection':
                            analysis = analyze_detection(outputs, self.conf_threshold)
                            if analysis:
                                summary_data.append({'Model': model_mode, **analysis})
                    
                    except Exception as e:
                        print(f"Could not run {model_mode}: {e}")
        
        return {
            'results': results,
            'preprocessed_image': preprocessed_image,
            'summary_data': summary_data,
            'task': task,
            'eval_aux_tasks': eval_aux_tasks
        }
    
    def visualize_results(self, process_output, save_path=None, show=True):
        """Visualize inference results."""
        if process_output is None:
            return
        
        results = process_output['results']
        preprocessed_image = process_output['preprocessed_image']
        task = process_output['task']
        eval_aux_tasks = process_output['eval_aux_tasks']
        
        # Build visualization list
        vis_list = [
            ('Preprocessed', preprocessed_image, 
             lambda ax, img: (ax.imshow(img), ax.set_axis_off()))
        ]
        
        vis_funcs = {
            'detection': visualize_detection,
            'segmentation': visualize_segmentation,
            'heatmap': visualize_heatmap
        }
        
        # Add primary task visualizations
        if task == 'roi_classif':
            for mode, data in results.items():
                vis_list.append((
                    f"{mode} ROI Classification",
                    preprocessed_image,
                    visualize_roi_classification,
                    data['roi_results'],
                    data['roi_num_classes']
                ))
        else:
            for mode, data in results.items():
                if task in data['outputs']:
                    if task == 'detection':
                        vis_list.append((
                            f"{mode} {task.capitalize()}",
                            preprocessed_image,
                            vis_funcs[task],
                            data['outputs'][task],
                            self.conf_threshold,
                            self.focus 
                        ))
                    else:
                        vis_list.append((
                            f"{mode} {task.capitalize()}",
                            preprocessed_image,
                            vis_funcs[task],
                            data['outputs'][task]
                        ))
        
        # Add auxiliary task visualizations for MTTL
        if eval_aux_tasks:
            for mode, data in results.items():
                if mode == 'MTTL' and 'config' in data:
                    for aux_task in data['config'].get('active_tasks', []):
                        if aux_task != task and aux_task in data['outputs'] and aux_task in vis_funcs:
                            if aux_task == 'detection':
                                vis_list.append((
                                    f"MTTL {aux_task.capitalize()} (Aux)",
                                    preprocessed_image,
                                    vis_funcs[aux_task],
                                    data['outputs'][aux_task],
                                    self.conf_threshold,
                                    self.focus 
                                ))
                            else:
                                vis_list.append((
                                    f"MTTL {aux_task.capitalize()} (Aux)",
                                    preprocessed_image,
                                    vis_funcs[aux_task],
                                    data['outputs'][aux_task]
                                ))
        
        # Create figure
        fig, axes = plt.subplots(
            1, len(vis_list), 
            figsize=(7 * len(vis_list), 7), 
            squeeze=False
        )
        
        for i, (title, img, func, *args) in enumerate(vis_list):
            axes[0, i].set_title(title, fontweight='bold')
            func(axes[0, i], img, *args)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        if show:
            plt.show()
        
        plt.close(fig)
    
    def run(self, image_path, task, num_classes, output_dir=None,
            eval_aux_tasks=False, show_plots=True, save_plots=False):
        """Complete inference pipeline."""
        # Handle directory or file
        if Path(image_path).is_dir():
            image_paths = list(Path(image_path).glob('*.png')) + \
                         list(Path(image_path).glob('*.jpg')) + \
                         list(Path(image_path).glob('*.jpeg'))
        else:
            image_paths = [Path(image_path)]
        
        for img_path in tqdm(image_paths, desc="Inference"):
            output = self.process_image(
                img_path, task, num_classes, eval_aux_tasks
            )
            
            if output is None:
                continue
            
            # Display summary
            if output['summary_data']:
                df = pd.DataFrame(output['summary_data']).set_index('Model')
                print("\n" + "="*60)
                print(df.to_string())
                print("="*60)
            
            # Visualize
            if show_plots or save_plots:
                save_path = None
                if save_plots and output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    filename = f"{img_path.stem}_inference.png"
                    save_path = Path(output_dir) / filename
                
                self.visualize_results(output, save_path, show_plots)

def main():
    parser = argparse.ArgumentParser(
        description='Malaria Cell Analysis Inference Engine'
    )
    
    parser.add_argument(
        'image_path',
        help='Path to image file or directory containing images'
    )
    
    parser.add_argument(
        '--task',
        required=True,
        choices=['detection', 'segmentation', 'heatmap', 'roi_classif'],
        help='Primary task to perform'
    )
    
    parser.add_argument(
        '--num-classes',
        type=int,
        required=True,
        help='Number of classes for the detection task'
    )
    
    parser.add_argument(
        '--models-dir',
        default='../../models',
        help='Base directory containing model folders (default: ../../models)'
    )
    
    parser.add_argument(
        '--mode',
        choices=['stl', 'mttl', 'both'],
        default='both',
        help='Model mode to use (default: both)'
    )
    
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for detection/classification (default: 0.5)'
    )
    
    parser.add_argument(
        '--det-classes-for-roi',
        type=int,
        default=3,
        help='Detection model classes to use for ROI stage 1 (default: 3)'
    )
    
    parser.add_argument(
        '--eval-aux',
        action='store_true',
        help='Evaluate and visualize auxiliary tasks for MTTL models'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Directory to save visualizations'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save visualization plots to file'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display plots (useful for batch processing)'
    )
    
    parser.add_argument(
        '--show-all-classes',
        action='store_true',
        help='If set, displays all classes (Healthy, WBC). Default behavior focuses only on Infected.'
    )
    
    args = parser.parse_args()
    
    models_path = Path(args.models_dir)
    if not models_path.exists() or not any(models_path.iterdir()):
        print(f"\nModel directory '{models_path}' is missing or empty.")
        success = download_models(interactive=True) 
        if not success:
            print("Cannot proceed without model weights.")
            sys.exit(1)
    
    # Create engine
    engine = InferenceEngine(
        models_dir=args.models_dir,
        mode=args.mode,
        conf_threshold=args.conf_threshold,
        det_num_classes_for_roi=args.det_classes_for_roi,
        focus=not args.show_all_classes 
    )
    
    # Run pipeline
    engine.run(
        image_path=args.image_path,
        task=args.task,
        num_classes=args.num_classes,
        output_dir=args.output_dir,
        eval_aux_tasks=args.eval_aux,
        show_plots=not args.no_show,
        save_plots=args.save_plots
    )

if __name__ == '__main__':
    main()