# ============================================================================
# Our Data Preprocessor and manager for STL and MTTL 
# Gwade Steve
# MTTL and Application to Malaria Detection
# April 2025
# ============================================================================
import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context('notebook')
from PIL import Image
import cv2
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
from tqdm.notebook import tqdm
import json
import pickle
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import random
import numpy as np
from tqdm.notebook import tqdm


# If staintools not available, we'll fall back to our custom implementations
try:
    import staintools
    STAINTOOLS_AVAILABLE = True
    print("staintools library available")
except ImportError:
    STAINTOOLS_AVAILABLE = False
    print("staintools library not available - using custom implementations")

# Function to set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeds set to {seed} for reproducibility")
    
# Worker seed function for DataLoader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
    print("Albumentations available for advanced augmentations")
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Albumentations not available - using basic augmentations")
    
# Function to crop cell from image using bounding box
def crop_cell_from_image(image, bbox_xyxy, target_size=(64, 64)):
    """Crops a cell using xyxy coordinates and resizes it."""
    x1, y1, x2, y2 = bbox_xyxy
    
    # Add some padding to the crop
    padding_x = (x2 - x1) * 0.1
    padding_y = (y2 - y1) * 0.1
    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(image.shape[1], x2 + padding_x)
    y2 = min(image.shape[0], y2 + padding_y)
    
    cell_patch = image[int(y1):int(y2), int(x1):int(x2)]
    if cell_patch.size == 0:
        return None
    
    return cv2.resize(cell_patch, target_size, interpolation=cv2.INTER_LANCZOS4)
    
# MalariaPreprocessor Class ( Our attempt for making images stain invariant )
class MalariaPreprocessor:
    """
    Preprocessing class for malaria microscopy images
    We implement stain normalization and image enhancement techniques
    """
    
    def __init__(self, target_size=(512,512)):
        self.target_size = target_size
        
        # Standard H&E stain matrix (Ruifrok & Johnston, 2001)
        self.he_matrix = np.array([
            [0.65, 0.70, 0.29],    # Hematoxylin (blue/purple)
            [0.07, 0.99, 0.11],    # Eosin (pink/red)
            [0.27, 0.57, 0.78]     # DAB (brown)
        ])
        
        # Reference target for Reinhard normalization 
        self.target_means = [134.2, 123.8, 166.4]  # LAB means
        self.target_stds = [52.3, 41.7, 48.9]      # LAB stds
        
    def resize_image(self, image, maintain_aspect=False):
        if maintain_aspect:
            h, w = image.shape[:2]
            scale = min(self.target_size[0]/h, self.target_size[1]/w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Pad with white background
            pad_h = (self.target_size[0] - new_h) // 2
            pad_w = (self.target_size[1] - new_w) // 2
            result = np.full((self.target_size[0], self.target_size[1], 3), 245, dtype=np.uint8)
            result[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
            return result
        else:
            return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
  
    def resize_annotations_to_target_size(self, annotations, original_size, target_size):
        original_h, original_w = original_size[:2]
        target_h, target_w = target_size
        
        # Scaling factors
        scale_x = target_w / original_w
        scale_y = target_h / original_h
        
        # Resize each annotation
        resized_annotations = []
        for ann in annotations:
            new_ann = ann.copy()
            
            if ann.get('shape') == 'Point':
                # Resize point coordinates
                x, y = ann.get('x', 0), ann.get('y', 0)
                new_x = x * scale_x
                new_y = y * scale_y
                
                # Ensure coordinates are within bounds
                new_x = max(0, min(new_x, target_w - 1))
                new_y = max(0, min(new_y, target_h - 1))

                new_ann['x'] = new_x
                new_ann['y'] = new_y
                
            elif ann.get('shape') == 'Polygon' and 'polygon' in ann:
                # Resize polygon coordinates
                old_coords = ann['polygon']
                new_coords = []
                for x, y in old_coords:
                    new_x = x * scale_x
                    new_y = y * scale_y
                    
                    # Ensure in bounds
                    new_x = max(0, min(new_x, target_w - 1))
                    new_y = max(0, min(new_y, target_h - 1))
                    
                    new_coords.append((new_x, new_y))
                new_ann['polygon'] = new_coords
                
                # Update bounding box
                xs, ys = zip(*new_coords)
                new_ann['bbox'] = [min(xs), min(ys), max(xs), max(ys)]
            
            # Metadata about the resizing
            new_ann['resizing_info'] = {
                'original_size': (original_w, original_h),
                'target_size': (target_w, target_h),
                'scale_x': scale_x,
                'scale_y': scale_y
            }
            
            resized_annotations.append(new_ann)
        
        return resized_annotations
        
    def clahe_normalization(self, image):
        # Convert to LAB color space for better uniformity
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel 
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16,16))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        # Enhance A and B channels for improved color contrast
        lab[:,:,1] = cv2.multiply(lab[:,:,1], 1.15)
        lab[:,:,2] = cv2.multiply(lab[:,:,2], 1.15)
        
        # Back to RGB
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return result
    
    def robust_stain_normalization(self, image):
        """
        Our implementation of stain normalization using optical density and Beer-Lambert law
        To handle the full range of microscopy staining variations
        """
        # Convert to optical density
        image_float = image.astype(np.float32)
        
        # Avoid log(0) by adding small epsilon
        image_float = np.maximum(image_float, 1.0)
        od = -np.log(image_float / 255.0)
        
        # Remove very dark pixels 
        od_flat = od.reshape((-1, 3))
        
        # Calculate statistics using percentiles to avoid outliers
        percentiles = np.percentile(od_flat, [5, 50, 95], axis=0)
        p5, p50, p95 = percentiles[0], percentiles[1], percentiles[2]
        
        # Normalize each channel using the statistics
        od_normalized = np.zeros_like(od)
        for c in range(3):
            if p95[c] > p5[c]:  # So as to zero division
                od_normalized[:,:,c] = (od[:,:,c] - p5[c]) / (p95[c] - p5[c])
            else:
                od_normalized[:,:,c] = od[:,:,c]
        
        # Clip range
        od_normalized = np.clip(od_normalized, 0, 1)
        
        # back to RGB
        result = (np.exp(-od_normalized) * 255).astype(np.uint8)
        
        # statistics
        orig_std = np.std(image)
        result_std = np.std(result)
        print(f"Rob. Stain Norm: Original std: {orig_std:.1f} -> Normalized: {result_std:.1f}")
        
        return result
    
    def macenko_normalization(self, image):
        """Macenko stain normalization with fallback"""
        if STAINTOOLS_AVAILABLE:
            try:
                # Prepare the image
                img_normalized = self._prepare_for_staintools(image)
                
                # staintools Macenko normalization
                normalizer = staintools.StainNormalizer(method='macenko')
                normalizer.fit(img_normalized)
                result = normalizer.transform(img_normalized)
                print(f"Macenko (staintools): Successfully normalized using staintools library")
                return result
                
            except Exception as e:
                print(f"Macenko: staintools failed ({str(e)}), using fallback...")
                return self._macenko_fallback(image)
        else:
            print("Macenko: Using custom implementation (staintools not available)")
            return self._macenko_fallback(image)
    
    def _prepare_for_staintools(self, image):
        # Ensure image is in the right format and range
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # Remove pure white pixels that can cause issues
        white_mask = np.all(image >= 250, axis=-1)
        image_copy = image.copy()
        image_copy[white_mask] = [240, 240, 240]
        
        return image_copy
    
    def _macenko_fallback(self, image):
        # Convert to optical density
        od = self._rgb_to_od(image)
        
        # Remove background (low OD pixels)
        od_flat = od.reshape((-1, 3))
        od_mean = np.mean(od_flat, axis=0)
        od_thresh = np.percentile(np.linalg.norm(od_flat, axis=1), 15)
        
        # Select tissue pixels (high enough OD)
        tissue_mask = np.linalg.norm(od_flat, axis=1) > od_thresh
        tissue_od = od_flat[tissue_mask]
        
        if len(tissue_od) < 100: 
            return self.robust_stain_normalization(image)
        
        # Compute eigenvectors of covariance matrix
        cov_matrix = np.cov(tissue_od.T)
        eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalue
        idx = np.argsort(eigenvals)[::-1]
        eigenvecs = eigenvecs[:, idx]
        
        # Project tissue OD onto the plane spanned by first two eigenvectors
        proj_coords = tissue_od @ eigenvecs[:, :2]
        
        # Find extreme points (represent pure stains)
        phi = np.arctan2(proj_coords[:, 1], proj_coords[:, 0])
        min_phi_idx = np.argmin(phi)
        max_phi_idx = np.argmax(phi)
        
        # Get stain vectors
        stain1 = tissue_od[min_phi_idx]
        stain2 = tissue_od[max_phi_idx]
        
        # Normalize stain vectors
        stain_matrix = np.array([stain1, stain2])
        stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=1, keepdims=True)
        
        # Separate stains
        od_flat_normalized = np.linalg.lstsq(stain_matrix.T, od_flat.T, rcond=None)[0].T
        
        # Normalize concentrations
        for i in range(2):
            p99 = np.percentile(od_flat_normalized[:, i], 99)
            if p99 > 0:
                od_flat_normalized[:, i] = np.clip(od_flat_normalized[:, i] / p99, 0, 1)
        
        # Reconstruct with normalized concentrations
        od_reconstructed = od_flat_normalized @ stain_matrix
        od_result = od_reconstructed.reshape(od.shape)
        
        # Convert back to RGB
        result = self._od_to_rgb(od_result)
        
        print(f"Macenko (custom): Extracted {len(tissue_od)} tissue pixels for normalization")
        return result
    
    def reinhard_normalization(self, image):
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # current statistics
        current_means = [np.mean(lab[:,:,i]) for i in range(3)]
        current_stds = [np.std(lab[:,:,i]) for i in range(3)]
        
        print(f"Reinhard: Current LAB means: {[f'{m:.1f}' for m in current_means]}")
        print(f"Reinhard: Target LAB means: {[f'{m:.1f}' for m in self.target_means]}")
        
        # Normalize each channel
        for i in range(3):
            if current_stds[i] > 0:
                lab[:,:,i] = ((lab[:,:,i] - current_means[i]) * 
                             (self.target_stds[i] / current_stds[i]) + 
                             self.target_means[i])
        
        # Clip ranges
        lab[:,:,0] = np.clip(lab[:,:,0], 0, 100)    # L: 0-100
        lab[:,:,1] = np.clip(lab[:,:,1], -127, 127) # A: -127 to 127
        lab[:,:,2] = np.clip(lab[:,:,2], -127, 127) # B: -127 to 127
        
        # convert back to RGB
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        return result
    
    def enhanced_color_deconvolution(self, image):
        od = self._rgb_to_od(image)
        
        # stain matrix
        stain_matrix = np.array([
            [0.644, 0.717, 0.267],  # Hematoxylin
            [0.092, 0.954, 0.283],  # Eosin
            [0.600, 0.420, 0.685]   # stain/background
        ])
        
        # Flatten 
        od_flat = od.reshape((-1, 3))
        
        # Separate stains using pseudo-inverse
        concentrations = np.linalg.pinv(stain_matrix) @ od_flat.T
        concentrations = concentrations.T
        
        # Enhance H&E channels
        h_enhanced = concentrations[:, 0] * 1.2  # Improve H visibility
        e_enhanced = concentrations[:, 1] * 0.8  # Slightly reduce E visibility
        
        # Reconstruct with computed concentrations
        enhanced_concentrations = np.column_stack([h_enhanced, e_enhanced, concentrations[:, 2]])
        
        # Reconstruct OD
        od_reconstructed = (stain_matrix @ enhanced_concentrations.T).T
        od_result = od_reconstructed.reshape(od.shape)
        
        # to RGB
        result = self._od_to_rgb(od_result)
        
        # Apply slight color enhancement
        result = self._enhance_microscopy_colors(result)
        
        return result
    
    def _rgb_to_od(self, rgb):
        """Convert RGB to optical density"""
        rgb_float = rgb.astype(np.float32)
        rgb_float = np.maximum(rgb_float, 1.0)  # to avoid log(0)
        return -np.log(rgb_float / 255.0)
    
    def _od_to_rgb(self, od):
        """Convert optical density to RGB"""
        rgb = np.exp(-od) * 255.0
        return np.clip(rgb, 0, 255).astype(np.uint8)
    
    def _enhance_microscopy_colors(self, image):
        """Enhance colors in microscopy images"""
        # Convert to HSV for color enhancement
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Enhance saturation slightly
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.15, 0, 255)
        
        # Slight value enhancement
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.05, 0, 255)
        
        # Convert back
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return result
    
# Class for handling parasitemia scores
class ParasitemiaScorer:
    """
    Parasitemia scoring system for our dataset
    """
    def __init__(self, target_image_size=(512,512)):
        self.target_image_size = target_image_size
        self.total_image_area = target_image_size[0] * target_image_size[1]
        
        # Matching for our dataset classes
        self.infected_types = {
            'parasitized',  
            'Parasitized'   
        }
        
        self.healthy_types = {
            'uninfected',
            'Uninfected'
        }
        
        self.wbc_types = {
            'white_blood_cell',   
            'White_Blood_Cell'
        }
        
        # Store scoring history for analysis
        self.scoring_history = []
        
        print(f"Cell type categories:")
        print(f"Infected: {self.infected_types}")
        print(f"Healthy:  {self.healthy_types}")
        print(f"WBC:      {self.wbc_types}")
    
    def extract_cell_information(self, annotations):
        cells_info = {
            'infected_cells': [],
            'healthy_cells': [],
            'wbc_cells': [],
            'unknown_cells': [],
            'total_cells': 0
        }
        
        for ann in annotations:
            cell_type = ann.get('cell_type', '').strip()
            
            # Calculate area
            if ann.get('shape') == 'Polygon' and 'polygon' in ann:
                coords = ann['polygon']
                x_coords = [p[0] for p in coords]
                y_coords = [p[1] for p in coords]
                area = 0.5 * abs(sum(x_coords[i] * y_coords[i+1] - x_coords[i+1] * y_coords[i] 
                                for i in range(-1, len(x_coords)-1)))
            elif ann.get('shape') == 'Point':
                # Different areas based on cell type
                if cell_type in self.infected_types:
                    area = np.random.normal(130, 24)  
                elif cell_type in self.healthy_types:
                    area = np.random.normal(90, 16)   
                elif cell_type in self.wbc_types:
                    area = np.random.normal(300, 50) 
                else:
                    area = 100
                area = max(area, 40)  # min area
            else:
                area = 0
            
            cell_info = {
                'cell_type': cell_type,
                'original_type': cell_type,
                'shape': ann.get('shape', ''),
                'area': area,
                'coordinates': ann.get('polygon', [(ann.get('x', 0), ann.get('y', 0))]),
                'annotation': ann
            }
            
            if cell_type in self.infected_types:
                cells_info['infected_cells'].append(cell_info)
            elif cell_type in self.healthy_types:
                cells_info['healthy_cells'].append(cell_info)
            elif cell_type in self.wbc_types:
                cells_info['wbc_cells'].append(cell_info)
            else:
                cells_info['unknown_cells'].append(cell_info)
                if cells_info['total_cells'] < 5:  # Only print first few
                    print(f"Unknown cell type: '{cell_type}'")
            
            cells_info['total_cells'] += 1
        
        return cells_info

    def density_based_scoring(self, cells_info):
        infected_count = len(cells_info['infected_cells'])
        
        # We will asume 512X512 pixels ≈ 0.4 mm² (heuristic)
        area_mm2_equivalent = 0.4
        density_score = infected_count / area_mm2_equivalent
        
        return {
            'score': density_score,
            'method': 'density_based',
            'infected_count': infected_count,
            'area_mm2_equivalent': area_mm2_equivalent,
            'density_per_mm2': density_score,
            'spatial_concentration': self._calculate_spatial_concentration(cells_info['infected_cells'])
        }
    
    def count_based_scoring(self, cells_info):
        infected_count = len(cells_info['infected_cells'])
        # we count RBCs (infected + healthy)
        rbc_count = infected_count + len(cells_info['healthy_cells'])
        
        if rbc_count == 0:
            return {
                'score': 0.0,
                'method': 'count_based',
                'infected_count': 0,
                'rbc_count': 0,
                'ratio': 0.0,
                'note': 'No RBCs found'
            }
        
        parasitemia_percentage = (infected_count / rbc_count) * 100
        
        return {
            'score': parasitemia_percentage,
            'method': 'count_based',
            'infected_count': infected_count,
            'rbc_count': rbc_count,
            'ratio': infected_count / rbc_count,
            'healthy_count': len(cells_info['healthy_cells']),
            'wbc_count': len(cells_info['wbc_cells']),
            'unknown_count': len(cells_info['unknown_cells'])
        }
    
    def area_based_scoring(self, cells_info):
        infected_area = sum(cell['area'] for cell in cells_info['infected_cells'])
        healthy_area = sum(cell['area'] for cell in cells_info['healthy_cells'])
        total_rbc_area = infected_area + healthy_area
        
        if total_rbc_area == 0:
            return {
                'score': 0.0,
                'method': 'area_based',
                'infected_area': 0,
                'rbc_area': 0,
                'ratio': 0.0,
                'note': 'No RBC area found'
            }
        
        parasitemia_percentage = (infected_area / total_rbc_area) * 100
        
        return {
            'score': parasitemia_percentage,
            'method': 'area_based',
            'infected_area': infected_area,
            'healthy_area': healthy_area,
            'rbc_area': total_rbc_area,
            'ratio': infected_area / total_rbc_area,
            'image_coverage': total_rbc_area / self.total_image_area
        }
    
    def _calculate_spatial_concentration(self, infected_cells):
        if len(infected_cells) < 2:
            return 0.0
        
        coordinates = [cell['coordinates'][0] if cell['coordinates'] else (0, 0) 
                      for cell in infected_cells]
        
        total_distance = 0
        pairs = 0
        
        for i, coord1 in enumerate(coordinates):
            for j, coord2 in enumerate(coordinates[i+1:], i+1):
                distance = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
                total_distance += distance
                pairs += 1
        
        avg_distance = total_distance / pairs if pairs > 0 else 0
        image_diagonal = np.sqrt(self.target_image_size[0]**2 + self.target_image_size[1]**2)
        concentration_score = 1 - (avg_distance / image_diagonal) if image_diagonal > 0 else 0
        
        return max(0, concentration_score)
    
    def calculate_all_scores(self, annotations, image_path=None):
        cells_info = self.extract_cell_information(annotations)
        
        count_score = self.count_based_scoring(cells_info)
        area_score = self.area_based_scoring(cells_info)
        density_score = self.density_based_scoring(cells_info)
        
        results = {
            'image_path': image_path,
            'cells_info': cells_info,
            'count_based': count_score,
            'area_based': area_score,
            'density_based': density_score,
            'timestamp': pd.Timestamp.now(),
            'summary': {
                'total_annotations': len(annotations),
                'infected_cells': len(cells_info['infected_cells']),
                'healthy_cells': len(cells_info['healthy_cells']),
                'wbc_cells': len(cells_info['wbc_cells']),
                'unknown_cells': len(cells_info['unknown_cells'])
            }
        }
        
        self.scoring_history.append(results)
        return results
    
    def analyze_scoring_methods(self, all_results):
        
        if not all_results:
            print("No results to analyze!")
            return None
        
        print("\n" + "="*60)
        print("PARASITEMIA SCORING METHODS ANALYSIS")
        print("="*60)
        
        count_scores = [r['count_based']['score'] for r in all_results]
        area_scores = [r['area_based']['score'] for r in all_results]
        density_scores = [r['density_based']['score'] for r in all_results]
        
        methods_stats = {
            'Count-based': {
                'scores': count_scores,
                'mean': np.mean(count_scores),
                'std': np.std(count_scores),
                'min': np.min(count_scores),
                'max': np.max(count_scores),
                'median': np.median(count_scores),
                'non_zero': sum(1 for s in count_scores if s > 0)
            },
            'Area-based': {
                'scores': area_scores,
                'mean': np.mean(area_scores),
                'std': np.std(area_scores),
                'min': np.min(area_scores),
                'max': np.max(area_scores),
                'median': np.median(area_scores),
                'non_zero': sum(1 for s in area_scores if s > 0)
            },
            'Density-based': {
                'scores': density_scores,
                'mean': np.mean(density_scores),
                'std': np.std(density_scores),
                'min': np.min(density_scores),
                'max': np.max(density_scores),
                'median': np.median(density_scores),
                'non_zero': sum(1 for s in density_scores if s > 0)
            }
        }
        
        print(f"\nSCORING METHODS STATISTICS (n={len(all_results)} images):")
        print("-" * 70)
        
        for method, stats in methods_stats.items():
            print(f"\n{method.upper()}:")
            print(f"Mean:     {stats['mean']:.3f}")
            print(f"Std:      {stats['std']:.3f}")
            print(f"Range:    {stats['min']:.3f} - {stats['max']:.3f}")
            print(f"Median:   {stats['median']:.3f}")
            print(f"Non-zero: {stats['non_zero']}/{len(all_results)} ({stats['non_zero']/len(all_results)*100:.1f}%)")
        
        # Calculate correlations
        from scipy.stats import pearsonr
        
        print(f"\nCORRELATION ANALYSIS:")
        print("-" * 30)
        
        correlations = {}
        
        if len(set(count_scores)) > 1 and len(set(area_scores)) > 1:
            corr_count_area, p_count_area = pearsonr(count_scores, area_scores)
            correlations['count_vs_area'] = {'correlation': corr_count_area, 'p_value': p_count_area}
            print(f"Count vs Area:    r = {corr_count_area:.3f} (p = {p_count_area:.3f})")
        
        if len(set(count_scores)) > 1 and len(set(density_scores)) > 1:
            corr_count_density, p_count_density = pearsonr(count_scores, density_scores)
            correlations['count_vs_density'] = {'correlation': corr_count_density, 'p_value': p_count_density}
            print(f"Count vs Density: r = {corr_count_density:.3f} (p = {p_count_density:.3f})")
        
        if len(set(area_scores)) > 1 and len(set(density_scores)) > 1:
            corr_area_density, p_area_density = pearsonr(area_scores, density_scores)
            correlations['area_vs_density'] = {'correlation': corr_area_density, 'p_value': p_area_density}
            print(f"Area vs Density:  r = {corr_area_density:.3f} (p = {p_area_density:.3f})")
        
        return {
            'statistics': methods_stats,
            'correlations': correlations,
            'all_results': all_results
        }

# Data preparation class for our tasks
class MalariaDataPreparator:
    def __init__(self, image_size=(512,512), heatmap_size=(64, 64), infection_sigma=8.0, healthy_sigma=20.0):
        self.image_size = image_size
        self.heatmap_size = heatmap_size 
        self.heatmap_scale = image_size[0] // heatmap_size[0]  
        self.infection_sigma = infection_sigma
        self.healthy_sigma = healthy_sigma
        
        print(f"MalariaDataPreparator initialized:")
        print(f"Image size: {image_size}")
        print(f"Heatmap size: {heatmap_size}")
        print(f"Heatmap scale: 1/{self.heatmap_scale}")
        print(f"Infection sigma: {infection_sigma}")
        
        
    def annotations_to_bboxes(self, annotations, image_size=(512,512)):
        bboxes = []
        labels = []
        
        for ann in annotations:
            if ann.get('shape') == 'Point':
                x, y = ann.get('x', 0), ann.get('y', 0)
                box_size = 28
                x1 = max(0, x - box_size//2)
                y1 = max(0, y - box_size//2)
                x2 = min(image_size[1], x + box_size//2)
                y2 = min(image_size[0], y + box_size//2)
                
            elif ann.get('shape') == 'Polygon' and 'polygon' in ann:
                coords = ann['polygon']
                xs, ys = zip(*coords)
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                
                x1 = max(0, min(x1, image_size[1]-1))
                y1 = max(0, min(y1, image_size[0]-1))
                x2 = max(x1+1, min(x2, image_size[1]))
                y2 = max(y1+1, min(y2, image_size[0]))
            else:
                continue
            
            # Normalize to [0, 1] range
            center_x = (x1 + x2) / 2 / image_size[1]
            center_y = (y1 + y2) / 2 / image_size[0]
            width = (x2 - x1) / image_size[1]
            height = (y2 - y1) / image_size[0]
            
            center_x = np.clip(center_x, 0, 1)
            center_y = np.clip(center_y, 0, 1)
            width = np.clip(width, 0, 1)
            height = np.clip(height, 0, 1)
            
            bboxes.append([center_x, center_y, width, height])
            
            cell_type = ann.get('cell_type', '').lower()
            if 'parasitized' in cell_type:
                labels.append(0)
            elif 'uninfected' in cell_type:
                labels.append(1)
            elif 'white_blood_cell' in cell_type:
                labels.append(2)
            else:
                labels.append(3)
        
        return np.array(bboxes), np.array(labels)
    
    def create_realistic_localization_heatmap(self, annotations, image_size=(512,512), heatmap_size=(64, 64)):
        """
        - Healthy images will be exact zero heatmap.
        - Parasite spots will be clean isotropic Gaussians with peak 1.0.
        - Gaussian tails truncated
        - Overlaps blended with np.maximum
        """
        heatmap = np.zeros(heatmap_size, dtype=np.float32)
        scale_x = heatmap_size[1] / image_size[1] 
        scale_y = heatmap_size[0] / image_size[0] 
        
        infected_points = []
        for ann in annotations:
            cell_type = ann.get('cell_type', '').lower().strip()
            if 'parasitized' not in cell_type:
                continue
                
            if ann.get('shape') == 'Point':
                x, y = ann.get('x', 0), ann.get('y', 0)
                hx, hy = int(x * scale_x), int(y * scale_y)
                
            elif ann.get('shape') == 'Polygon' and 'polygon' in ann and ann['polygon']:
                xs, ys = zip(*ann['polygon'])
                center_x, center_y = np.mean(xs) * scale_x, np.mean(ys) * scale_y
                hx, hy = int(center_x), int(center_y)
            
            else:
                continue
            
            # Clamp coords safely inside [0, W-1], [0, H-1]
            hx = max(0, min(hx, heatmap_size[1] - 1))
            hy = max(0, min(hy, heatmap_size[0] - 1))
            infected_points.append((hx, hy))

        if not infected_points:
            return heatmap  # strictly zero for healthy
        
        # Precompute Gaussian kernel (single reusable patch)
        heatmap_sigma = self.infection_sigma * (scale_x + scale_y) / 2.0
        sigma_sq = heatmap_sigma**2
        cutoff_radius = int(3 * heatmap_sigma)  # truncate beyond 3σ

        # Build a reusable local grid for the Gaussian kernel
        y = np.arange(-cutoff_radius, cutoff_radius + 1)
        x = np.arange(-cutoff_radius, cutoff_radius + 1)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma_sq)).astype(np.float32)

        kh, kw = kernel.shape
        kh_half, kw_half = kh // 2, kw // 2

        # Place kernel at each infected point using slicing
        for hx, hy in infected_points:
            y1 = max(0, hy - kh_half); y2 = min(heatmap_size[0], hy + kh_half + 1)
            x1 = max(0, hx - kw_half); x2 = min(heatmap_size[1], hx + kw_half + 1)

            ky1 = max(0, kh_half - hy); ky2 = ky1 + (y2 - y1)
            kx1 = max(0, kw_half - hx); kx2 = kx1 + (x2 - x1)

            # Blend with maximum (no overlaps > 1.0)
            heatmap[y1:y2, x1:x2] = np.maximum(
                heatmap[y1:y2, x1:x2], kernel[ky1:ky2, kx1:kx2]
            )

        return heatmap
        
    def prepare_training_data(self, dataset_records):
        print(f"Preparing {len(dataset_records)} samples for training...")
        
        training_samples = []
        
        for record in tqdm(dataset_records, desc="Processing samples"):
            try:
                resized_annotations = record.get('resized_annotations', [])
                
                if resized_annotations:
                    bboxes, labels = self.annotations_to_bboxes(resized_annotations, self.image_size)
                    heatmap = self.create_realistic_localization_heatmap(resized_annotations, self.image_size, self.heatmap_size)
                    mask = self.create_cell_segmentation_mask(resized_annotations, self.image_size)
                    severity_class, severity_label = self.assign_severity_label(record['parasitemia_score'])
                    
                    sample = {
                        'image_id': record['image_id'],
                        'image': record['processed_image'],
                        
                        'detection': {
                            'bboxes': bboxes,
                            'labels': labels,
                            'num_objects': len(bboxes)
                        },
                        
                        'regression': {
                            'parasitemia_score': record['parasitemia_score']
                        },
                        
                        'localization': {
                            'heatmap': heatmap
                        },
                        
                        'heatmap': {
                            'heatmap': heatmap
                        },
                        
                        'segmentation': {  
                            'mask': mask
                        },
                        
                        'severity': {      
                            'severity_class': severity_class,
                            'severity_label': severity_label
                        },
                        
                        'multi_task': {
                            'parasitemia_score': record['parasitemia_score'],
                            'bboxes': bboxes,
                            'bbox_labels': labels,
                            'heatmap': heatmap,
                            'mask': mask,                
                            'severity_class': severity_class,  
                            'severity_label': severity_label   
                        },
                        
                        'cell_counts': record['cell_counts'],
                        'metadata': {
                            'original_path': record['original_path'],
                            'preprocessing_method': record['preprocessing_method'],
                            'scoring_method': record['scoring_method'],
                            'image_size': self.image_size,    
                            'heatmap_size': self.heatmap_size, 
                            'heatmap_scale': self.heatmap_scale
                        }
                    }
                    
                    training_samples.append(sample)
                    
            except Exception as e:
                print(f"Failed to process {record['image_id']}: {str(e)}")
                continue
        
        print(f"Successfully prepared {len(training_samples)} training samples")
        return training_samples

    def create_cell_segmentation_mask(self, annotations, image_size=(512,512)):
        """
        Generate a binary mask for cell segmentation.
        - 0: background
        - 1: cell (infected, healthy, WBC)
        """
        mask = np.zeros(image_size, dtype=np.uint8)
        for ann in annotations:
            if ann.get('shape') == 'Point':
                x, y = int(ann.get('x', 0)), int(ann.get('y', 0))
                rr, cc = np.ogrid[
                    max(0, y-16):min(image_size[0], y+16),
                    max(0, x-16):min(image_size[1], x+16)
                ]
                mask[rr, cc] = 1
            elif ann.get('shape') == 'Polygon' and 'polygon' in ann:
                from skimage.draw import polygon
                xs, ys = zip(*ann['polygon'])
                rr, cc = polygon(ys, xs, shape=image_size)
                mask[rr, cc] = 1
        return mask
    
    def create_parasite_segmentation_mask(self, annotations, image_size=(512,512)):
        """
        Generate a binary mask for segmenting ONLY parasitized pixels.
        This creates a "hard negative" learning target.
        - 0: background AND healthy cells/WBCs (all non-parasitized areas)
        - 1: only pixels belonging to an infected cell
        """
        mask = np.zeros(image_size, dtype=np.uint8)
        
        for ann in annotations:
            # Check the cell type first to see if we should process this annotation
            cell_type = ann.get('cell_type', '').lower().strip()
            
            # --- The Key Logic Change ---
            # If the cell is NOT parasitized, skip it entirely.
            if 'parasitized' not in cell_type:
                continue

            # --- The rest of the logic is the same, but only runs for infected cells ---
            if ann.get('shape') == 'Point':
                x, y = int(ann.get('x', 0)), int(ann.get('y', 0))
                # Create a small circular mask for point annotations
                # np.ogrid is a fast way to generate coordinate grids
                rr, cc = np.ogrid[
                    max(0, y-16):min(image_size[0], y+16),
                    max(0, x-16):min(image_size[1], x+16)
                ]
                # Draw the circle onto the mask
                circle_mask = (rr - y)**2 + (cc - x)**2 <= 16**2
                mask[max(0, y-16):min(image_size[0], y+16), max(0, x-16):min(image_size[1], x+16)][circle_mask] = 1

            elif ann.get('shape') == 'Polygon' and 'polygon' in ann:
                # Use skimage to draw the filled polygon for this infected cell
                try:
                    from skimage.draw import polygon
                    xs, ys = zip(*ann['polygon'])
                    rr, cc = polygon(ys, xs, shape=image_size)
                    mask[rr, cc] = 1
                except ImportError:
                    print("Warning: skimage not available. Polygon masks for parasite segmentation will be empty.")
                    
        return mask
    
    def assign_severity_label(self, parasitemia_score):
        """
        Assign severity class based on parasitemia score.
        Returns integer class and string label.
        """
        if parasitemia_score == 0:
            return 0, 'negative'
        elif parasitemia_score <= 2:
            return 1, 'low'
        elif parasitemia_score <= 10:
            return 2, 'moderate'
        else:
            return 3, 'high'

# Our specialist that loads data based on the mode(STL and MTTL)
class FlexibleMalariaDataset(Dataset):
    """
    Flexible dataset class for various malaria tasks;
    - 'detection': object detection of cells
    - 'segmentation': cell segmentation
    - 'regression': parasitemia score regression
    - 'localization': heatmap localization of parasites
    - 'multi_task': combined tasks
    - 'severity': severity classification
    - 'cell_classif': individual cell patch classification
    """
    def __init__(self, samples, task_mode='multi_task', transform=None, augment=True):
        self.samples = samples
        self.task_mode = task_mode
        self.transform = transform
        self.augment = augment
        
        valid_tasks = ['detection', 'regression', 'localization', 'multi_task', 'segmentation', 'severity', 'cell_classif', 'heatmap']  
        if task_mode not in valid_tasks:
            raise ValueError(f"Invalid task_mode '{task_mode}'. Valid options: {valid_tasks}")
        
        if self.task_mode == 'cell_classif':
            self.cell_data = []
            print("Pre-cropping all cell patches...")
            for sample in tqdm(self.samples, desc="Cropping cells"):
                full_image = sample['image'].astype(np.uint8)
                img_h, img_w, _ = full_image.shape
                
                detection_data = sample.get('detection', {})
                bboxes_yolo = detection_data.get('bboxes', [])
                labels = detection_data.get('labels', [])
                
                for bbox, label in zip(bboxes_yolo, labels):
                    cx, cy, bw, bh = bbox
                    x1, y1, x2, y2 = (cx - bw / 2) * img_w, (cy - bh / 2) * img_h, \
                                     (cx + bw / 2) * img_w, (cy + bh / 2) * img_h
                    
                    cell_patch = crop_cell_from_image(full_image, [x1, y1, x2, y2])
                    if cell_patch is not None:
                        self.cell_data.append({'patch': cell_patch, 'label': int(label)})
            print(f"Created {len(self.cell_data)} individual cell patches.")
        
        # Augmentations
        if ALBUMENTATIONS_AVAILABLE and augment:
            if task_mode in ['detection', 'multi_task']:
                self.augmentation = A.Compose([
                    #A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=0.75),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                    A.OneOf([
                        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    ], p=0.5),
                    #A.CoarseDropout(max_holes=8, max_height=64, max_width=64, min_holes=1, min_height=16, min_width=16, fill_value=0, p=0.5),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['bbox_labels'], min_visibility=0.3))
            else:
                self.augmentation = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                    A.OneOf([
                        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    ], p=0.5),
                    #A.CoarseDropout(max_holes=8, max_height=64, max_width=64, min_holes=1, min_height=16, min_width=16, fill_value=0, p=0.5),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
        else:
            if augment:
                self.augmentation = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.08),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.augmentation = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    
    def __len__(self):
        if self.task_mode == 'cell_classif':
            return len(self.cell_data)
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.task_mode == 'cell_classif':
            data_item = self.cell_data[idx]
            patch_image = data_item['patch']
            label = data_item['label']

            if ALBUMENTATIONS_AVAILABLE and self.augment:
                augmented = self.augmentation(image=patch_image)
                image_tensor = augmented['image']
            else:
                image_tensor = self.augmentation(patch_image)

            return {
                'image': image_tensor,
                'label': torch.tensor(label, dtype=torch.long)
            }
            
        if self.task_mode == 'cell_classif':
            data_item = self.cell_data[idx]
            patch_image = data_item['patch']
            label = data_item['label']

            if ALBUMENTATIONS_AVAILABLE:
                augmented = self.augmentation(patch_image) 
                image_tensor = augmented['image']
            else:
                image_tensor = self.augmentation(patch_image)

            return {
                'image': image_tensor,
                'label': torch.tensor(label, dtype=torch.long)
            }
        
        sample = self.samples[idx]
        image = sample['image'].astype(np.uint8)
        
        if ALBUMENTATIONS_AVAILABLE:
            if self.task_mode == 'detection' and self.augment:
                bboxes = sample['detection']['bboxes']
                labels = sample['detection']['labels']
                
                if len(bboxes) > 0:
                    try:
                        augmented = self.augmentation(
                            image=image,
                            bboxes=bboxes,
                            bbox_labels=labels
                        )
                        image = augmented['image']
                        bboxes = np.array(augmented['bboxes'])
                        labels = np.array(augmented['bbox_labels'])
                    except:
                        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
                else:
                    image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
            else:
                try:
                    augmentation = A.Compose([
                        A.HorizontalFlip(p=0.5) if self.augment else A.NoOp(),
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2()
                    ])
                    augmented = augmentation(image=image)
                    image = augmented['image']
                except:
                    image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        else:
            image = self.augmentation(image)
        
        if self.task_mode == 'detection':
            return {
                'image': image,
                'bboxes': torch.from_numpy(sample['detection']['bboxes']).float() if len(sample['detection']['bboxes']) > 0 else torch.empty(0, 4),
                'labels': torch.from_numpy(sample['detection']['labels']).float() if len(sample['detection']['labels']) > 0 else torch.empty(0, dtype=torch.float),
                'num_objects': len(sample['detection']['bboxes']),
                'image_id': sample['image_id']
            }
        
        elif self.task_mode == 'regression':
            return {
                'image': image,
                'parasitemia_score': torch.tensor(sample['regression']['parasitemia_score'], dtype=torch.float32),
                'image_id': sample['image_id']
            }
        
        elif self.task_mode == 'localization' or self.task_mode == 'heatmap':
            heatmap_data = sample.get('localization', {}).get('heatmap')
            if heatmap_data is None:
                return {
                    'image': image,
                    'heatmap': torch.zeros(self.samples[0]['localization']['heatmap'].shape, dtype=torch.float), # Return a zero tensor
                    'image_id': sample.get('image_id', 'unknown')
                }
                
            return {
                'image': image,
                'heatmap': torch.from_numpy(heatmap_data).float(),
                'image_id': sample.get('image_id', 'unknown')
            }
            
        elif self.task_mode == 'segmentation': 
            return {
                'image': image,
                'mask': torch.from_numpy(sample['segmentation']['mask']).float(),
                'image_id': sample['image_id']
            }
        
        elif self.task_mode == 'severity':     
            return {
                'image': image,
                'severity_class': sample['severity']['severity_class'],
                'severity_label': sample['severity']['severity_label'],
                'image_id': sample['image_id']
            }
        
        elif self.task_mode == 'multi_task':
            return {
                'image': image,
                'parasitemia_score': torch.tensor(sample['multi_task']['parasitemia_score'], dtype=torch.float32),
                'bboxes': torch.from_numpy(sample['multi_task']['bboxes']).float() if len(sample['multi_task']['bboxes']) > 0 else torch.empty(0, 4),
                'bbox_labels': torch.from_numpy(sample['multi_task']['bbox_labels']).float() if len(sample['multi_task']['bbox_labels']) > 0 else torch.empty(0, dtype=torch.float),
                'heatmap': torch.from_numpy(sample['multi_task']['heatmap']).float().unsqueeze(0),
                'mask': torch.from_numpy(sample['multi_task']['mask']).float(),
                'severity_class': sample['multi_task']['severity_class'],
                'severity_label': sample['multi_task']['severity_label'],
                'num_objects': len(sample['multi_task']['bboxes']),
                'image_id': sample['image_id']
            }
            
# Function for creating a custom weighted random sampler for class balancing with strict priority on inf class
def create_class_balanced_sampler(data_samples, priority_class=0, num_classes=3):
    print("\nCreating a robust, strictly prioritized class-balanced sampler...")
    
    class_counts = Counter()
    for sample in data_samples:
        labels = sample.get('detection', {}).get('labels', 
                 sample.get('multi_task', {}).get('bbox_labels'))
        if labels is not None and len(labels) > 0:
            class_counts.update(int(l) for l in labels)
    
    if not class_counts:
        print("Warning: No labels found for sampler. Returning None.")
        return None

    for i in range(num_classes):
        if i not in class_counts:
            class_counts[i] = 0

    print(f"Total class counts in the provided sample set: {dict(class_counts)}")
    
    # Calculate raw class weights inversely proportional to frequency
    num_total_instances = sum(class_counts.values())
    raw_class_weights = {
        class_id: num_total_instances / count if count > 0 else 0
        for class_id, count in class_counts.items()
    }
    print(f"Raw calculated class weights: {raw_class_weights}")
    
    # Determine priority weight and cap for other classes
    priority_weight = raw_class_weights.get(priority_class, 1.0)
    weight_cap = priority_weight * 0.8
    
    print(f"Priority Class: {priority_class}, Final Priority Weight: {priority_weight:.2f}")
    print(f"Cap for other classes set to: {weight_cap:.2f}")

    final_class_weights = {}
    all_possible_classes = list(range(num_classes))
    
    # Adjust weights with strict priority
    for class_id in all_possible_classes:
        current_raw_weight = raw_class_weights.get(class_id, 0)
        
        if class_id == priority_class:
            final_class_weights[class_id] = priority_weight 
        else: 
            final_class_weights[class_id] = min(current_raw_weight, weight_cap)
    if 1 != priority_class and 1 in final_class_weights:
        final_class_weights[1] = max(final_class_weights[1], 1.0)

    print(f"Final adjusted class weights for ALL classes: {final_class_weights}")

    # Assign sample weights based on the maximum class weight of labels present
    sample_weights = []
    for sample in data_samples:
        max_weight_in_sample = final_class_weights.get(1, 1.0) 
        
        labels = sample.get('detection', {}).get('labels', 
                 sample.get('multi_task', {}).get('bbox_labels'))

        if labels is not None and len(labels) > 0:
            current_max_weight = 0.0
            found_scorable_label = False
            for label in labels:
                weight = final_class_weights.get(int(label), 0)
                if weight > 0:
                    found_scorable_label = True
                if weight > current_max_weight:
                    current_max_weight = weight
            
            if found_scorable_label:
                max_weight_in_sample = current_max_weight

        sample_weights.append(max_weight_in_sample)
    
    # Create the WeightedRandomSampler instance
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print("Final priority sampler created successfully.")
    return sampler

