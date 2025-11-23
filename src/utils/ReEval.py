# ==============================================================
# Re-evaluate any run
# Gwade Steve
# MTTL and Application to Malaria Detection
# April 2025
# ====================================================================
import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
from experiment_manager import ExperimentManager
from Evaluator import AdvancedEvaluator
from IPython.display import display

# Configs and paths
print("="*80)
print("STEP 1: INITIALIZING CONFIGURATION")
print("="*80)

BASE_EXPERIMENTS_DIR = "experiments"
DATA_DIR = os.path.join("..", "data", "preprocessed_NLM", "mttl_training_data")

# Verify paths exist
if not os.path.exists(BASE_EXPERIMENTS_DIR):
    print(f"FATAL: Experiments directory '{BASE_EXPERIMENTS_DIR}' not found.")
    sys.exit(1)

if not os.path.exists(DATA_DIR):
    print(f"FATAL: Data directory '{DATA_DIR}' not found.")
    sys.exit(1)

print(f"Experiments directory: {BASE_EXPERIMENTS_DIR}")
print(f"Data directory: {DATA_DIR}")

# Load all experiments
print("\n" + "="*80)
print("STEP 2: LOADING ALL EXPERIMENTS")
print("="*80)

try:
    manager = ExperimentManager(base_dir=BASE_EXPERIMENTS_DIR)
    all_runs_df = manager.load_all_experiments()
    print(f"Successfully loaded experiments database")
except Exception as e:
    print(f"FATAL: Failed to load experiments. Error: {e}")
    sys.exit(1)

if all_runs_df.empty:
    print(f"FATAL: No experiments found in '{BASE_EXPERIMENTS_DIR}'.")
    sys.exit(1)

print(f"Found {len(all_runs_df)} total experiment records")

# get evaluatable runs
print("\n" + "="*80)
print("STEP 3: IDENTIFYING EVALUATABLE RUNS")
print("="*80)

# Create identifiers
all_runs_df['task_name'] = all_runs_df['tasks'].apply(
    lambda t: t[0] if isinstance(t, list) and len(t) > 0 else 'unknown'
)
all_runs_df['uri'] = (all_runs_df['mode'].astype(str) + '-' + 
                      all_runs_df['task_name'].astype(str) + '-' + 
                      all_runs_df['run_id'].astype(str))

# Check for saved models
all_runs_df['has_model'] = all_runs_df['directory'].apply(
    lambda d: os.path.exists(os.path.join(d, 'best_model.pth')) if pd.notna(d) else False
)

evaluable_runs = all_runs_df[all_runs_df['has_model']].copy().reset_index(drop=True)

if evaluable_runs.empty:
    print("FATAL: No evaluatable experiments with 'best_model.pth' found.")
    sys.exit(1)

print(f"Found {len(evaluable_runs)} evaluatable experiments")

# show available runs
print("\n" + "="*80)
print("STEP 4: AVAILABLE EVALUABLE EXPERIMENTS")
print("="*80)

display_cols = ['uri', 'strategy', 'status', 'timestamp_start']
display_df = evaluable_runs[display_cols].copy()
display_df.index = np.arange(len(display_df))

print("\n")
display(display_df.style.set_properties(**{'text-align': 'left'}))

# User select
print("\n" + "="*80)
print("STEP 5: SELECT EXPERIMENT TO EVALUATE")
print("="*80)

try:
    selection = int(input(f"\nEnter the index number (0-{len(evaluable_runs)-1}): "))
    
    if not (0 <= selection < len(evaluable_runs)):
        print(f"Invalid selection. Please enter a number between 0 and {len(evaluable_runs)-1}.")
        sys.exit(1)
        
except ValueError:
    print("Invalid input. Please enter a valid integer.")
    sys.exit(1)

selected_run_info = evaluable_runs.iloc[selection]
run_directory = selected_run_info['directory']

print(f"\nSelected experiment:")
print(f"  - URI: {selected_run_info['uri']}")
print(f"  - Strategy: {selected_run_info['strategy']}")
print(f"  - Mode: {selected_run_info['mode']}")
print(f"  - Status: {selected_run_info['status']}")
print(f"  - Directory: {run_directory}")

#Check selection
print("\n" + "="*80)
print("STEP 6: VERIFYING SELECTED RUN")
print("="*80)

# Check config exists
config_path = os.path.join(run_directory, 'config.json')
if not os.path.exists(config_path):
    print(f"FATAL: Config file not found at {config_path}")
    sys.exit(1)

try:
    with open(config_path, 'r') as f:
        config_run = json.load(f)
    print(f"Config loaded successfully")
    print(f"  - Mode: {config_run.get('mode', 'Unknown')}")
    print(f"  - Active tasks: {config_run.get('active_tasks', [])}")
except Exception as e:
    print(f"FATAL: Failed to load config. Error: {e}")
    sys.exit(1)

# Check model exists
model_path = os.path.join(run_directory, 'best_model.pth')
if not os.path.exists(model_path):
    print(f"FATAL: Model file not found at {model_path}")
    sys.exit(1)

print(f"Model file exists")

# load test samples
print("\n" + "="*80)
print("STEP 7: LOADING TEST DATA")
print("="*80)

try:
    test_sample_path = os.path.join(DATA_DIR, "test", "test_samples.pkl")
    
    if not os.path.exists(test_sample_path):
        print(f"FATAL: Test samples file not found at {test_sample_path}")
        sys.exit(1)
    
    with open(test_sample_path, "rb") as f:
        test_samples_full = pickle.load(f)
    
    print(f"Successfully loaded test dataset")
    print(f"  - Total test samples: {len(test_samples_full)}")
    
except Exception as e:
    print(f"FATAL: Failed to load test samples. Error: {e}")
    sys.exit(1)

# setup evaluator
print("\n" + "="*80)
print("STEP 8: INITIALIZING EVALUATOR")
print("="*80)

try:
    evaluator = AdvancedEvaluator(
        run_directory=run_directory,
        test_samples=test_samples_full
    )
    print(f"Evaluator initialized successfully")
    print(f"  - Device: {evaluator.device}")
    print(f"  - Test samples loaded: {len(evaluator.test_samples)}")
    
except Exception as e:
    print(f"FATAL: Failed to initialize evaluator. Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# evaluate
print("\n" + "="*80)
print("STEP 9: RUNNING COMPREHENSIVE EVALUATION")
print("="*80)

try:
    if config_run.get('mode') == 'MTTL':
        print(f"\nInitiating MTTL evaluation on FULL TEST SET ({len(test_samples_full)} samples)")
        print(f"  Tasks: {config_run.get('active_tasks', [])}")
        
        evaluator.run_mttl(
            iou_thresh=0.5, 
            conf_thresh=0.5, 
            conf_thresh_vis=0.75, 
            focus_infected=True, 
            use_roi_classifier=False
        )
        
    else:  # STL
        print(f"\nInitiating STL evaluation on FULL TEST SET ({len(test_samples_full)} samples)")
        print(f"  Task: {config_run.get('active_tasks', [None])[0]}")
        
        evaluator.run(
            iou_thresh=0.5, 
            conf_thresh=0.5, 
            conf_thresh_vis=0.75, 
            focus_infected=True
        )
    
    print(f"\nEvaluation completed successfully")
    
except Exception as e:
    print(f"ERROR: Evaluation failed. Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# results
print("\n" + "="*80)
print("STEP 10: EVALUATION COMPLETE")
print("="*80)

print(f"\nAll results have been saved to: {run_directory}")
print("\n" + "="*80)