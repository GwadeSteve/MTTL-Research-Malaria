import os
import sys
import zipfile
import shutil
import gdown
from pathlib import Path

# config and structure
GDRIVE_FILE_ID = '1DBx0nYCAQZqMLUMBDDSbV98fyeFnKmrc'

REQUIRED_STRUCTURE = {
    'MTTL': {
        'detection': ['1-Class', '2-Class', '3-Class'],
        'roi_classif': ['2-Class', '3-Class'],
        'heatmap': [],       
        'segmentation': []   
    },
    'STL': {
        'detection': ['1-Class', '2-Class', '3-Class'],
        'roi_classif': ['2-Class', '3-Class'],
        'heatmap': [],
        'segmentation': []
    },
    'Yolo_Models': ['best_yolov8n.pt', 'best_yolov8s.pt']
}

# get project root dir
def get_project_root():
    try:
        current = Path(__file__).resolve()
        for _ in range(4): 
            if (current / 'requirements.txt').exists() or (current / 'src').exists():
                return current
            current = current.parent
    except Exception:
        pass
    
    return Path('.')

# check dirs structure
def verify_structure(models_dir, verbose=True):
    
    if not models_dir.exists():
        if verbose: print(f"Critical Error: Missing root directory: {models_dir}")
        return False
    
    missing_found = False
    print(f"Verifying structure in: {models_dir} ...")

    for category, content in REQUIRED_STRUCTURE.items():
        cat_path = models_dir / category
        
        # 1. Check Category Folder
        if not cat_path.exists():
            if verbose: print(f"Missing folder: {cat_path}")
            missing_found = True
            continue

        # Special yolo case
        if category == 'Yolo_Models':
            for filename in content:
                if not (cat_path / filename).exists():
                    if verbose: print(f"Missing Yolo file: {cat_path / filename}")
                    missing_found = True
            continue

        # standard cases
        for task, sub_folders in content.items():
            task_path = cat_path / task
            
            if not task_path.exists():
                if verbose: print(f"Missing task folder: {task_path}")
                missing_found = True
                continue

            paths_to_check = []
            
            if not sub_folders:
                paths_to_check.append(task_path)
            else:
                for sub in sub_folders:
                    sub_path = task_path / sub
                    if not sub_path.exists():
                        if verbose: print(f"Missing class folder: {sub_path}")
                        missing_found = True
                    else:
                        paths_to_check.append(sub_path)

            for p in paths_to_check:
                if not (p / 'best_model.pth').exists():
                    if verbose: print(f"Missing file: {p / 'best_model.pth'}")
                    missing_found = True
                if not (p / 'config.json').exists():
                    if verbose: print(f"Missing file: {p / 'config.json'}")
                    missing_found = True

    if missing_found:
        print("\nVerification Failed. See missing files above.")
        return False
    else:
        print("\nVerification Successful! All structure matches.")
        return True

# download model weights
def download_models(force=False, interactive=True):
    root_dir = get_project_root()
    models_dir = root_dir / 'models'
    zip_path = root_dir / 'models.zip'
    
    is_valid = verify_structure(models_dir, verbose=False)
    
    if is_valid and not force:
        print(f"Models verified at: {models_dir.resolve()}")
        return True

    if interactive:
        print(f"Model weights status: {'Valid' if is_valid else 'Missing/Incomplete'}")
        print(f"Target directory: {models_dir.resolve()}")
        print(f"This will download ~1.9GB of data.")
        response = input("Do you want to proceed? (y/n): ").strip().lower()
        if response != 'y':
            print("\nDownload cancelled.")
            print(f"You can download manually here: https://drive.google.com/uc?id={GDRIVE_FILE_ID}")
            return False

    if models_dir.exists() and not is_valid:
        print("Removing incomplete models directory...")
        shutil.rmtree(models_dir)
    
    root_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading models.zip...")
    url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
    
    try:
        if not zip_path.exists():
            gdown.download(url, str(zip_path), quiet=False)
        
        print("\nExtracting models...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            top_level_items = {item.split('/')[0] for item in zip_ref.namelist()}
            
            if 'models' in top_level_items:
                zip_ref.extractall(root_dir)
            else:
                models_dir.mkdir(exist_ok=True)
                zip_ref.extractall(models_dir)
            
        print("Cleaning up zip file...")
        if zip_path.exists():
            os.remove(zip_path)
            
        if verify_structure(models_dir, verbose=True):
            print(f"Models setup complete at {models_dir.resolve()}")
            return True
        else:
            print("Verification failed after download. See missing files above.")
            return False
        
    except Exception as e:
        print(f"\nError: {e}")
        print(f"Please download manually from: https://drive.google.com/uc?id={GDRIVE_FILE_ID}")
        return False

if __name__ == "__main__":
    force_download = '--force' in sys.argv
    download_models(force=force_download, interactive=not force_download)