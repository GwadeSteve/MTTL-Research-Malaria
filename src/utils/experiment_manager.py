# ====================================================================
# Experiment manager for this research MTTL for Malaria Detection
# Gwade Steve
# MTTL and Application to Malaria Detection
# April 2025
# ====================================================================
import os
import json
import uuid
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

class ExperimentManager:
    """
    Experiment Manager that automatically organizes
    experiments by mode (STL/MTTL) and task.
    """
    def __init__(self, base_dir: str = "experiments"):
        self.root_dir = base_dir
        os.makedirs(self.root_dir, exist_ok=True)
        self.current_experiment_dir = None
        self.current_uuid = None

    def _get_next_run_id(self, experiment_base_path: str) -> int:
        os.makedirs(experiment_base_path, exist_ok=True)
        existing_runs = [d for d in os.listdir(experiment_base_path) if d.startswith('run_')]
        if not existing_runs:
            return 1
        max_id = 0
        for run in existing_runs:
            try:
                run_id = int(run.split('_')[1])
                if run_id > max_id:
                    max_id = run_id
            except (IndexError, ValueError):
                continue
        return max_id + 1

    def new_experiment(self, config: dict, model, notes: str = ""):
        mode = config.get('mode', 'STL').upper()
        tasks = config.get('active_tasks', ['unknown'])
        strategy = config.get('strategy', 'default')

        if mode == 'STL':
            task_name = tasks[0] if tasks else 'unknown'
        else:
            task_name = "MTTL_" + "_".join(sorted(tasks))

        experiment_base_path = os.path.join(self.root_dir, mode, task_name)
        run_id = self._get_next_run_id(experiment_base_path)
        
        self.current_uuid = str(uuid.uuid4())
        run_name = f"run_{run_id}_{strategy}"
        
        self.current_experiment_dir = os.path.join(experiment_base_path, run_name)
        os.makedirs(self.current_experiment_dir, exist_ok=True)
        
        if hasattr(model, 'get_parameter_breakdown'):
            param_breakdown = model.get_parameter_breakdown()
            with open(os.path.join(self.current_experiment_dir, 'parameters.json'), 'w') as f:
                json.dump(param_breakdown, f, indent=4)
        
        metadata = {
            'uuid': self.current_uuid,
            'run_id': run_id,
            'run_name': run_name,
            'timestamp_start': datetime.now().isoformat(),
            'mode': mode,
            'tasks': tasks,
            'strategy': strategy,
            'status': 'running',
            'notes': notes
        }
        with open(os.path.join(self.current_experiment_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
            
        with open(os.path.join(self.current_experiment_dir, 'config.json'), 'w') as f:
            def safe_json_dump(obj): return str(obj)
            json.dump(config, f, indent=4, default=safe_json_dump)
            
        print(f"--- New Experiment Started ---")
        print(f"  Run Name: {run_name}")
        print(f"  Directory: {self.current_experiment_dir}")
        print(f"-----------------------------")
        return self.current_experiment_dir

    def get_model_path(self):
        if not self.current_experiment_dir:
            raise Exception("Experiment not initialized. Call .new_experiment() first.")
        return os.path.join(self.current_experiment_dir, "best_model.pth")

    def save_history(self, history: dict):
        if not self.current_experiment_dir: return
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(self.current_experiment_dir, 'history.csv'), index_label='epoch')

    def save_plot(self, fig, plot_name: str):
        if not self.current_experiment_dir: return
        plots_dir = os.path.join(self.current_experiment_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        fig.savefig(os.path.join(plots_dir, f"{plot_name}.png"), bbox_inches='tight')
        #plt.close(fig)

    def log_final_results(self, test_metrics: dict, filename: str = "test_results.json"):
        if not self.current_experiment_dir: return
        
        def make_json_serializable(data):
            if isinstance(data, dict):
                return {key: make_json_serializable(value) for key, value in data.items()}
            elif isinstance(data, list):
                return [make_json_serializable(item) for item in data]
            elif hasattr(data, 'item'):
                return data.item()
            elif isinstance(data, (int, float, str, bool)) or data is None:
                return data
            else:
                return str(data)

        serializable_metrics = make_json_serializable(test_metrics)
        
        output_path = os.path.join(self.current_experiment_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)

        metadata_path = os.path.join(self.current_experiment_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata['status'] = 'completed'
            metadata['timestamp_end'] = datetime.now().isoformat()
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
        uuid_str = self.current_uuid
        if uuid_str is None:
             if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                uuid_str = metadata.get('uuid', 'UNKNOWN')
             else:
                uuid_str = 'UNKNOWN'
        
        print(f"--- Experiment {uuid_str[:8]} Completed & Logged ---")

    @staticmethod
    def _get_all_exp_dirs(base_dir: str = "experiments"):
        """A helper to walk the directory tree and find all individual experiment folders."""
        exp_dirs = []
        if not os.path.isdir(base_dir): return exp_dirs
        for mode in os.listdir(base_dir):
            mode_dir = os.path.join(base_dir, mode)
            if not os.path.isdir(mode_dir): continue
            for task_name in os.listdir(mode_dir):
                task_dir = os.path.join(mode_dir, task_name)
                if not os.path.isdir(task_dir): continue
                for exp_name in os.listdir(task_dir):
                    exp_dir = os.path.join(task_dir, exp_name)
                    if os.path.isdir(exp_dir):
                        exp_dirs.append(exp_dir)
        return exp_dirs

    @staticmethod
    def load_all_experiments(base_dir: str = "experiments") -> pd.DataFrame:
        all_metadata = []
        exp_dirs = ExperimentManager._get_all_exp_dirs(base_dir)

        for exp_dir in exp_dirs:
            metadata_path = os.path.join(exp_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    metadata['directory'] = exp_dir

                    results_path = None
                    mttl_results_path = os.path.join(exp_dir, 'mttl_test_results.json')
                    stl_results_path = os.path.join(exp_dir, 'test_results.json')
                    
                    if os.path.exists(mttl_results_path):
                        results_path = mttl_results_path
                    elif os.path.exists(stl_results_path):
                        results_path = stl_results_path

                    if results_path:
                        with open(results_path, 'r') as f:
                            results = json.load(f)
                        
                        mode = metadata.get('mode', 'STL')
                        tasks = metadata.get('tasks', [])
                        
                        if mode == 'MTTL':
                            # For MTTL, results are already nested by task
                            for task, metrics in results.items():
                                if isinstance(metrics, dict):
                                    for metric_name, value in metrics.items():
                                        metadata[f"{task}_{metric_name}"] = value
                        elif mode == 'STL' and tasks:
                            # For STL, results are flat. We add the task prefix.
                            task_name = tasks[0]
                            for metric_name, value in results.items():
                                metadata[f"{task_name}_{metric_name}"] = value
                        else:
                             # Fallback 
                            metadata.update(results)
                    
                    all_metadata.append(metadata)
                except Exception as e:
                    print(f"Warning: Could not read metadata/results in {exp_dir}. Error: {e}")
        
        df = pd.DataFrame(all_metadata)
        if 'timestamp_start' in df.columns:
            df['timestamp_start'] = pd.to_datetime(df['timestamp_start'])
            df = df.sort_values(by='timestamp_start', ascending=False).reset_index(drop=True)
        return df
    
    @staticmethod
    def clean_incomplete_runs(base_dir: str = "experiments", dry_run: bool = True):
        print("\n" + "="*60)
        print("Scanning for incomplete or failed experiment runs...")
        print(f"Mode: {'DRY RUN (no files will be deleted)' if dry_run else 'DELETION ENABLED'}")
        print("="*60)

        all_exp_dirs = ExperimentManager._get_all_exp_dirs(base_dir)
        dirs_to_delete = []

        for exp_dir in all_exp_dirs:
            model_path = os.path.join(exp_dir, "best_model.pth")
            history_path = os.path.join(exp_dir, "history.csv")
            
            if not os.path.exists(model_path) or not os.path.exists(history_path):
                dirs_to_delete.append(exp_dir)

        if not dirs_to_delete:
            print("No incomplete runs found. Directory is clean.")
            return

        print(f"Found {len(dirs_to_delete)} incomplete run(s) to delete:")
        for d in dirs_to_delete:
            print(f" - {d}")
        
        if not dry_run:
            confirm = input("\nAre you sure you want to permanently delete these directories? (y/n): ")
            if confirm.lower() == 'y':
                for d in dirs_to_delete:
                    try:
                        shutil.rmtree(d)
                        print(f"Deleted: {d}")
                    except OSError as e:
                        print(f"Error deleting {d}: {e}")
                print("\nCleanup complete.")
            else:
                print("\nCleanup aborted by user.")
        else:
            print("\nDry run complete. No files were deleted.")
            print("To perform the deletion, call this function with `dry_run=False`.")