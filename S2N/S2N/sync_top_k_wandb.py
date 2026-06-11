import argparse
import os
import glob
import re
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Sync top K wandb runs from an optuna sweep.")
    parser.add_argument("--multi-run-dir", type=str, required=True, help="Path to the hydra multirun dir (e.g. logs_multi/GCN-S2N/2026-...)")
    parser.add_argument("--wandb-dir", type=str, required=True, help="Path to the wandb logs directory")
    parser.add_argument("--top-k", type=int, default=3, help="Number of best runs to sync")
    args = parser.parse_args()

    results = []

    # Iterate over all trial directories 0/, 1/, 2/...
    for trial_dir in glob.glob(os.path.join(args.multi_run_dir, "*")):
        if not os.path.isdir(trial_dir) or not os.path.basename(trial_dir).isdigit():
            continue
        
        log_file = os.path.join(trial_dir, "run_main.log")
        if not os.path.isfile(log_file):
            continue
        
        with open(log_file, "r") as f:
            content = f.read()
            
        # Extract objective value
        val_match = re.search(r"HPARAMS_SEARCH OBJECTIVE: [^,]+, VALUE: ([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", content)
        if not val_match:
            continue
        value = float(val_match.group(1))

        # Extract wandb run ID (optional, needed only for syncing)
        run_id_match = re.search(r"Wandb run ID: ([a-zA-Z0-9]+)", content)
        run_id = run_id_match.group(1) if run_id_match else None
        
        results.append((value, run_id, trial_dir))
    
    if not results:
        print("No valid runs with wandb run IDs and objective values found.")
        return

    # Sort results. Assuming maximization!
    results.sort(key=lambda x: x[0], reverse=True)
    
    top_k_results = results[:args.top_k]
    print(f"--- Top {len(top_k_results)} Trials (by objective value) ---")
    for rank, (val, run_id, d) in enumerate(top_k_results, start=1):
        trial_id = os.path.basename(d)
        run_id_text = run_id if run_id is not None else "N/A"
        print(f"Rank {rank} | Trial {trial_id} | Value: {val:.8f} | Run ID: {run_id_text}")
    
    # Sync to wandb
    for _, run_id, _ in top_k_results:
        if run_id is None:
            print("Skipping sync for trial without WandB run ID.")
            continue

        # wandb offline directories look like offline-run-YYYYMMDD_HHMMSS-run_id
        offline_dirs = glob.glob(os.path.join(args.wandb_dir, f"offline-run-*-{run_id}"))
        if not offline_dirs:
            print(f"Could not find offline directory for run {run_id} in {args.wandb_dir}")
            continue
        
        offline_dir = offline_dirs[0]
        print(f"\nSyncing offline run: {offline_dir}")
        subprocess.run(["wandb", "sync", offline_dir])

if __name__ == "__main__":
    main()
