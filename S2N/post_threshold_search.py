import sys
import os
import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "S2N"))
import S2N.data as data
import S2N.model as model
import hydra
from S2N.run_main import _run_threshold_search

def run_post_search(run_dir, ckpt_name):
    print(f"Loading config from {run_dir}/.hydra/config.yaml")
    config = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    
    datamodule = hydra.utils.instantiate(config.datamodule)
    
    ckpt_path = os.path.join(run_dir, "checkpoints", ckpt_name)
    print(f"Loading model from {ckpt_path}")
    
    # Load model
    my_model = model.GraphNeuralModel.load_from_checkpoint(ckpt_path, given_datamodule=datamodule)
    
    trainer = log_trainer = Trainer(
        accelerator=config.trainer.get("accelerator", "gpu"),
        devices=config.trainer.get("devices", 1),
        logger=False
    )
    
    my_model, best_threshold, best_metric = _run_threshold_search(
        trainer=trainer,
        model=my_model,
        datamodule=datamodule,
        best_ckpt_path=ckpt_path,
        metric_name="illicit_f1",
        n_trials=50,
        lower=0.01,
        upper=0.99
    )
    
    print()
    print("=" * 60)
    print(f"OPTUNA SEARCH FINISHED")
    print(f"Best Valid illicit_f1: {best_metric:.6f} at Threshold: {best_threshold:.4f}")
    print("=" * 60)
    print()
    
    # Evaluate Test Set with Best Threshold
    print("Running test evaluation with optimized threshold...")
    trainer.test(model=my_model, datamodule=datamodule)

if __name__ == "__main__":
    RUN_DIR = "/home/ghonkoop/repos/S2N/logs_multi/GCN-S2N/2026-05-18-21-19-52/0"
    CKPT_NAME = "epoch_epoch=016.ckpt"
    run_post_search(RUN_DIR, CKPT_NAME)
