import argparse
import os
import tempfile

import yaml

from train import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on validation/test splits.")
    parser.add_argument("--config", required=True, help="Path to the training config YAML.")
    parser.add_argument("--checkpoint-path", default=None, help="Checkpoint path. Defaults to save_dir/run_name/best_checkpoint.pth.")
    parser.add_argument("--run-name", default=None, help="Run name to use during evaluation.")
    parser.add_argument("--eval-node-batch-size", type=int, default=None, help="Chunk size for eval node aggregation.")
    parser.add_argument("--threshold-lower", type=float, default=None, help="Override threshold search lower bound.")
    parser.add_argument("--threshold-upper", type=float, default=None, help="Override threshold search upper bound.")
    parser.add_argument("--threshold-n-trials", type=int, default=None, help="Override threshold search number of trials.")
    parser.add_argument("--threshold-metric", default=None, help="Override threshold search metric.")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable W&B logging for this evaluation job.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Config is empty or invalid YAML: {args.config}")

    cfg.setdefault("logging", {})
    cfg.setdefault("train", {})

    if args.run_name:
        cfg["logging"]["run_name"] = args.run_name
    if args.eval_node_batch_size is not None:
        cfg["train"]["eval_node_batch_size"] = args.eval_node_batch_size
    if (
        args.threshold_lower is not None
        or args.threshold_upper is not None
        or args.threshold_n_trials is not None
        or args.threshold_metric is not None
    ):
        cfg.setdefault("threshold_search", {})
        cfg["threshold_search"]["enabled"] = True
        if args.threshold_lower is not None:
            cfg["threshold_search"]["lower"] = args.threshold_lower
        if args.threshold_upper is not None:
            cfg["threshold_search"]["upper"] = args.threshold_upper
        if args.threshold_n_trials is not None:
            cfg["threshold_search"]["n_trials"] = args.threshold_n_trials
        if args.threshold_metric is not None:
            cfg["threshold_search"]["metric"] = args.threshold_metric
    if args.disable_wandb:
        cfg["logging"]["use_wandb"] = False

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
        eval_config_path = f.name

    try:
        eval_args = argparse.Namespace(config=eval_config_path, resume=False)
        evaluate_checkpoint(eval_args, checkpoint_path=args.checkpoint_path)
    finally:
        try:
            os.unlink(eval_config_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
