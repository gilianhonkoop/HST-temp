import argparse
import json
import os

from train import evaluate_checkpoint_from_config, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a PNA checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    metrics = evaluate_checkpoint_from_config(cfg, checkpoint_path=args.checkpoint)
    run_dir = os.path.join(str(cfg.get("logging", {}).get("save_dir", "outputs")), str(cfg.get("logging", {}).get("run_name", "pna_run")))
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "eval_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
