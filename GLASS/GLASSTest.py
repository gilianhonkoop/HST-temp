from impl import models, SubGDataset, train, metrics, utils, config
import datasets
import torch
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import argparse
import torch.nn as nn
import functools
import numpy as np
import time
import random
import copy
import yaml
import os
import wandb
from dotenv import load_dotenv

parser = argparse.ArgumentParser(description='')
# Dataset settings
parser.add_argument('--dataset', type=str, default='ppi_bp')
parser.add_argument('--config', type=str, default=None)
# Node feature settings. 
# deg means use node degree. one means use homogeneous embeddings.
# nodeid means use pretrained node embeddings in ./Emb
parser.add_argument('--use_deg', action='store_true')
parser.add_argument('--use_one', action='store_true')
parser.add_argument('--use_nodeid', action='store_true')
# node label settings
parser.add_argument('--use_maxzeroone', action='store_true')
# Edge feature settings.
parser.add_argument('--use_edge_features', action='store_true')
parser.add_argument('--edge_feature_cols', type=str, default=None)

parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--use_seed', action='store_true')
parser.add_argument('--seed', type=int, default=None)

args = parser.parse_args()
config_path = args.config if args.config else f"config/{args.dataset}.yml"


def resolve_dataset_loader_key(dataset_name: str) -> str:
    normalized = dataset_name.strip().lower().replace("-", "_")
    aml_aliases = {
        "hi_small",
        "hi_medium",
        "li_small",
        "li_medium",
        "aml_hi_small",
        "aml_hi_medium",
        "aml_li_small",
        "aml_li_medium",
        "saml_d",
        "samld",
        "aml",
    }
    if normalized in aml_aliases:
        return "aml"
    return normalized


dataset_loader_key = resolve_dataset_loader_key(args.dataset)
config.set_device(args.device)

load_dotenv()
# read configuration early so run_name can be sourced from YAML
with open(config_path) as f:
    params = yaml.safe_load(f) or {}
base_seed = args.seed if args.seed is not None else int(params.get("seed", 0))


def parse_edge_feature_cols(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [col.strip() for col in value.split(",") if col.strip()]
    return list(value)


use_edge_features = args.use_edge_features or bool(
    params.get("use_edge_features", False))
edge_feature_cols = parse_edge_feature_cols(
    args.edge_feature_cols
    if args.edge_feature_cols is not None else params.get("edge_feature_cols"))
edge_feature_clip = params.get("edge_feature_clip", None)
if edge_feature_clip is not None:
    edge_feature_clip = float(edge_feature_clip)


def resolve_split_ratios(config_params):
    split = config_params.get("split", None)
    if split is not None:
        if len(split) != 3:
            raise ValueError("Config key 'split' must have three values: [train, val, test].")
        train_ratio, val_ratio, test_ratio = [float(v) for v in split]
    else:
        train_ratio = float(
            config_params.get("train_ratio",
                              config_params.get("aml_train_ratio", 0.7)))
        val_ratio = float(
            config_params.get("val_ratio",
                              config_params.get("aml_val_ratio", 0.15)))
        test_ratio = float(config_params.get("test_ratio",
                                             1.0 - train_ratio - val_ratio))

    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError(
            f"Split ratios must be positive, got train={train_ratio}, val={val_ratio}, test={test_ratio}."
        )
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got train={train_ratio}, val={val_ratio}, test={test_ratio}."
        )
    return train_ratio, val_ratio, test_ratio


train_ratio, val_ratio, test_ratio = resolve_split_ratios(params)
os.environ["AML_TRAIN_RATIO"] = str(train_ratio)
os.environ["AML_VAL_RATIO"] = str(val_ratio)
split_mode = str(params.get("split_mode", "temporal")).strip().lower()
os.environ["AML_SPLIT_MODE"] = split_mode

USE_WANDB = os.getenv("USE_WANDB", "0") == "1"
WANDB_PROJECT = os.getenv("WANDB_PROJECT",
                          "hierarchical-subgraph-transformer")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)
WANDB_RUN_NAME = os.getenv("RUN_NAME") or params.get("run_name") or f"{args.dataset}_run"
if USE_WANDB:
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise ValueError(
            "WANDB_API_KEY environment variable is not set. Please export it in your SLURM script, or set USE_WANDB=0."
        )
    wandb.login(key=api_key)
    wandb.init(project=WANDB_PROJECT,
               entity=WANDB_ENTITY,
               name=WANDB_RUN_NAME,
               config={
                   "dataset": args.dataset,
                   "config_path": config_path,
                   "use_deg": args.use_deg,
                   "use_one": args.use_one,
                   "use_nodeid": args.use_nodeid,
                   "use_maxzeroone": args.use_maxzeroone,
                   "use_edge_features": use_edge_features,
                   "edge_feature_cols": edge_feature_cols,
                   "repeat": args.repeat,
                   "device": args.device,
                   "use_seed": args.use_seed,
                   "seed": base_seed,
                   "train_ratio": train_ratio,
                   "val_ratio": val_ratio,
                   "test_ratio": test_ratio,
                   "split_mode": split_mode,
               })

def set_seed(seed: int):
    print("seed ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu


if args.use_seed:
    set_seed(base_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

baseG = datasets.load_dataset(dataset_loader_key,
                              use_edge_features=use_edge_features,
                              edge_feature_cols=edge_feature_cols,
                              edge_feature_clip=edge_feature_clip)

trn_dataset, val_dataset, tst_dataset = None, None, None
max_deg, output_channels = 0, 1
score_fn = None
use_aml_metrics = dataset_loader_key == "aml"
pos_weight_values = None
threshold_search = False
threshold_candidates = None
model_selection_metric = "val/minor_prauc"
threshold_selection_metric = "val/minor_f1"
train_eval_threshold = 0.5


def log_wandb_metrics(step,
                      train_loss=None,
                      train_metrics=None,
                      val_metrics=None,
                      best_threshold=None,
                      end_epoch=None,
                      avg_score=None,
                      avg_error=None):
    if not USE_WANDB:
        return

    payload = {}
    if train_loss is not None:
        payload["train/loss"] = float(train_loss)
    if best_threshold is not None:
        payload["val/best_threshold"] = float(best_threshold)
    if end_epoch is not None:
        payload["summary/end_epoch"] = int(end_epoch)
    if avg_score is not None:
        payload["summary/avg_score"] = float(avg_score)
    if avg_error is not None:
        payload["summary/avg_error"] = float(avg_error)

    for split_name, split_metrics in (("train", train_metrics),
                                      ("val", val_metrics)):
        if not split_metrics:
            continue
        for metric_name, metric_value in split_metrics.items():
            payload[f"{split_name}/{metric_name}"] = float(metric_value)

    wandb.log(payload, step=step)


def log_final_test_metrics(step, test_metrics):
    if not USE_WANDB or not test_metrics:
        return
    wandb.log({f"test/{k}": float(v)
               for k, v in test_metrics.items()},
              step=step)
    for metric_name, metric_value in test_metrics.items():
        wandb.summary[f"test/{metric_name}"] = float(metric_value)


def canonical_metric_name(metric_name):
    name = str(metric_name).strip().lower()
    if "/" in name:
        name = name.split("/", 1)[1]

    aliases = {
        "minor_pr_auc": "minor_prauc",
        "major_pr_auc": "major_prauc",
        "minor_auprc": "minor_prauc",
        "major_auprc": "major_prauc",
        "minor_roc_auc": "minor_rocauc",
        "major_roc_auc": "major_rocauc",
        "pr_auc": "prauc",
        "roc_auc": "rocauc",
        "f1": "minor_f1",
    }
    return aliases.get(name, name)


def metric_direction(metric_name):
    key = canonical_metric_name(metric_name)
    if key.endswith("loss") or key == "loss":
        return "min"
    return "max"


def metric_value(metric_name, metrics_dict, loss_value=None):
    key = canonical_metric_name(metric_name)
    if key == "loss":
        if loss_value is None:
            raise ValueError("Requested metric 'loss' but no loss value was provided")
        return float(loss_value)
    if key not in metrics_dict:
        available = ", ".join(sorted(metrics_dict.keys()))
        raise KeyError(
            f"Metric '{metric_name}' (canonical '{key}') not found. Available metrics: {available}"
        )
    return float(metrics_dict[key])


def count_trainable_params(model):
    return sum(param.numel() for param in model.parameters()
               if param.requires_grad)

if baseG.y.unique().shape[0] == 2:
    # binary classification task
    def loss_fn(x, y):
        logits = x.flatten()
        targets = y.flatten()
        if pos_weight_values and len(pos_weight_values) == 2:
            licit_weight = torch.as_tensor(pos_weight_values[0], device=targets.device)
            illicit_weight = torch.as_tensor(pos_weight_values[1], device=targets.device)
            weights = torch.where(targets > 0.5, illicit_weight, licit_weight)
            loss = BCEWithLogitsLoss(reduction="none")(logits, targets)
            return (loss * weights).mean()
        return BCEWithLogitsLoss()(logits, targets)

    baseG.y = baseG.y.to(torch.float)
    if baseG.y.ndim > 1:
        output_channels = baseG.y.shape[1]
    else:
        output_channels = 1
    score_fn = metrics.binaryf1
else:
    # multi-class classification task
    baseG.y = baseG.y.to(torch.int64)
    loss_fn = CrossEntropyLoss()
    output_channels = baseG.y.unique().shape[0]
    score_fn = metrics.microf1

loader_fn = SubGDataset.GDataloader
tloader_fn = SubGDataset.GDataloader


def split():
    '''
    load and split dataset.
    '''
    # initialize and split dataset
    global trn_dataset, val_dataset, tst_dataset, baseG
    global max_deg, output_channels, loader_fn, tloader_fn
    baseG = datasets.load_dataset(dataset_loader_key,
                                  use_edge_features=use_edge_features,
                                  edge_feature_cols=edge_feature_cols,
                                  edge_feature_clip=edge_feature_clip)
    if baseG.y.unique().shape[0] == 2:
        baseG.y = baseG.y.to(torch.float)
    else:
        baseG.y = baseG.y.to(torch.int64)
    # initialize node features
    if args.use_deg:
        baseG.setDegreeFeature()
    elif args.use_one:
        baseG.setOneFeature()
    elif args.use_nodeid:
        baseG.setNodeIdFeature()
    else:
        raise NotImplementedError

    max_deg = torch.max(baseG.x)
    baseG.to(config.device)
    # split data
    trn_dataset = SubGDataset.GDataset(*baseG.get_split("train"))
    val_dataset = SubGDataset.GDataset(*baseG.get_split("valid"))
    tst_dataset = SubGDataset.GDataset(*baseG.get_split("test"))
    # choice of dataloader
    if args.use_maxzeroone:

        def tfunc(ds, bs, shuffle=True, drop_last=True):
            return SubGDataset.ZGDataloader(ds,
                                            bs,
                                            z_fn=utils.MaxZOZ,
                                            shuffle=shuffle,
                                            drop_last=drop_last)

        def loader_fn(ds, bs):
            return tfunc(ds, bs)

        def tloader_fn(ds, bs):
            return tfunc(ds, bs, True, False)
    else:

        def loader_fn(ds, bs):
            return SubGDataset.GDataloader(ds, bs)

        def tloader_fn(ds, bs):
            return SubGDataset.GDataloader(ds, bs, shuffle=True)


def buildModel(hidden_dim, conv_layer, dropout, jk, pool, z_ratio, aggr):
    '''
    Build a GLASS model.
    Args:
        jk: whether to use Jumping Knowledge Network.
        conv_layer: number of GLASSConv.
        pool: pooling function transfer node embeddings to subgraph embeddings.
        z_ratio: see GLASSConv in impl/model.py. Z_ratio in [0.5, 1].
        aggr: aggregation method. mean, sum, or gcn. 
    '''
    edge_feature_dim = 0
    if use_edge_features and baseG.edge_attr.dim() == 2:
        edge_feature_dim = baseG.edge_attr.size(-1)

    conv = models.EmbZGConv(hidden_dim,
                            hidden_dim,
                            conv_layer,
                            max_deg=max_deg,
                            activation=nn.ELU(inplace=True),
                            jk=jk,
                            dropout=dropout,
                            conv=functools.partial(models.GLASSConv,
                                                   aggr=aggr,
                                                   z_ratio=z_ratio,
                                                   dropout=dropout,
                                                   edge_feature_dim=edge_feature_dim),
                            gn=True)

    # use pretrained node embeddings.
    if args.use_nodeid:
        print("load ", f"./Emb/{dataset_loader_key}_{hidden_dim}.pt")
        emb = torch.load(f"./Emb/{dataset_loader_key}_{hidden_dim}.pt",
                         map_location=torch.device('cpu')).detach()
        conv.input_emb = nn.Embedding.from_pretrained(emb, freeze=False)

    mlp = nn.Linear(hidden_dim * (conv_layer) if jk else hidden_dim,
                    output_channels)

    pool_fn_fn = {
        "mean": models.MeanPool,
        "max": models.MaxPool,
        "sum": models.AddPool,
        "size": models.SizePool
    }
    if pool in pool_fn_fn:
        pool_fn1 = pool_fn_fn[pool]()
    else:
        raise NotImplementedError

    gnn = models.GLASS(conv, torch.nn.ModuleList([mlp]),
                       torch.nn.ModuleList([pool_fn1])).to(config.device)
    return gnn


def test(pool="size",
         aggr="mean",
         hidden_dim=64,
         conv_layer=8,
         dropout=0.3,
         jk=1,
         lr=1e-3,
         z_ratio=0.8,
         batch_size=None,
         resi=0.7):
    '''
    Test a set of hyperparameters in a task.
    Args:
        jk: whether to use Jumping Knowledge Network.
        z_ratio: see GLASSConv in impl/model.py. A hyperparameter of GLASS.
        resi: the lr reduce factor of ReduceLROnPlateau.
    '''
    outs = []
    t1 = time.time()
    # we set batch_size = tst_dataset.y.shape[0] // num_div.
    num_div = tst_dataset.y.shape[0] / batch_size
    print(f'num_div: {num_div}')
    # we use num_div to calculate the number of iteration per epoch and count the number of iteration.
    if dataset_loader_key in ["density", "component", "cut_ratio", "coreness"]:
        num_div /= 5
    patience_limit = (early_stop_patience
                      if early_stop_patience is not None else 150 / num_div)
    print(f'early_stop_patience: {patience_limit}')

    outs = []
    last_step = 0
    final_test_metrics = None
    for repeat in range(args.repeat):
        if args.use_seed:
            set_seed(base_seed + repeat)
        print(f"repeat {repeat}")
        split()
        gnn = buildModel(hidden_dim, conv_layer, dropout, jk, pool, z_ratio,
                         aggr)
        trainable_params = count_trainable_params(gnn)
        print(f"trainable params: {trainable_params:,}", flush=True)
        trn_loader = loader_fn(trn_dataset, batch_size)
        val_loader = tloader_fn(val_dataset, batch_size)
        tst_loader = tloader_fn(tst_dataset, batch_size)
        optimizer = Adam(gnn.parameters(), lr=lr)
        scd = lr_scheduler.ReduceLROnPlateau(optimizer,
                                             factor=resi,
                                             min_lr=5e-5)
        selection_direction = metric_direction(
            model_selection_metric) if use_aml_metrics else "max"
        if use_aml_metrics:
            val_score = float("-inf") if selection_direction == "max" else float(
                "inf")
        else:
            val_score = 0
        best_state = copy.deepcopy(gnn.state_dict())
        best_epoch = 0
        val_metrics = {
            "minor_f1": 0.0,
            "major_f1": 0.0,
            "minor_precision": 0.0,
            "major_precision": 0.0,
            "minor_recall": 0.0,
            "major_recall": 0.0,
            "minor_rocauc": 0.0,
            "major_rocauc": 0.0,
            "minor_prauc": 0.0,
            "major_prauc": 0.0,
            "rocauc": 0.0,
            "prauc": 0.0,
        }
        tst_metrics = {
            "minor_f1": 0.0,
            "major_f1": 0.0,
            "minor_precision": 0.0,
            "major_precision": 0.0,
            "minor_recall": 0.0,
            "major_recall": 0.0,
            "minor_rocauc": 0.0,
            "major_rocauc": 0.0,
            "minor_prauc": 0.0,
            "major_prauc": 0.0,
            "rocauc": 0.0,
            "prauc": 0.0,
        }
        early_stop = 0
        trn_time = []
        repeat_start_time = time.time()

        def eval_aml(loader, threshold):
            pred, y = train.predict(gnn, loader)
            metrics_dict = metrics.aml_metrics(pred.cpu().numpy(),
                                               y.cpu().numpy(),
                                               threshold=threshold)
            return metrics_dict, float(loss_fn(pred, y).item())

        def threshold_search_on_validation(loader, thresholds, objective_metric):
            pred, y = train.predict(gnn, loader)
            pred_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()
            val_loss = float(loss_fn(pred, y).item())

            if not thresholds:
                metrics_dict = metrics.aml_metrics(pred_np,
                                                   y_np,
                                                   threshold=train_eval_threshold)
                return train_eval_threshold, metrics_dict, metric_value(
                    objective_metric, metrics_dict, val_loss)

            best_threshold = thresholds[0]
            best_metrics = metrics.aml_metrics(pred_np,
                                               y_np,
                                               threshold=best_threshold)
            best_score = metric_value(objective_metric, best_metrics, val_loss)
            direction = metric_direction(objective_metric)

            for threshold in thresholds:
                metrics_dict = metrics.aml_metrics(pred_np,
                                                   y_np,
                                                   threshold=threshold)
                score = metric_value(objective_metric, metrics_dict, val_loss)
                is_better = (score > best_score) if direction == "max" else (
                    score < best_score)
                if is_better:
                    best_score = score
                    best_threshold = threshold
                    best_metrics = metrics_dict
            return best_threshold, best_metrics, best_score

        def format_metrics(tag, m):
            return (
                f"{tag} minor_f1 {m['minor_f1']:.4f} major_f1 {m['major_f1']:.4f} "
                f"minor_precision {m['minor_precision']:.4f} major_precision {m['major_precision']:.4f} "
                f"minor_recall {m['minor_recall']:.4f} major_recall {m['major_recall']:.4f} "
                f"minor_rocauc {m['minor_rocauc']:.4f} major_rocauc {m['major_rocauc']:.4f} "
                f"minor_prauc {m['minor_prauc']:.4f} major_prauc {m['major_prauc']:.4f}")

        def timing_prefix(iter_time):
            elapsed = time.time() - repeat_start_time
            return f"iter_time {iter_time:.2f}s elapsed {elapsed:.2f}s"

        for i in range(200):
            step = repeat * 300 + i
            last_step = step
            t1 = time.time()
            loss = train.train(optimizer, gnn, trn_loader, loss_fn)
            iter_time = time.time() - t1
            trn_time.append(iter_time)
            scd.step(loss)

            # if i >= 100 / num_div:
            if i > -1:
                if use_aml_metrics:
                    train_metrics, _ = eval_aml(trn_loader, train_eval_threshold)
                    val_metrics, val_loss = eval_aml(val_loader,
                                                     train_eval_threshold)
                    score = metric_value(model_selection_metric, val_metrics,
                                         val_loss)
                    best_score_value = val_score
                    is_better = (score > best_score_value
                                 ) if selection_direction == "max" else (
                                     score < best_score_value)
                    is_close = abs(score - best_score_value) <= 1e-5

                    if is_better:
                        early_stop = 0
                        val_score = score
                        best_state = copy.deepcopy(gnn.state_dict())
                        best_epoch = i + 1
                        print(
                            f"iter {i} loss {loss:.4f} {timing_prefix(iter_time)} {format_metrics('val', val_metrics)}",
                            flush=True)
                        log_wandb_metrics(step=step,
                                          train_loss=loss,
                                          train_metrics=train_metrics,
                                          val_metrics=val_metrics,
                                          best_threshold=train_eval_threshold)
                    elif is_close:
                        print(
                            f"iter {i} loss {loss:.4f} {timing_prefix(iter_time)} {format_metrics('val', val_metrics)}",
                            flush=True)
                        log_wandb_metrics(step=step,
                                          train_loss=loss,
                                          train_metrics=train_metrics,
                                          val_metrics=val_metrics,
                                          best_threshold=train_eval_threshold)
                    else:
                        early_stop += 1
                        # if i % 5 == 0:
                        if i % 2 == 0:
                            print(
                                f"iter {i} loss {loss:.4f} {timing_prefix(iter_time)} {format_metrics('val', val_metrics)}",
                                flush=True)
                            log_wandb_metrics(step=step,
                                              train_loss=loss,
                                              train_metrics=train_metrics,
                                              val_metrics=val_metrics,
                                              best_threshold=train_eval_threshold)
                else:
                    train_score, _ = train.test(gnn,
                                                trn_loader,
                                                score_fn,
                                                loss_fn=loss_fn)
                    score, _ = train.test(gnn,
                                          val_loader,
                                          score_fn,
                                          loss_fn=loss_fn)

                    if score > val_score:
                        early_stop = 0
                        val_score = score
                        best_state = copy.deepcopy(gnn.state_dict())
                        best_epoch = i + 1
                        print(
                            f"iter {i} loss {loss:.4f} {timing_prefix(iter_time)} val {val_score:.4f}",
                            flush=True)
                        log_wandb_metrics(step=step,
                                          train_loss=loss,
                                          train_metrics={"score": train_score},
                                          val_metrics={"score": val_score})
                    elif score >= val_score - 1e-5:
                        print(
                            f"iter {i} loss {loss:.4f} {timing_prefix(iter_time)} val {val_score:.4f}",
                            flush=True)
                        log_wandb_metrics(step=step,
                                          train_loss=loss,
                                          train_metrics={"score": train_score},
                                          val_metrics={"score": val_score})
                    else:
                        early_stop += 1
                        if i % 5 == 0:
                            print(
                                f"iter {i} loss {loss:.4f} {timing_prefix(iter_time)} val {score:.4f}",
                                flush=True)
                            log_wandb_metrics(step=step,
                                              train_loss=loss,
                                              train_metrics={"score": train_score},
                                              val_metrics={"score": score})
            if ((not use_aml_metrics) or selection_direction == "max") and (
                    val_score >= 1 - 1e-5):
                early_stop += 1
            # if early_stop > 150 / num_div:
            if early_stop > patience_limit:
                print(f"Early stopping at iteration {i} with best epoch {best_epoch} and best val score {val_score:.4f}",)
                break

        # Post-training phase: threshold search on best model using configurable objective.
        gnn.load_state_dict(best_state)
        if use_aml_metrics and threshold_search:
            chosen_threshold, val_metrics, threshold_obj = threshold_search_on_validation(
                val_loader, threshold_candidates, threshold_selection_metric)
        elif use_aml_metrics:
            chosen_threshold = train_eval_threshold
            val_metrics, _ = eval_aml(val_loader, chosen_threshold)
            threshold_obj = metric_value(threshold_selection_metric,
                                         val_metrics)
        else:
            chosen_threshold = train_eval_threshold
            threshold_obj = float("nan")
            tst_score, _ = train.test(gnn,
                                      tst_loader,
                                      score_fn,
                                      loss_fn=loss_fn)

        if use_aml_metrics:
            tst_metrics, _ = eval_aml(tst_loader, chosen_threshold)
            print(
                f"end: epoch {best_epoch}, train time {sum(trn_time):.2f} s, best_threshold {chosen_threshold:.4f}, threshold_obj({threshold_selection_metric}) {threshold_obj:.4f}, {format_metrics('val', val_metrics)}, {format_metrics('tst', tst_metrics)}",
                flush=True)
            outs.append(tst_metrics["minor_f1"])
            log_wandb_metrics(step=repeat * 300 + i + 1,
                              val_metrics=val_metrics,
                              best_threshold=chosen_threshold,
                              end_epoch=best_epoch)
            final_test_metrics = tst_metrics
        else:
            print(
                f"end: epoch {best_epoch}, train time {sum(trn_time):.2f} s, val {val_score:.3f}, tst {tst_score:.3f}",
                flush=True)
            outs.append(tst_score)
            log_wandb_metrics(step=repeat * 300 + i + 1,
                              val_metrics={"score": val_score},
                              end_epoch=best_epoch)
            final_test_metrics = {"score": tst_score}
    print(
        f"average {np.average(outs):.3f} error {np.std(outs) / np.sqrt(len(outs)):.3f}"
    )
    if USE_WANDB:
        log_final_test_metrics(last_step + 1, final_test_metrics)
        log_wandb_metrics(step=last_step + 1,
                          avg_score=np.average(outs),
                          avg_error=np.std(outs) / np.sqrt(len(outs)))
        wandb.finish()


print(args)
threshold_search = bool(params.get("threshold_search", False))
threshold_candidates = params.get("thresholds", None)
if threshold_candidates is None:
    threshold_candidates = np.linspace(0.05, 0.5, 30).tolist()
pos_weight_values = params.get("pos_weight", None)
model_selection_metric = params.get("model_selection_metric",
                                    "val/minor_prauc")
threshold_selection_metric = params.get("threshold_selection_metric",
                                        "val/minor_f1")
train_eval_threshold = float(
    params.get("train_eval_threshold", params.get("threshold", 0.5)))
early_stop_patience = params.get("early_stop_patience", None)
if early_stop_patience is not None:
    early_stop_patience = float(early_stop_patience)
    if early_stop_patience <= 0:
        raise ValueError(
            f"early_stop_patience must be positive, got {early_stop_patience}."
        )
params.pop("threshold", None)
params.pop("threshold_search", None)
params.pop("thresholds", None)
params.pop("pos_weight", None)
params.pop("model_selection_metric", None)
params.pop("threshold_selection_metric", None)
params.pop("train_eval_threshold", None)
params.pop("early_stop_patience", None)
params.pop("run_name", None)
params.pop("seed", None)
params.pop("split", None)
params.pop("split_mode", None)
params.pop("train_ratio", None)
params.pop("val_ratio", None)
params.pop("test_ratio", None)
params.pop("aml_train_ratio", None)
params.pop("aml_val_ratio", None)
params.pop("use_edge_features", None)
params.pop("edge_feature_cols", None)
params.pop("edge_feature_clip", None)

print("params", params, flush=True)
split()
test(**(params))
