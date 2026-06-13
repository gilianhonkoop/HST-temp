import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.utils import negative_sampling, to_undirected


class LinkEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, conv_type, dropout):
        super().__init__()
        self.conv_type = conv_type
        self.dropout = dropout

        if conv_type == "gcn":
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)
        elif conv_type == "gin":
            self.conv1 = GINConv(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
            self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate AML global node embeddings for SubGNN.")
    parser.add_argument("--data_dir", required=True, help="Directory containing background_nodes.csv/background_edges.csv.")
    parser.add_argument("--out_path", default=None, help="Where to save the embedding tensor.")
    parser.add_argument("--conv", choices=["gin", "gcn"], default="gin")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--neg_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def edge_dot(z, edge_index):
    src, dst = edge_index
    return (z[src] * z[dst]).sum(dim=-1)


def split_edges(edge_index, val_ratio, seed):
    num_edges = edge_index.shape[1]
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_edges, generator=generator)
    val_count = int(num_edges * val_ratio)
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    return edge_index[:, train_idx], edge_index[:, val_idx]


def score_edges(z, pos_edge_index, full_edge_index, num_nodes, neg_ratio):
    num_neg = max(1, int(pos_edge_index.shape[1] * neg_ratio))
    neg_edge_index = negative_sampling(
        edge_index=full_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg,
    )
    logits = torch.cat([edge_dot(z, pos_edge_index), edge_dot(z, neg_edge_index)], dim=0)
    labels = torch.cat([
        torch.ones(pos_edge_index.shape[1], device=z.device),
        torch.zeros(neg_edge_index.shape[1], device=z.device),
    ])
    loss = F.binary_cross_entropy_with_logits(logits, labels)

    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    auroc = roc_auc_score(labels_np, probs)
    auprc = average_precision_score(labels_np, probs)
    return loss, auroc, auprc


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = Path(args.data_dir)
    out_path = Path(args.out_path) if args.out_path else data_dir / f"{args.conv}_embeddings.pth"

    nodes_df = pd.read_csv(data_dir / "background_nodes.csv")
    edges_df = pd.read_csv(data_dir / "background_edges.csv")
    nodes_df["clId"] = nodes_df["clId"].astype(str)
    edges_df["clId1"] = edges_df["clId1"].astype(str)
    edges_df["clId2"] = edges_df["clId2"].astype(str)

    node_mapping = {cl_id: idx for idx, cl_id in enumerate(nodes_df["clId"])}
    feat_cols = [c for c in nodes_df.columns if c.startswith("feat")]
    if len(feat_cols) == 0:
        raise ValueError("background_nodes.csv has no feat* columns.")

    valid_edges = edges_df[edges_df["clId1"].isin(node_mapping) & edges_df["clId2"].isin(node_mapping)]
    src = valid_edges["clId1"].map(node_mapping).to_numpy(dtype=np.int64)
    dst = valid_edges["clId2"].map(node_mapping).to_numpy(dtype=np.int64)

    x = torch.tensor(nodes_df[feat_cols].values, dtype=torch.float)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=x.shape[0])
    train_pos_edge_index, val_pos_edge_index = split_edges(edge_index, args.val_ratio, args.seed)

    device = torch.device(args.device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    train_pos_edge_index = train_pos_edge_index.to(device)
    val_pos_edge_index = val_pos_edge_index.to(device)

    model = LinkEncoder(
        in_dim=x.shape[1],
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        conv_type=args.conv,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_auprc = -1.0
    best_embeddings = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model(x, edge_index)
        train_loss, train_auroc, train_auprc = score_edges(
            z,
            train_pos_edge_index,
            edge_index,
            num_nodes=x.shape[0],
            neg_ratio=args.neg_ratio,
        )
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z = model(x, edge_index)
            val_loss, val_auroc, val_auprc = score_edges(
                z,
                val_pos_edge_index,
                edge_index,
                num_nodes=x.shape[0],
                neg_ratio=args.neg_ratio,
            )

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_embeddings = z.detach().cpu()

        print(
            f"epoch={epoch:03d} "
            f"train_loss={float(train_loss.item()):.5f} "
            f"train_auroc={train_auroc:.5f} "
            f"train_auprc={train_auprc:.5f} "
            f"val_loss={float(val_loss.item()):.5f} "
            f"val_auroc={val_auroc:.5f} "
            f"val_auprc={val_auprc:.5f}",
            flush=True,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_embeddings, out_path)
    print(f"Saved embeddings: {out_path} shape={tuple(best_embeddings.shape)}", flush=True)


if __name__ == "__main__":
    main()
