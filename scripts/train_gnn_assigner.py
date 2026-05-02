"""Behaviour-cloning trainer for the GNN frontier assigner.

Reads JSONL traces produced by :mod:`utils.gpt_trace`, trains the
:class:`agents.gnn_assigner.CrossAttentionAssigner` to predict the GPT
label for each robot, and saves a checkpoint loadable by ``GNNAssigner``.

Usage:
    python scripts/train_gnn_assigner.py \\
        --trace-dir data/gnn_traces \\
        --out outputs/gnn/assigner.pt \\
        --epochs 30 --batch-size 32
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Allow running as a script from project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from agents.gnn_assigner import CrossAttentionAssigner  # noqa: E402
from utils.graph_builder import features_from_jsonable  # noqa: E402


class TraceDataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        feats = features_from_jsonable(rec["features"])
        labels = torch.tensor(rec["labels"], dtype=torch.long)
        return {**feats, "labels": labels}


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad to max R and F in the batch with -inf masks for padded frontiers."""
    max_R = max(item["robot_feat"].shape[0] for item in batch)
    max_F = max(item["frontier_feat"].shape[0] for item in batch)
    Dr = batch[0]["robot_feat"].shape[1]
    Df = batch[0]["frontier_feat"].shape[1]
    De = batch[0]["edge_feat"].shape[2]
    B = len(batch)

    rf = torch.zeros(B, max_R, Dr)
    ff = torch.zeros(B, max_F, Df)
    ef = torch.zeros(B, max_R, max_F, De)
    labels = torch.full((B, max_R), -100, dtype=torch.long)
    frontier_mask = torch.zeros(B, max_F, dtype=torch.bool)

    for i, item in enumerate(batch):
        R, F_ = item["robot_feat"].shape[0], item["frontier_feat"].shape[0]
        rf[i, :R] = item["robot_feat"]
        ff[i, :F_] = item["frontier_feat"]
        ef[i, :R, :F_] = item["edge_feat"]
        labels[i, :R] = item["labels"]
        frontier_mask[i, :F_] = True
    return {
        "robot_feat": rf,
        "frontier_feat": ff,
        "edge_feat": ef,
        "labels": labels,
        "frontier_mask": frontier_mask,
    }


def load_traces(trace_dir: str) -> List[Dict]:
    paths = sorted(glob.glob(os.path.join(trace_dir, "*.jsonl")))
    records: List[Dict] = []
    for p in paths:
        with open(p, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def split(records: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n_val = max(1, int(len(records) * val_ratio)) if records else 0
    val_idx = set(indices[:n_val])
    train = [records[i] for i in range(len(records)) if i not in val_idx]
    val = [records[i] for i in range(len(records)) if i in val_idx]
    return train, val


def run_epoch(
    model: CrossAttentionAssigner,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float]:
    is_train = optimiser is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch in loader:
        rf = batch["robot_feat"].to(device)
        ff = batch["frontier_feat"].to(device)
        ef = batch["edge_feat"].to(device)
        labels = batch["labels"].to(device)
        f_mask = batch["frontier_mask"].to(device)  # (B, F)

        logits = model(rf, ff, ef)  # (B, R, F)
        # Mask out padded frontiers
        mask = f_mask.unsqueeze(1).expand_as(logits)
        logits = logits.masked_fill(~mask, float("-inf"))

        B, R, F_ = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * R, F_),
            labels.reshape(B * R),
            ignore_index=-100,
        )
        if is_train:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            valid = labels != -100
            total_correct += int(((preds == labels) & valid).sum())
            total_count += int(valid.sum())
            total_loss += float(loss) * int(valid.sum())

    avg_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)
    return avg_loss, acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--out", default="outputs/gnn/assigner.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    records = load_traces(args.trace_dir)
    if not records:
        raise SystemExit(f"no traces found under {args.trace_dir}")
    print(f"loaded {len(records)} trace records")

    train_recs, val_recs = split(records, args.val_ratio, args.seed)
    train_loader = DataLoader(
        TraceDataset(train_recs),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        TraceDataset(val_recs),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    ) if val_recs else None

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = {"hidden": args.hidden, "num_layers": args.num_layers, "num_heads": args.num_heads}
    model = CrossAttentionAssigner(**cfg).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimiser, device)
        if val_loader is not None:
            val_loss, val_acc = run_epoch(model, val_loader, None, device)
            print(
                f"epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} "
                f"| val loss {val_loss:.4f} acc {val_acc:.3f}"
            )
            improved = val_loss < best_val
            best_val = min(best_val, val_loss)
        else:
            print(f"epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.3f}")
            improved = True

        if improved:
            torch.save({"model": model.state_dict(), "config": cfg}, args.out)
            print(f"  -> saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
