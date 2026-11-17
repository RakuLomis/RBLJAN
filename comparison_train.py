#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

import comparison_config as cfg
from comparison_data import make_loader
from comparison_runtime import build_flow_model, checkpoint_path, load_label_map, progress, set_global_seed


def run_epoch(model, loader, optimizer, device, utils):
    model.train()
    total_loss, y_true, y_pred = 0.0, [], []
    loss_func = utils.FocalLoss(alpha=None)
    for x, y in progress(loader, desc="train", unit="batch", leave=False):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out1, out2 = model(x)
        num_groups = max(1, int(out2.size(0) / utils.NUM_LABELS))
        label_target = torch.tensor(list(utils.LABELS.values()), device=device).long().repeat(num_groups)
        loss = loss_func(out1, y) + utils.LAMBDA * F.cross_entropy(out2, label_target)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(out1.argmax(dim=1).detach().cpu().tolist())
    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "accuracy": accuracy_score(y_true, y_pred) if y_true else 0.0,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, y_true, y_pred = 0.0, [], []
    for x, y in progress(loader, desc="eval", unit="batch", leave=False):
        x = x.to(device)
        y = y.to(device)
        out, _ = model(x)
        loss = F.cross_entropy(out, y)
        total_loss += float(loss.item()) * x.size(0)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(out.argmax(dim=1).detach().cpu().tolist())
    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "accuracy": accuracy_score(y_true, y_pred) if y_true else 0.0,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Train RBLJAN-Flow on comparison splits.")
    parser.add_argument("--dataset", required=True, choices=list(cfg.DATASETS))
    parser.add_argument("--data-root", type=Path, default=cfg.GENERATED_ROOT)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LR)
    parser.add_argument("--cpu", action="store_true", help="Train on CPU even when CUDA is available.")
    args = parser.parse_args()

    set_global_seed(cfg.SEED)
    dataset_dir = args.data_root / args.dataset
    label_map = load_label_map(dataset_dir)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, utils = build_flow_model(label_map, device=device, batch_size=args.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = make_loader(dataset_dir, "train", args.batch_size, shuffle=True)
    val_loader = make_loader(dataset_dir, "val", args.batch_size, shuffle=False)

    out_dir = checkpoint_path(args.dataset).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch, stale = -1.0, 0, 0
    history = []
    started = time.time()
    for epoch in progress(range(1, args.epochs + 1), desc="epochs", unit="epoch"):
        train_metrics = run_epoch(model, train_loader, optimizer, device, utils)
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch, stale = val_metrics["macro_f1"], epoch, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_macro_f1": best_f1,
                    "label_map": label_map,
                    "dataset": args.dataset,
                    "model": cfg.MODEL_NAME,
                },
                checkpoint_path(args.dataset),
            )
        else:
            stale += 1
        if stale >= cfg.EARLY_STOP_PATIENCE:
            break

    report = {
        "dataset": args.dataset,
        "model": cfg.MODEL_NAME,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "elapsed_seconds": time.time() - started,
        "history": history,
        "checkpoint": str(checkpoint_path(args.dataset)),
    }
    with open(out_dir / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
