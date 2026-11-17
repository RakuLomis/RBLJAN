#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import comparison_config as cfg
from comparison_data import make_loader
from comparison_runtime import load_checkpoint_model, progress


@torch.no_grad()
def evaluate(model, loader, device, num_labels):
    total_loss, y_true, y_pred = 0.0, [], []
    model.eval()
    for x, y in progress(loader, desc="evaluate", unit="batch"):
        x = x.to(device)
        y = y.to(device)
        out, _ = model(x)
        loss = F.cross_entropy(out, y)
        total_loss += float(loss.item()) * x.size(0)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(out.argmax(dim=1).detach().cpu().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_labels)))
    return {
        "accuracy": accuracy_score(y_true, y_pred) if y_true else 0.0,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0,
        "loss": total_loss / max(1, len(loader.dataset)),
        "confusion_matrix": cm,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RBLJAN comparison checkpoint.")
    parser.add_argument("--dataset", required=True, choices=list(cfg.DATASETS))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-root", type=Path, default=cfg.GENERATED_ROOT)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.data_root / args.dataset
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, _, ckpt = load_checkpoint_model(args.dataset, dataset_dir, device=device, batch_size=args.batch_size)
    loader = make_loader(dataset_dir, args.split, args.batch_size)
    metrics = evaluate(model, loader, device, len(ckpt["label_map"]))

    out_dir = cfg.RESULTS_ROOT / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "{}_metrics.json".format(args.dataset)
    cm_path = out_dir / "{}_confusion_matrix.csv".format(args.dataset)
    payload = {
        "dataset": args.dataset,
        "model": cfg.MODEL_NAME,
        "split": args.split,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "loss": metrics["loss"],
        "checkpoint": str(cfg.RESULTS_ROOT / "checkpoints" / args.dataset / "best.pt"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(metrics["confusion_matrix"].tolist())
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
