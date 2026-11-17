#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import csv
import json
from pathlib import Path

import comparison_config as cfg


def read_json(path, default=None):
    if not Path(path).exists():
        return default if default is not None else {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_row(dataset, data_root):
    metrics = read_json(cfg.RESULTS_ROOT / "metrics" / "{}_metrics.json".format(dataset))
    profile = read_json(cfg.RESULTS_ROOT / "profiles" / "{}_profile.json".format(dataset))
    latency = read_json(cfg.RESULTS_ROOT / "latency" / "{}_cpu_latency.json".format(dataset))
    preprocess = read_json(Path(data_root) / dataset / "preprocess_report.json")
    return {
        "dataset": dataset,
        "model": cfg.MODEL_NAME,
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "loss": metrics.get("loss"),
        "total_params": profile.get("total_params"),
        "trainable_params": profile.get("trainable_params"),
        "active_params_estimated": profile.get("active_params_estimated"),
        "active_param_ratio": profile.get("active_param_ratio"),
        "dense_equivalent_flops": profile.get("dense_equivalent_flops"),
        "effective_flops": profile.get("effective_flops"),
        "cpu_latency_p50_ms": latency.get("latency_p50_ms"),
        "cpu_latency_p90_ms": latency.get("latency_p90_ms"),
        "cpu_latency_p95_ms": latency.get("latency_p95_ms"),
        "cpu_latency_p99_ms": latency.get("latency_p99_ms"),
        "cpu_latency_mean_ms": latency.get("latency_mean_ms"),
        "cpu_throughput_samples_per_s": latency.get("throughput_samples_per_s"),
        "preprocess_policy": preprocess.get("preprocess_policy"),
        "flow_leakage_count": preprocess.get("flow_leakage_count"),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize RBLJAN comparison results.")
    parser.add_argument("--data-root", type=Path, default=cfg.GENERATED_ROOT)
    parser.add_argument("--datasets", nargs="+", default=list(cfg.DATASETS), choices=list(cfg.DATASETS))
    args = parser.parse_args()

    cfg.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [build_row(dataset, args.data_root) for dataset in args.datasets]
    csv_path = cfg.RESULTS_ROOT / "comparison_summary.csv"
    json_path = cfg.RESULTS_ROOT / "comparison_summary.json"
    fields = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
