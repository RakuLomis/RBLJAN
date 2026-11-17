#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import json
import statistics
import time
from itertools import cycle, islice
from pathlib import Path

import torch

import comparison_config as cfg
from comparison_data import make_loader
from comparison_runtime import force_cpu, load_checkpoint_model, progress


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


@torch.no_grad()
def measure_latency(model, dataset_dir, warmup, measured):
    loader = make_loader(dataset_dir, "test", cfg.LATENCY_BATCH_SIZE)
    samples = list(loader)
    for x, _ in progress(islice(cycle(samples), warmup), total=warmup, desc="latency warmup", unit="sample"):
        model(x)
    latencies = []
    for x, _ in progress(islice(cycle(samples), measured), total=measured, desc="latency measured", unit="sample"):
        start = time.perf_counter()
        model(x)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


@torch.no_grad()
def measure_throughput(model, dataset_dir, warmup, measured, batch_size):
    loader = make_loader(dataset_dir, "test", batch_size)
    batches = list(loader)
    warmup_batches = max(1, warmup // batch_size)
    measured_batches = max(1, measured // batch_size)
    for x, _ in progress(islice(cycle(batches), warmup_batches), total=warmup_batches, desc="throughput warmup", unit="batch"):
        model(x)
    seen, start = 0, time.perf_counter()
    for x, _ in progress(islice(cycle(batches), measured_batches), total=measured_batches, desc="throughput measured", unit="batch"):
        model(x)
        seen += x.size(0)
    elapsed = time.perf_counter() - start
    return seen / elapsed if elapsed > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark RBLJAN CPU inference latency and throughput.")
    parser.add_argument("--dataset", required=True, choices=list(cfg.DATASETS))
    parser.add_argument("--data-root", type=Path, default=cfg.GENERATED_ROOT)
    parser.add_argument("--threads", type=int, default=cfg.CPU_THREADS)
    parser.add_argument("--warmup-samples", type=int, default=cfg.WARMUP_SAMPLES)
    parser.add_argument("--measured-samples", type=int, default=cfg.MEASURED_SAMPLES)
    args = parser.parse_args()

    force_cpu(args.threads)
    dataset_dir = args.data_root / args.dataset
    model, _, _ = load_checkpoint_model(args.dataset, dataset_dir, device="cpu", batch_size=1)
    latencies = measure_latency(model, dataset_dir, args.warmup_samples, args.measured_samples)
    throughput = measure_throughput(
        model, dataset_dir, args.warmup_samples, args.measured_samples, cfg.THROUGHPUT_BATCH_SIZE
    )
    result = {
        "dataset": args.dataset,
        "model": cfg.MODEL_NAME,
        "device": "cpu",
        "cpu_threads": torch.get_num_threads(),
        "batch_size_latency": cfg.LATENCY_BATCH_SIZE,
        "batch_size_throughput": cfg.THROUGHPUT_BATCH_SIZE,
        "seq_length": cfg.PKT_MAX_LEN,
        "input_length": cfg.PKT_MAX_LEN,
        "flow_max_len": cfg.FLOW_MAX_LEN,
        "warmup_samples": args.warmup_samples,
        "measured_samples": args.measured_samples,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p90_ms": percentile(latencies, 90),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_p99_ms": percentile(latencies, 99),
        "latency_mean_ms": statistics.mean(latencies) if latencies else 0.0,
        "throughput_samples_per_s": throughput,
    }
    out_dir = cfg.RESULTS_ROOT / "latency"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "{}_cpu_latency.json".format(args.dataset), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
