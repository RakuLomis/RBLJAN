#!/usr/bin/env python
# -*- coding:utf-8 -*-

from pathlib import Path

MODEL_NAME = "RBLJAN"

RAW_ROOT = Path(r"E:\Coding\TrafficData\datasets_raw_add2")
GENERATED_ROOT = Path(r"E:\Coding\TrafficData\generated_for_comparison") / MODEL_NAME
RESULTS_ROOT = Path("results")

DATASETS = ("cstnet_tls_1.3", "CipherSpectrum")

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42

PYTHON_CMD = ["conda", "run", "-n", "Pytorch_env", "python"]

PKT_MAX_LEN = 1500
HEADER_LEN = 50
FLOW_MAX_LEN = 10
FLOW_MIN_LEN = 2
PADDING_IDX = 256

BATCH_SIZE = 32
EPOCHS = 120
LR = 0.001
EARLY_STOP_PATIENCE = 8

CPU_THREADS = 1
LATENCY_BATCH_SIZE = 1
THROUGHPUT_BATCH_SIZE = 32
WARMUP_SAMPLES = 100
MEASURED_SAMPLES = 1000

PREPROCESS_POLICY = (
    "no_eth_ip_packet_only_mask_ipv4_ipv6_addresses_mask_tcp_udp_ports_"
    "mask_tls_sni_zero_checksums_ids_seq_ack_flow_safe_split_seed42"
)

EXCLUDE_PORTS = {5353, 137, 138, 5355, 17500, 1900}
