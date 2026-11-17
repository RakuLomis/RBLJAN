#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import comparison_config as cfg


class FlowComparisonDataset(Dataset):
    def __init__(self, dataset_dir, split, max_flow_len=cfg.FLOW_MAX_LEN, max_pkt_len=cfg.PKT_MAX_LEN):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.max_flow_len = max_flow_len
        self.max_pkt_len = max_pkt_len
        with open(self.dataset_dir / "{}.pkl".format(split), "rb") as f:
            self.records = pickle.load(f)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        flow = record["x"][: self.max_flow_len]
        padded = []
        for pkt in flow:
            pkt = list(pkt[: self.max_pkt_len])
            if len(pkt) < self.max_pkt_len:
                pkt = pkt + [cfg.PADDING_IDX] * (self.max_pkt_len - len(pkt))
            padded.append(pkt)
        while len(padded) < self.max_flow_len:
            padded.append([cfg.PADDING_IDX] * self.max_pkt_len)
        return torch.tensor(padded, dtype=torch.long), torch.tensor(record["y"], dtype=torch.long)


def make_loader(dataset_dir, split, batch_size, shuffle=False, num_workers=0):
    ds = FlowComparisonDataset(dataset_dir, split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def class_counts(dataset_dir, split):
    ds = FlowComparisonDataset(dataset_dir, split)
    counts = {}
    for item in ds.records:
        counts[item["y"]] = counts.get(item["y"], 0) + 1
    return np.array([counts.get(i, 0) for i in range(max(counts) + 1)], dtype=np.float32)
