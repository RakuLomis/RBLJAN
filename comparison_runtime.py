#!/usr/bin/env python
# -*- coding:utf-8 -*-

import importlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

import comparison_config as cfg


def progress(iterable, **kwargs):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


def set_global_seed(seed=cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def force_cpu(num_threads=cfg.CPU_THREADS):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(int(num_threads))
    try:
        torch.set_num_interop_threads(max(1, int(num_threads)))
    except RuntimeError:
        pass


def load_label_map(dataset_dir):
    with open(Path(dataset_dir) / "label_map.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): int(v) for k, v in raw.items()}


def configure_repo_globals(label_map, *, force_cpu_mode=False, batch_size=cfg.BATCH_SIZE):
    if force_cpu_mode:
        force_cpu()
    import utils

    labels = [k for k, _ in sorted(label_map.items(), key=lambda item: item[1])]
    utils.CLASSES = labels
    utils.LABELS = {label: idx for idx, label in enumerate(labels)}
    utils.NUM_LABELS = len(labels)
    utils.DATA_MAP[len(labels)] = "comparison"
    utils.PKT_MAX_LEN = cfg.PKT_MAX_LEN
    utils.HEADER_LEN = cfg.HEADER_LEN
    utils.PAYLOAD_LEN = cfg.PKT_MAX_LEN - cfg.HEADER_LEN
    utils.FLOW_MAX_LEN = cfg.FLOW_MAX_LEN
    utils.FLOW_MIN_LEN = cfg.FLOW_MIN_LEN
    utils.PADDING_IDX = cfg.PADDING_IDX
    utils.BATCH_SIZE = batch_size
    utils.LR = cfg.LR
    utils.EPOCHS = cfg.EPOCHS
    utils.CUDA = False if force_cpu_mode else torch.cuda.is_available()
    utils.DEVICE = 0
    utils.DEVICE_COUNT = max(1, torch.cuda.device_count())
    utils.ALL_DEVICE = ["cuda:{}".format(i) for i in range(torch.cuda.device_count())]

    import model
    importlib.reload(model)
    return utils, model


def build_flow_model(label_map, *, device="cpu", batch_size=cfg.BATCH_SIZE):
    utils, model_mod = configure_repo_globals(
        label_map, force_cpu_mode=(str(device) == "cpu"), batch_size=batch_size
    )
    matrix_path = Path("tmp_variable") / ("p_e_matrix_{}.variable".format(utils.EMBEDDING_DIM))
    if matrix_path.exists():
        import pickle

        with open(matrix_path, "rb") as f:
            p_e_matrix = pickle.load(f)[: utils.PKT_MAX_LEN + utils.NGRAM_PAYLOAD]
    else:
        p_e_matrix = utils.get_position_encoding_matrix(utils.EMBEDDING_DIM)
    overall_label_idx = torch.arange(0, utils.NUM_LABELS).long()
    net = model_mod.RBLJAN_Classifier_FLOW(p_e_matrix, overall_label_idx)
    return net.to(device), utils


def checkpoint_path(dataset):
    return cfg.RESULTS_ROOT / "checkpoints" / dataset / "best.pt"


def load_checkpoint_model(dataset, dataset_dir, *, device="cpu", batch_size=cfg.BATCH_SIZE):
    label_map = load_label_map(dataset_dir)
    net, utils = build_flow_model(label_map, device=device, batch_size=batch_size)
    ckpt = torch.load(checkpoint_path(dataset), map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    net.load_state_dict(state)
    net.eval()
    return net, utils, ckpt
