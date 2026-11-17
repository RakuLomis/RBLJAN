#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import json
from pathlib import Path

import torch

import comparison_config as cfg
from comparison_runtime import load_checkpoint_model


def linear_flops(in_features, out_features, rows=1, bias=True):
    flops = rows * out_features * (2 * in_features)
    if bias:
        flops += rows * out_features
    return flops


def cosine_flops(rows, seq_len, num_labels, embedding_dim):
    dot = rows * num_labels * seq_len * (2 * embedding_dim)
    byte_norm = rows * seq_len * (2 * embedding_dim)
    label_norm = num_labels * (2 * embedding_dim)
    norm_outer = rows * num_labels * seq_len
    divide = rows * num_labels * seq_len
    return dot + byte_norm + label_norm + norm_outer + divide


def att_byte_flops(seq_len, hidden_size, ngram, out_channels, embedding_dim):
    out_len = seq_len - ngram + 1
    conv = out_channels * out_len * (2 * ngram * hidden_size)
    batch_norm_relu = out_channels * out_len * 4
    max_pool = out_channels * out_len
    softmax = out_len * 5
    avg_pool = out_len * embedding_dim * ngram
    weighted_sum = out_len * (2 * embedding_dim)
    return conv + batch_norm_relu + max_pool + softmax + avg_pool + weighted_sum


def att_label_flow_flops(seq_len, num_labels, flow_len, embedding_dim):
    hidden_size = num_labels * flow_len
    fc1 = linear_flops(seq_len, 1, rows=hidden_size, bias=True)
    fc2 = linear_flops(hidden_size, num_labels, rows=1, bias=True)
    softmax = num_labels * 5
    label_weighted_sum = num_labels * (2 * embedding_dim)
    return fc1 + fc2 + softmax + label_weighted_sum


def manual_rbljan_flow_flops(utils):
    """Dense inference FLOPs for RBLJAN_Classifier_FLOW, batch_size=1.

    Conventions:
    - multiply-add counts as 2 FLOPs;
    - embedding lookups are treated as memory lookups, not arithmetic FLOPs;
    - dropout is inactive in eval mode;
    - the training-only out2 regularizer branch is excluded.
    """
    k = utils.NUM_LABELS
    f = cfg.FLOW_MAX_LEN
    e = utils.EMBEDDING_DIM
    header_len = utils.HEADER_LEN
    payload_len = utils.PKT_MAX_LEN - utils.HEADER_LEN
    ngram_h = utils.NGRAM_HEADER
    ngram_p = utils.NGRAM_PAYLOAD
    kernel_h = utils.KERNEL_NUM_HEADER
    kernel_p = utils.KERNEL_NUM_PAYLOAD
    hidden_size = k * f

    header = {
        "position_add": f * header_len * e,
        "cosine_similarity": cosine_flops(f, header_len, k, e),
        "att_byte": att_byte_flops(header_len, hidden_size, ngram_h, kernel_h, e),
        "flow_sum": f * header_len * e,
        "att_label": att_label_flow_flops(header_len, k, f, e),
    }
    payload = {
        "position_add": f * payload_len * e,
        "cosine_similarity": cosine_flops(f, payload_len, k, e),
        "att_byte": att_byte_flops(payload_len, hidden_size, ngram_p, kernel_p, e),
        "flow_sum": f * payload_len * e,
        "att_label": att_label_flow_flops(payload_len, k, f, e),
    }
    classifier = {
        "fc": linear_flops(4 * e, k, rows=1, bias=True),
    }
    breakdown = {
        "header": header,
        "payload": payload,
        "classifier": classifier,
    }
    total = sum(sum(group.values()) for group in breakdown.values())
    return int(total), breakdown


def try_tool_flops(model):
    dummy = torch.zeros((1, cfg.FLOW_MAX_LEN, cfg.PKT_MAX_LEN), dtype=torch.long)
    try:
        from thop import profile

        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(flops), "thop"
    except Exception as exc:
        return None, "not_available: {}".format(exc)


def main():
    parser = argparse.ArgumentParser(description="Profile RBLJAN model size and dense-equivalent FLOPs.")
    parser.add_argument("--dataset", required=True, choices=list(cfg.DATASETS))
    parser.add_argument("--data-root", type=Path, default=cfg.GENERATED_ROOT)
    args = parser.parse_args()

    model, utils, _ = load_checkpoint_model(args.dataset, args.data_root / args.dataset, device="cpu", batch_size=1)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dense_flops, flops_breakdown = manual_rbljan_flow_flops(utils)
    tool_flops, tool_method = try_tool_flops(model)
    profile = {
        "dataset": args.dataset,
        "model": cfg.MODEL_NAME,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "active_params_estimated": int(total_params),
        "active_param_ratio": 1.0,
        "dense_equivalent_flops": dense_flops,
        "effective_flops": dense_flops,
        "flops_method": "manual_formula_rbljan_flow_inference_bsz1_multiply_add_as_2_flops_embedding_lookup_excluded",
        "flops_scope": "single_sample_eval_forward",
        "flops_breakdown": flops_breakdown,
        "dense_equivalent_flops_tool": tool_flops,
        "flops_tool_method": tool_method,
    }
    out_dir = cfg.RESULTS_ROOT / "profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "{}_profile.json".format(args.dataset), "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
