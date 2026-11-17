#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import binascii
import ipaddress
import json
import pickle
import random
import socket
import time
from pathlib import Path

import dpkt

import comparison_config as cfg
from comparison_runtime import progress


def _mask_sni_in_client_hello(buf, start, end):
    changed = False
    p = start
    if p + 34 > end:
        return changed
    p += 34
    if p + 1 > end:
        return changed
    sid_len = buf[p]
    p += 1 + sid_len
    if p + 2 > end:
        return changed
    cs_len = (buf[p] << 8) | buf[p + 1]
    p += 2 + cs_len
    if p + 1 > end:
        return changed
    comp_len = buf[p]
    p += 1 + comp_len
    if p + 2 > end:
        return changed
    ext_total_len = (buf[p] << 8) | buf[p + 1]
    p += 2
    ext_end = min(p + ext_total_len, end)
    while p + 4 <= ext_end:
        ext_type = (buf[p] << 8) | buf[p + 1]
        ext_len = (buf[p + 2] << 8) | buf[p + 3]
        p += 4
        e_end = p + ext_len
        if e_end > ext_end:
            break
        if ext_type == 0 and ext_len >= 2:
            list_len = (buf[p] << 8) | buf[p + 1]
            q = p + 2
            list_end = min(q + list_len, e_end)
            while q + 3 <= list_end:
                name_type = buf[q]
                name_len = (buf[q + 1] << 8) | buf[q + 2]
                q += 3
                if q + name_len > list_end:
                    break
                if name_type == 0:
                    buf[q : q + name_len] = b"\x00" * name_len
                    changed = True
                q += name_len
        p = e_end
    return changed


def mask_tls_sni_in_payload(payload):
    buf = bytearray(payload)
    changed = False
    i = 0
    while i + 5 <= len(buf):
        content_type = buf[i]
        tls_major = buf[i + 1]
        rec_len = (buf[i + 3] << 8) | buf[i + 4]
        rec_end = i + 5 + rec_len
        if rec_end > len(buf):
            break
        if content_type == 22 and tls_major == 3:
            j = i + 5
            while j + 4 <= rec_end:
                hs_type = buf[j]
                hs_len = (buf[j + 1] << 16) | (buf[j + 2] << 8) | buf[j + 3]
                hs_end = j + 4 + hs_len
                if hs_end > rec_end:
                    break
                if hs_type == 1 and _mask_sni_in_client_hello(buf, j + 4, hs_end):
                    changed = True
                j = hs_end
        i = rec_end
    return bytes(buf), changed


def ip_to_text(raw):
    try:
        return str(ipaddress.ip_address(raw))
    except ValueError:
        return binascii.hexlify(raw).decode("ascii")


def canonical_flow_key(src, dst, sport, dport, proto):
    a = "{}:{}".format(ip_to_text(src), sport)
    b = "{}:{}".format(ip_to_text(dst), dport)
    lo, hi = sorted([a, b])
    return "{}|{}|{}".format(lo, hi, proto)


def iter_packets(path):
    with open(path, "rb") as f:
        try:
            reader = dpkt.pcap.Reader(f)
        except (ValueError, dpkt.dpkt.NeedData):
            f.seek(0)
            reader = dpkt.pcapng.Reader(f)
        datalink = reader.datalink()
        for _, buf in reader:
            yield datalink, buf


def extract_ip_packet(datalink, buf):
    if datalink == dpkt.pcap.DLT_EN10MB:
        eth = dpkt.ethernet.Ethernet(buf)
        return eth.data if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)) else None
    dlt_raw = getattr(dpkt.pcap, "DLT_RAW", 12)
    if datalink in (dlt_raw, 101, 228):
        version = buf[0] >> 4 if buf else 0
        if version == 4:
            return dpkt.ip.IP(buf)
        if version == 6:
            return dpkt.ip6.IP6(buf)
    return None


def transport_info(ip_pkt):
    if isinstance(ip_pkt, dpkt.ip.IP):
        proto = ip_pkt.p
        src, dst = ip_pkt.src, ip_pkt.dst
    elif isinstance(ip_pkt, dpkt.ip6.IP6):
        proto = ip_pkt.nxt
        src, dst = ip_pkt.src, ip_pkt.dst
    else:
        return None
    if proto not in (socket.IPPROTO_TCP, socket.IPPROTO_UDP):
        return None
    trans = ip_pkt.data
    if not hasattr(trans, "sport") or not hasattr(trans, "dport"):
        return None
    return src, dst, int(trans.sport), int(trans.dport), int(proto), trans


def sanitized_packet_bytes(ip_pkt):
    info = transport_info(ip_pkt)
    if info is None:
        return None
    src, dst, sport, dport, proto, trans = info
    if sport in cfg.EXCLUDE_PORTS or dport in cfg.EXCLUDE_PORTS:
        return None
    if isinstance(trans, dpkt.tcp.TCP) and (trans.flags & (dpkt.tcp.TH_SYN | dpkt.tcp.TH_FIN)):
        return None
    payload = bytes(trans.data) if hasattr(trans, "data") else b""
    if len(ip_pkt) <= 50 or len(payload) <= 0:
        return None

    if isinstance(ip_pkt, dpkt.ip.IP):
        ip_pkt.src = b"\x00" * 4
        ip_pkt.dst = b"\x00" * 4
        ip_pkt.sum = 0
        ip_pkt.id = 0
        ip_pkt.off = 0
    elif isinstance(ip_pkt, dpkt.ip6.IP6):
        ip_pkt.src = b"\x00" * 16
        ip_pkt.dst = b"\x00" * 16
    trans.sport = 0
    trans.dport = 0
    if hasattr(trans, "sum"):
        trans.sum = 0
    if isinstance(trans, dpkt.tcp.TCP):
        trans.seq = 0
        trans.ack = 0
        masked_payload, _ = mask_tls_sni_in_payload(bytes(trans.data))
        trans.data = masked_payload
    return bytes(ip_pkt)


def packet_to_indices(pkt_bytes):
    values = list(pkt_bytes[: cfg.PKT_MAX_LEN])
    if len(values) < cfg.HEADER_LEN:
        return None
    return values


def collect_dataset(raw_dataset_dir, dataset_name):
    records_by_flow = {}
    skipped = []
    labels = sorted([p.name for p in raw_dataset_dir.iterdir() if p.is_dir()])
    label_map = {label: idx for idx, label in enumerate(labels)}

    for label in progress(labels, desc="labels:{}".format(dataset_name), unit="label"):
        label_dir = raw_dataset_dir / label
        pcap_files = [
            p for p in label_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".pcap", ".pcapng", ".cap"}
        ]
        for pcap_file in progress(pcap_files, desc="pcap:{}".format(label), unit="file", leave=False):
            try:
                for datalink, buf in iter_packets(pcap_file):
                    try:
                        ip_pkt = extract_ip_packet(datalink, buf)
                        if ip_pkt is None:
                            continue
                        info = transport_info(ip_pkt)
                        if info is None:
                            continue
                        src, dst, sport, dport, proto, _ = info
                        flow_key = canonical_flow_key(src, dst, sport, dport, proto)
                        pkt_bytes = sanitized_packet_bytes(ip_pkt)
                        if pkt_bytes is None:
                            continue
                        pkt_idx = packet_to_indices(pkt_bytes)
                        if pkt_idx is None:
                            continue
                        rec = records_by_flow.setdefault(
                            (label, flow_key),
                            {
                                "x": [],
                                "y": label_map[label],
                                "label": label,
                                "flow_key": flow_key,
                                "source_file": str(pcap_file),
                            },
                        )
                        if len(rec["x"]) < cfg.FLOW_MAX_LEN:
                            rec["x"].append(pkt_idx)
                    except Exception as exc:
                        skipped.append("{}\tpacket_error\t{}".format(pcap_file, exc))
            except Exception as exc:
                skipped.append("{}\tfile_error\t{}".format(pcap_file, exc))

    records = [rec for rec in records_by_flow.values() if len(rec["x"]) >= cfg.FLOW_MIN_LEN]
    return records, label_map, skipped


def assign_splits(records):
    flow_to_split = {}
    by_label = {}
    for rec in records:
        by_label.setdefault(rec["label"], set()).add(rec["flow_key"])

    rng = random.Random(cfg.SEED)
    for label in sorted(by_label):
        keys = sorted(by_label[label])
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(n * cfg.SPLIT_RATIOS["train"])
        n_val = int(n * cfg.SPLIT_RATIOS["val"])
        for idx, key in enumerate(keys):
            split = "train" if idx < n_train else "val" if idx < n_train + n_val else "test"
            flow_to_split.setdefault(key, split)

    splits = {"train": [], "val": [], "test": []}
    for rec in records:
        splits[flow_to_split[rec["flow_key"]]].append(rec)
    return splits


def leakage_count(splits):
    split_keys = {split: {rec["flow_key"] for rec in rows} for split, rows in splits.items()}
    leaks = 0
    names = list(split_keys)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            leaks += len(split_keys[a].intersection(split_keys[b]))
    return leaks


def write_dataset(dataset, records, label_map, skipped, output_root):
    dataset_dir = output_root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    splits = assign_splits(records)
    for split, rows in splits.items():
        with open(dataset_dir / "{}.pkl".format(split), "wb") as f:
            pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(dataset_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    with open(dataset_dir / "split_manifest.tsv", "w", encoding="utf-8") as f:
        f.write("split\tlabel\tsource_file\tflow_key\n")
        for split, rows in splits.items():
            for rec in rows:
                f.write("{}\t{}\t{}\t{}\n".format(split, rec["label"], rec["source_file"], rec["flow_key"]))
    with open(dataset_dir / "skipped_files.log", "w", encoding="utf-8") as f:
        f.write("\n".join(skipped))
    report = {
        "dataset": dataset,
        "num_labels": len(label_map),
        "records_total": sum(len(v) for v in splits.values()),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "flow_leakage_count": leakage_count(splits),
        "preprocess_policy": cfg.PREPROCESS_POLICY,
        "seed": cfg.SEED,
        "flow_min_len": cfg.FLOW_MIN_LEN,
        "flow_max_len": cfg.FLOW_MAX_LEN,
        "pkt_max_len": cfg.PKT_MAX_LEN,
    }
    with open(dataset_dir / "preprocess_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description="Build flow-safe no-leakage RBLJAN comparison datasets from PCAP.")
    parser.add_argument("--raw-root", type=Path, default=cfg.RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=cfg.GENERATED_ROOT)
    parser.add_argument("--datasets", nargs="+", default=list(cfg.DATASETS), choices=list(cfg.DATASETS))
    parser.add_argument("--dry-run", action="store_true", help="Only inspect dataset folders and PCAP counts.")
    args = parser.parse_args()

    started = time.time()
    for dataset in progress(args.datasets, desc="datasets", unit="dataset"):
        raw_dataset_dir = args.raw_root / dataset
        if not raw_dataset_dir.exists():
            raise FileNotFoundError(raw_dataset_dir)
        if args.dry_run:
            labels = sorted([p for p in raw_dataset_dir.iterdir() if p.is_dir()])
            pcap_count = sum(
                1
                for label_dir in labels
                for p in label_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in {".pcap", ".pcapng", ".cap"}
            )
            print(json.dumps({
                "dataset": dataset,
                "label_count": len(labels),
                "pcap_file_count": pcap_count,
                "output_dir": str(args.output_root / dataset),
                "preprocess_policy": cfg.PREPROCESS_POLICY,
            }, indent=2))
            continue
        records, label_map, skipped = collect_dataset(raw_dataset_dir, dataset)
        report = write_dataset(dataset, records, label_map, skipped, args.output_root)
        print(json.dumps(report, indent=2))
    print("comparison_preprocess.py finished with {:.2f}s".format(time.time() - started))


if __name__ == "__main__":
    main()
