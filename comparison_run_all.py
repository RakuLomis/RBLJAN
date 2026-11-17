#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import shlex
import subprocess
import sys

import comparison_config as cfg
from comparison_runtime import progress


def run(cmd):
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run the full RBLJAN comparison pipeline.")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=list(cfg.DATASETS), choices=list(cfg.DATASETS))
    parser.add_argument("--python", nargs="+", default=[sys.executable], help="Python command prefix.")
    parser.add_argument(
        "--python-cmd",
        default=None,
        help='Python command prefix as one quoted string, e.g. "conda run -n Pytorch_env python".',
    )
    args = parser.parse_args()

    py = shlex.split(args.python_cmd) if args.python_cmd else args.python
    if not args.skip_preprocess:
        run(py + ["comparison_preprocess.py", "--datasets"] + args.datasets)
    for dataset in progress(args.datasets, desc="pipeline datasets", unit="dataset"):
        if not args.skip_train:
            run(py + ["comparison_train.py", "--dataset", dataset])
        run(py + ["comparison_evaluate.py", "--dataset", dataset])
        run(py + ["comparison_profile.py", "--dataset", dataset])
        run(py + ["comparison_cpu_benchmark.py", "--dataset", dataset])
    run(py + ["comparison_summarize.py", "--datasets"] + args.datasets)


if __name__ == "__main__":
    main()
