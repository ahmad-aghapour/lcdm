from __future__ import annotations

import argparse

from lcdm.registries import build_dataset, build_model, build_operator
from lcdm.runner import run_experiment
from lcdm.utils import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # build with a temporary device; runner rebuilds pipeline on accelerator device
    dataset = build_dataset(cfg)
    operator = build_operator(cfg)
    model = build_model(cfg, device="cpu")

    run_experiment(cfg, dataset, model, operator)


if __name__ == "__main__":
    main()