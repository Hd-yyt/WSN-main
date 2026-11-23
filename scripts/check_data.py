"""Check the arrays and spatial metadata required by a dataset configuration."""

import argparse
import configparser
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def check_config(path):
    config = configparser.ConfigParser()
    if not config.read(path, encoding="utf-8"):
        raise ValueError(f"Configuration not found: {path}")
    files = {key: ROOT / config["file"][key] for key in ("traffic", "meta", "adj")}
    for key, value in files.items():
        if not value.is_file():
            raise ValueError(f"Missing {key} file: {value}. See docs/DATASETS.md.")
    with np.load(files["traffic"], allow_pickle=False) as source:
        if "data" not in source:
            raise ValueError("The NPZ archive must contain an array named 'data'.")
        data = source["data"]
    nodes = config.getint("param", "nodes")
    if data.ndim != 3 or data.shape[1] != nodes or data.shape[2] < 1:
        raise ValueError(f"Expected (time, {nodes}, features), found {data.shape}.")
    if np.isinf(data).any():
        raise ValueError("The data contain infinite values; use NaN for missing observations.")
    meta = pd.read_csv(files["meta"])
    if len(meta) != nodes or not {"Lat", "Lng"}.issubset(meta.columns):
        raise ValueError(f"Metadata must contain {nodes} rows and Lat/Lng columns.")
    if not np.isfinite(meta[["Lat", "Lng"]].to_numpy()).all():
        raise ValueError("Coordinates must be finite.")
    adjacency = np.load(files["adj"], allow_pickle=False)
    if adjacency.shape != (nodes, nodes):
        raise ValueError(f"Expected ({nodes}, {nodes}) adjacency, found {adjacency.shape}.")
    steps = data.shape[0]
    train = round(config.getfloat("data", "train_ratio") * steps)
    test = round(config.getfloat("data", "test_ratio") * steps)
    split_lengths = (train, steps - train - test, test)
    window = config.getint("data", "input_len") + config.getint("data", "output_len")
    if min(split_lengths) < window:
        raise ValueError(f"Each time partition must contain at least {window} steps.")
    print(f"OK {Path(path).stem}: shape={data.shape}, dtype={data.dtype}, "
          f"time partitions={split_lengths}, observed={np.isfinite(data).mean():.2%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config", type=Path)
    selection.add_argument("--all", action="store_true", help="Check all dataset configurations.")
    args = parser.parse_args()
    configs = sorted((ROOT / "config").glob("*.conf")) if args.all else [args.config]
    errors = []
    for config in configs:
        try:
            check_config(config)
        except (ValueError, OSError, KeyError) as error:
            errors.append(f"{config.name}: {error}")
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
