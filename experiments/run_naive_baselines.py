"""Evaluate persistence, seasonal-lag, and station-climatology baselines."""

import argparse
import configparser
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.data_pipeline import build_spatial_masks, seq2instance
from lib.utils import (
    precipitation_distribution_metrics,
    precipitation_metrics,
    precipitation_stratified_metrics,
    read_meta1,
)


CONFIGS = {
    key: ROOT / "config" / f"{key}.conf"
    for key in (
        "wind", "temp", "rfusa", "rfuk", "rfcn", "rfaus", "meteonet_nw", "meteonet_se"
    )
}


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def predict(method, history, climatology, output_len):
    if method == "persistence":
        return np.repeat(history[:, -1:, :, :], output_len, axis=1)
    if method == "seasonal-lag":
        if history.shape[1] < output_len:
            raise ValueError("seasonal-lag requires input_len >= output_len")
        return history[:, -output_len:, :, :]
    if method == "station-climatology":
        return np.broadcast_to(climatology[None, None, :, :], (len(history), output_len, *climatology.shape))
    raise ValueError(f"Unknown baseline: {method}")


def evaluate(method, inputs, labels, climatology, batch_size, collect_arrays, node_mask=None):
    abs_sum = sq_sum = mape_sum = 0.0
    valid_count = mape_count = 0
    sample_mae_parts = []
    horizon_abs_sum = np.zeros(labels.shape[1], dtype=np.float64)
    horizon_count = np.zeros(labels.shape[1], dtype=np.int64)
    collected_predictions = []
    collected_labels = []

    for start in range(0, len(inputs), batch_size):
        history = np.asarray(inputs[start:start + batch_size], dtype=np.float32)
        target = np.asarray(labels[start:start + batch_size], dtype=np.float32)
        if node_mask is not None:
            history = history[:, :, node_mask, :]
            target = target[:, :, node_mask, :]
            batch_climatology = climatology[node_mask]
        else:
            batch_climatology = climatology
        prediction = np.asarray(predict(method, history, batch_climatology, labels.shape[1]), dtype=np.float32)
        valid = np.isfinite(target) & np.isfinite(prediction)
        error = prediction - target
        absolute = np.where(valid, np.abs(error), np.nan)
        squared = np.where(valid, error ** 2, np.nan)
        abs_sum += float(np.nansum(absolute, dtype=np.float64))
        sq_sum += float(np.nansum(squared, dtype=np.float64))
        valid_count += int(valid.sum())
        relative_valid = valid & (np.abs(target) > 1e-2)
        mape_sum += float(np.sum(np.abs(error[relative_valid]) / np.abs(target[relative_valid]), dtype=np.float64))
        mape_count += int(relative_valid.sum())
        sample_mae_parts.append(np.nanmean(absolute, axis=(1, 2, 3)).astype(np.float32))
        horizon_abs_sum += np.nansum(absolute, axis=(0, 2, 3), dtype=np.float64)
        horizon_count += np.sum(valid, axis=(0, 2, 3))
        if collect_arrays:
            collected_predictions.append(prediction)
            collected_labels.append(target)

    mae = abs_sum / max(valid_count, 1)
    mse = sq_sum / max(valid_count, 1)
    metrics = {"mae": mae, "rmse": float(np.sqrt(mse)), "mape": 100.0 * mape_sum / max(mape_count, 1), "mse": mse}
    arrays = (
        np.concatenate(collected_predictions) if collected_predictions else None,
        np.concatenate(collected_labels) if collected_labels else None,
    )
    return metrics, np.concatenate(sample_mae_parts), horizon_abs_sum / np.maximum(horizon_count, 1), arrays


def run(dataset, method, spatial_split, spatial_seed, args):
    config_path = CONFIGS[dataset]
    if not config_path.exists():
        print(f"SKIP {dataset}: missing {config_path}")
        return
    config = configparser.ConfigParser()
    config.read(config_path)
    traffic_path = (ROOT / config["file"]["traffic"]).resolve()
    if not traffic_path.exists():
        print(f"SKIP {dataset}: missing {traffic_path}")
        return

    tag = f"naive__{dataset}__{method}"
    if spatial_split != "none":
        tag += f"__spatial-{spatial_split}__spatial-seed-{spatial_seed}"
    output_dir = args.output_root / "naive" / tag
    result_path = output_dir / "result.json"
    if result_path.exists() and not args.rerun:
        print(f"DONE {dataset} {method}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    values = np.load(traffic_path)["data"][..., :1].astype(np.float32)
    input_len = config.getint("data", "input_len")
    output_len = config.getint("data", "output_len")
    train_steps = round(config.getfloat("data", "train_ratio") * len(values))
    test_steps = round(config.getfloat("data", "test_ratio") * len(values))
    train = values[:train_steps]
    test = values[-test_steps:]
    inputs, labels = seq2instance(test, input_len, output_len)
    climatology = np.nanmean(train, axis=0).astype(np.float32)
    node_mask = None
    if spatial_split != "none":
        if method == "station-climatology":
            print(f"SKIP {dataset} {spatial_split} {method}: station climatology requires observations from the training period")
            return
        locations = read_meta1((ROOT / config["file"]["meta"]).resolve())
        _, _, node_mask = build_spatial_masks(locations, spatial_split, 0.1, 0.2, spatial_seed)
    is_precipitation = config.getboolean("data", "is_precipitation", fallback=False)
    metrics, sample_mae, horizon_mae, (prediction, target) = evaluate(
        method, inputs, labels, climatology, args.batch_size, is_precipitation, node_mask
    )

    artifact_path = output_dir / "diagnostics.npz"
    np.savez_compressed(artifact_path, sample_mae=sample_mae, horizon_mae=horizon_mae.astype(np.float32))
    event_metrics = {}
    strata = {}
    distribution = {}
    quantile_thresholds = {}
    if is_precipitation:
        wet = train[np.isfinite(train) & (train > 0)]
        quantiles = parse_float_list(config.get("data", "rain_quantiles", fallback="0.90,0.95,0.99"))
        quantile_thresholds = {str(q): float(np.quantile(wet, q)) for q in quantiles if wet.size}
        thresholds = parse_float_list(config.get("data", "rain_thresholds", fallback="0.1,1,5,10"))
        thresholds = sorted(set(thresholds + list(quantile_thresholds.values())))
        event_metrics = {str(threshold): precipitation_metrics(prediction, target, threshold) for threshold in thresholds}
        strata = precipitation_stratified_metrics(prediction, target, thresholds)
        distribution = precipitation_distribution_metrics(prediction, target)

    result = {
        "dataset": dataset,
        "study_name": "naive",
        "seed": 0,
        "spatial_split": spatial_split,
        "spatial_seed": spatial_seed,
        "train_missing_rate": 0.0,
        "eval_missing_rate": 0.0,
        "missing_pattern": "none",
        "loss_type": "none",
        "event_loss_threshold": None,
        "event_loss_weight": None,
        "position_encoder": "none",
        "use_elevation": False,
        "branch_mode": method,
        "fusion_mode": "none",
        "metrics": metrics,
        "event_metrics": event_metrics,
        "stratified_rainfall_metrics": strata,
        "precipitation_distribution": distribution,
        "rain_quantile_thresholds": quantile_thresholds,
        "parameters": 0,
        "peak_gpu_memory_mb": 0.0,
        "test_time_seconds": None,
        "best_epoch": 0,
        "training_max_epoch": 0,
        "artifact_file": str(artifact_path.resolve()),
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    print(
        f"{dataset:12s} {spatial_split:8s} {method:20s} "
        f"MAE {metrics['mae']:.6f} RMSE {metrics['rmse']:.6f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(CONFIGS))
    parser.add_argument("--methods", default="persistence,seasonal-lag,station-climatology")
    parser.add_argument("--spatial-splits", default="none")
    parser.add_argument("--spatial-seeds", default="1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    spatial_splits = [value.strip() for value in args.spatial_splits.split(",") if value.strip()]
    spatial_seeds = [int(value) for value in args.spatial_seeds.split(",") if value.strip()]
    if not set(methods).issubset({"persistence", "seasonal-lag", "station-climatology"}):
        raise SystemExit(f"Unknown baseline method: {methods}")
    if args.batch_size < 1:
        raise SystemExit("batch-size must be positive.")
    if not set(spatial_splits).issubset({"none", "random", "block"}):
        raise SystemExit(f"Unknown spatial split: {spatial_splits}")
    for dataset in datasets:
        if dataset not in CONFIGS:
            raise SystemExit(f"Unknown dataset: {dataset}")
        for method in methods:
            for spatial_split in spatial_splits:
                seeds = [0] if spatial_split == "none" else spatial_seeds
                for spatial_seed in seeds:
                    run(dataset, method, spatial_split, spatial_seed, args)


if __name__ == "__main__":
    main()
