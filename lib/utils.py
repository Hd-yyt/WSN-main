"""Forecasting metrics, geographic metadata, and spatial partition helpers."""

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity


def log_string(log, string: str) -> None:
    log.write(string + "\n")
    log.flush()
    print(string)


def metric(pred, label):
    with np.errstate(divide="ignore", invalid="ignore"):
        epsilon = 1e-8
        mask = ~np.isnan(label)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask) + 1e-10

        abs_error = np.abs(pred - label).astype(np.float32)
        mae = np.nan_to_num(abs_error * mask).mean()

        squared_error = np.square(pred - label).astype(np.float32)
        mse = np.nan_to_num(squared_error * mask).mean()
        rmse = np.sqrt(mse)

        non_zero_mask = np.abs(label) > 1e-2
        combined_mask = mask * non_zero_mask.astype(np.float32)
        if np.any(combined_mask):
            mape = np.divide(abs_error, np.abs(label) + epsilon, where=non_zero_mask)
            mape = np.nan_to_num(mape * combined_mask).mean() * 100
        else:
            mape = 0.0

    return float(mae), float(rmse), float(mape), float(mse)


def precipitation_metrics(pred, label, threshold):
    """Event-based precipitation scores with NaN-aware observations."""
    pred = np.asarray(pred, dtype=np.float32)
    label = np.asarray(label, dtype=np.float32)
    valid = np.isfinite(label) & np.isfinite(pred)
    if not np.any(valid):
        return {name: float("nan") for name in ("csi", "pod", "far", "precision", "f1", "ets", "hss", "brier")}

    observed = label[valid] >= threshold
    forecast = pred[valid] >= threshold
    hits = float(np.sum(observed & forecast))
    misses = float(np.sum(observed & ~forecast))
    false_alarms = float(np.sum(~observed & forecast))
    correct_negatives = float(np.sum(~observed & ~forecast))
    total = hits + misses + false_alarms + correct_negatives

    def safe_div(numerator, denominator):
        return float(numerator / denominator) if denominator > 0 else float("nan")

    csi = safe_div(hits, hits + misses + false_alarms)
    pod = safe_div(hits, hits + misses)
    far = safe_div(false_alarms, hits + false_alarms)
    precision = safe_div(hits, hits + false_alarms)
    f1 = safe_div(2.0 * hits, 2.0 * hits + misses + false_alarms)
    random_hits = safe_div((hits + misses) * (hits + false_alarms), total)
    ets = safe_div(hits - random_hits, hits + misses + false_alarms - random_hits)
    hss = safe_div(
        2.0 * (hits * correct_negatives - misses * false_alarms),
        (hits + misses) * (misses + correct_negatives)
        + (hits + false_alarms) * (false_alarms + correct_negatives),
    )
    # Deterministic forecasts are interpreted as probabilities in {0, 1}.
    brier = float(np.mean((forecast.astype(np.float32) - observed.astype(np.float32)) ** 2))
    return {
        "csi": csi,
        "pod": pod,
        "far": far,
        "precision": precision,
        "f1": f1,
        "ets": ets,
        "hss": hss,
        "brier": brier,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
    }


def precipitation_stratified_metrics(pred, label, thresholds):
    """Regression errors by observed rainfall-intensity interval."""
    pred = np.asarray(pred, dtype=np.float32)
    label = np.asarray(label, dtype=np.float32)
    valid = np.isfinite(label) & np.isfinite(pred)
    boundaries = [-np.inf, *sorted(set(float(value) for value in thresholds)), np.inf]
    output = {}
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        selected = valid & (label >= lower) & (label < upper)
        key = f"[{lower:g},{upper:g})"
        if not np.any(selected):
            output[key] = {"count": 0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
            continue
        errors = pred[selected] - label[selected]
        output[key] = {
            "count": int(selected.sum()),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "bias": float(np.mean(errors)),
        }
    return output


def precipitation_distribution_metrics(pred, label):
    """Diagnostics for frequency bias and over-smoothed precipitation fields."""
    pred = np.asarray(pred, dtype=np.float32)
    label = np.asarray(label, dtype=np.float32)
    valid = np.isfinite(label) & np.isfinite(pred)
    if not np.any(valid):
        return {}
    predicted = np.maximum(pred[valid], 0.0)
    observed = np.maximum(label[valid], 0.0)
    result = {
        "predicted_nonzero_frequency": float(np.mean(predicted >= 0.1)),
        "observed_nonzero_frequency": float(np.mean(observed >= 0.1)),
        "predicted_variance": float(np.var(predicted)),
        "observed_variance": float(np.var(observed)),
        "variance_ratio": float(np.var(predicted) / max(np.var(observed), 1e-12)),
        "predicted_max": float(np.max(predicted)),
        "observed_max": float(np.max(observed)),
    }
    for quantile in (0.90, 0.95, 0.99, 0.999):
        result[f"predicted_q{quantile:g}"] = float(np.quantile(predicted, quantile))
        result[f"observed_q{quantile:g}"] = float(np.quantile(observed, quantile))
    return result


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        mask = target != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.nan_to_num(mask)
    loss = torch.nan_to_num((prediction - target) ** 2 * mask)
    return torch.mean(loss)


def _compute_loss(
    y_true,
    y_predicted,
    loss_type="huber",
    event_threshold=5.0,
    event_weight=4.0,
):
    """NaN-aware regression losses, including an extreme-event weighted variant."""
    valid = torch.isfinite(y_true) & torch.isfinite(y_predicted)
    if not torch.any(valid):
        # Preserve a differentiable zero for degenerate mini-batches.
        return y_predicted.sum() * 0.0
    target = y_true[valid]
    prediction = y_predicted[valid]
    if loss_type == "mae":
        losses = torch.abs(prediction - target)
    elif loss_type == "mse":
        losses = (prediction - target) ** 2
    else:
        losses = torch.nn.functional.huber_loss(prediction, target, delta=1.0, reduction="none")
    if loss_type == "weighted-huber":
        if event_weight < 1.0:
            raise ValueError("event_weight must be at least 1.0")
        weights = torch.where(
            target >= event_threshold,
            torch.as_tensor(event_weight, dtype=losses.dtype, device=losses.device),
            torch.ones((), dtype=losses.dtype, device=losses.device),
        )
        return torch.sum(losses * weights) / torch.sum(weights)
    elif loss_type not in {"huber", "mae", "mse"}:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return losses.mean()


def parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def discretize_altitude(altitudes, num_bins=50):
    if altitudes is None:
        return None
    altitudes = np.asarray(altitudes, dtype=np.float32)
    bins = np.quantile(altitudes, np.linspace(0, 1, num_bins + 1))
    bins = np.unique(bins)
    if len(bins) < num_bins + 1:
        bins = np.linspace(np.min(altitudes), np.max(altitudes), num_bins + 1)
    indices = np.digitize(altitudes, bins[:-1]) - 1
    return np.clip(indices, 0, num_bins - 1)


def read_meta1(path):
    meta = pd.read_csv(path)
    lat = meta["Lat"].to_numpy(dtype=np.float32)
    lng = meta["Lng"].to_numpy(dtype=np.float32)
    return np.stack([lat, lng], axis=0)


def read_meta2(path) -> Optional[np.ndarray]:
    meta = pd.read_csv(path)
    for column in ("Altitude", "altitude", "Elevation", "elevation", "col3"):
        if column in meta.columns and not meta[column].isnull().all():
            return meta[column].to_numpy(dtype=np.float32)
    return None


def construct_adj(data, num_node):
    day_slices = data.shape[0] // (24 * 12)
    data_mean = np.mean(
        [data[24 * 12 * i: 24 * 12 * (i + 1)] for i in range(day_slices)],
        axis=0,
    )
    data_mean = data_mean.squeeze().T
    # Fill missing profile values before computing cosine similarity.
    data_mean = np.asarray(data_mean, dtype=np.float32)
    finite_mean = np.nanmean(data_mean, axis=0, keepdims=True)
    finite_mean = np.where(np.isfinite(finite_mean), finite_mean, 0.0)
    data_mean = np.where(np.isfinite(data_mean), data_mean, finite_mean)
    tem_matrix = cosine_similarity(data_mean, data_mean)
    tem_matrix = np.exp((tem_matrix - tem_matrix.mean()) / (tem_matrix.std() + 1e-6))
    return tem_matrix


def construct_geo_adj(locations):
    """Target-independent geographic affinity used by spatial holdout runs."""
    locations = np.asarray(locations, dtype=np.float32)
    radians = np.deg2rad(locations.T)
    lat = radians[:, 0:1]
    lon = radians[:, 1:2]
    dlat = lat - lat.T
    dlon = lon - lon.T
    hav = np.sin(dlat / 2.0) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2.0) ** 2
    distance_km = 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
    nonzero = distance_km[distance_km > 0]
    scale = float(np.median(nonzero)) if nonzero.size else 1.0
    return np.exp(-distance_km / max(scale, 1e-6)).astype(np.float32)


def augmentAlign(dist_matrix, auglen):
    sorted_idx = np.argsort(dist_matrix.reshape(-1) * -1)
    sorted_idx = sorted_idx % dist_matrix.shape[-1]
    augidx = []
    for idx in sorted_idx:
        if idx not in augidx:
            augidx.append(idx)
        if len(augidx) == auglen:
            break
    return np.array(augidx, dtype=np.int32)


def reorderData(parts_idx, mxlen, adj, sps):
    ori_parts_idx = np.array([], dtype=np.int32)
    reo_parts_idx = np.array([], dtype=np.int32)
    reo_all_idx = np.array([], dtype=np.int32)
    for i, part_idx in enumerate(parts_idx):
        part_dist = adj[part_idx, :].copy()
        part_dist[:, part_idx] = 0
        if sps - part_idx.shape[0] > 0:
            local_part_idx = augmentAlign(part_dist, sps - part_idx.shape[0])
            auged_part_idx = np.concatenate([part_idx, local_part_idx], axis=0)
        else:
            auged_part_idx = part_idx
        reo_parts_idx = np.concatenate([reo_parts_idx, np.arange(part_idx.shape[0]) + sps * i], axis=0)
        ori_parts_idx = np.concatenate([ori_parts_idx, part_idx], axis=0)
        reo_all_idx = np.concatenate([reo_all_idx, auged_part_idx], axis=0)
    return ori_parts_idx, reo_parts_idx, reo_all_idx


def kdTree(locations, times, axis):
    sorted_idx = np.argsort(locations[axis])
    split = locations.shape[1] // 2
    part1 = np.sort(sorted_idx[:split])
    part2 = np.sort(sorted_idx[split:])
    parts = []
    if times == 1:
        return [part1, part2], max(part1.shape[0], part2.shape[0])

    left_parts, lmxlen = kdTree(locations[:, part1], times - 1, axis ^ 1)
    right_parts, rmxlen = kdTree(locations[:, part2], times - 1, axis ^ 1)
    for part in left_parts:
        parts.append(part1[part])
    for part in right_parts:
        parts.append(part2[part])
    return parts, max(lmxlen, rmxlen)


def load_adj(adjpath, train_data, num_nodes):
    if os.path.exists(adjpath):
        return np.load(adjpath)
    adj = construct_adj(train_data, num_nodes)
    np.save(adjpath, adj)
    return adj
