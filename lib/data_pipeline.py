"""Temporal windows, station splits, and data loaders for forecasting."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lib.utils import construct_geo_adj, kdTree, load_adj, log_string, read_meta1, reorderData


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    mean: float
    std: float
    ori_parts_idx: np.ndarray
    reo_parts_idx: np.ndarray
    reo_all_idx: np.ndarray
    train_size: int
    val_size: int
    test_size: int
    train_node_mask: np.ndarray
    val_node_mask: np.ndarray
    test_node_mask: np.ndarray
    input_channels: int
    rain_quantile_thresholds: dict


def seq2instance(data, input_len, output_len):
    """Create overlapping windows without copying the full source array."""
    num_step, num_nodes, num_dims = data.shape
    num_sample = num_step - input_len - output_len + 1
    if num_sample <= 0:
        raise ValueError(
            f"Not enough time steps ({num_step}) for input_len={input_len} and output_len={output_len}."
        )
    shape = (num_sample, input_len + output_len, num_nodes, num_dims)
    strides = (data.strides[0], data.strides[0], data.strides[1], data.strides[2])
    windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return windows[:, :input_len], windows[:, input_len:input_len + output_len]


def _prepare_inputs(values, use_missing_mask):
    observation_mask = np.isfinite(values).astype(np.float32)
    filled = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if use_missing_mask:
        return np.concatenate([filled, observation_mask], axis=-1)
    return filled


def _mask_nodes(values, allowed_nodes):
    masked = values.copy()
    masked[:, ~allowed_nodes, :] = np.nan
    return masked


def inject_missingness(values, rate, pattern="random", block_length=24, seed=1):
    """Mask input observations for robustness experiments without changing labels."""
    if rate <= 0:
        return values
    if rate >= 1:
        raise ValueError("Missingness rate must be smaller than 1.")
    masked = values.copy()
    rng = np.random.default_rng(seed)
    finite = np.isfinite(masked[..., 0])
    if pattern == "random":
        remove = rng.random(finite.shape) < rate
    elif pattern == "block":
        remove = np.zeros(finite.shape, dtype=bool)
        target = int(np.ceil(rate * finite.size))
        removed = 0
        attempts = 0
        while removed < target and attempts < target * 2 + 100:
            node = int(rng.integers(0, finite.shape[1]))
            start = int(rng.integers(0, max(1, finite.shape[0] - block_length + 1)))
            before = int(remove[start:start + block_length, node].sum())
            remove[start:start + block_length, node] = True
            removed += block_length - before
            attempts += 1
    else:
        raise ValueError(f"Unknown missingness pattern: {pattern}")
    masked[..., 0][finite & remove] = np.nan
    return masked


def build_spatial_masks(locations, split, val_ratio, test_ratio, seed):
    """Return disjoint train/validation/test station masks.

    ``block`` projects coordinates onto a deterministic seeded axis and holds out
    the two tails, producing geographically contiguous validation/test regions.
    """
    node_count = locations.shape[1]
    if split == "none":
        all_nodes = np.ones(node_count, dtype=bool)
        return all_nodes.copy(), all_nodes.copy(), all_nodes.copy()

    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Spatial val/test ratios must be positive and sum to less than 1.")

    val_count = max(1, round(node_count * val_ratio))
    test_count = max(1, round(node_count * test_ratio))
    rng = np.random.default_rng(seed)

    if split == "random":
        order = rng.permutation(node_count)
    elif split == "block":
        angle = rng.uniform(0.0, 2.0 * np.pi)
        lat = locations[0]
        lon = locations[1] * np.cos(np.deg2rad(np.mean(lat)))
        projection = np.sin(angle) * lat + np.cos(angle) * lon
        order = np.argsort(projection, kind="stable")
    else:
        raise ValueError(f"Unknown spatial split: {split}")

    val_indices = order[:val_count]
    test_indices = order[-test_count:]
    train_mask = np.ones(node_count, dtype=bool)
    train_mask[val_indices] = False
    train_mask[test_indices] = False
    val_mask = np.zeros(node_count, dtype=bool)
    val_mask[val_indices] = True
    test_mask = np.zeros(node_count, dtype=bool)
    test_mask[test_indices] = True
    return train_mask, val_mask, test_mask


def _window_split(
    values,
    input_len,
    output_len,
    input_visible_nodes,
    target_nodes,
    use_missing_mask,
    missing_rate=0.0,
    missing_pattern="random",
    missing_block_length=24,
    missing_seed=1,
):
    inputs_source = _mask_nodes(values, input_visible_nodes)
    inputs_source = inject_missingness(
        inputs_source, missing_rate, missing_pattern, missing_block_length, missing_seed
    )
    inputs, _ = seq2instance(_prepare_inputs(inputs_source, use_missing_mask), input_len, output_len)
    _, targets = seq2instance(values, input_len, output_len)
    targets = _mask_nodes(targets.reshape(-1, targets.shape[2], targets.shape[3]), target_nodes).reshape(targets.shape)
    return inputs, targets


def _build_loader(inputs, targets, batch_size, shuffle, num_workers, pin_memory, prefetch_factor):
    dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)


def load_data_bundle(
    filepath,
    metapath,
    input_len,
    output_len,
    train_ratio,
    test_ratio,
    adjpath,
    recurtimes,
    sps,
    batch_size,
    num_workers=0,
    pin_memory=False,
    prefetch_factor=2,
    spatial_split="none",
    spatial_val_ratio=0.1,
    spatial_test_ratio=0.2,
    spatial_seed=1,
    use_missing_mask=False,
    rain_quantiles=(0.90, 0.95, 0.99),
    train_missing_rate=0.0,
    eval_missing_rate=0.0,
    missing_pattern="random",
    missing_block_length=24,
    missing_seed=1,
    log: Optional[object] = None,
):
    traffic = np.load(filepath)["data"][..., :1].astype(np.float32)
    locations = read_meta1(metapath)
    num_step = traffic.shape[0]
    if traffic.shape[1] != locations.shape[1]:
        raise ValueError(f"Data has {traffic.shape[1]} nodes but metadata has {locations.shape[1]} rows.")

    if log is not None:
        log_string(log, f"Shape of data: {traffic.shape}")
        log_string(log, f"Shape of locations: {locations.shape}")

    train_steps = round(train_ratio * num_step)
    test_steps = round(test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train_data = traffic[:train_steps]
    val_data = traffic[train_steps:train_steps + val_steps]
    test_data = traffic[-test_steps:]

    train_nodes, val_nodes, test_nodes = build_spatial_masks(
        locations, spatial_split, spatial_val_ratio, spatial_test_ratio, spatial_seed
    )
    if log is not None:
        log_string(
            log,
            f"Spatial split: {spatial_split} | train/val/test nodes: "
            f"{train_nodes.sum()}/{val_nodes.sum()}/{test_nodes.sum()}",
        )

    # Use geographic affinities to pad spatial holdout partitions.
    if spatial_split == "none":
        adj = load_adj(adjpath, train_data, locations.shape[1])
    else:
        adj = construct_geo_adj(locations)
    parts_idx, mxlen = kdTree(locations, recurtimes, 0)
    if mxlen > sps:
        raise ValueError(f"spa_patchsize={sps} is smaller than KD-tree leaf size {mxlen}.")
    ori_parts_idx, reo_parts_idx, reo_all_idx = reorderData(parts_idx, mxlen, adj, sps)

    # Training sees training stations only; evaluation uses recent observations
    # from all stations and scores the designated target stations.
    all_nodes = np.ones(traffic.shape[1], dtype=bool)
    train_x, train_y = _window_split(
        train_data,
        input_len,
        output_len,
        train_nodes,
        train_nodes,
        use_missing_mask,
        train_missing_rate,
        missing_pattern,
        missing_block_length,
        missing_seed,
    )
    val_target_nodes = all_nodes if spatial_split == "none" else val_nodes
    test_target_nodes = all_nodes if spatial_split == "none" else test_nodes
    val_x, val_y = _window_split(
        val_data,
        input_len,
        output_len,
        all_nodes,
        val_target_nodes,
        use_missing_mask,
        eval_missing_rate,
        missing_pattern,
        missing_block_length,
        missing_seed + 1,
    )
    test_x, test_y = _window_split(
        test_data,
        input_len,
        output_len,
        all_nodes,
        test_target_nodes,
        use_missing_mask,
        eval_missing_rate,
        missing_pattern,
        missing_block_length,
        missing_seed + 2,
    )

    visible_train = train_data[:, train_nodes, :]
    mean = float(np.nanmean(visible_train, dtype=np.float64))
    std = float(np.nanstd(visible_train, dtype=np.float64))
    wet_values = visible_train[np.isfinite(visible_train) & (visible_train > 0)]
    rain_quantile_thresholds = {
        str(quantile): float(np.quantile(wet_values, quantile))
        for quantile in rain_quantiles
        if wet_values.size
    }

    if log is not None:
        log_string(log, f"Shape of Train: {train_y.shape}")
        log_string(log, f"Shape of Validation: {val_y.shape}")
        log_string(log, f"Shape of Test: {test_y.shape}")
        log_string(log, f"Mean: {mean:.6f} & Std: {std:.6f}")

    return DataBundle(
        train_loader=_build_loader(train_x, train_y, batch_size, True, num_workers, pin_memory, prefetch_factor),
        val_loader=_build_loader(val_x, val_y, batch_size, False, num_workers, pin_memory, prefetch_factor),
        test_loader=_build_loader(test_x, test_y, batch_size, False, num_workers, pin_memory, prefetch_factor),
        mean=mean,
        std=std,
        ori_parts_idx=ori_parts_idx,
        reo_parts_idx=reo_parts_idx,
        reo_all_idx=reo_all_idx,
        train_size=train_x.shape[0],
        val_size=val_x.shape[0],
        test_size=test_x.shape[0],
        train_node_mask=train_nodes,
        val_node_mask=val_nodes,
        test_node_mask=test_nodes,
        input_channels=train_x.shape[-1],
        rain_quantile_thresholds=rain_quantile_thresholds,
    )
