import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def discretize_altitude(altitudes, num_bins=50):
    # 使用分位数分桶
    bins = np.quantile(altitudes, np.linspace(0, 1, num_bins + 1))
    bins = np.unique(bins)  # 避免重复边界
    if len(bins) < num_bins + 1:
        bins = np.linspace(np.min(altitudes), np.max(altitudes), num_bins + 1)
    return np.digitize(altitudes, bins[:-1]) - 1

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        epsilon = 1e-8
        mask = np.not_equal(label, np.nan)
        mask = mask.astype(np.float16)
        mask /= np.mean(mask + 1e-10)
        mae = np.abs(np.subtract(pred, label)).astype(np.float16)
        wape = np.divide(np.sum(mae), np.sum(np.abs(label)))
        wape = np.nan_to_num(wape * mask)
        rmse = np.square(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        error = np.square(np.subtract(pred, label)).astype(np.float16)
        mse = np.nan_to_num(error * mask)
        mse = np.mean(mse)
        non_zero_mask = np.abs(label) > 1e-2
        combined_mask = mask * non_zero_mask.astype(np.float16)
        if np.sum(combined_mask) > 0:
            mape = np.divide(mae, label + epsilon, where=non_zero_mask)
            mape = np.nan_to_num(mape * combined_mask)
            mape = np.mean(mape) * 100
        else:
            mape = 0.0
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
    return mae, rmse, mape, mse

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mse(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        mask = (target != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.nan_to_num(mask)
    loss = (prediction - target) ** 2
    loss *= mask
    loss = torch.nan_to_num(loss)
    return torch.mean(loss)

def _compute_loss(y_true, y_predicted):
    criterion = torch.nn.HuberLoss(delta=1.0)
    loss = criterion(y_predicted, y_true)
    return loss

def seq2instance(data, P, Q):
    num_step, nodes, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape=(num_sample, P, nodes, dims), dtype=np.float16)
    y = np.zeros(shape=(num_sample, Q, nodes, dims), dtype=np.float16
                 )
    for i in range(num_sample):
        x[i] = data[i: i + P]
        y[i] = data[i + P: i + P + Q]
    return x, y

def read_meta1(path):
    meta = pd.read_csv(path)
    lat = meta['Lat'].values
    lng = meta['Lng'].values
    locations = np.stack([lat, lng], 0)
    return locations

def read_meta2(path):
    try:
        meta = pd.read_csv(path)
        if 'col3' in meta.columns and not meta['col3'].isnull().all():
            attitude = meta['col3'].values
            attitudes = np.stack(attitude, 0)
            return attitudes
        else:
            return None  # Return None if 'col3' is missing or contains only NaN
    except Exception as e:
        print(f"Error reading meta2 data: {e}")
        return None  # Return None on any error (e.g., file not found, invalid format)

def construct_adj(data, num_node):
    data_mean = np.mean([data[24 * 12 * i: 24 * 12 * (i + 1)] for i in range(data.shape[0] // (24 * 12))], axis=0)
    data_mean = data_mean.squeeze().T
    tem_matrix = cosine_similarity(data_mean, data_mean)
    tem_matrix = np.exp((tem_matrix - tem_matrix.mean()) / tem_matrix.std())
    return tem_matrix

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
        part_dist = adj[part_idx, :]
        part_dist[:, part_idx] = 0
        if sps - part_idx.shape[0] > 0:
            local_part_idx = augmentAlign(part_dist, sps - part_idx.shape[0])
            auged_part_idx = np.concatenate([part_idx, local_part_idx], 0)
        else:
            auged_part_idx = part_idx
        reo_parts_idx = np.concatenate([reo_parts_idx, np.arange(part_idx.shape[0]) + sps * i])
        ori_parts_idx = np.concatenate([ori_parts_idx, part_idx])
        reo_all_idx = np.concatenate([reo_all_idx, auged_part_idx])
    return ori_parts_idx, reo_parts_idx, reo_all_idx

def kdTree(locations, times, axis):
    sorted_idx = np.argsort(locations[axis])
    part1, part2 = np.sort(sorted_idx[:locations.shape[1] // 2]), np.sort(sorted_idx[locations.shape[1] // 2:])
    parts = []
    if times == 1:
        return [part1, part2], max(part1.shape[0], part2.shape[0])
    else:
        left_parts, lmxlen = kdTree(locations[:, part1], times - 1, axis ^ 1)
        right_parts, rmxlen = kdTree(locations[:, part2], times - 1, axis ^ 1)
        for part in left_parts:
            parts.append(part1[part])
        for part in right_parts:
            parts.append(part2[part])
    return parts, max(lmxlen, rmxlen)

def loadData(filepath, metapath, P, Q, train_ratio, test_ratio, adjpath, recurtimes, sps, log):
    # Traffic
    Traffic = np.load(filepath)['data'][..., :1].astype(np.float16)
    locations = read_meta1(metapath)
    num_step = Traffic.shape[0]
    log_string(log, f'Shape of data: {Traffic.shape}')
    log_string(log, f'Shape of locations: {locations.shape}')
    # train/val/test
    train_steps = round(train_ratio * num_step)
    test_steps = round(test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    trainData = Traffic[:train_steps]
    valData = Traffic[train_steps:train_steps + val_steps]
    testData = Traffic[-test_steps:]
    # load adj for padding
    if os.path.exists(adjpath):
        adj = np.load(adjpath)
    else:
        adj = construct_adj(trainData, locations.shape[1])
        np.save(adjpath, adj)
    # partition and pad data with new indices
    parts_idx, mxlen = kdTree(locations, recurtimes, 0)
    ori_parts_idx, reo_parts_idx, reo_all_idx = reorderData(parts_idx, mxlen, adj, sps)
    # X, Y
    trainX, trainY = seq2instance(trainData, P, Q)
    valX, valY = seq2instance(valData, P, Q)
    testX, testY = seq2instance(testData, P, Q)
    # normalization
    mean, std = np.mean(trainX), np.std(trainX)
    # log
    log_string(log, f'Shape of Train: {trainY.shape}')
    log_string(log, f'Shape of Validation: {valY.shape}')
    log_string(log, f'Shape of Test: {testY.shape}')
    log_string(log, f'Mean: {mean} & Std: {std}')
    return trainX, trainY, valX, valY, testX, testY, mean, std, ori_parts_idx, reo_parts_idx, reo_all_idx

