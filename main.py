import math
import time

import pandas as pd
import torch
import random
import argparse
import numpy as np
import configparser
import os
from tqdm import tqdm
from models.model import PatchSTG
from lib.utils import log_string, loadData, _compute_loss, metric


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        log_string(log, '\n------------ 加载数据 -------------')
        self.trainX, self.trainY, self.valX, self.valY, self.testX, self.testY, \
            self.mean, self.std, self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx = loadData(
            self.traffic_file, self.meta_file,
            self.input_len, self.output_len,
            self.train_ratio, self.test_ratio,
            self.adj_file, self.recur_times,
            self.spa_patchsize, log)
        log_string(log, '------------ 结束 -------------\n')

        self.best_epoch = 0

        self.device = torch.device(f"cuda:{self.cuda}" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):
        self.model = PatchSTG(self.tem_patchsize, self.tem_patchnum,
                              self.node_num, self.spa_patchsize, self.spa_patchnum,
                              self.layers, self.factors,
                              self.input_dims, self.node_dims,
                              self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx, self.meta_file).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.learning_rate, weight_decay=self.weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[1, 35, 40],
            gamma=0.5,
        )

    def vali(self):
        self.model.eval()
        num_val = self.valX.shape[0]
        pred = []
        label = []

        num_batch = math.ceil(num_val / self.batch_size)
        with torch.no_grad():
            for batch_idx in range(num_batch):
                start_idx = batch_idx * self.batch_size
                end_idx = min(num_val, (batch_idx + 1) * self.batch_size)

                X = self.valX[start_idx: end_idx]
                Y = self.valY[start_idx: end_idx]

                NormX = torch.from_numpy(X).float().to(self.device)

                y_hat = self.model(NormX)

                pred.append(y_hat.cpu().numpy())
                label.append(Y)

        pred = np.concatenate(pred, axis=0)
        label = np.concatenate(label, axis=0)

        maes = []
        rmses = []
        mapes = []
        mses = []
        for i in range(pred.shape[1]):
            mae, rmse, mape, mse = metric(pred[:, i, :], label[:, i, :])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            mses.append(mse)
            log_string(log, '步骤 %d, MAE: %.4f, RMSE: %.4f, MAPE: %.4f, MSE: %.4f' % (i + 1, mae, rmse, mape, mse))

        mae, rmse, mape, mse = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        mses.append(mse)
        log_string(log, '平均值, MAE: %.4f, RMSE: %.4f, MAPE: %.4f, MSE: %.4f' % (mae, rmse, mape, mse))

        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0), np.stack(mses, 0)

    def train(self):
        log_string(log, "======================训练模式======================")
        min_loss = 10000000.0
        num_train = self.trainX.shape[0]

        for epoch in tqdm(range(1, self.max_epoch + 1)):
            self.model.train()
            train_l_sum, batch_count, start = 0.0, 0, time.time()
            permutation = np.random.permutation(num_train)
            self.trainX = self.trainX[permutation]
            self.trainY = self.trainY[permutation]
            num_batch = math.ceil(num_train / self.batch_size)

            epoch_preds = []
            epoch_labels = []

            with tqdm(total=num_batch) as pbar:
                for batch_idx in range(num_batch):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_train, (batch_idx + 1) * self.batch_size)

                    X = self.trainX[start_idx: end_idx]
                    Y = self.trainY[start_idx: end_idx]

                    NormX = torch.from_numpy(X).float().to(self.device)
                    Y = torch.from_numpy(Y).float().to(self.device)

                    self.optimizer.zero_grad()

                    y_hat = self.model(NormX)

                    pred_np = y_hat.detach().cpu().numpy()
                    label_np = Y.detach().cpu().numpy()
                    epoch_preds.append(pred_np)
                    epoch_labels.append(label_np)

                    loss = _compute_loss(Y, y_hat)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()

                    train_l_sum += loss.cpu().item()
                    batch_count += 1
                    pbar.update(1)

            log_string(log, 'epoch %d, 学习率 %.6f, 损失 %.4f, 时间 %.1f 秒'
                       % (epoch, self.optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
            mae, rmse, mape, mse = self.vali()
            self.lr_scheduler.step()
            if mae[-1] < min_loss:
                self.best_epoch = epoch
                min_loss = mae[-1]
                torch.save(self.model.state_dict(), self.model_file)

        log_string(log, f'最佳 epoch 为: {self.best_epoch}')

    def test(self):
        log_string(log, "======================TEST MODE======================")
        self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
        self.model.eval()
        num_val = self.testX.shape[0]
        pred = []
        label = []

        num_batch = math.ceil(num_val / self.batch_size)
        with torch.no_grad():
            for batch_idx in range(num_batch):
                if isinstance(self.model, torch.nn.Module):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_val, (batch_idx + 1) * self.batch_size)

                    X = self.testX[start_idx:end_idx]
                    Y = self.testY[start_idx:end_idx]

                    NormX = torch.from_numpy(X).float().to(self.device)
                    y_hat = self.model(NormX)

                    pred.append(y_hat.cpu().numpy())
                    label.append(Y)

        pred = np.concatenate(pred, axis=0)
        label = np.concatenate(label, axis=0)
        log_string(log, f"pred 形狀: {pred.shape}, label 形狀: {label.shape}")

        maes, rmses, mapes, mses = [], [], [], []
        # Initialize lists to store data for each step
        data_dict = {
            'pred_step_6': [],
            'label_step_6': [],
            'pred_step_12': [],
            'label_step_12': [],
            'pred_step_24': [],
            'label_step_24': []
        }

        for i in range(pred.shape[1]):  # 假设 time_steps 包含 24 小时
            mae, rmse, mape, mse = metric(pred[:, i, :, :], label[:, i, :, :])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            mses.append(mse)
            log_string(log, f'步驟 {i + 1}, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}, mse: {mse:.4f}')

            if i in [5, 11, 23]:  # Steps 6, 12, and 24
                pred_subset = pred[0:1, i:i + 1, :, :]
                label_subset = label[0:1, i:i + 1, :, :]
                print(f"Step {i + 1} pred_subset 形状: {pred_subset.shape}, 值: \n{pred_subset}")
                print(f"Step {i + 1} label_subset 形状: {label_subset.shape}, 值: \n{label_subset}")

                # Store flattened data in dictionary
                step_key = f'step_{i + 1}'
                data_dict[f'pred_{step_key}'] = pred_subset.reshape(-1)
                data_dict[f'label_{step_key}'] = label_subset.reshape(-1)

        # Create DataFrame from dictionary
        df = pd.DataFrame(data_dict)

        # Ensure all columns have the same length by padding with NaN if necessary
        max_length = max(len(data_dict[key]) for key in data_dict)
        for key in data_dict:
            if len(data_dict[key]) < max_length:
                data_dict[key] = np.pad(data_dict[key], (0, max_length - len(data_dict[key])),
                                        mode='constant', constant_values=np.nan)

        # Save to single CSV file
        df.to_csv('subsets_temp.csv', index=False)



        mae, rmse, mape, mse = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        mses.append(mse)
        log_string(log, f'平均值, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}, mse: {mse:.4f}')

        # 返回原始指标
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0), np.stack(mses, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='配置文件')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    parser.add_argument('--cuda', type=str, default=config['train']['cuda'])
    parser.add_argument('--seed', type=int, default=config['train']['seed'])
    parser.add_argument('--batch_size', type=int, default=config['train']['batch_size'])
    parser.add_argument('--max_epoch', type=int, default=config['train']['max_epoch'])
    parser.add_argument('--learning_rate', type=float, default=config['train']['learning_rate'])
    parser.add_argument('--weight_decay', type=float, default=config['train']['weight_decay'])

    parser.add_argument('--input_len', type=int, default=config['data']['input_len'])
    parser.add_argument('--output_len', type=int, default=config['data']['output_len'])
    parser.add_argument('--train_ratio', type=float, default=config['data']['train_ratio'])
    parser.add_argument('--val_ratio', type=float, default=config['data']['val_ratio'])
    parser.add_argument('--test_ratio', type=float, default=config['data']['test_ratio'])

    parser.add_argument('--layers', type=int, default=config['param']['layers'])
    parser.add_argument('--tem_patchsize', type=int, default=config['param']['tps'])
    parser.add_argument('--tem_patchnum', type=int, default=config['param']['tpn'])
    parser.add_argument('--factors', type=int, default=config['param']['factors'])
    parser.add_argument('--recur_times', type=int, default=config['param']['recur'])
    parser.add_argument('--spa_patchsize', type=int, default=config['param']['sps'])
    parser.add_argument('--spa_patchnum', type=int, default=config['param']['spn'])
    parser.add_argument('--node_num', type=int, default=config['param']['nodes'])
    parser.add_argument('--input_dims', type=int, default=config['param']['id'])
    parser.add_argument('--node_dims', type=int, default=config['param']['nd'])

    parser.add_argument('--traffic_file', default=config['file']['traffic'])
    parser.add_argument('--meta_file', default=config['file']['meta'])
    parser.add_argument('--adj_file', default=config['file']['adj'])
    parser.add_argument('--model_file', default=config['file']['model'])
    parser.add_argument('--log_file', default=config['file']['log'])
    parser.add_argument('--dataset_name', default=config.get('file', 'dataset_name', fallback=
    os.path.basename(config['file']['traffic']).split('.')[0]))

    args = parser.parse_args()

    log = open(args.log_file, 'w')

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    log_string(log, '------------ 配置选项 -------------')
    for k, v in vars(args).items():
        log_string(log, '%s: %s' % (str(k), str(v)))
    log_string(log, '-------------- 结束 ----------------')

    solver = Solver(vars(args))

    solver.train()
    solver.test()