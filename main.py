import argparse
import configparser
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lib.data_pipeline import load_data_bundle
from lib.utils import (
    _compute_loss,
    log_string,
    metric,
    parse_bool,
    precipitation_distribution_metrics,
    precipitation_metrics,
    precipitation_stratified_metrics,
)
from models.weatherscalenet import WeatherScaleNet


def parse_float_list(value):
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    if value is None or not str(value).strip():
        return []
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def finite_mean(values, axis):
    finite = np.isfinite(values)
    counts = np.sum(finite, axis=axis)
    sums = np.sum(np.where(finite, values, 0.0), axis=axis)
    output = np.full(np.shape(sums), np.nan, dtype=np.float64)
    return np.divide(sums, counts, out=output, where=counts > 0)


class Solver:
    def __init__(self, config, log):
        self.__dict__.update(config)
        self.log = log
        self.best_epoch = 0
        self.best_val_mae = float("inf")
        self.train_epoch_times = []
        self.val_times = []
        self.fusion_weight_history = []

        if self.device == "auto":
            selected_device = f"cuda:{self.cuda}" if torch.cuda.is_available() and str(self.cuda) != "-1" else "cpu"
        else:
            selected_device = self.device
        self.device = torch.device(selected_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use --device cpu or install a CUDA-enabled PyTorch build.")
        self.pin_memory = bool(self.pin_memory and self.device.type == "cuda")
        self.use_amp = bool(self.amp and self.device.type == "cuda")

        log_string(self.log, "\n------------ Loading data ------------")
        data_bundle = load_data_bundle(
            filepath=self.traffic_file,
            metapath=self.meta_file,
            input_len=self.input_len,
            output_len=self.output_len,
            train_ratio=self.train_ratio,
            test_ratio=self.test_ratio,
            adjpath=self.adj_file,
            recurtimes=self.recur_times,
            sps=self.spa_patchsize,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            spatial_split=self.spatial_split,
            spatial_val_ratio=self.spatial_val_ratio,
            spatial_test_ratio=self.spatial_test_ratio,
            spatial_seed=self.spatial_seed,
            use_missing_mask=self.use_missing_mask,
            rain_quantiles=self.rain_quantiles,
            train_missing_rate=self.train_missing_rate,
            eval_missing_rate=self.eval_missing_rate,
            missing_pattern=self.missing_pattern,
            missing_block_length=self.missing_block_length,
            missing_seed=self.missing_seed,
            log=self.log,
        )
        self.train_loader = data_bundle.train_loader
        self.val_loader = data_bundle.val_loader
        self.test_loader = data_bundle.test_loader
        self.mean = data_bundle.mean
        self.std = data_bundle.std
        self.ori_parts_idx = data_bundle.ori_parts_idx
        self.reo_parts_idx = data_bundle.reo_parts_idx
        self.reo_all_idx = data_bundle.reo_all_idx
        self.train_size = data_bundle.train_size
        self.val_size = data_bundle.val_size
        self.test_size = data_bundle.test_size
        self.input_channels = data_bundle.input_channels
        self.rain_quantile_thresholds = data_bundle.rain_quantile_thresholds
        self.event_thresholds = list(self.rain_thresholds)
        self.event_thresholds.extend(data_bundle.rain_quantile_thresholds.values())
        self.event_thresholds = sorted(set(float(value) for value in self.event_thresholds))
        log_string(self.log, "------------ Data ready ------------\n")

        self.build_model()

    def _sync_device(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _autocast_context(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def build_model(self):
        self.base_model = WeatherScaleNet(
            self.tem_patchsize,
            self.tem_patchnum,
            self.node_num,
            self.spa_patchsize,
            self.spa_patchnum,
            self.layers,
            self.factors,
            self.input_dims,
            self.node_dims,
            self.ori_parts_idx,
            self.reo_parts_idx,
            self.reo_all_idx,
            self.meta_file,
            ablation_mode=self.ablation_mode,
            position_encoder=self.position_encoder,
            use_elevation=self.use_elevation,
            elevation_encoder=self.elevation_encoder,
            branch_mode=self.branch_mode,
            fusion_mode=self.fusion_mode,
            input_channels=self.input_channels,
            feature_seed=self.seed,
            output_len=self.output_len,
        ).to(self.device)

        if self.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.base_model)
        else:
            self.model = self.base_model

        self.optimizer = torch.optim.AdamW(
            self.base_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        decay_1 = max(1, int(self.max_epoch * 0.6))
        decay_2 = max(decay_1 + 1, int(self.max_epoch * 0.85))
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=sorted(set([decay_1, decay_2])),
            gamma=0.5,
        )
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.param_count = sum(p.numel() for p in self.base_model.parameters())

    def _move_batch(self, batch):
        inputs, labels = batch
        inputs = inputs.to(self.device, non_blocking=self.pin_memory)
        labels = labels.to(self.device, non_blocking=self.pin_memory)
        return inputs, labels

    def _summarize_metrics(self, pred, label, split_name, log_steps):
        maes, rmses, mapes, mses = [], [], [], []
        for step in range(pred.shape[1]):
            mae, rmse, mape, mse = metric(pred[:, step], label[:, step])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            mses.append(mse)
            if log_steps:
                log_string(
                    self.log,
                    f"[{split_name}] step {step + 1:02d} | MAE {mae:.4f} | RMSE {rmse:.4f} | MAPE {mape:.4f} | MSE {mse:.4f}",
                )

        mae, rmse, mape, mse = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        mses.append(mse)
        log_string(
            self.log,
            f"[{split_name}] avg | MAE {mae:.4f} | RMSE {rmse:.4f} | MAPE {mape:.4f} | MSE {mse:.4f}",
        )
        return np.array(maes), np.array(rmses), np.array(mapes), np.array(mses)

    def evaluate(self, data_loader, split_name="val", log_steps=False, return_arrays=False):
        self.model.eval()
        preds = []
        labels = []

        self._sync_device()
        start_time = time.perf_counter()
        with torch.inference_mode():
            for batch in data_loader:
                inputs, target = self._move_batch(batch)
                with self._autocast_context():
                    prediction = self.model(inputs)
                preds.append(prediction.float().cpu().numpy())
                labels.append(target.cpu().numpy())
        self._sync_device()
        elapsed = time.perf_counter() - start_time

        pred = np.concatenate(preds, axis=0)
        label = np.concatenate(labels, axis=0)
        metrics = self._summarize_metrics(pred, label, split_name, log_steps)
        if self.is_precipitation:
            for threshold in self.event_thresholds:
                scores = precipitation_metrics(pred, label, threshold)
                summary = " | ".join(
                    f"{name.upper()} {scores[name]:.4f}" for name in ("csi", "pod", "far", "f1", "ets", "hss", "brier")
                )
                log_string(self.log, f"[{split_name}] rain >= {threshold:g} | {summary}")
        if return_arrays:
            return metrics, elapsed, pred, label
        return metrics, elapsed

    def train(self):
        log_string(self.log, "====================== TRAIN MODE ======================")
        patience_counter = 0

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for epoch in range(1, self.max_epoch + 1):
            self.model.train()
            running_loss = 0.0

            self._sync_device()
            epoch_start = time.perf_counter()
            progress = None
            data_iter = self.train_loader
            if self.progress_bar:
                progress = tqdm(
                    self.train_loader,
                    total=len(self.train_loader),
                    desc=f"Epoch {epoch}/{self.max_epoch}",
                    leave=False,
                    mininterval=self.progress_mininterval,
                )
                data_iter = progress

            for step, batch in enumerate(data_iter, start=1):
                inputs, target = self._move_batch(batch)

                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast_context():
                    prediction = self.model(inputs)
                    loss = _compute_loss(
                        target,
                        prediction,
                        loss_type=self.loss_type,
                        event_threshold=self.event_loss_threshold,
                        event_weight=self.event_loss_weight,
                    )

                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += loss.item()
                if (
                    progress is not None
                    and self.progress_update_steps > 0
                    and step % self.progress_update_steps == 0
                ):
                    progress.set_postfix(loss=f"{loss.item():.4f}")

            if progress is not None:
                progress.close()

            self._sync_device()
            epoch_time = time.perf_counter() - epoch_start
            self.train_epoch_times.append(epoch_time)

            avg_loss = running_loss / max(1, len(self.train_loader))
            current_lr = self.optimizer.param_groups[0]["lr"]
            log_string(
                self.log,
                f"epoch {epoch:03d} | lr {current_lr:.6f} | train_loss {avg_loss:.4f} | train_time {epoch_time:.2f}s",
            )
            self.fusion_weight_history.append(self.base_model.get_fusion_weights())

            should_validate = (epoch % self.val_interval == 0) or (epoch == self.max_epoch)
            if should_validate:
                (mae, _, _, _), val_time = self.evaluate(self.val_loader, split_name="val", log_steps=self.verbose_val_steps)
                self.val_times.append(val_time)
                log_string(self.log, f"epoch {epoch:03d} | val_time {val_time:.2f}s")

                current_val_mae = float(mae[-1])
                if current_val_mae < self.best_val_mae:
                    self.best_val_mae = current_val_mae
                    self.best_epoch = epoch
                    patience_counter = 0
                    Path(self.model_file).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.base_model.state_dict(), self.model_file)
                else:
                    patience_counter += 1

                if self.early_stop_patience > 0 and patience_counter >= self.early_stop_patience:
                    log_string(self.log, f"Early stop at epoch {epoch}, best epoch {self.best_epoch}")
                    break

            self.lr_scheduler.step()

        if self.best_epoch == 0:
            self.best_epoch = self.max_epoch
            Path(self.model_file).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.base_model.state_dict(), self.model_file)

        log_string(self.log, f"Best epoch: {self.best_epoch}")

    def test(self):
        log_string(self.log, "====================== TEST MODE ======================")
        self.base_model.load_state_dict(torch.load(self.model_file, map_location=self.device))
        (maes, rmses, mapes, mses), test_time, pred, label = self.evaluate(
            self.test_loader, split_name="test", log_steps=True, return_arrays=True
        )

        peak_memory_mb = 0.0
        if self.device.type == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        avg_train_time = float(np.mean(self.train_epoch_times)) if self.train_epoch_times else 0.0
        avg_val_time = float(np.mean(self.val_times)) if self.val_times else 0.0

        log_string(self.log, "\n==================== Performance Metrics ====================")
        log_string(self.log, f"Batch Size (BS)        : {self.batch_size}")
        log_string(self.log, f"Train Time (avg/epoch) : {avg_train_time:.4f} s")
        log_string(self.log, f"Val Time (avg/run)     : {avg_val_time:.4f} s")
        log_string(self.log, f"Test Time (total)      : {test_time:.4f} s")
        log_string(self.log, f"Peak GPU Memory        : {peak_memory_mb:.2f} MB")
        log_string(self.log, f"Model Parameters       : {self.param_count:,}")
        fusion_weights = self.base_model.get_fusion_weights()
        log_string(self.log, f"Fusion Weights (far/local): {fusion_weights}")
        log_string(self.log, "=============================================================\n")

        event_results = {}
        stratified_results = {}
        distribution_results = {}
        if self.is_precipitation:
            event_results = {
                str(threshold): precipitation_metrics(pred, label, threshold)
                for threshold in self.event_thresholds
            }
            stratified_results = precipitation_stratified_metrics(pred, label, self.event_thresholds)
            distribution_results = precipitation_distribution_metrics(pred, label)

        valid = np.isfinite(label) & np.isfinite(pred)
        abs_error = np.where(valid, np.abs(pred - label), np.nan)
        sq_error = np.where(valid, (pred - label) ** 2, np.nan)
        sample_mae = finite_mean(abs_error, axis=(1, 2, 3)).astype(np.float32)
        sample_mse = finite_mean(sq_error, axis=(1, 2, 3)).astype(np.float32)
        horizon_mae = finite_mean(abs_error, axis=(0, 2, 3)).astype(np.float32)
        node_mae = finite_mean(abs_error, axis=(0, 1, 3)).astype(np.float32)

        if self.artifact_file:
            artifact_dir = os.path.dirname(self.artifact_file)
            if artifact_dir:
                os.makedirs(artifact_dir, exist_ok=True)
            np.savez_compressed(
                self.artifact_file,
                sample_mae=sample_mae,
                sample_mse=sample_mse,
                horizon_mae=horizon_mae,
                node_mae=node_mae,
                fusion_weights=np.asarray(
                    [[np.nan, np.nan] if weights is None else weights for weights in fusion_weights], dtype=np.float32
                ),
                fusion_weight_history=np.asarray(
                    [
                        [[np.nan, np.nan] if weights is None else weights for weights in epoch_weights]
                        for epoch_weights in self.fusion_weight_history
                    ],
                    dtype=np.float32,
                ),
            )

        result = {
            "dataset": self.dataset_name,
            "metric_value_scale_to_physical": 0.1 if self.dataset_name in ("wind", "temp") else 1.0,
            "metric_squared_scale_to_physical": 0.01 if self.dataset_name in ("wind", "temp") else 1.0,
            "study_name": self.study_name,
            "seed": self.seed,
            "spatial_seed": self.spatial_seed,
            "spatial_split": self.spatial_split,
            "train_missing_rate": self.train_missing_rate,
            "eval_missing_rate": self.eval_missing_rate,
            "missing_pattern": self.missing_pattern,
            "loss_type": self.loss_type,
            "event_loss_threshold": self.event_loss_threshold,
            "event_loss_weight": self.event_loss_weight,
            "position_encoder": self.position_encoder,
            "use_elevation": self.use_elevation,
            "elevation_encoder": self.elevation_encoder,
            "branch_mode": self.branch_mode,
            "fusion_mode": self.fusion_mode,
            "ablation_mode": self.ablation_mode,
            "factors": self.factors,
            "recur_times": self.recur_times,
            "spa_patchsize": self.spa_patchsize,
            "spa_patchnum": self.spa_patchnum,
            "effective_partition_count": self.spa_patchnum // self.factors,
            "effective_partition_size": self.spa_patchsize * self.factors,
            "metrics": {
                "mae": float(maes[-1]),
                "rmse": float(rmses[-1]),
                "mape": float(mapes[-1]),
                "mse": float(mses[-1]),
            },
            "event_metrics": event_results,
            "stratified_rainfall_metrics": stratified_results,
            "precipitation_distribution": distribution_results,
            "rain_quantile_thresholds": self.rain_quantile_thresholds,
            "fusion_weights_far_local": fusion_weights,
            "fusion_weight_history_far_local": self.fusion_weight_history,
            "parameters": self.param_count,
            "peak_gpu_memory_mb": peak_memory_mb,
            "test_time_seconds": test_time,
            "average_train_epoch_seconds": avg_train_time,
            "average_validation_seconds": avg_val_time,
            "batch_size": self.batch_size,
            "epochs_executed": len(self.train_epoch_times),
            "best_epoch": self.best_epoch,
            "training_max_epoch": self.max_epoch,
            "artifact_file": self.artifact_file,
        }
        if self.result_file:
            result_dir = os.path.dirname(self.result_file)
            if result_dir:
                os.makedirs(result_dir, exist_ok=True)
            with open(self.result_file, "w", encoding="utf-8") as result_handle:
                json.dump(result, result_handle, indent=2, ensure_ascii=False, allow_nan=True)

        return maes, rmses, mapes, mses


def build_parser(config):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=str, required=True, help="config file path")
    parser.add_argument(
        "--run_mode",
        choices=["train-test", "test-only"],
        default="train-test",
        help="Use test-only to recompute metrics from an existing checkpoint.",
    )

    parser.add_argument(
        "--device", default="auto", help="Execution device: auto, cpu, cuda, or cuda:<index>."
    )
    parser.add_argument("--cuda", type=str, default=config["train"]["cuda"], help="GPU index used with --device auto; -1 selects CPU.")
    parser.add_argument("--seed", type=int, default=config["train"]["seed"])
    parser.add_argument("--batch_size", type=int, default=config["train"]["batch_size"])
    parser.add_argument("--max_epoch", type=int, default=config["train"]["max_epoch"])
    parser.add_argument("--learning_rate", type=float, default=config["train"]["learning_rate"])
    parser.add_argument("--weight_decay", type=float, default=config["train"]["weight_decay"])
    parser.add_argument("--num_workers", type=int, default=config.getint("train", "num_workers", fallback=0))
    parser.add_argument("--prefetch_factor", type=int, default=config.getint("train", "prefetch_factor", fallback=2))
    parser.add_argument("--val_interval", type=int, default=config.getint("train", "val_interval", fallback=2))
    parser.add_argument("--early_stop_patience", type=int, default=config.getint("train", "early_stop_patience", fallback=5))
    parser.add_argument("--grad_clip", type=float, default=config.getfloat("train", "grad_clip", fallback=0.0))
    parser.add_argument("--amp", type=str, default=config.get("train", "amp", fallback="true"))
    parser.add_argument("--pin_memory", type=str, default=config.get("train", "pin_memory", fallback="true"))
    parser.add_argument("--compile_model", type=str, default=config.get("train", "compile_model", fallback="false"))
    parser.add_argument("--deterministic", type=str, default=config.get("train", "deterministic", fallback="false"))
    parser.add_argument("--allow_tf32", type=str, default=config.get("train", "allow_tf32", fallback="true"))
    parser.add_argument("--verbose_val_steps", type=str, default=config.get("train", "verbose_val_steps", fallback="false"))
    parser.add_argument("--progress_bar", type=str, default=config.get("train", "progress_bar", fallback="false"))
    parser.add_argument("--progress_update_steps", type=int, default=config.getint("train", "progress_update_steps", fallback=50))
    parser.add_argument("--progress_mininterval", type=float, default=config.getfloat("train", "progress_mininterval", fallback=1.0))
    parser.add_argument(
        "--loss_type",
        choices=["huber", "weighted-huber", "mae", "mse"],
        default=config.get("train", "loss_type", fallback="huber"),
    )
    parser.add_argument(
        "--event_loss_threshold",
        type=float,
        default=config.getfloat("train", "event_loss_threshold", fallback=5.0),
    )
    parser.add_argument(
        "--event_loss_weight",
        type=float,
        default=config.getfloat("train", "event_loss_weight", fallback=4.0),
    )

    parser.add_argument("--input_len", type=int, default=config["data"]["input_len"])
    parser.add_argument("--output_len", type=int, default=config["data"]["output_len"])
    parser.add_argument("--train_ratio", type=float, default=config["data"]["train_ratio"])
    parser.add_argument("--val_ratio", type=float, default=config["data"]["val_ratio"])
    parser.add_argument("--test_ratio", type=float, default=config["data"]["test_ratio"])
    parser.add_argument(
        "--spatial_split",
        choices=["none", "random", "block"],
        default=config.get("data", "spatial_split", fallback="none"),
    )
    parser.add_argument(
        "--spatial_val_ratio", type=float, default=config.getfloat("data", "spatial_val_ratio", fallback=0.1)
    )
    parser.add_argument(
        "--spatial_test_ratio", type=float, default=config.getfloat("data", "spatial_test_ratio", fallback=0.2)
    )
    parser.add_argument(
        "--spatial_seed", type=int, default=config.getint("data", "spatial_seed", fallback=1)
    )
    parser.add_argument(
        "--use_missing_mask", type=str, default=config.get("data", "use_missing_mask", fallback="false")
    )
    parser.add_argument(
        "--is_precipitation", type=str, default=config.get("data", "is_precipitation", fallback="false")
    )
    parser.add_argument(
        "--rain_thresholds", type=parse_float_list, default=parse_float_list(config.get("data", "rain_thresholds", fallback=""))
    )
    parser.add_argument(
        "--rain_quantiles",
        type=parse_float_list,
        default=parse_float_list(config.get("data", "rain_quantiles", fallback="0.90,0.95,0.99")),
    )
    parser.add_argument(
        "--train_missing_rate", type=float, default=config.getfloat("data", "train_missing_rate", fallback=0.0)
    )
    parser.add_argument(
        "--eval_missing_rate", type=float, default=config.getfloat("data", "eval_missing_rate", fallback=0.0)
    )
    parser.add_argument(
        "--missing_pattern",
        choices=["random", "block"],
        default=config.get("data", "missing_pattern", fallback="random"),
    )
    parser.add_argument(
        "--missing_block_length", type=int, default=config.getint("data", "missing_block_length", fallback=24)
    )
    parser.add_argument(
        "--missing_seed", type=int, default=config.getint("data", "missing_seed", fallback=1)
    )

    parser.add_argument("--layers", type=int, default=config["param"]["layers"])
    parser.add_argument("--tem_patchsize", type=int, default=config["param"]["tps"])
    parser.add_argument("--tem_patchnum", type=int, default=config["param"]["tpn"])
    parser.add_argument("--factors", type=int, default=config["param"]["factors"])
    parser.add_argument("--recur_times", type=int, default=config["param"]["recur"])
    parser.add_argument("--spa_patchsize", type=int, default=config["param"]["sps"])
    parser.add_argument("--spa_patchnum", type=int, default=config["param"]["spn"])
    parser.add_argument("--node_num", type=int, default=config["param"]["nodes"])
    parser.add_argument("--input_dims", type=int, default=config["param"]["id"])
    parser.add_argument("--node_dims", type=int, default=config["param"]["nd"])
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default=config.get("param", "ablation_mode", fallback="full"),
        choices=["full", "history-only", "GeoCLIP-only", "elevation-only", "local-only", "far-only"],
    )
    parser.add_argument(
        "--position_encoder",
        choices=["none", "geoclip", "random-geoclip", "raw-mlp", "sincos", "rff", "station-id"],
        default=config.get("param", "position_encoder", fallback="geoclip"),
    )
    parser.add_argument(
        "--use_elevation", type=str, default=config.get("param", "use_elevation", fallback="true")
    )
    parser.add_argument(
        "--elevation_encoder",
        choices=["none", "discrete", "raw-mlp"],
        default=config.get("param", "elevation_encoder", fallback="discrete"),
    )
    parser.add_argument(
        "--branch_mode",
        choices=["full", "local", "far"],
        default=config.get("param", "branch_mode", fallback="full"),
    )
    parser.add_argument(
        "--fusion_mode",
        choices=["learned", "fixed", "concat"],
        default=config.get("param", "fusion_mode", fallback="learned"),
    )

    parser.add_argument("--traffic_file", default=config["file"]["traffic"])
    parser.add_argument("--meta_file", default=config["file"]["meta"])
    parser.add_argument("--adj_file", default=config["file"]["adj"])
    parser.add_argument("--model_file", default=config["file"]["model"])
    parser.add_argument("--log_file", default=config["file"]["log"])
    parser.add_argument("--result_file", default=config.get("file", "result", fallback=""))
    parser.add_argument("--artifact_file", default=config.get("file", "artifact", fallback=""))
    parser.add_argument(
        "--dataset_name",
        default=config.get("file", "dataset_name", fallback=os.path.basename(config["file"]["traffic"]).split(".")[0]),
    )
    parser.add_argument("--study_name", default="main")
    return parser


def select_evaluation_paths(args, argv):
    """Keep evaluation outputs separate from training files by default."""
    if args.run_mode != "test-only":
        return
    explicit_options = {item.split("=", 1)[0] for item in argv if item.startswith("--")}
    for field, filename in (
        ("log_file", "test.log"),
        ("result_file", "test_result.json"),
        ("artifact_file", "test_diagnostics.npz"),
    ):
        if f"--{field}" not in explicit_options:
            configured_path = getattr(args, field)
            parent = Path(configured_path).parent if configured_path else Path(args.model_file).parent
            setattr(args, field, str(parent / filename))


def set_runtime_flags(args):
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str)
    bootstrap_args, _ = bootstrap.parse_known_args()
    config_path = bootstrap_args.config
    if config_path is None:
        if "--help" in sys.argv or "-h" in sys.argv:
            config_path = Path(__file__).resolve().parent / "config" / "rfuk.conf"
        else:
            bootstrap.error("the following argument is required: --config")

    config = configparser.ConfigParser()
    if not config.read(config_path):
        bootstrap.error(f"Cannot read configuration file: {config_path}")

    parser = build_parser(config)
    args = parser.parse_args()
    select_evaluation_paths(args, sys.argv[1:])
    if args.max_epoch < 1 or args.batch_size < 1 or args.val_interval < 1:
        parser.error("max_epoch, batch_size, and val_interval must be positive.")

    args.amp = parse_bool(args.amp, True)
    args.pin_memory = parse_bool(args.pin_memory, True)
    args.compile_model = parse_bool(args.compile_model, False)
    args.deterministic = parse_bool(args.deterministic, False)
    args.allow_tf32 = parse_bool(args.allow_tf32, True)
    args.verbose_val_steps = parse_bool(args.verbose_val_steps, False)
    args.progress_bar = parse_bool(args.progress_bar, False)
    args.use_missing_mask = parse_bool(args.use_missing_mask, False)
    args.is_precipitation = parse_bool(args.is_precipitation, False)
    args.use_elevation = parse_bool(args.use_elevation, True)
    if not args.use_elevation or args.elevation_encoder == "none":
        args.use_elevation = False
        args.elevation_encoder = "none"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_runtime_flags(args)

    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log_file, "w", encoding="utf-8") as log:
        log_string(log, "------------ Config ------------")
        for key, value in vars(args).items():
            log_string(log, f"{key}: {value}")
        log_string(log, "------------ Start ------------")

        solver = Solver(vars(args), log)
        log_string(log, f"Current ablation mode: {solver.base_model.ablation_mode}")
        if args.run_mode == "train-test":
            solver.train()
        solver.test()
