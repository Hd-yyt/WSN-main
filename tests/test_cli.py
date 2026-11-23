"""Exercise CPU training and checkpoint evaluation through the public CLI."""

import argparse
import configparser
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from main import select_evaluation_paths


ROOT = Path(__file__).resolve().parents[1]


class CommandLineTests(unittest.TestCase):
    def test_explicit_evaluation_paths_take_precedence(self):
        args = argparse.Namespace(
            run_mode="test-only",
            model_file="outputs/model.pt",
            log_file="custom.log",
            result_file="custom.json",
            artifact_file="custom.npz",
        )
        select_evaluation_paths(
            args,
            ["--log_file=custom.log", "--result_file", "custom.json", "--artifact_file=custom.npz"],
        )
        self.assertEqual(args.log_file, "custom.log")
        self.assertEqual(args.result_file, "custom.json")
        self.assertEqual(args.artifact_file, "custom.npz")

    def test_cpu_training_then_evaluation_preserves_training_files(self):
        output_root = ROOT / "outputs"
        output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="cli-test-", dir=output_root) as temporary:
            directory = Path(temporary)
            rng = np.random.default_rng(7)
            observations = rng.normal(size=(80, 8, 1)).astype(np.float32)
            np.savez_compressed(directory / "observations.npz", data=observations)
            np.save(directory / "adjacency.npy", np.eye(8, dtype=np.float32))
            with (directory / "stations.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Lat", "Lng"])
                writer.writerows(zip(np.linspace(40, 48, 8), np.linspace(-5, 5, 8)))

            config = configparser.ConfigParser()
            config.read_dict({
                "file": {
                    "traffic": str(directory / "observations.npz"),
                    "meta": str(directory / "stations.csv"),
                    "adj": str(directory / "adjacency.npy"),
                    "model": str(directory / "model.pt"),
                    "log": str(directory / "train.log"),
                    "result": str(directory / "result.json"),
                    "artifact": str(directory / "diagnostics.npz"),
                    "dataset_name": "synthetic",
                },
                "data": {
                    "input_len": "4", "output_len": "2",
                    "train_ratio": "0.7", "val_ratio": "0.1", "test_ratio": "0.2",
                },
                "train": {
                    "cuda": "0", "max_epoch": "1", "batch_size": "16",
                    "learning_rate": "0.001", "weight_decay": "0.001", "seed": "7",
                    "amp": "false", "val_interval": "1", "num_workers": "0",
                },
                "param": {
                    "factors": "1", "layers": "1", "tps": "2", "tpn": "2",
                    "recur": "2", "sps": "2", "spn": "4", "nodes": "8",
                    "id": "8", "nd": "4", "position_encoder": "raw-mlp",
                    "use_elevation": "false",
                },
            })
            config_path = directory / "experiment.conf"
            with config_path.open("w", encoding="utf-8") as stream:
                config.write(stream)
            command = [sys.executable, str(ROOT / "main.py"), "--config", str(config_path), "--device", "cpu"]
            environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            training = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
            self.assertEqual(training.returncode, 0, training.stdout + training.stderr)
            paths = [directory / name for name in ("model.pt", "train.log", "result.json", "diagnostics.npz")]
            hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
            training_result = json.loads((directory / "result.json").read_text(encoding="utf-8"))

            evaluation = subprocess.run(
                [*command, "--run_mode", "test-only"],
                cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(evaluation.returncode, 0, evaluation.stdout + evaluation.stderr)
            for path, expected_hash in hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected_hash, str(path))
            self.assertTrue((directory / "test.log").is_file())
            self.assertTrue((directory / "test_diagnostics.npz").is_file())
            test_result = json.loads((directory / "test_result.json").read_text(encoding="utf-8"))
            for metric in ("mae", "rmse", "mse", "mape"):
                self.assertAlmostEqual(training_result["metrics"][metric], test_result["metrics"][metric], places=6)


if __name__ == "__main__":
    unittest.main()
