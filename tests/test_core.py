"""Small CPU checks using synthetic observations and a local coordinate encoder."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from lib.data_pipeline import load_data_bundle
from lib.utils import (
    _compute_loss,
    construct_geo_adj,
    kdTree,
    metric,
    precipitation_metrics,
    read_meta1,
    reorderData,
)
from models.weatherscalenet import WeatherScaleNet


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.metadata_path = self.directory / "stations.csv"
        with self.metadata_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Lat", "Lng"])
            writer.writerows(zip(np.linspace(40, 48, 8), np.linspace(-5, 5, 8)))

        locations = read_meta1(self.metadata_path)
        parts, maximum_size = kdTree(locations, 2, 0)
        self.indices = reorderData(parts, maximum_size, construct_geo_adj(locations), 2)
        self.inputs = torch.randn(2, 4, 8, 1)

    def make_model(self, branch_mode="full"):
        return WeatherScaleNet(
            tem_patchsize=2,
            tem_patchnum=2,
            node_num=8,
            spa_patchsize=2,
            spa_patchnum=4,
            layers=2,
            factors=1,
            input_dims=8,
            node_dims=4,
            ori_parts_idx=self.indices[0],
            reo_parts_idx=self.indices[1],
            reo_all_idx=self.indices[2],
            meta_file=self.metadata_path,
            position_encoder="raw-mlp",
            branch_mode=branch_mode,
            output_len=3,
        )

    def test_forward_backward_without_elevation(self):
        model = self.make_model()
        self.assertFalse(bool(model.has_altitude))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        before = model.input_st_fc.weight.detach().clone()
        prediction = model(self.inputs)
        self.assertEqual(tuple(prediction.shape), (2, 3, 8, 1))
        self.assertTrue(bool(torch.isfinite(prediction).all()))
        target = torch.randn_like(prediction)
        _compute_loss(target, prediction).backward()
        self.assertTrue(bool(torch.isfinite(model.input_st_fc.weight.grad).all()))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.input_st_fc.weight))

    def test_branch_selection_gradients(self):
        for branch_mode in ("local", "far", "full"):
            with self.subTest(branch_mode=branch_mode):
                model = self.make_model(branch_mode)
                prediction = model(self.inputs)
                prediction.square().mean().backward()
                block = model.spa_encoder[0]
                local_gradient = block.sattn.qkv.weight.grad
                far_gradient = block.nattn.qkv.weight.grad
                self.assertEqual(local_gradient is not None, branch_mode != "far")
                self.assertEqual(far_gradient is not None, branch_mode != "local")
                for gradient in (local_gradient, far_gradient):
                    if gradient is not None:
                        self.assertTrue(bool(torch.isfinite(gradient).all()))
                        self.assertGreater(float(gradient.abs().sum()), 0)

    def test_checkpoint_roundtrip(self):
        model = self.make_model().eval()
        with torch.no_grad():
            expected = model(self.inputs)
        checkpoint_path = self.directory / "model.pt"
        torch.save(model.state_dict(), checkpoint_path)
        restored_model = self.make_model().eval()
        restored_model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
        with torch.no_grad():
            actual = restored_model(self.inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_spatial_holdout_inputs_and_targets(self):
        data_path = self.directory / "observations.npz"
        values = np.arange(80 * 8, dtype=np.float32).reshape(80, 8, 1) + 1
        np.savez_compressed(data_path, data=values)
        for split in ("random", "block"):
            with self.subTest(split=split):
                bundle = load_data_bundle(
                    filepath=data_path,
                    metapath=self.metadata_path,
                    input_len=4,
                    output_len=2,
                    train_ratio=0.6,
                    test_ratio=0.2,
                    adjpath=self.directory / "adjacency.npy",
                    recurtimes=2,
                    sps=2,
                    batch_size=2,
                    spatial_split=split,
                    spatial_val_ratio=0.25,
                    spatial_test_ratio=0.25,
                    use_missing_mask=True,
                    rain_quantiles=(),
                )
                masks = [bundle.train_node_mask, bundle.val_node_mask, bundle.test_node_mask]
                np.testing.assert_array_equal(np.sum(masks, axis=0), np.ones(8))
                self.assertTrue(all(mask.any() for mask in masks))
                train_inputs, train_targets = bundle.train_loader.dataset.tensors
                held_out = torch.from_numpy(~bundle.train_node_mask)
                self.assertTrue(bool((train_inputs[:, :, held_out, :] == 0).all()))
                self.assertTrue(bool(torch.isnan(train_targets[:, :, held_out, :]).all()))
                for loader, target_nodes in (
                    (bundle.val_loader, bundle.val_node_mask),
                    (bundle.test_loader, bundle.test_node_mask),
                ):
                    inputs, targets = loader.dataset.tensors
                    selected = torch.from_numpy(target_nodes)
                    self.assertTrue(bool((inputs[..., 1] == 1).all()))
                    self.assertTrue(bool(torch.isfinite(targets[:, :, selected, :]).all()))
                    self.assertTrue(bool(torch.isnan(targets[:, :, ~selected, :]).all()))

    def test_nan_aware_metrics_and_loss(self):
        prediction = np.array([2, 4, 1000, 1], dtype=np.float32)
        target = np.array([1, 2, np.nan, 0], dtype=np.float32)
        mae, rmse, mape, mse = metric(prediction, target)
        self.assertAlmostEqual(mae, 4 / 3, places=5)
        self.assertAlmostEqual(mse, 2, places=5)
        self.assertAlmostEqual(rmse, np.sqrt(2), places=5)
        self.assertTrue(np.isfinite(mape))
        scores = precipitation_metrics(
            np.array([1, 8, 7, 999]), np.array([0, 2, 6, np.nan]), threshold=5
        )
        self.assertEqual(scores["hits"], 1)
        self.assertEqual(scores["false_alarms"], 1)
        self.assertEqual(scores["csi"], 0.5)
        prediction_tensor = torch.tensor(prediction, requires_grad=True)
        loss = _compute_loss(torch.tensor(target), prediction_tensor)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertEqual(float(prediction_tensor.grad[2]), 0)
        self.assertTrue(bool(torch.isfinite(prediction_tensor.grad).all()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
