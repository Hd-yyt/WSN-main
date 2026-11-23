# Training and evaluation

Run commands from the repository root. `python main.py --help` lists all options; settings supplied on the command line override the selected `.conf` file.

## Forecasting settings

Global and regional gridded tasks use chronological training, validation, and test partitions in a 7:1:2 ratio. MeteoNet uses calendar years 2016, 2017, and 2018, respectively. Sliding windows stay within their time partition.

`--spatial_split none` evaluates the same station set across the time partitions. `random` and `block` additionally hold out stations, with default validation and test fractions of 0.1 and 0.2. Training inputs and targets exclude held-out stations. At evaluation, recent observations from the station network are available and metrics use the selected target stations. `block` orders stations along a seeded geographic projection. Use `--use_missing_mask true` for a spatial holdout run.

## Default global training configurations

| Setting | Wind | Temperature |
| --- | --- | --- |
| Batch size | 8 | 8 |
| Maximum epochs | 20 | 20 |
| Learning rate | 0.002 | 0.002 |
| Weight decay | 0.001 | 0.0001 |
| Spatial layers | 5 | 4 |
| Gradient clipping | Disabled | Disabled |

Validation runs every two epochs, with an early-stopping patience of five validation checks. CUDA runs enable mixed precision and TF32 by default. The temperature branch-comparison preset uses batch size 16; the main preset uses batch size 8.

## Model options

| Option | Values | Purpose |
| --- | --- | --- |
| `--branch_mode` | `full`, `local`, `far` | Both directions, within-region only, or cross-region only |
| `--fusion_mode` | `learned`, `fixed`, `concat` | How the two directions are combined |
| `--position_encoder` | `geoclip`, `raw-mlp`, `sincos`, `rff`, `none`, `station-id`, `random-geoclip` | Coordinate representation or station-identity control |
| `--elevation_encoder` | `discrete`, `raw-mlp`, `none` | Elevation representation |
| `--use_missing_mask` | `true`, `false` | Add an observed-value indicator to the input |
| `--factors` | Divisor of `spn` | Merge tree leaves to set the exchange layout |

`geoclip` uses fixed location features followed by a learned projection. `random-geoclip` uses fixed random 512-dimensional features as a control. Elevation features are zero when metadata do not contain elevation. The default decoder predicts all requested horizons jointly.

## Repeated runs

The runner accepts `main`, `branch`, `spatial`, `position`, `elevation`, `partition`, `rainfall`, `rainfall-loss`, `missingness`, and `meteonet` presets. Choose datasets and seeds explicitly:

```bash
python experiments/run_experiments.py --preset spatial --datasets wind,temp --seeds 1,2,3,4,5
python experiments/run_experiments.py --preset position --datasets wind --position-encoders none,raw-mlp,geoclip --seeds 1
```

Runs are placed in separate `outputs/` directories and completed runs with the same epoch budget are skipped. Use `--rerun` after changing a configuration, or choose a new `--output-root`. `--max-epoch` overrides the epoch budget; `--max-runs` limits the number of runs.

Simple persistence, seasonal-lag, and station-climatology baselines are available through:

```bash
python experiments/run_naive_baselines.py --datasets rfuk --methods persistence,seasonal-lag
```

## Outputs and units

A training run saves the checkpoint selected by validation MAE. `--run_mode test-only` loads that checkpoint and evaluates the test split, writing `test.log`, `test_result.json`, and `test_diagnostics.npz` without replacing the training outputs. Keep the same model, feature, and seed settings when loading a checkpoint.

- `model.pt`: model parameters.
- `train.log`: configuration, optimization progress, and evaluation metrics.
- `result.json`: aggregate errors, precipitation event metrics where applicable, and runtime measurements.
- `diagnostics.npz`: errors by horizon, sample, and station, and fusion weights.

Metrics use the values stored in the dataset arrays. Global wind and temperature arrays are in tenths of their physical units: multiply MAE and RMSE by 0.1, and MSE by 0.01, to obtain m/s and degrees Celsius scales. The corresponding multipliers are included in `result.json`. Rainfall errors use the supplied precipitation units.

Missing targets are excluded from loss and error averages. Precipitation event metrics include CSI at configured thresholds. Runtime measurements depend on the device and batch size supplied to the run.

## Validation

```bash
python scripts/smoke_test.py --verbose
python -m unittest discover -s tests -v
```

These checks exercise model gradients, checkpoint loading, spatial holdout masks, and missing-value metrics on small CPU inputs.
