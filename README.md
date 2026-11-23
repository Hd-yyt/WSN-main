# WeatherScaleNet

PyTorch implementation of **WeatherScaleNet: Collaborative Station-Level Forecasting for Large Irregular Weather Station Networks**.

WeatherScaleNet forecasts individual stations by combining a shared history and geographic encoder with within-region and cross-region exchange. Stations retain separate representations throughout the network, and a shared decoder produces their multistep forecasts.

## Installation

Use Python 3.11. Install PyTorch and torchvision with a build suitable for your device from the [PyTorch installation guide](https://pytorch.org/get-started/locally/). The CPU commands below provide a tested environment:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For GPU training, install a PyTorch CUDA build that supports your GPU instead of the CPU build above. Training uses CUDA when available. Select a device with `--device cuda:0` or `--device cpu`.

The default geographic encoder uses the pretrained location weights included in GeoCLIP 1.2.0. The installation check below uses a small coordinate MLP and synthetic data, so it does not need geographic model downloads.

## Quick start

Run the CPU installation check:

```bash
python scripts/smoke_test.py
```

The compact **RainfallUK** dataset is included. From the repository root, check the files and train the default model:

```bash
python scripts/check_data.py --config config/rfuk.conf
python main.py --config config/rfuk.conf
```

Results are saved under `outputs/rfuk/`: the best validation checkpoint, training log, test metrics, and diagnostic arrays. To evaluate the saved checkpoint:

```bash
python main.py --config config/rfuk.conf --run_mode test-only
```

## Datasets

| Task | Configuration | Input / forecast length | Data |
| --- | --- | --- | --- |
| Global Wind | `config/wind.conf` | 48 / 24 hours | Prepared data package; NOAA ISD source |
| Global Temperature | `config/temp.conf` | 48 / 24 hours | Prepared data package; NOAA ISD source |
| Rain-AUS | `config/rfaus.conf` | 30 / 30 days | Prepared data package |
| Rain-CN | `config/rfcn.conf` | 30 / 30 days | Prepared data package |
| Rain-UK | `config/rfuk.conf` | 30 / 30 days | Included |
| Rain-USA | `config/rfusa.conf` | 30 / 30 days | Prepared data package |
| MeteoNet NW | `config/meteonet_nw.conf` | 48 / 24 hours | Prepared data package or official download |
| MeteoNet SE | `config/meteonet_se.conf` | 48 / 24 hours | Prepared data package or official download |

Large datasets are stored separately from the Git repository. See [docs/DATASETS.md](docs/DATASETS.md) for sources, preparation, and file formats. Install the prepared data archive with:

```bash
python scripts/install_data.py --archive /path/to/WeatherScaleNet_Data.zip
python scripts/check_data.py --all
```

## Experiments

Configuration files specify the training settings for each task. Command-line options override them. For example:

```bash
python main.py --config config/wind.conf --seed 1 --device cuda:0
python main.py --config config/wind.conf --spatial_split random --spatial_seed 1 --use_missing_mask true
python main.py --config config/wind.conf --branch_mode local
python main.py --config config/wind.conf --position_encoder raw-mlp
```

Use separate output paths for independent runs, or use the experiment runner to create a directory for each configuration:

```bash
python experiments/run_experiments.py --preset main --datasets wind,temp --seeds 1,2,3,4,5 --dry-run
python experiments/run_experiments.py --preset main --datasets wind,temp --seeds 1,2,3,4,5
python experiments/run_experiments.py --preset branch --datasets wind,temp --seeds 1,2,3,4,5
```

`--dry-run` prints commands without starting training. Other presets cover spatial holdout, coordinate and elevation representations, partition size, and missing observations. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for options and metric units.

## Repository layout

```text
config/                 Dataset configurations
models/weatherscalenet.py
lib/                    Data loading, spatial organization, and metrics
experiments/            Experiment runner and simple forecasting baselines
scripts/                Data preparation, installation, and checks
tests/                  Lightweight CPU tests
data/RainfallUK/         Included example dataset
docs/                   Data and experiment documentation
main.py                 Training and evaluation entry point
```

## Acknowledgments and license

The implementation builds on spatial organization utilities from [PatchSTG](https://github.com/LMissher/PatchSTG), the [GeoCLIP](https://github.com/VicenteVivan/geo-clip) location encoder, and [timm](https://github.com/huggingface/pytorch-image-models). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE](LICENSE). Dataset terms are separate from the code license.
