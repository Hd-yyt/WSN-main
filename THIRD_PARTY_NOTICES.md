# Third-party acknowledgments

## PatchSTG

The spatial partition helpers and the implementation foundation of this project are adapted from [PatchSTG](https://github.com/LMissher/PatchSTG), distributed under the MIT License. The original copyright notice is retained in [LICENSE](LICENSE).

WeatherScaleNet uses this spatial organization with shared geographic station representations and within-region and cross-region forecasting updates.

## GeoCLIP

The coordinate branch uses the location encoder from [GeoCLIP](https://github.com/VicenteVivan/geo-clip). GeoCLIP is installed as a dependency; its source and model weights retain their original terms. WeatherScaleNet computes fixed location features and trains its own projection.

## Other dependencies

Attention and feed-forward layers use [timm](https://github.com/huggingface/pytorch-image-models). PyTorch, NumPy, pandas, scikit-learn, and tqdm retain their respective licenses.

## Datasets

The code license does not replace the terms associated with the source datasets. Public data sources and prepared benchmark files are described in [docs/DATASETS.md](docs/DATASETS.md).
