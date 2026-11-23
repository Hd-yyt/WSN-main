# Datasets

## Prepared data

Rain-UK is included in `data/RainfallUK` for a complete training and evaluation example. The separate prepared data package contains all eight datasets. Install it from the repository root:

```bash
python scripts/install_data.py --archive ../WeatherScaleNet_Data.zip
```

The installer verifies the file sizes and SHA-256 hashes before placing files under `data/`. Alternatively, extract the package and merge its `data` directory into the repository root. The paths then match the configuration files under `config/`.

The prepared package contains approximately 1.44 GiB of arrays and metadata before compression. Raw downloads, model checkpoints, and prediction outputs are not needed to run these datasets.

| Task | Configuration | Array shape `(time, locations, features)` | Input / forecast length |
| --- | --- | --- | --- |
| Global Wind | `config/wind.conf` | `(17544, 3850, 1)` | 48 / 24 hours |
| Global Temperature | `config/temp.conf` | `(17544, 3850, 1)` | 48 / 24 hours |
| Rain-AUS | `config/rfaus.conf` | `(9404, 1230, 1)` | 30 / 30 days |
| Rain-CN | `config/rfcn.conf` | `(9404, 1800, 1)` | 30 / 30 days |
| Rain-UK | `config/rfuk.conf` | `(9404, 100, 1)` | 30 / 30 days |
| Rain-USA | `config/rfusa.conf` | `(9404, 1392, 1)` | 30 / 30 days |
| MeteoNet NW | `config/meteonet_nw.conf` | `(26304, 231, 1)` | 48 / 24 hours |
| MeteoNet SE | `config/meteonet_se.conf` | `(26304, 451, 1)` | 48 / 24 hours |

Each configuration specifies the observation array, location metadata, and spatial adjacency file. MeteoNet also includes timestamp arrays and dataset summaries. The regional precipitation arrays use a regular one-degree grid; their metadata lists the grid coordinates.

## Public source datasets

### Global Wind and Temperature

The source observations come from the [NOAA Integrated Surface Database (ISD)](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database). The prepared tasks contain hourly observations from 3,850 stations during 2019 and 2020. Use the prepared package for the station selection and array ordering used by the supplied configurations.

The stored wind and temperature values use a scale of 0.1: multiply wind values by 0.1 to obtain m/s and temperature values by 0.1 to obtain degrees Celsius. MAE and RMSE use the same conversion; MSE uses a factor of 0.01. Training retains the stored scale.

### Regional precipitation

Rain-AUS, Rain-CN, Rain-UK, and Rain-USA are supplied as prepared regional precipitation benchmarks in the separate data package. Rain-UK is also included in this repository. The NPZ arrays and coordinate CSV files can be used directly with the corresponding configurations.

### MeteoNet

MeteoNet is provided by METEO FRANCE:

- [Dataset website and downloads](https://meteonet.umr-cnrm.fr/)
- [Ground-observation documentation](https://meteofrance.github.io/meteonet/english/data/ground-observations/)
- [Official toolbox](https://github.com/meteofrance/meteonet)

The prepared NW and SE tasks use ground-station precipitation observations from 2016 through 2018. Six-minute observations are summed into hourly totals; an hour requires ten valid reports. Stations are selected with at least 80% observation completeness in 2016. The years 2016, 2017, and 2018 form the training, validation, and test splits, respectively. Missing observations are stored as NaN and handled by the observation mask in the data loader.

To download and prepare MeteoNet from the public source, run:

```bash
python scripts/prepare_meteonet.py --zone NW
python scripts/prepare_meteonet.py --zone SE
```

These commands download the source files into `data/MeteoNet/raw`, write prepared arrays into `data/MeteoNet`, and generate the corresponding configuration files. Downloading requires an internet connection and `curl` on the system path. Raw source files require considerably more storage than the prepared arrays. `--download-dir` and `--output-dir` select alternative locations. Use `--help` for the complete options.

MeteoNet is distributed under the [Etalab Open Licence 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/). Attribution: METEO FRANCE, MeteoNet, 2020. The repository's software license does not replace the source dataset's license.

## Array and metadata format

An observation file is an NPZ archive with a `data` array of shape `(T, N, D)`:

- `T`: ordered time steps.
- `N`: stations or grid locations in the same order as the metadata rows.
- `D`: observation features; the supplied tasks have one target feature.

Metadata CSV files contain latitude and longitude columns named `Lat` and `Lng`. Elevation is read from `Altitude`, `altitude`, `Elevation`, `elevation`, or `col3` when available; MeteoNet metadata retains the source station identifier. Adjacency arrays use the same location order. To use another dataset, prepare these files and create a configuration with the matching number of locations and forecasting window.
