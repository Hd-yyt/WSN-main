"""Download and prepare MeteoNet rain-gauge observations.

The output follows WeatherScaleNet's [time, station, feature] NPZ convention.
Missing observations remain NaN and are accompanied by an observation mask at
training time; they are never interpreted as zero rainfall.
"""

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


BASE_URL = "https://meteonet.umr-cnrm.fr/dataset/data/{zone}/ground_stations/{zone}_ground_stations_{year}.tar.gz"
YEARS = (2016, 2017, 2018)
REQUIRED_COLUMNS = ("number_sta", "lat", "lon", "height_sta", "date", "precip")


def _http_opener(proxy=None):
    proxies = {"http": proxy, "https": proxy} if proxy else {}
    return urllib.request.build_opener(urllib.request.ProxyHandler(proxies))


def _download_range(url, segment_path, start, end, proxy=None):
    expected_size = end - start + 1
    completed = segment_path.stat().st_size if segment_path.exists() else 0
    if completed > expected_size:
        raise IOError(f"Oversized partial segment: {segment_path}")
    # The MeteoNet host serves short ranges much faster and more reliably than
    # a single range spanning hundreds of MiB. Keep each request bounded while
    # maintaining one contiguous, resumable file per worker.
    request_size = 256 * 1024
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for reliable segmented MeteoNet downloads.")
    chunk_path = Path(f"{segment_path}.chunk")
    while completed < expected_size:
        range_start = start + completed
        range_end = min(end, range_start + request_size - 1)
        requested = range_end - range_start + 1
        command = [
            curl,
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "3",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--max-time",
            "120",
            "--range",
            f"{range_start}-{range_end}",
            "--output",
            str(chunk_path),
        ]
        if proxy:
            command.extend(["--proxy", proxy])
        else:
            command.extend(["--noproxy", "*"])
        command.append(url)
        subprocess.run(command, check=True)
        received = chunk_path.stat().st_size
        if received != requested:
            raise IOError(
                f"Incomplete request for {segment_path.name}: got {received} of {requested} bytes"
            )
        with segment_path.open("ab") as handle, chunk_path.open("rb") as chunk:
            shutil.copyfileobj(chunk, handle, length=64 * 1024)
        chunk_path.unlink()
        completed += received
    actual_size = segment_path.stat().st_size
    if actual_size != expected_size:
        raise IOError(
            f"Incomplete segment {segment_path.name}: got {actual_size} of {expected_size} bytes"
        )


def download(url, destination, parallel_segments=1, proxy=None):
    if destination.exists():
        print(f"Using cached {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if parallel_segments > 1:
        head = urllib.request.Request(url, method="HEAD")
        with _http_opener(proxy).open(head, timeout=120) as response:
            length = response.headers.get("Content-Length")
        if not length or not length.isdigit():
            raise IOError(f"Cannot determine remote size for segmented download: {url}")
        total = int(length)
        if offset > total:
            raise IOError(f"Partial archive is larger than remote file: {partial}")
        if offset < total:
            remaining = total - offset
            segment_size = math.ceil(remaining / parallel_segments)
            ranges = []
            for index in range(parallel_segments):
                start = offset + index * segment_size
                if start >= total:
                    break
                end = min(total - 1, start + segment_size - 1)
                segment_path = Path(f"{partial}.seg{index:02d}")
                ranges.append((segment_path, start, end))
            print(
                f"Downloading {destination.name} in {len(ranges)} resumable ranges "
                f"({offset / 1024**2:.1f}/{total / 1024**2:.1f} MiB already complete)"
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(ranges)) as executor:
                futures = {
                    executor.submit(
                        _download_range, url, segment_path, start, end, proxy
                    ): segment_path
                    for segment_path, start, end in ranges
                }
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    print(f"  completed {futures[future].name}")
            with partial.open("ab") as destination_handle:
                for segment_path, _, _ in ranges:
                    with segment_path.open("rb") as segment_handle:
                        shutil.copyfileobj(segment_handle, destination_handle, length=1024**2)
                    segment_path.unlink()
        if partial.stat().st_size != total:
            raise IOError(
                f"Incomplete merged archive {destination.name}: "
                f"got {partial.stat().st_size} of {total} bytes"
            )
        partial.replace(destination)
        return

    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
        print(f"Resuming {url} from {offset / 1024**2:.1f} MiB")
    else:
        print(f"Downloading {url}")
    with _http_opener(proxy).open(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        # A server that ignores Range returns 200; restart rather than append a
        # duplicate full archive to the existing partial file.
        append = bool(offset and status == 206)
        if offset and not append:
            print("Server ignored the Range request; restarting this archive.")
            offset = 0
        content_range = response.headers.get("Content-Range", "")
        total = None
        if "/" in content_range:
            try:
                total = int(content_range.rsplit("/", 1)[1])
            except ValueError:
                total = None
        if total is None:
            length = response.headers.get("Content-Length")
            total = offset + int(length) if length and length.isdigit() else None
        downloaded = offset
        next_report = downloaded + 64 * 1024**2
        with partial.open("ab" if append else "wb") as handle:
            while True:
                block = response.read(1024**2)
                if not block:
                    break
                handle.write(block)
                downloaded += len(block)
                if downloaded >= next_report:
                    if total:
                        print(
                            f"  {destination.name}: {downloaded / 1024**2:.1f}/"
                            f"{total / 1024**2:.1f} MiB ({downloaded / total:.1%})"
                        )
                    else:
                        print(f"  {destination.name}: {downloaded / 1024**2:.1f} MiB")
                    next_report += 64 * 1024**2
        if total is not None and downloaded != total:
            raise IOError(
                f"Incomplete download for {destination.name}: got {downloaded} of {total} bytes"
            )
    partial.replace(destination)


def download_kaggle(zone, year, destination, proxy=None):
    file_name = f"{zone}_Ground_Stations/{zone}_Ground_Stations/{zone}_Ground_Stations_{year}.csv"
    executable = shutil.which("kaggle")
    if executable is None:
        executable_name = "kaggle.exe" if os.name == "nt" else "kaggle"
        candidates = [
            Path(sys.executable).parent / executable_name,
            Path(sys.executable).parent / "Scripts" / executable_name,
        ]
        executable = next((str(candidate) for candidate in candidates if candidate.exists()), None)
    if executable is None:
        raise RuntimeError("Kaggle CLI is required for --source kaggle; install it with `pip install kaggle`.")

    destination.mkdir(parents=True, exist_ok=True)
    zip_path = destination / f"{zone}_Ground_Stations_{year}.csv.zip"
    marker = zip_path.with_suffix(zip_path.suffix + ".kaggle-partial")
    if zip_path.exists() and not marker.exists():
        print(f"Using cached {zip_path.name}")
        return zip_path
    environment = os.environ.copy()
    if proxy:
        environment["HTTP_PROXY"] = proxy
        environment["HTTPS_PROXY"] = proxy
    print(f"Downloading Kaggle mirror file for {zone} {year}")
    subprocess.run(
        [executable, "datasets", "download", "katerpillar/meteonet", "-f", file_name, "-p", str(destination)],
        check=True,
        env=environment,
    )
    if not zip_path.exists() or marker.exists():
        raise RuntimeError(f"Kaggle download did not complete: {zip_path}")
    return zip_path


def safe_extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe path in archive: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported archive entry: {member.name}")
        handle.extractall(destination)


def find_csv(extracted, zone, year):
    matches = list(extracted.rglob(f"{zone}{year}.csv"))
    if not matches:
        matches = list(extracted.rglob("*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one CSV under {extracted}, found {len(matches)}")
    return matches[0]


def aggregate_year(csv_path, chunksize, min_reports_per_hour):
    aggregated = []
    for chunk_index, chunk in enumerate(
        pd.read_csv(csv_path, usecols=REQUIRED_COLUMNS, chunksize=chunksize), start=1
    ):
        chunk["date"] = pd.to_datetime(chunk["date"], utc=True, errors="coerce")
        chunk["precip"] = pd.to_numeric(chunk["precip"], errors="coerce")
        chunk = chunk.dropna(subset=["number_sta", "date"])
        chunk["hour"] = chunk["date"].dt.floor("h")
        grouped = chunk.groupby(["hour", "number_sta"], sort=False).agg(
            precip=("precip", lambda values: values.sum(min_count=1)),
            reports=("precip", "count"),
            Lat=("lat", "first"),
            Lng=("lon", "first"),
            Elevation=("height_sta", "first"),
        )
        aggregated.append(grouped.reset_index())
        print(f"  {csv_path.name}: processed chunk {chunk_index}")

    hourly = pd.concat(aggregated, ignore_index=True)
    # A station-hour can straddle CSV chunk boundaries.
    hourly = hourly.groupby(["hour", "number_sta"], sort=True).agg(
        precip=("precip", lambda values: values.sum(min_count=1)),
        reports=("reports", "sum"),
        Lat=("Lat", "first"),
        Lng=("Lng", "first"),
        Elevation=("Elevation", "first"),
    ).reset_index()
    hourly.loc[hourly["reports"] < min_reports_per_hour, "precip"] = np.nan
    return hourly


def select_stations_from_training_year(hourly, training_year, expected_hours, min_completeness):
    """Select stations by observation completeness in the training year."""
    training = hourly[hourly["hour"].dt.year == training_year]
    valid_counts = training.groupby("number_sta")["precip"].count()
    return valid_counts[valid_counts / expected_hours >= min_completeness].index


def build_dataset(
    frames, output_dir, zone, min_station_completeness, source, min_reports_per_hour
):
    hourly = pd.concat(frames, ignore_index=True)
    full_time = pd.date_range(
        f"{YEARS[0]}-01-01 00:00:00+00:00",
        f"{YEARS[-1]}-12-31 23:00:00+00:00",
        freq="h",
    )
    # Use the training year to determine the station set.
    train_time = full_time[full_time.year == YEARS[0]]
    selected = select_stations_from_training_year(
        hourly, YEARS[0], len(train_time), min_station_completeness
    )
    if len(selected) < 16:
        raise ValueError(
            f"Only {len(selected)} stations meet completeness={min_station_completeness}; lower the threshold."
        )

    hourly = hourly[hourly["number_sta"].isin(selected)]
    metadata = (
        hourly.groupby("number_sta", sort=True)[["Lat", "Lng", "Elevation"]]
        .median()
        .reset_index()
        .rename(columns={"number_sta": "StationID"})
    )
    station_order = metadata["StationID"].tolist()
    matrix = hourly.pivot_table(index="hour", columns="number_sta", values="precip", aggfunc="first")
    matrix = matrix.reindex(index=full_time, columns=station_order)
    data = matrix.to_numpy(dtype=np.float32)[..., None]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"meteonet_{zone.lower()}.npz", data=data)
    metadata.to_csv(output_dir / f"meteonet_{zone.lower()}_meta.csv", index=False)
    np.save(output_dir / f"meteonet_{zone.lower()}_timestamps.npy", full_time.astype(str).to_numpy())

    missing_rate = float(np.mean(~np.isfinite(data)))
    split_slices = {
        "train_2016": full_time.year == 2016,
        "validation_2017": full_time.year == 2017,
        "test_2018": full_time.year == 2018,
    }
    finite_values = data[np.isfinite(data)]
    wet_values = finite_values[finite_values > 0]
    summary = {
        "dataset": f"meteonet_{zone.lower()}",
        "source": source,
        "years": list(YEARS),
        "temporal_split": {"train": 2016, "validation": 2017, "test": 2018},
        "time_steps": int(len(full_time)),
        "stations": int(len(station_order)),
        "minimum_training_year_completeness": min_station_completeness,
        "minimum_six_minute_reports_per_hour": int(min_reports_per_hour),
        "overall_missing_rate": missing_rate,
        "split_missing_rates": {
            name: float(np.mean(~np.isfinite(data[mask]))) for name, mask in split_slices.items()
        },
        "observed_rainfall_mean": float(np.mean(finite_values)) if finite_values.size else float("nan"),
        "wet_observation_frequency": float(np.mean(finite_values > 0)) if finite_values.size else float("nan"),
        "wet_rainfall_quantiles": {
            str(quantile): float(np.quantile(wet_values, quantile))
            for quantile in (0.5, 0.9, 0.95, 0.99, 0.999)
            if wet_values.size
        },
    }
    (output_dir / f"meteonet_{zone.lower()}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Prepared zone={zone}: data={data.shape}, stations={len(station_order)}, missing={missing_rate:.2%}")
    return data.shape[1]


def write_config(root, output_dir, zone, node_count):
    patch_size = 4
    leaves = 2 ** math.ceil(math.log2(node_count / patch_size))
    recurrence = int(math.log2(leaves))
    actual_patch_size = math.ceil(node_count / leaves)
    factor = next(value for value in (32, 16, 8, 4, 2, 1) if leaves % value == 0 and value <= leaves)
    name = f"meteonet_{zone.lower()}"
    year_hours = [366 * 24, 365 * 24, 365 * 24]
    total_hours = sum(year_hours)
    train_ratio = year_hours[0] / total_hours
    val_ratio = year_hours[1] / total_hours
    test_ratio = year_hours[2] / total_hours
    data_dir = Path(os.path.relpath(output_dir.resolve(), root.resolve())).as_posix()
    config = f"""[file]
traffic = {data_dir}/{name}.npz
meta = {data_dir}/{name}_meta.csv
adj = {data_dir}/{name}_adj.npy
model = ./outputs/{name}/model.pt
log = ./outputs/{name}/train.log
result = ./outputs/{name}/result.json
artifact = ./outputs/{name}/diagnostics.npz
dataset_name = {name}

[data]
input_len = 48
output_len = 24
train_ratio = {train_ratio:.12f}
val_ratio = {val_ratio:.12f}
test_ratio = {test_ratio:.12f}
use_missing_mask = true
is_precipitation = true
rain_thresholds = 0.1,1,5,10
rain_quantiles = 0.90,0.95,0.99

[train]
cuda = 0
max_epoch = 20
batch_size = 16
learning_rate = 0.002
weight_decay = 0.0001
seed = 1
num_workers = 0

[param]
factors = {factor}
layers = 5
tps = 24
tpn = 2
recur = {recurrence}
sps = {actual_patch_size}
spn = {leaves}
nodes = {node_count}
id = 64
nd = 32
ablation_mode = full
position_encoder = geoclip
use_elevation = true
branch_mode = full
fusion_mode = learned
"""
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / f"{name}.conf").write_text(config, encoding="utf-8")
    print(f"Wrote config/{name}.conf (leaves={leaves}, patch_size={actual_patch_size}, factor={factor})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", choices=["NW", "SE"], required=True)
    parser.add_argument("--source", choices=["official", "kaggle"], default="official")
    parser.add_argument("--proxy", help="Optional HTTP proxy, e.g. http://127.0.0.1:7890")
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument(
        "--min-reports-per-hour",
        type=int,
        default=10,
        choices=range(1, 11),
        metavar="1..10",
        help="Required valid six-minute reports for an hourly total (default: 10).",
    )
    parser.add_argument("--min-station-completeness", type=float, default=0.80)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    download_dir = args.download_dir or root / "data" / "MeteoNet" / "raw"
    output_dir = args.output_dir or root / "data" / "MeteoNet"
    frames = []
    for year in YEARS:
        if args.source == "kaggle":
            csv_path = download_kaggle(args.zone, year, download_dir / "kaggle", args.proxy)
        else:
            archive = download_dir / f"{args.zone}_ground_stations_{year}.tar.gz"
            extracted = download_dir / f"{args.zone}_{year}"
            # The official host throttles individual connections and becomes
            # unresponsive with high fan-out; two ranges improve throughput
            # while remaining reliable.
            download(
                BASE_URL.format(zone=args.zone, year=year),
                archive,
                parallel_segments=2,
                proxy=args.proxy,
            )
            if not extracted.exists():
                safe_extract(archive, extracted)
            csv_path = find_csv(extracted, args.zone, year)
        frames.append(aggregate_year(csv_path, args.chunksize, args.min_reports_per_hour))

    node_count = build_dataset(
        frames,
        output_dir,
        args.zone,
        args.min_station_completeness,
        args.source,
        args.min_reports_per_hour,
    )
    write_config(root, output_dir, args.zone, node_count)
    print(f"Prepared data: {output_dir}")
    print(f"Downloaded source files: {download_dir}")


if __name__ == "__main__":
    main()
