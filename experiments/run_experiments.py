"""Run forecasting and model ablation experiments with WeatherScaleNet."""

import argparse
import configparser
import itertools
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "wind": ROOT / "config" / "wind.conf",
    "temp": ROOT / "config" / "temp.conf",
    "rfusa": ROOT / "config" / "rfusa.conf",
    "rfuk": ROOT / "config" / "rfuk.conf",
    "rfcn": ROOT / "config" / "rfcn.conf",
    "rfaus": ROOT / "config" / "rfaus.conf",
    "meteonet_nw": ROOT / "config" / "meteonet_nw.conf",
    "meteonet_se": ROOT / "config" / "meteonet_se.conf",
}


@dataclass
class Run:
    study: str
    dataset: str
    seed: int
    overrides: dict = field(default_factory=dict)

    @property
    def tag(self):
        fields = [self.study, self.dataset, f"seed{self.seed}"]
        for key in (
            "spatial_split", "spatial_seed", "position_encoder", "branch_mode", "fusion_mode", "factors",
            "train_missing_rate", "eval_missing_rate", "missing_pattern", "missing_seed",
            "loss_type", "event_loss_threshold", "event_loss_weight",
            "use_elevation", "elevation_encoder",
        ):
            if key in self.overrides:
                fields.append(f"{key.replace('_', '-')}-{self.overrides[key]}")
        return "__".join(str(value) for value in fields)


def position_runs(seeds):
    encoders = ["none", "raw-mlp", "sincos", "rff", "station-id", "random-geoclip", "geoclip"]
    return [
        Run("position", dataset, seed, {"position_encoder": encoder, "use_elevation": "false"})
        for dataset, seed, encoder in itertools.product(("wind", "temp"), seeds, encoders)
    ]


def elevation_runs(seeds):
    variants = []
    for position_encoder in ("none", "rff", "geoclip"):
        variants.extend(
            [
                {"position_encoder": position_encoder, "use_elevation": "false", "elevation_encoder": "none"},
                {"position_encoder": position_encoder, "use_elevation": "true", "elevation_encoder": "discrete"},
                {"position_encoder": position_encoder, "use_elevation": "true", "elevation_encoder": "raw-mlp"},
            ]
        )
    return [
        Run("elevation", dataset, seed, dict(variant))
        for dataset, seed, variant in itertools.product(("wind", "temp"), seeds, variants)
    ]


def spatial_runs(seeds):
    runs = []
    encoders = ["none", "raw-mlp", "rff", "station-id", "geoclip"]
    for dataset, seed, split, encoder in itertools.product(
        ("wind", "temp"), seeds, ("random", "block"), encoders
    ):
        runs.append(
            Run(
                "spatial",
                dataset,
                seed,
                {
                    "spatial_split": split,
                    "spatial_seed": seed,
                    "spatial_val_ratio": 0.1,
                    "spatial_test_ratio": 0.2,
                    "position_encoder": encoder,
                    "use_missing_mask": "true",
                },
            )
        )
    return runs


def branch_runs(seeds):
    variants = [
        {"branch_mode": "local"},
        {"branch_mode": "far"},
        {"branch_mode": "full", "fusion_mode": "fixed"},
        {"branch_mode": "full", "fusion_mode": "learned"},
        {"branch_mode": "full", "fusion_mode": "concat"},
    ]
    return [
        Run("branch", dataset, seed, dict(variant, **({"batch_size": 16} if dataset == "temp" else {})))
        for dataset, seed, variant in itertools.product(
            ("wind", "temp", "meteonet_nw", "meteonet_se"), seeds, variants
        )
    ]


def partition_runs(seeds):
    return [
        Run("partition", dataset, seed, {"factors": factors})
        for dataset, seed, factors in itertools.product(("wind", "temp"), seeds, (8, 16, 32, 64))
    ]


def rainfall_runs(seeds):
    return [
        Run("rainfall-events", dataset, seed)
        for dataset, seed in itertools.product(("rfusa", "rfuk", "rfcn", "rfaus"), seeds)
    ]


def rainfall_loss_runs(seeds):
    variants = [{"loss_type": "huber"}]
    variants.extend(
        {
            "loss_type": "weighted-huber",
            "event_loss_threshold": 5.0,
            "event_loss_weight": weight,
        }
        for weight in (2.0, 4.0, 8.0)
    )
    return [
        Run("rainfall-loss", dataset, seed, dict(variant))
        for dataset, seed, variant in itertools.product(
            ("rfuk", "meteonet_nw", "meteonet_se"), seeds, variants
        )
    ]


def missingness_runs(seeds):
    runs = [
        Run(
            "missingness",
            dataset,
            seed,
            {
                "use_missing_mask": "true",
                "train_missing_rate": 0.0,
                "eval_missing_rate": 0.0,
                "missing_pattern": "random",
                "missing_seed": seed,
            },
        )
        for dataset, seed in itertools.product(("wind", "temp"), seeds)
    ]
    for dataset, seed, pattern, rate in itertools.product(
        ("wind", "temp"), seeds, ("random", "block"), (0.1, 0.3, 0.5)
    ):
        runs.append(
            Run(
                "missingness",
                dataset,
                seed,
                {
                    "use_missing_mask": "true",
                    "train_missing_rate": rate,
                    "eval_missing_rate": rate,
                    "missing_pattern": pattern,
                    "missing_seed": seed,
                },
            )
        )
        if rate in (0.3, 0.5):
            runs.append(
                Run(
                    "missingness",
                    dataset,
                    seed,
                    {
                        "use_missing_mask": "true",
                        "train_missing_rate": 0.0,
                        "eval_missing_rate": rate,
                        "missing_pattern": pattern,
                        "missing_seed": seed,
                    },
                )
            )
    return runs


def meteonet_runs(seeds):
    variants = [
        {"spatial_split": "none", "position_encoder": "geoclip"},
        {"spatial_split": "random", "position_encoder": "geoclip", "use_missing_mask": "true"},
        {"spatial_split": "block", "position_encoder": "geoclip", "use_missing_mask": "true"},
        {"spatial_split": "block", "position_encoder": "rff", "use_missing_mask": "true"},
        {"spatial_split": "block", "position_encoder": "raw-mlp", "use_missing_mask": "true"},
        {"spatial_split": "block", "position_encoder": "station-id", "use_missing_mask": "true"},
    ]
    return [
        Run("meteonet", dataset, seed, dict(variant, spatial_seed=seed))
        for dataset, seed, variant in itertools.product(("meteonet_nw", "meteonet_se"), seeds, variants)
    ]


def main_runs(seeds):
    return [Run("main", dataset, seed) for dataset, seed in itertools.product(CONFIGS, seeds)]


PRESETS = {
    "main": main_runs,
    "position": position_runs,
    "elevation": elevation_runs,
    "spatial": spatial_runs,
    "branch": branch_runs,
    "partition": partition_runs,
    "rainfall": rainfall_runs,
    "rainfall-loss": rainfall_loss_runs,
    "missingness": missingness_runs,
    "meteonet": meteonet_runs,
}


def config_available(run):
    config_path = CONFIGS[run.dataset]
    if not config_path.exists():
        return False, f"missing config {config_path}"
    parser = configparser.ConfigParser()
    parser.read(config_path)
    traffic = (ROOT / parser["file"]["traffic"]).resolve()
    if not traffic.exists():
        return False, f"missing data {traffic}"
    return True, ""


def command_for(run, python, output_root, max_epoch):
    run_dir = output_root / run.study / run.tag
    command = [
        python,
        str(ROOT / "main.py"),
        "--config",
        str(CONFIGS[run.dataset]),
        "--seed",
        str(run.seed),
        "--dataset_name",
        run.dataset,
        "--study_name",
        run.study,
        "--model_file",
        str(run_dir / "model.pt"),
        "--log_file",
        str(run_dir / "train.log"),
        "--result_file",
        str(run_dir / "result.json"),
        "--artifact_file",
        str(run_dir / "diagnostics.npz"),
    ]
    if max_epoch is not None:
        command.extend(["--max_epoch", str(max_epoch)])
    for key, value in run.overrides.items():
        command.extend([f"--{key}", str(value)])
    return command, run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["all", *PRESETS], required=True)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="auto", help="Device passed to each training run.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--max-epoch", type=int)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--datasets", help="Comma-separated dataset keys, e.g. wind,temp or rfuk")
    parser.add_argument("--position-encoders", help="Comma-separated position encoders to keep")
    parser.add_argument("--missing-rates", help="Comma-separated injected missingness rates to keep")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    names = list(PRESETS) if args.preset == "all" else [args.preset]
    runs = [run for name in names for run in PRESETS[name](seeds)]
    if args.datasets:
        selected_datasets = {value.strip() for value in args.datasets.split(",") if value.strip()}
        unknown = selected_datasets.difference(CONFIGS)
        if unknown:
            raise SystemExit(f"Unknown datasets: {sorted(unknown)}")
        runs = [run for run in runs if run.dataset in selected_datasets]
    if args.position_encoders:
        selected_encoders = {value.strip() for value in args.position_encoders.split(",") if value.strip()}
        runs = [
            run for run in runs
            if run.overrides.get("position_encoder", "geoclip") in selected_encoders
        ]
    if args.missing_rates:
        selected_rates = {float(value.strip()) for value in args.missing_rates.split(",") if value.strip()}
        runs = [
            run for run in runs
            if float(run.overrides.get("eval_missing_rate", 0.0)) in selected_rates
        ]
    if args.max_runs is not None:
        runs = runs[: args.max_runs]

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.jsonl"
    print(f"Prepared {len(runs)} runs under {args.output_root}")
    completed = skipped = failed = 0

    for index, run in enumerate(runs, 1):
        available, reason = config_available(run)
        command, run_dir = command_for(run, args.python, args.output_root, args.max_epoch)
        command.extend(["--device", args.device])
        result_path = run_dir / "result.json"
        if not available and not args.dry_run:
            print(f"[{index}/{len(runs)}] SKIP {run.tag}: {reason}")
            skipped += 1
            continue
        if result_path.exists() and not args.rerun and not args.dry_run:
            with result_path.open("r", encoding="utf-8") as handle:
                previous = json.load(handle)
            configured_epoch = configparser.ConfigParser()
            configured_epoch.read(CONFIGS[run.dataset])
            expected_epoch = args.max_epoch or configured_epoch.getint("train", "max_epoch")
            if previous.get("training_max_epoch") == expected_epoch:
                print(f"[{index}/{len(runs)}] DONE {run.tag}")
                completed += 1
                continue
            print(f"[{index}/{len(runs)}] RERUN {run.tag}: epoch budget changed")
        print(f"[{index}/{len(runs)}] RUN  {run.tag}")
        if args.dry_run:
            print(" ".join(command))
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        record = {"tag": run.tag, "command": command, "status": "running"}
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        process = subprocess.run(command, cwd=ROOT)
        record["status"] = "complete" if process.returncode == 0 else "failed"
        record["returncode"] = process.returncode
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if process.returncode != 0:
            failed += 1
            print(f"FAILED: {run.tag}; continuing with the remaining independent runs.")
        else:
            completed += 1

    print(f"Finished: completed={completed}, skipped={skipped}, failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
