"""Run a CPU installation check without downloading GeoCLIP weights."""

import argparse
from pathlib import Path
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="Print each check name.")
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root))

    import torch

    torch.set_num_threads(1)
    print("Running CPU checks on synthetic observations with the raw-mlp coordinate encoder.")
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(repository_root / "tests"), pattern="test_core.py"
    )
    result = unittest.TextTestRunner(
        stream=sys.stdout, verbosity=2 if args.verbose else 1
    ).run(suite)
    if result.wasSuccessful():
        print("CPU checks passed. Temporary inputs and checkpoints have been removed.")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
