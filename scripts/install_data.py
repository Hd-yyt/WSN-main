"""Install prepared dataset files from a ZIP archive and verify their checksums."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install(archive, destination, overwrite=False):
    destination = destination.resolve()
    with ZipFile(archive) as source:
        entries = json.loads(source.read("manifest.json"))["files"]
        plans = []
        seen = set()
        for entry in entries:
            name = entry["path"]
            relative = PurePosixPath(name)
            if (relative.is_absolute() or not relative.parts or relative.parts[0] != "data"
                    or ".." in relative.parts or "\\" in name or ":" in name):
                raise ValueError(f"Invalid dataset path: {name}")
            if name in seen:
                raise ValueError(f"Duplicate dataset path: {name}")
            seen.add(name)
            target = (destination / name).resolve()
            if not target.is_relative_to(destination / "data"):
                raise ValueError(f"Dataset path escapes the data directory: {name}")
            if source.getinfo(name).file_size != entry["size"]:
                raise ValueError(f"Archive size mismatch: {name}")
            if target.is_file() and sha256(target) == entry["sha256"]:
                print(f"Already installed: {name}")
                continue
            if target.exists() and not overwrite:
                raise ValueError(f"A different file exists at {target}; use --overwrite to replace it.")
            plans.append((entry, target))
        for entry, target in plans:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            try:
                with source.open(entry["path"]) as reader, temporary.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
                if sha256(temporary) != entry["sha256"]:
                    raise ValueError(f"Checksum mismatch: {entry['path']}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            print(f"Installed: {entry['path']}")
    print(f"Dataset installation complete: {destination / 'data'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=ROOT, help="Repository root.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing files with different checksums.")
    args = parser.parse_args()
    try:
        install(args.archive, args.destination, args.overwrite)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Data installation failed: {error}\n")


if __name__ == "__main__":
    main()
