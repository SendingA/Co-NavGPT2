#!/usr/bin/env python3
"""Package runtime data with relative names, materialized aliases and checksums."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import zipfile


EXCLUDED = {".git", ".ssh", ".env", "__pycache__", ".DS_Store"}


def source_files(root: Path):
    """Follow only internal, acyclic aliases; never include local Git state."""
    root = root.resolve(strict=True)

    def walk(path: Path, relative: Path, ancestors: frozenset[Path]):
        if path.name in EXCLUDED:
            return
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError(f"Asset alias leaves the dataset: {relative}")
        if any(part in EXCLUDED for part in resolved.relative_to(root).parts):
            raise ValueError(f"Asset alias targets excluded metadata: {relative}")
        if resolved.is_dir():
            if resolved in ancestors:
                raise ValueError(f"Cyclic asset alias: {relative}")
            for child in sorted(path.iterdir()):
                yield from walk(child, relative / child.name, ancestors | {resolved})
        elif resolved.is_file():
            # These installer records contain machine-specific extraction paths.
            if "versioned_data" in resolved.relative_to(root).parts and path.name.endswith("-files.json.gz"):
                return
            if path.name.startswith(".env.") or path.suffix in {".key", ".pem", ".pyc"}:
                raise ValueError(f"Unexpected private/cache file: {relative}")
            yield relative, resolved
        else:
            raise ValueError(f"Unsupported dataset entry: {relative}")

    yield from walk(root, Path("data"), frozenset())


def package(data_dir: Path, output: Path) -> dict:
    data_dir = data_dir.resolve(strict=True)
    output = output.resolve()
    if output.is_relative_to(data_dir):
        raise ValueError("Write the release archive outside the dataset.")
    if output.exists():
        raise FileExistsError("The output archive already exists.")
    entries = list(source_files(data_dir))
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    fd, temporary_name = tempfile.mkstemp(prefix=".dataset-", suffix=".partial", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for relative, source in entries:
                before = source.stat()
                info = zipfile.ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | (0o755 if before.st_mode & 0o111 else 0o644)) << 16
                digest = hashlib.sha256()
                size = 0
                with source.open("rb") as reader, archive.open(info, "w", force_zip64=True) as writer:
                    while chunk := reader.read(8 * 1024 * 1024):
                        writer.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                after = source.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                    raise RuntimeError(f"Dataset changed while packaging: {relative}")
                records.append({"path": relative.as_posix(), "bytes": size, "sha256": digest.hexdigest()})
        manifest = {
            "archive": output.name,
            "format": "ZIP64, stored entries",
            "aliases": "Materialized as ordinary files inside data/",
            "timestamps": "Fixed to 1980-01-01; no owner names, user IDs or group IDs stored",
            "files": records,
        }
        temporary.replace(output)
        output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = package(args.data_dir, args.output)
    print(json.dumps({"archive": str(args.output), "files": len(manifest["files"]),
                      "data_bytes": sum(row["bytes"] for row in manifest["files"])}), flush=True)


if __name__ == "__main__":
    main()
