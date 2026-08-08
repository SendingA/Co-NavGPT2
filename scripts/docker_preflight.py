#!/usr/bin/env python3
"""Validate the Docker runtime before starting an expensive Habitat run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Dict, List


EXPECTED_VERSIONS = {
    "python": "3.9",
    "torch": "2.0.1+cu118",
    "torchvision": "0.15.2+cu118",
    "habitat-sim": "0.3.3",
    "habitat-lab": "0.3.3",
    "numpy": "1.26.4",
    "open3d": "0.19.0",
    "opencv-python": "4.10.0.84",
    "scikit-fmm": "2023.4.2",
    "scikit-image": "0.24.0",
    "hydra-core": "1.3.4",
    "omegaconf": "2.3.1",
    "openai": "2.44.0",
    "ultralytics": "8.4.88",
    "supervision": "0.19.0",
}

MODEL_HASHES = {
    "mobile_sam": (
        "CONAV_MOBILE_SAM_PATH",
        "mobile_sam.pt",
        "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
    ),
    "yolo_world": (
        "CONAV_YOLO_WORLD_PATH",
        "yolov8l-world.pt",
        "8bdfaef999116760247d6fb0b0f8fca064b43e94598b3d4a807ebae9bcf0cdd5",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Checks:
    def __init__(self) -> None:
        self.rows: List[Dict[str, str]] = []

    def add(self, name: str, status: str, detail: str) -> None:
        self.rows.append(
            {"name": str(name), "status": str(status), "detail": str(detail)}
        )

    def require(self, name: str, condition: bool, detail: str) -> None:
        self.add(name, "ok" if condition else "error", detail)

    def warn(self, name: str, condition: bool, detail: str) -> None:
        self.add(name, "ok" if condition else "warning", detail)

    @property
    def errors(self) -> List[Dict[str, str]]:
        return [row for row in self.rows if row["status"] == "error"]


def _check_versions(checks: Checks) -> None:
    checks.require(
        "python",
        sys.version.startswith(EXPECTED_VERSIONS["python"] + "."),
        sys.version.splitlines()[0],
    )
    for package, expected in EXPECTED_VERSIONS.items():
        if package == "python":
            continue
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            checks.require(package, False, "not installed")
            continue
        checks.require(package, actual == expected, f"{actual}; expected {expected}")
    if os.environ.get("CONAV_CONTAINER") == "1":
        try:
            headless_version = importlib.metadata.version(
                "opencv-python-headless"
            )
        except importlib.metadata.PackageNotFoundError:
            checks.require("opencv-python-headless", False, "not installed")
        else:
            checks.require(
                "opencv-python-headless",
                headless_version == "4.10.0.84",
                f"{headless_version}; expected 4.10.0.84",
            )


def _check_habitat_patch(checks: Checks) -> None:
    try:
        import habitat
        from habitat.config.default_structured_configs import SimulatorConfig

        config = SimulatorConfig()
        checks.require(
            "habitat_vulcan_patch",
            hasattr(config, "tilt_angle"),
            f"habitat={habitat.__file__}; tilt_angle={getattr(config, 'tilt_angle', None)}",
        )
    except Exception as error:  # pragma: no cover - diagnostic boundary
        checks.require("habitat_vulcan_patch", False, repr(error))


def _check_gpu(checks: Checks) -> None:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        detail = "CUDA unavailable"
        if available:
            detail = "; ".join(
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            )
        checks.require("nvidia_gpu", available, detail)
    except Exception as error:  # pragma: no cover - diagnostic boundary
        checks.require("nvidia_gpu", False, repr(error))


def _check_models(checks: Checks, workspace: Path) -> None:
    for name, (env_name, relative, expected_hash) in MODEL_HASHES.items():
        path = Path(os.environ.get(env_name, str(workspace / relative)))
        if not path.is_file():
            checks.require(name, False, f"missing file: {path}")
            continue
        actual_hash = _sha256(path)
        checks.require(
            name,
            actual_hash == expected_hash,
            f"{path}; sha256={actual_hash}",
        )


def _check_data(checks: Checks, workspace: Path) -> None:
    data_root = Path(
        os.environ.get("CONAV_DATA_ROOT", str(workspace / "data"))
    )
    required = {
        "hm3d_scene_config": data_root
        / "scene_datasets/hm3d_v0.2/"
        "hm3d_annotated_basis.scene_dataset_config.json",
        "objectnav_dataset": data_root
        / "datasets/objectnav_hm3d_v2/val/val.json.gz",
    }
    for name, path in required.items():
        checks.require(name, path.is_file(), str(path))

    checks.warn(
        "humanoid_assets",
        (data_root / "humanoids/humanoid_data").is_dir(),
        "optional; required for pedestrians and static-person ObjectNav",
    )
    checks.warn(
        "pointnav_checkpoint",
        (data_root / "ddppo-models/gibson-2plus-resnet50.pth").is_file(),
        "optional; required only for --local_planner pointnav",
    )


def _check_output(checks: Checks, workspace: Path) -> None:
    output_root = Path(
        os.environ.get("CONAV_OUTPUT_ROOT", str(workspace / "outputs"))
    )
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output_root, delete=True):
            pass
    except OSError as error:
        checks.require("output_writable", False, f"{output_root}: {error}")
    else:
        checks.require("output_writable", True, str(output_root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("base", "navigation", "gpt"),
        default="navigation",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checks = Checks()
    _check_versions(checks)
    _check_habitat_patch(checks)
    if args.mode != "base":
        _check_gpu(checks)
        _check_models(checks, args.workspace)
        _check_data(checks, args.workspace)
        _check_output(checks, args.workspace)
    if args.mode == "gpt":
        checks.require(
            "openai_api_key",
            bool(os.environ.get("OPENAI_API_KEY")),
            "set" if os.environ.get("OPENAI_API_KEY") else "not set",
        )

    payload = {
        "mode": args.mode,
        "ok": not checks.errors,
        "checks": checks.rows,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for row in checks.rows:
            print(
                "[{status:7}] {name}: {detail}".format(**row),
                flush=True,
            )
        print("preflight: {}".format("PASS" if payload["ok"] else "FAIL"))
    return 1 if args.strict and checks.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
