"""Static and host-equivalent validation for Docker reproduction artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class DockerPackagingTests(unittest.TestCase):
    def test_dockerfile_pins_runtime_and_applies_patch(self) -> None:
        source = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        environment = (ROOT / "docker/environment.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04",
            source,
        )
        self.assertIn("python=3.9.19", environment)
        self.assertIn("habitat-sim=0.3.3", environment)
        requirements = (ROOT / "docker/requirements.lock.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("opencv-python-headless==4.10.0.84", requirements)
        self.assertIn("--force-reinstall --no-deps", source)
        self.assertIn(
            "094d6be2f9d057e4781a68ae792132895fd4d3d0",
            source,
        )
        self.assertIn("git -C /opt/habitat-lab-src apply --check", source)
        self.assertNotIn("COPY data", source)
        self.assertNotIn("COPY mobile_sam.pt", source)

    def test_heavy_and_sensitive_paths_are_outside_context(self) -> None:
        ignored = {
            line.strip()
            for line in (ROOT / ".dockerignore").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        for expected in (
            ".git",
            ".env",
            "data",
            "outputs",
            "tmp",
            "weights",
            "mobile_sam.pt",
            "yolov8l-world.pt",
        ):
            self.assertIn(expected, ignored)

    def test_compose_mounts_assets_and_exposes_gpu(self) -> None:
        payload = yaml.safe_load(
            (ROOT / "compose.yaml").read_text(encoding="utf-8")
        )
        service = payload["services"]["conav"]
        self.assertEqual(service["gpus"], "all")
        self.assertEqual(service["command"][-1], "--strict")
        targets = {
            volume["target"]: volume for volume in service["volumes"]
        }
        self.assertTrue(targets["/workspace/data"]["read_only"])
        self.assertTrue(targets["/workspace/mobile_sam.pt"]["read_only"])
        self.assertTrue(targets["/workspace/yolov8l-world.pt"]["read_only"])
        self.assertNotIn("OPENAI_API_KEY", str(service.get("build", {})))

    def test_entrypoint_has_valid_bash_syntax(self) -> None:
        source = (ROOT / "scripts/docker_entrypoint.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("tests.test_docker_packaging", source)
        self.assertIn("tests.test_global_planners", source)
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts/docker_entrypoint.sh")],
            check=True,
        )

    def test_base_preflight_matches_host_equivalent_environment(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/docker_preflight.py"),
                "--mode",
                "base",
                "--strict",
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        statuses = {row["name"]: row["status"] for row in payload["checks"]}
        self.assertEqual(statuses["habitat_vulcan_patch"], "ok")
        self.assertEqual(statuses["habitat-sim"], "ok")


if __name__ == "__main__":
    unittest.main()
