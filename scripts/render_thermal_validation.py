"""Render a deterministic thermal-camera validation montage.

The fixture deliberately puts a 700 C hot-air curtain across most camera
rays. A correct surface-dominant thermal model keeps the ambient room dark
while still highlighting a flame, a warm person, and a heated object.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.fire_sensors.humans_thermal import (
    HumanThermalTarget,
    add_humans_to_thermal_image,
)
from utils.fire_sensors.voxel_render import VoxelRenderParams, volumetric_composite


def _label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return out


def render(output_dir: Path) -> dict:
    height, width = 240, 320
    rgb = np.full((height, width, 3), (118, 132, 145), dtype=np.uint8)
    rgb[150:] = (105, 82, 62)
    rgb[65:165, 205:280] = (70, 92, 126)       # cabinet / heated object
    rgb[55:200, 65:115] = (82, 96, 108)        # dark furniture
    rgb[:, 155:160] = (75, 68, 62)              # structural edge
    depth = np.full((height, width), 4.0, dtype=np.float32)
    depth[65:165, 205:280] = 3.8

    shape = (40, 25, 45)
    origin = np.array([-2.0, 0.0, 0.0], dtype=np.float32)
    voxel_m = 0.1
    temp = np.full(shape, 25.0, dtype=np.float32)
    flame = np.zeros(shape, dtype=np.float32)
    smoke = np.zeros(shape, dtype=np.float32)

    # A wide hot-air curtain tests that line-of-sight gas cannot repaint the
    # ambient rear wall as a uniformly hot surface.
    temp[:, :, 27:29] = 700.0
    # Heated cabinet surface near the depth endpoint.
    temp[25:34, 4:18, 4:10] = 180.0
    # Local flame and its near-source heat.
    flame[10:16, 1:12, 17:23] = 0.95
    temp[8:18, 0:15, 15:25] = np.maximum(
        temp[8:18, 0:15, 15:25], 620.0
    )

    camera_k = SimpleNamespace(
        fx=250.0, fy=250.0, cx=width / 2.0, cy=height / 2.0
    )
    camera_position = np.array([0.0, 1.4, 4.5], dtype=np.float32)
    out = volumetric_composite(
        rgb_clean=rgb,
        depth_m=depth,
        cam_pos_world=camera_position,
        R_cam2world=np.eye(3, dtype=np.float32),
        flame_field=flame,
        smoke_field=smoke,
        temp_field=temp,
        origin=origin,
        voxel_m=voxel_m,
        grid_shape=shape,
        ambient_c=25.0,
        camera_K=camera_k,
        params=VoxelRenderParams(
            max_depth_m=5.0,
            n_steps=32,
            render_scale=1.0,
            thermal_color_blend=0.85,
            flame_noise_strength=0.0,
            flame_edge_break=0.0,
            flame_color_jitter=0.0,
            smoke_noise_strength=0.0,
        ),
        t_sim=0.0,
    )

    sensor_state = SimpleNamespace(
        position=camera_position.astype(np.float64),
        rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    agent_state = SimpleNamespace(sensor_states={"depth": sensor_state})
    person = HumanThermalTarget(
        position=np.array([0.05, 0.9, 1.45], dtype=np.float64),
        height_m=1.7,
        radius_m=0.25,
        excess_c=25.0,
    )
    thermal, temperature = add_humans_to_thermal_image(
        out["thermal_image"],
        out["thermal_temperature"],
        [person],
        agent_state,
        camera_k,
        depth_m=None,
        max_depth_m=5.0,
    )

    temperature_norm = np.log1p(
        np.maximum(temperature - 25.0, 0.0) / 5.0
    ) / np.log1p(600.0 / 5.0)
    temperature_color = cv2.applyColorMap(
        np.clip(temperature_norm * 255.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_INFERNO,
    )
    montage = np.concatenate(
        [
            _label(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "Clean RGB"),
            _label(thermal, "Localized Thermal IR"),
            _label(temperature_color, "Apparent Temperature"),
        ],
        axis=1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "thermal_localized_validation.png"
    metrics_path = output_dir / "thermal_localized_metrics.json"
    if not cv2.imwrite(str(image_path), montage):
        raise RuntimeError(f"failed to write {image_path}")

    gray = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
    metrics = {
        "temperature_min_c": float(temperature.min()),
        "temperature_max_c": float(temperature.max()),
        "temperature_p50_c": float(np.percentile(temperature, 50.0)),
        "temperature_p95_c": float(np.percentile(temperature, 95.0)),
        "bright_pixel_fraction": float(np.mean(gray > 120)),
        "dark_pixel_fraction": float(np.mean(gray < 55)),
        "output_image": str(image_path),
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/thermal_validation"),
    )
    args = parser.parse_args()
    print(json.dumps(render(args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
