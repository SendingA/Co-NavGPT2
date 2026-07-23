"""Headless visualisation helpers for dynamic navigation risk maps.

The renderer is intentionally independent from Habitat and Open3D so the same
artifact format can be used in unit tests, batch benchmarks and interactive
runs.  Values are displayed as normalized simulator hazard indices, not as
physiological probabilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


_PANEL_NAMES = (
    ("flame", "Flame risk"),
    ("temperature", "Temperature risk"),
    ("smoke", "Smoke risk"),
    ("physical_risk", "Physical risk"),
    ("planning_risk", "Planning cost"),
    ("confidence", "Confidence"),
)


def _unit_map(values, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if values is None:
        if shape is None:
            raise ValueError("shape is required when a risk panel is missing")
        return np.zeros(shape, dtype=np.float32)
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"risk panels must be 2-D, got {array.shape}")
    return np.clip(
        np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )


def colorize_risk(
    values: np.ndarray,
    *,
    unknown: Optional[np.ndarray] = None,
    invert: bool = False,
) -> np.ndarray:
    """Return a BGR heatmap, rendering unknown cells in neutral grey."""
    unit = _unit_map(values)
    intensity = np.rint((1.0 - unit if invert else unit) * 255.0).astype(np.uint8)
    image = cv2.applyColorMap(intensity, cv2.COLORMAP_INFERNO)
    if unknown is not None:
        unknown_mask = np.asarray(unknown, dtype=bool)
        if unknown_mask.shape != unit.shape:
            raise ValueError("unknown mask must match the risk-map shape")
        image[unknown_mask] = (72, 72, 72)
    return image


def _label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 27), (0, 0, 0), -1)
    cv2.putText(
        result,
        text,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def _draw_context(
    image: np.ndarray,
    *,
    obstacle_map: Optional[np.ndarray],
    hard_unsafe: Optional[np.ndarray],
    agent_cells: Optional[Sequence[Sequence[int]]],
    frontier_points: Optional[Iterable[Sequence[int]]],
) -> np.ndarray:
    result = image.copy()
    source_shape = None
    for value in (obstacle_map, hard_unsafe):
        if value is not None:
            source_shape = np.asarray(value).shape
            break
    if source_shape is None:
        source_shape = result.shape[:2]
    scale_y = result.shape[0] / max(1, source_shape[0])
    scale_x = result.shape[1] / max(1, source_shape[1])

    if obstacle_map is not None:
        obstacles = cv2.resize(
            np.asarray(obstacle_map, dtype=np.uint8),
            (result.shape[1], result.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        result[obstacles] = (20, 20, 20)
    if hard_unsafe is not None:
        unsafe = cv2.resize(
            np.asarray(hard_unsafe, dtype=np.uint8),
            (result.shape[1], result.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        result[unsafe] = (255, 0, 255)

    for agent_id, cell in enumerate(agent_cells or []):
        row, col = int(cell[0]), int(cell[1])
        centre = (int((col + 0.5) * scale_x), int((row + 0.5) * scale_y))
        cv2.circle(result, centre, 6, (255, 255, 0), -1)
        cv2.putText(
            result, f"R{agent_id}", (centre[0] + 5, centre[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
            cv2.LINE_AA,
        )
    for frontier_id, cell in enumerate(frontier_points or []):
        row, col = int(cell[0]), int(cell[1])
        centre = (int((col + 0.5) * scale_x), int((row + 0.5) * scale_y))
        cv2.circle(result, centre, 5, (0, 255, 255), 1)
        cv2.putText(
            result, f"F{frontier_id}", (centre[0] + 5, centre[1] + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1,
            cv2.LINE_AA,
        )
    return result


def compose_risk_dashboard(
    layers,
    *,
    planning_risk: Optional[np.ndarray] = None,
    obstacle_map: Optional[np.ndarray] = None,
    agent_cells: Optional[Sequence[Sequence[int]]] = None,
    frontier_points: Optional[Iterable[Sequence[int]]] = None,
    panel_size: Tuple[int, int] = (360, 360),
    title: str = "Dynamic Risk Assessment",
) -> np.ndarray:
    """Render flame/temperature/smoke/total/cost/confidence in a 2x3 grid."""
    physical = _unit_map(getattr(layers, "physical_risk"))
    shape = physical.shape
    unknown = getattr(layers, "unknown", None)
    hard_unsafe = getattr(layers, "hard_unsafe", None)
    values = {
        "flame": getattr(layers, "flame", None),
        "temperature": getattr(layers, "temperature", None),
        "smoke": getattr(layers, "smoke", None),
        "physical_risk": physical,
        "planning_risk": planning_risk if planning_risk is not None else physical,
        "confidence": getattr(layers, "confidence", None),
    }
    panels = []
    for key, label in _PANEL_NAMES:
        unit = _unit_map(values[key], shape)
        panel_unknown = unknown if key != "confidence" else None
        panel = colorize_risk(
            unit,
            unknown=panel_unknown,
            invert=(key == "confidence"),
        )
        panel = cv2.resize(panel, panel_size, interpolation=cv2.INTER_NEAREST)
        panel = _draw_context(
            panel,
            obstacle_map=obstacle_map,
            hard_unsafe=hard_unsafe,
            agent_cells=agent_cells,
            frontier_points=frontier_points,
        )
        panels.append(_label(panel, label))

    dashboard = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))
    title_bar = np.zeros((34, dashboard.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        title_bar, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return np.vstack((title_bar, dashboard))


def save_risk_snapshot(
    path: Path,
    layers,
    **dashboard_kwargs,
) -> Path:
    """Atomically create the parent directory and write one dashboard PNG."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = compose_risk_dashboard(layers, **dashboard_kwargs)
    if not cv2.imwrite(str(output), image):
        raise OSError(f"failed to write risk dashboard: {output}")
    return output
