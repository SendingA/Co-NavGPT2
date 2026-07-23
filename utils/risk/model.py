"""Small NumPy data models shared by the risk pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class GridFrame:
    """Explicit affine between world coordinates and a navigation grid.

    The existing navigation maps use axis 0 for local ``x`` and axis 1 for
    local ``z``.  ``origin_xz`` is the lower corner of cell ``(0, 0)`` in
    that map coordinate system.  ``world_to_map_matrix`` may encode the
    initial-pose translation/rotation used by the Open3D map; using identity
    makes map coordinates identical to Habitat world coordinates.
    """

    shape: Tuple[int, int]
    resolution_m: float
    origin_xz: Tuple[float, float]
    world_to_map_matrix: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.shape = (int(self.shape[0]), int(self.shape[1]))
        if self.shape[0] <= 0 or self.shape[1] <= 0:
            raise ValueError("GridFrame shape must be positive")
        self.resolution_m = float(self.resolution_m)
        if self.resolution_m <= 0.0:
            raise ValueError("GridFrame resolution_m must be positive")
        self.origin_xz = (float(self.origin_xz[0]), float(self.origin_xz[1]))
        matrix = (
            np.eye(4, dtype=np.float64)
            if self.world_to_map_matrix is None
            else np.asarray(self.world_to_map_matrix, dtype=np.float64)
        )
        if matrix.shape != (4, 4):
            raise ValueError("world_to_map_matrix must have shape (4, 4)")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("world_to_map_matrix must be finite")
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError("world_to_map_matrix must be invertible") from exc
        self.world_to_map_matrix = matrix
        self._map_to_world_matrix = inverse

    @classmethod
    def centered(
        cls,
        shape: Tuple[int, int],
        resolution_m: float,
        world_to_map_matrix: Optional[np.ndarray] = None,
    ) -> "GridFrame":
        """Construct a map centred on local coordinate ``(0, 0)``."""

        return cls(
            shape=shape,
            resolution_m=resolution_m,
            origin_xz=(
                -0.5 * int(shape[0]) * float(resolution_m),
                -0.5 * int(shape[1]) * float(resolution_m),
            ),
            world_to_map_matrix=world_to_map_matrix,
        )

    def _transform(self, points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        single = points.ndim == 1
        points = points.reshape(-1, 3)
        homogeneous = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1
        )
        transformed = homogeneous @ matrix.T
        out = transformed[:, :3] / transformed[:, 3:4]
        return out[0] if single else out

    def world_to_map_points(self, points_world: np.ndarray) -> np.ndarray:
        return self._transform(points_world, self.world_to_map_matrix)

    def map_to_world_points(self, points_map: np.ndarray) -> np.ndarray:
        return self._transform(points_map, self._map_to_world_matrix)

    def world_to_grid(
        self, points_world: np.ndarray, *, clip: bool = False
    ) -> np.ndarray:
        points_map = self.world_to_map_points(points_world)
        single = points_map.ndim == 1
        points_map = points_map.reshape(-1, 3)
        grid = np.stack(
            [
                np.floor(
                    (points_map[:, 0] - self.origin_xz[0]) / self.resolution_m
                ),
                np.floor(
                    (points_map[:, 2] - self.origin_xz[1]) / self.resolution_m
                ),
            ],
            axis=1,
        ).astype(np.int64)
        if clip:
            grid[:, 0] = np.clip(grid[:, 0], 0, self.shape[0] - 1)
            grid[:, 1] = np.clip(grid[:, 1], 0, self.shape[1] - 1)
        return grid[0] if single else grid

    def grid_to_world(
        self, indices: np.ndarray, *, map_y_m: float = 0.0
    ) -> np.ndarray:
        indices = np.asarray(indices)
        single = indices.ndim == 1
        indices = indices.reshape(-1, 2)
        points_map = np.stack(
            [
                self.origin_xz[0] + (indices[:, 0] + 0.5) * self.resolution_m,
                np.full(indices.shape[0], float(map_y_m)),
                self.origin_xz[1] + (indices[:, 1] + 0.5) * self.resolution_m,
            ],
            axis=1,
        )
        points_world = self.map_to_world_points(points_map)
        return points_world[0] if single else points_world

    def in_bounds(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices).reshape(-1, 2)
        return (
            (indices[:, 0] >= 0)
            & (indices[:, 0] < self.shape[0])
            & (indices[:, 1] >= 0)
            & (indices[:, 1] < self.shape[1])
        )

    def cell_centers_world(self, *, map_y_m: float = 0.0) -> np.ndarray:
        rows, cols = np.indices(self.shape, dtype=np.int64)
        indices = np.stack([rows.ravel(), cols.ravel()], axis=1)
        return self.grid_to_world(indices, map_y_m=map_y_m).reshape(
            self.shape + (3,)
        )


@dataclass
class RiskLayers:
    """A floor-aware 2-D risk snapshot.

    ``temperature_c`` preserves the physical sensor/simulator unit while
    ``temperature`` is its bounded risk normalisation.  ``physical_risk``
    excludes uncertainty so ground-truth exposure and planner uncertainty
    remain separable.
    """

    flame: np.ndarray
    temperature_c: np.ndarray
    temperature: np.ndarray
    smoke: np.ndarray
    physical_risk: np.ndarray
    hard_unsafe: np.ndarray
    confidence: np.ndarray
    uncertainty: np.ndarray
    last_update: np.ndarray
    unknown: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "flame": self.flame,
            "temperature_c": self.temperature_c,
            "temperature": self.temperature,
            "smoke": self.smoke,
            "physical_risk": self.physical_risk,
            "hard_unsafe": self.hard_unsafe,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "last_update": self.last_update,
            "unknown": self.unknown,
        }
        shape = np.asarray(self.flame).shape
        if len(shape) != 2:
            raise ValueError("risk layers must be 2-D")
        for name, array in arrays.items():
            if np.asarray(array).shape != shape:
                raise ValueError(f"risk layer {name} has inconsistent shape")

    @property
    def risk(self) -> np.ndarray:
        """Short alias for the uncertainty-free physical risk."""

        return self.physical_risk

    def copy(self) -> "RiskLayers":
        return RiskLayers(**{
            name: np.asarray(getattr(self, name)).copy()
            for name in (
                "flame", "temperature_c", "temperature", "smoke",
                "physical_risk", "hard_unsafe", "confidence", "uncertainty",
                "last_update", "unknown",
            )
        })


@dataclass
class RiskEvidence:
    """Per-agent world-space evidence, with no reference to FireWorld GT."""

    agent_id: int
    timestamp_s: float
    points_world: np.ndarray
    flame: np.ndarray
    temperature_c: np.ndarray
    smoke: np.ndarray
    confidence: Optional[np.ndarray] = None
    uncertainty: Optional[np.ndarray] = None
    privileged_transmittance: bool = False

    def __post_init__(self) -> None:
        points = np.asarray(self.points_world, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_world must have shape (N, 3)")
        count = points.shape[0]

        def vector(value: np.ndarray, name: str, default: float) -> np.ndarray:
            if value is None:
                return np.full(count, default, dtype=np.float32)
            array = np.asarray(value, dtype=np.float32)
            if array.ndim == 0:
                array = np.full(count, float(array), dtype=np.float32)
            else:
                array = array.reshape(-1)
            if array.size != count:
                raise ValueError(f"{name} must contain one value per point")
            return array

        self.points_world = points
        self.flame = np.clip(np.nan_to_num(
            vector(self.flame, "flame", 0.0), nan=0.0, posinf=1.0, neginf=0.0
        ), 0.0, 1.0)
        self.temperature_c = vector(self.temperature_c, "temperature_c", 25.0)
        self.smoke = np.clip(np.nan_to_num(
            vector(self.smoke, "smoke", 0.0), nan=0.0, posinf=1.0, neginf=0.0
        ), 0.0, 1.0)
        self.confidence = np.clip(
            np.nan_to_num(
                vector(self.confidence, "confidence", 1.0),
                nan=0.0, posinf=1.0, neginf=0.0,
            ),
            0.0,
            1.0,
        )
        self.uncertainty = np.clip(
            np.nan_to_num(
                vector(self.uncertainty, "uncertainty", 0.0),
                nan=1.0, posinf=1.0, neginf=0.0,
            ),
            0.0,
            1.0,
        )
        self.agent_id = int(self.agent_id)
        self.timestamp_s = float(self.timestamp_s)

    @property
    def size(self) -> int:
        return int(self.points_world.shape[0])


@dataclass
class RiskPointSamples:
    """Risk components sampled at a sequence of world positions."""

    flame: np.ndarray
    temperature_c: np.ndarray
    temperature: np.ndarray
    smoke: np.ndarray
    physical_risk: np.ndarray
    hard_unsafe: np.ndarray
    confidence: np.ndarray

    def __post_init__(self) -> None:
        size = np.asarray(self.physical_risk).reshape(-1).size
        for name in (
            "flame", "temperature_c", "temperature", "smoke",
            "physical_risk", "hard_unsafe", "confidence",
        ):
            array = np.asarray(getattr(self, name)).reshape(-1)
            if array.size != size:
                raise ValueError(f"point sample {name} has inconsistent length")
            setattr(self, name, array)
