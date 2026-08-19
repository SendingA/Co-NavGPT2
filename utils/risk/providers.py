"""Ground-truth risk provider used by oracle ablations and evaluation.

This module is intentionally separate from :mod:`utils.risk.map`: sensed
belief maps cannot acquire a FireWorld reference through their constructor.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .config import RiskConfig
from .model import GridFrame, RiskLayers, RiskPointSamples
from .projection import project_fire_fields


class GroundTruthRiskProvider:
    """Project time-indexed FireWorld fields into an explicit map frame.

    Supplying this object to the planner constitutes ``oracle`` mode.  The
    evaluator may always use it because its output is kept independent from
    the sensed planner belief.
    """

    is_privileged = True

    def __init__(
        self,
        fire_world,
        frame: GridFrame,
        config: Optional[RiskConfig] = None,
        *,
        floor_y_m: float = 0.0,
        map_floor_y_m: Optional[float] = None,
        cache_frames: int = 2,
    ) -> None:
        # Accept FireScene for convenience but retain only its FireWorld data
        # layer.  Main should normally pass ``fire_scene.fw`` explicitly.
        self.fire_world = getattr(fire_world, "fw", fire_world)
        self.frame = frame
        self.config = config or RiskConfig()
        self.floor_y_m = float(floor_y_m)
        self.map_floor_y_m = float(
            floor_y_m if map_floor_y_m is None else map_floor_y_m
        )
        self.cache_frames = max(1, int(cache_frames))
        for name in ("query", "origin", "voxel_m", "shape"):
            if not hasattr(self.fire_world, name):
                raise TypeError(f"fire_world is missing required attribute {name!r}")
        # A complete 480x480 RiskLayers snapshot is several MB.  Keep only a
        # tiny LRU (planner + evaluator usually request the same current
        # frame) instead of retaining an entire long fire timeline.
        self._cache: "OrderedDict[Tuple[object, float], RiskLayers]" = OrderedDict()

    def clear_cache(self) -> None:
        self._cache.clear()

    def _frame_cache_key(self, timestamp_s: float, floor_y_m: float) -> tuple:
        if hasattr(self.fire_world, "frame_index"):
            time_key = int(self.fire_world.frame_index(float(timestamp_s)))
        else:
            time_key = float(timestamp_s)
        return time_key, round(float(floor_y_m), 6)

    def snapshot(
        self,
        timestamp_s: float,
        *,
        floor_y_m: Optional[float] = None,
    ) -> RiskLayers:
        floor = self.floor_y_m if floor_y_m is None else float(floor_y_m)
        key = self._frame_cache_key(timestamp_s, floor)
        cached = self._cache.get(key)
        if cached is None:
            flame, smoke, temperature = self.fire_world.query(float(timestamp_s))
            cached = project_fire_fields(
                flame,
                smoke,
                temperature,
                voxel_origin=np.asarray(self.fire_world.origin, dtype=np.float64),
                voxel_m=float(self.fire_world.voxel_m),
                frame=self.frame,
                floor_y_m=floor,
                map_floor_y_m=self.map_floor_y_m,
                config=self.config,
                timestamp_s=float(timestamp_s),
            )
            self._cache[key] = cached
            while len(self._cache) > self.cache_frames:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        # Providers never leak mutable cached arrays to a caller.
        return cached.copy()

    def sample_positions(
        self,
        timestamp_s: float,
        positions_world: np.ndarray,
        *,
        floor_y_m: Optional[
            Union[float, Sequence[float], np.ndarray]
        ] = None,
    ) -> RiskPointSamples:
        positions = np.asarray(positions_world, dtype=np.float64)
        if positions.ndim == 1:
            positions = positions.reshape(1, 3)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions_world must have shape (N, 3)")

        floors = np.asarray(floor_y_m) if floor_y_m is not None else None
        if floors is None or floors.ndim == 0:
            scalar_floor = None if floors is None else float(floors)
            return self._sample_positions_on_floor(
                float(timestamp_s), positions, floor_y_m=scalar_floor
            )

        floors = np.asarray(floors, dtype=np.float64).reshape(-1)
        if floors.size != positions.shape[0]:
            raise ValueError(
                "floor_y_m must be a scalar or contain one value per position"
            )
        if not np.all(np.isfinite(floors)):
            raise ValueError("floor_y_m values must be finite")

        count = positions.shape[0]
        combined = {
            "flame": np.zeros(count, dtype=np.float32),
            "temperature_c": np.full(
                count,
                float(self.config.temperature_ambient_c),
                dtype=np.float32,
            ),
            "temperature": np.zeros(count, dtype=np.float32),
            "smoke": np.zeros(count, dtype=np.float32),
            "physical_risk": np.zeros(count, dtype=np.float32),
            "hard_unsafe": np.zeros(count, dtype=bool),
            "confidence": np.zeros(count, dtype=np.float32),
        }
        # A separate vertical projection prevents hazards on one storey from
        # leaking into an agent that shares its x-z coordinates on another.
        for floor in np.unique(floors):
            selection = np.flatnonzero(floors == floor)
            samples = self._sample_positions_on_floor(
                float(timestamp_s),
                positions[selection],
                floor_y_m=float(floor),
            )
            for name in combined:
                combined[name][selection] = getattr(samples, name)
        return RiskPointSamples(**combined)

    def _sample_positions_on_floor(
        self,
        timestamp_s: float,
        positions: np.ndarray,
        *,
        floor_y_m: Optional[float],
    ) -> RiskPointSamples:
        """Sample one batch whose agents share a vertical projection band."""

        layers = self.snapshot(timestamp_s, floor_y_m=floor_y_m)
        indices = self.frame.world_to_grid(positions)
        valid = self.frame.in_bounds(indices)
        safe = indices.copy()
        safe[:, 0] = np.clip(safe[:, 0], 0, self.frame.shape[0] - 1)
        safe[:, 1] = np.clip(safe[:, 1], 0, self.frame.shape[1] - 1)
        rc = (safe[:, 0], safe[:, 1])

        def gather(array: np.ndarray, outside) -> np.ndarray:
            result = np.asarray(array[rc]).copy()
            result[~valid] = outside
            return result

        return RiskPointSamples(
            flame=gather(layers.flame, 0.0),
            temperature_c=gather(
                layers.temperature_c, self.config.temperature_ambient_c
            ),
            temperature=gather(layers.temperature, 0.0),
            smoke=gather(layers.smoke, 0.0),
            physical_risk=gather(layers.physical_risk, 0.0),
            hard_unsafe=gather(layers.hard_unsafe, False).astype(bool),
            confidence=gather(layers.confidence, 0.0),
        )
