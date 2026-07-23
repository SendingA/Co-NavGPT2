"""Dynamic multi-agent risk belief map built only from sensed evidence."""
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np

from .config import RiskConfig
from .model import GridFrame, RiskEvidence, RiskLayers
from .projection import (
    combine_normalized_risk,
    hard_unsafe_mask,
    normalize_temperature_c,
)


class DynamicRiskMap:
    """Confidence-aware shared risk belief for all robots.

    The class intentionally accepts only :class:`RiskEvidence`; it has no
    FireWorld/provider constructor parameter.  This makes the primary sensed
    benchmark's no-oracle boundary easy to audit and test.

    Fusion is conservative: a newly observed higher hazard takes effect
    immediately, while lower values only replace it through exponential
    temporal decay.  This is a simple "rise fast, fall slowly" model suitable
    for dynamic fire, and avoids averaging a dangerous report away with many
    low-confidence safe reports.
    """

    def __init__(self, frame: GridFrame, config: Optional[RiskConfig] = None):
        self.frame = frame
        self.config = config or RiskConfig()
        self._time_s: Optional[float] = None
        self._flame = np.zeros(frame.shape, dtype=np.float32)
        self._temperature_c = np.full(
            frame.shape,
            float(self.config.temperature_ambient_c),
            dtype=np.float32,
        )
        self._smoke = np.zeros(frame.shape, dtype=np.float32)
        self._confidence = np.zeros(frame.shape, dtype=np.float32)
        self._uncertainty = np.ones(frame.shape, dtype=np.float32)
        self._last_update = np.full(frame.shape, -np.inf, dtype=np.float64)

    @property
    def current_time_s(self) -> Optional[float]:
        return self._time_s

    def reset(self) -> None:
        self._time_s = None
        self._flame.fill(0.0)
        self._temperature_c.fill(float(self.config.temperature_ambient_c))
        self._smoke.fill(0.0)
        self._confidence.fill(0.0)
        self._uncertainty.fill(1.0)
        self._last_update.fill(-np.inf)

    def advance_time(self, timestamp_s: float) -> None:
        """Decay stale physical evidence and confidence to ``timestamp_s``."""

        now = float(timestamp_s)
        if self._time_s is None:
            self._time_s = now
            return
        if now < self._time_s - 1e-9:
            raise ValueError(
                "DynamicRiskMap timestamps must be monotonic; sample one shared "
                "fire time for every multi-agent navigation step"
            )
        dt = now - self._time_s
        if dt <= 0.0:
            return
        physical_factor = math.exp(-dt / float(self.config.decay_tau_s))
        confidence_factor = math.exp(
            -dt / float(self.config.confidence_decay_tau_s)
        )
        observed = np.isfinite(self._last_update)
        self._flame[observed] *= physical_factor
        self._smoke[observed] *= physical_factor
        ambient = float(self.config.temperature_ambient_c)
        self._temperature_c[observed] = ambient + (
            self._temperature_c[observed] - ambient
        ) * physical_factor
        self._confidence[observed] *= confidence_factor
        self._uncertainty[observed] = np.maximum(
            self._uncertainty[observed], 1.0 - self._confidence[observed]
        )
        self._time_s = now

    def update(self, evidence: RiskEvidence) -> RiskLayers:
        """Fuse one agent's evidence and return a defensive snapshot copy."""

        self.advance_time(evidence.timestamp_s)
        self._fuse(evidence)
        return self.snapshot()

    def _fuse(self, evidence: RiskEvidence) -> None:
        """Fuse evidence already advanced to its timestamp, without rendering."""

        if evidence.size == 0:
            return

        indices = self.frame.world_to_grid(evidence.points_world)
        valid = self.frame.in_bounds(indices)
        valid &= np.all(np.isfinite(evidence.points_world), axis=1)
        valid &= np.isfinite(evidence.temperature_c)
        if not valid.any():
            return

        indices = indices[valid]
        flat_cells = np.ravel_multi_index(
            (indices[:, 0], indices[:, 1]), self.frame.shape
        )
        total_cells = int(np.prod(self.frame.shape))

        flame_new = np.zeros(total_cells, dtype=np.float32)
        smoke_new = np.zeros(total_cells, dtype=np.float32)
        temp_new = np.full(
            total_cells,
            float(self.config.temperature_ambient_c),
            dtype=np.float32,
        )
        confidence_new = np.zeros(total_cells, dtype=np.float32)
        uncertainty_new = np.ones(total_cells, dtype=np.float32)
        np.maximum.at(flame_new, flat_cells, evidence.flame[valid])
        np.maximum.at(smoke_new, flat_cells, evidence.smoke[valid])
        np.maximum.at(temp_new, flat_cells, evidence.temperature_c[valid])
        np.maximum.at(confidence_new, flat_cells, evidence.confidence[valid])
        np.minimum.at(uncertainty_new, flat_cells, evidence.uncertainty[valid])
        touched_flat = np.zeros(total_cells, dtype=bool)
        touched_flat[flat_cells] = True

        touched = touched_flat.reshape(self.frame.shape)
        flame_new = flame_new.reshape(self.frame.shape)
        smoke_new = smoke_new.reshape(self.frame.shape)
        temp_new = temp_new.reshape(self.frame.shape)
        confidence_new = confidence_new.reshape(self.frame.shape)
        uncertainty_new = uncertainty_new.reshape(self.frame.shape)
        previously_observed = np.isfinite(self._last_update[touched])

        # Conservative physical fusion: higher readings take effect now;
        # decreases occur through explicit age decay rather than accidental
        # averaging across agents or repeated pixels.
        self._flame[touched] = np.maximum(
            self._flame[touched], flame_new[touched]
        )
        self._smoke[touched] = np.maximum(
            self._smoke[touched], smoke_new[touched]
        )
        self._temperature_c[touched] = np.where(
            previously_observed,
            np.maximum(self._temperature_c[touched], temp_new[touched]),
            temp_new[touched],
        )
        old_confidence = self._confidence[touched]
        new_confidence = confidence_new[touched]
        self._confidence[touched] = 1.0 - (
            1.0 - old_confidence
        ) * (1.0 - new_confidence)
        # First evidence establishes uncertainty.  Later agents fuse
        # conservatively: disagreement/low confidence may increase it, while a
        # single optimistic report cannot erase existing uncertainty.
        prior_uncertainty = self._uncertainty[touched]
        self._uncertainty[touched] = np.where(
            previously_observed,
            np.maximum(prior_uncertainty, uncertainty_new[touched]),
            uncertainty_new[touched],
        )
        self._last_update[touched] = float(evidence.timestamp_s)

    def update_many(self, evidence_items: Iterable[RiskEvidence]) -> RiskLayers:
        """Fuse a set of agent reports, which should share one timestamp."""

        items = list(evidence_items)
        if not items:
            return self.snapshot()
        times = np.asarray([item.timestamp_s for item in items], dtype=np.float64)
        if np.max(times) - np.min(times) > 1e-6:
            raise ValueError(
                "multi-agent evidence must share one synchronized fire timestamp"
            )
        self.advance_time(float(times[0]))
        for item in items:
            self._fuse(item)
        return self.snapshot()

    def snapshot(self, timestamp_s: Optional[float] = None) -> RiskLayers:
        if timestamp_s is not None:
            self.advance_time(timestamp_s)
        temperature = normalize_temperature_c(
            self._temperature_c,
            self.config.temperature_reference_c,
            self.config.temperature_hazard_c,
        )
        physical_risk = combine_normalized_risk(
            temperature, self._smoke, self.config
        )
        hard_unsafe = hard_unsafe_mask(
            self._flame, self._temperature_c, self.frame, self.config
        )
        unknown = (
            self._confidence < float(self.config.minimum_known_confidence)
        )
        return RiskLayers(
            flame=self._flame.copy(),
            temperature_c=self._temperature_c.copy(),
            temperature=temperature.copy(),
            smoke=self._smoke.copy(),
            physical_risk=physical_risk.copy(),
            hard_unsafe=hard_unsafe.copy(),
            confidence=self._confidence.copy(),
            uncertainty=self._uncertainty.copy(),
            last_update=self._last_update.copy(),
            unknown=unknown.copy(),
        )

    def planning_risk(self, timestamp_s: Optional[float] = None) -> np.ndarray:
        """Return risk with unknown/uncertainty priors for path planning.

        Evaluators must use ``physical_risk`` from a GT provider instead;
        planner uncertainty is intentionally absent from CHE.
        """

        layers = self.snapshot(timestamp_s)
        return self.planning_risk_from_layers(layers)

    def planning_risk_from_layers(self, layers: RiskLayers) -> np.ndarray:
        """Apply planner uncertainty to an already synchronized snapshot."""
        if layers.physical_risk.shape != self.frame.shape:
            raise ValueError("risk layers do not match the belief-map frame")
        unknown_penalty = float(self.config.unknown_risk_prior) * (
            1.0 - layers.confidence
        )
        uncertainty_penalty = float(self.config.uncertainty_weight) * (
            layers.uncertainty
        )
        return np.clip(
            np.maximum(layers.physical_risk, unknown_penalty)
            + uncertainty_penalty,
            0.0,
            1.0,
        ).astype(np.float32)
