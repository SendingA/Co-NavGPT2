"""Configuration and units for dynamic fire-risk assessment.

The defaults deliberately keep risk-aware navigation disabled.  Enabling the
module is therefore an explicit benchmark choice and cannot silently change
the existing ObjectNav policy.

All component risks are dimensionless and bounded to ``[0, 1]``.  Temperature
is the only input with a physical unit (degrees Celsius); it is normalised
between :attr:`RiskConfig.temperature_reference_c` and
:attr:`RiskConfig.temperature_hazard_c` before fusion.  These thresholds are
simulation/benchmark calibration parameters, *not* a claim about human
survivability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


_VALID_SOURCES = frozenset({"none", "oracle", "sensed"})


@dataclass(frozen=True)
class RiskWeights:
    """Weights in ``H = clip(wT*rT + wS*rS, 0, 1)``.

    Explicit flame is intentionally absent from the continuous score. A
    visible flame already produces high temperature, so counting both as
    weighted terms double-counted one physical event. Flame remains the
    strongest safety signal through :func:`hard_unsafe_mask`, where it creates
    a distance-inflated no-go region.
    """

    temperature: float = 0.60
    smoke: float = 0.40

    def __post_init__(self) -> None:
        values = (float(self.temperature), float(self.smoke))
        if any(value < 0.0 for value in values):
            raise ValueError("risk weights must be non-negative")
        if abs(sum(values) - 1.0) > 1e-6:
            raise ValueError("risk weights must sum to 1.0")


@dataclass(frozen=True)
class RiskConfig:
    """Risk contract shared by GT evaluation and sensed belief maps.

    ``source`` controls which map may be supplied to the planner:

    ``none``
        Risk-aware planning is disabled (the backward-compatible default).
    ``oracle``
        The planner may consume a complete FireWorld map.  This is only an
        upper-bound/debug ablation.
    ``sensed``
        The planner consumes :class:`~utils.risk.map.DynamicRiskMap`, built
        exclusively from per-agent observations.

    The independent evaluator always consumes a ground-truth provider; the
    distinction is enforced structurally by keeping that provider out of the
    sensed map class.
    """

    enabled: bool = False
    source: str = "none"
    weights: RiskWeights = field(default_factory=RiskWeights)

    # Temperature input/normalisation (degrees Celsius).
    temperature_ambient_c: float = 25.0
    temperature_reference_c: float = 25.0
    temperature_hazard_c: float = 150.0
    temperature_hard_c: float = 250.0
    temperature_hard_enabled: bool = False

    # The default hard exclusion is only the high-intensity flame core.
    # Surrounding heat/smoke remains continuous soft risk. A safety dilation
    # or temperature hard cutoff is an explicit experiment choice.
    flame_hard_threshold: float = 0.80
    flame_safety_distance_m: float = 0.0

    # 3-D FireWorld -> current-floor 2-D navigation-map projection.
    floor_min_offset_m: float = 0.0
    floor_max_offset_m: float = 1.8

    # Evaluation thresholds on the normalised composite risk.
    danger_threshold: float = 0.60
    critical_threshold: float = 0.85

    # Dynamic sensed-map semantics.
    decay_tau_s: float = 30.0
    confidence_decay_tau_s: float = 20.0
    unknown_risk_prior: float = 0.50
    uncertainty_weight: float = 0.20
    minimum_known_confidence: float = 0.10
    sensor_stride: int = 4

    # Beer-Lambert conversion for an explicitly privileged transmittance
    # ablation.  Normal sensed mode should pass its own smoke estimate.
    smoke_extinction_coefficient: float = 1.0

    def __post_init__(self) -> None:
        source = str(self.source).lower().strip()
        object.__setattr__(self, "source", source)
        if source not in _VALID_SOURCES:
            raise ValueError(
                f"risk source must be one of {sorted(_VALID_SOURCES)}, got {source!r}"
            )
        if self.temperature_hazard_c <= self.temperature_reference_c:
            raise ValueError("temperature_hazard_c must exceed the reference")
        if not all(map(math.isfinite, (
            float(self.temperature_ambient_c),
            float(self.temperature_reference_c),
            float(self.temperature_hazard_c),
            float(self.temperature_hard_c),
        ))):
            raise ValueError("temperature thresholds must be finite")
        if self.temperature_hard_c < self.temperature_reference_c:
            raise ValueError("temperature_hard_c must not be below the reference")
        if self.flame_safety_distance_m < 0.0:
            raise ValueError("flame_safety_distance_m must be non-negative")
        if self.floor_max_offset_m <= self.floor_min_offset_m:
            raise ValueError("floor_max_offset_m must exceed floor_min_offset_m")
        if self.decay_tau_s <= 0.0 or self.confidence_decay_tau_s <= 0.0:
            raise ValueError("risk decay time constants must be positive")
        if int(self.sensor_stride) < 1:
            raise ValueError("sensor_stride must be at least 1")
        if self.smoke_extinction_coefficient <= 0.0:
            raise ValueError("smoke_extinction_coefficient must be positive")
        for name in (
            "flame_hard_threshold",
            "danger_threshold",
            "critical_threshold",
            "unknown_risk_prior",
            "uncertainty_weight",
            "minimum_known_confidence",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.critical_threshold < self.danger_threshold:
            raise ValueError("critical_threshold must be >= danger_threshold")

    @property
    def effective_source(self) -> str:
        """Return ``none`` whenever the feature gate is disabled."""

        return self.source if self.enabled else "none"

    @property
    def body_height_m(self) -> float:
        return float(self.floor_max_offset_m - self.floor_min_offset_m)

    @classmethod
    def from_namespace(cls, args: Any) -> "RiskConfig":
        """Build from an argparse/OmegaConf-like object.

        Every lookup has a safe fallback so older call sites and configs do
        not need to define any risk arguments.  ``None`` is also treated as
        missing, which is useful when a config system materialises optional
        keys with null values.
        """

        defaults = cls()

        def value(name: str, default: Any) -> Any:
            candidate = getattr(args, name, default)
            return default if candidate is None else candidate

        enabled = bool(int(value("risk_enabled", int(defaults.enabled))))
        source_default = "sensed" if enabled else defaults.source
        source = str(value("risk_source", source_default)).lower()
        weights = RiskWeights(
            temperature=float(
                value("risk_weight_temperature", defaults.weights.temperature)
            ),
            smoke=float(value("risk_weight_smoke", defaults.weights.smoke)),
        )
        return cls(
            enabled=enabled,
            source=source,
            weights=weights,
            temperature_ambient_c=float(value(
                "risk_temperature_ambient_c", defaults.temperature_ambient_c
            )),
            temperature_reference_c=float(value(
                "risk_temperature_reference_c", defaults.temperature_reference_c
            )),
            temperature_hazard_c=float(value(
                "risk_temperature_hazard_c", defaults.temperature_hazard_c
            )),
            temperature_hard_c=float(value(
                "risk_temperature_hard_c", defaults.temperature_hard_c
            )),
            temperature_hard_enabled=bool(int(value(
                "risk_temperature_hard_enabled",
                int(defaults.temperature_hard_enabled),
            ))),
            flame_hard_threshold=float(value(
                "risk_flame_hard_threshold", defaults.flame_hard_threshold
            )),
            flame_safety_distance_m=float(value(
                "risk_flame_safety_distance_m", defaults.flame_safety_distance_m
            )),
            danger_threshold=float(value(
                "risk_danger_threshold", defaults.danger_threshold
            )),
            critical_threshold=float(value(
                "risk_critical_threshold", defaults.critical_threshold
            )),
            decay_tau_s=float(value("risk_decay_tau_s", defaults.decay_tau_s)),
            confidence_decay_tau_s=float(value(
                "risk_confidence_decay_tau_s", defaults.confidence_decay_tau_s
            )),
            unknown_risk_prior=float(value(
                "risk_unknown_risk_prior", defaults.unknown_risk_prior
            )),
            uncertainty_weight=float(value(
                "risk_uncertainty_weight", defaults.uncertainty_weight
            )),
            sensor_stride=int(value("risk_sensor_stride", defaults.sensor_stride)),
            floor_min_offset_m=float(value(
                "risk_floor_min_offset_m", defaults.floor_min_offset_m
            )),
            floor_max_offset_m=float(value(
                "risk_floor_max_offset_m", defaults.floor_max_offset_m
            )),
        )
