"""Canonical, human-readable identifiers for FireWorld plans."""

from __future__ import annotations

import re


PLAN_HASH_RE = re.compile(r"^[0-9a-f]{12}$")
_COMPONENT_RE = re.compile(r"[^A-Za-z0-9]+")


def _component(value: str, *, lowercase: bool = False) -> str:
    """Normalize one plan-id component without hiding empty values."""
    normalized = _COMPONENT_RE.sub("_", str(value).strip()).strip("_")
    if lowercase:
        normalized = normalized.lower()
    if not normalized:
        raise ValueError(f"plan-id component is empty: {value!r}")
    return normalized


def validate_plan_hash(plan_hash: str) -> str:
    """Return a normalized 12-hex plan hash or raise."""
    normalized = str(plan_hash).strip().lower()
    if not PLAN_HASH_RE.fullmatch(normalized):
        raise ValueError(
            f"plan_hash must contain exactly 12 lowercase hex digits, "
            f"got {plan_hash!r}"
        )
    return normalized


def semantic_plan_id(
    scene_id: str,
    fire_type: str,
    intensity: str,
    plan_hash: str,
) -> str:
    """Build ``scene_type_intensity_hash`` with stable components."""
    return "_".join(
        (
            _component(scene_id),
            _component(fire_type, lowercase=True),
            _component(intensity, lowercase=True),
            validate_plan_hash(plan_hash),
        )
    )


def extract_plan_hash(plan_id: str) -> str:
    """Extract the stable hash from a legacy or semantic plan ID."""
    value = str(plan_id).strip()
    if PLAN_HASH_RE.fullmatch(value):
        return value
    suffix = value.rsplit("_", 1)[-1].lower()
    return validate_plan_hash(suffix)
