"""Dynamic FireWorld risk assessment.

The public surface keeps privileged ground truth, sensed belief, planning and
evaluation as distinct objects so experiments can audit simulator-state use.
"""

from .config import RiskConfig, RiskWeights
from .frontier import (
    AssignmentGuardResult,
    FrontierRiskReport,
    FrontierScore,
    SeverityThresholds,
    UtilityWeights,
    assign_frontiers,
    build_frontier_risk_reports,
    frontier_risk_report,
    guard_frontier_assignments,
    risk_context_payload,
    score_frontiers,
)
from .map import DynamicRiskMap
from .metrics import MultiAgentRiskEvaluator, PRIMARY_BENCHMARK_METRICS
from .model import GridFrame, RiskEvidence, RiskLayers, RiskPointSamples
from .projection import (
    combine_normalized_risk,
    compute_physical_risk,
    dilate_disk,
    evidence_from_sensor_images,
    hard_unsafe_mask,
    normalize_temperature_c,
    project_fire_fields,
)
from .providers import GroundTruthRiskProvider

__all__ = [
    "AssignmentGuardResult",
    "DynamicRiskMap",
    "FrontierRiskReport",
    "FrontierScore",
    "GridFrame",
    "GroundTruthRiskProvider",
    "MultiAgentRiskEvaluator",
    "PRIMARY_BENCHMARK_METRICS",
    "RiskConfig",
    "RiskEvidence",
    "RiskLayers",
    "RiskPointSamples",
    "RiskWeights",
    "SeverityThresholds",
    "UtilityWeights",
    "assign_frontiers",
    "build_frontier_risk_reports",
    "combine_normalized_risk",
    "compute_physical_risk",
    "dilate_disk",
    "evidence_from_sensor_images",
    "frontier_risk_report",
    "guard_frontier_assignments",
    "hard_unsafe_mask",
    "normalize_temperature_c",
    "project_fire_fields",
    "risk_context_payload",
    "score_frontiers",
]
