"""Deterministic FireWorld plan selection for multi-scene benchmarks.

An explicit ``--fire_world_plan_id`` still selects exactly one asset.  When
the ID is omitted (or set to ``auto``), the active Habitat episode scene is
matched to a runnable plan with the requested fire type/intensity.  A plan is
runnable only when both its JSON and baked ``timeline.npz`` exist.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTO_PLAN_IDS = {"", "auto", "none"}
AUTO_FIRE_TYPE_PRIORITY = (
    "multi_origin",
    "kitchen_grease_fire",
    "bedroom_textile",
    "living_room_electric",
)


@dataclass(frozen=True)
class FirePlanSelection:
    """One plan/timeline pair ready for runtime playback."""

    scene_id: str
    plan_id: str
    fire_type: str
    intensity: str
    template_version: int
    seed: int
    plan_path: Path
    timeline_path: Path


def resolve_project_path(path: str | Path) -> Path:
    """Resolve CLI asset roots relative to the repository, not the caller."""

    result = Path(path).expanduser()
    if not result.is_absolute():
        result = PROJECT_ROOT / result
    return result


def is_auto_plan_id(plan_id: Optional[str]) -> bool:
    return str(plan_id or "").strip().lower() in AUTO_PLAN_IDS


def scene_id_from_config(config) -> str:
    """Return the current Habitat scene short ID after ``env.reset()``."""

    if hasattr(config, "habitat"):
        scene_path = config.habitat.simulator.scene
    else:
        scene_path = config.SIMULATOR.SCENE
    scene_name = Path(str(scene_path)).name
    for suffix in (".basis.glb", ".semantic.glb", ".glb"):
        if scene_name.endswith(suffix):
            return scene_name[: -len(suffix)]
    return scene_name


def find_scene_for_plan(
    plan_id: str,
    *,
    scenes_root: str | Path = "scenes",
) -> Optional[str]:
    """Find the unique scene owning an explicit plan ID."""

    root = resolve_project_path(scenes_root)
    matches = sorted(root.glob(f"*/plans/{plan_id}.json"))
    if not matches:
        return None
    if len(matches) > 1:
        owners = ", ".join(path.parents[1].name for path in matches)
        raise RuntimeError(
            f"FireWorld plan_id={plan_id!r} is ambiguous across scenes: "
            f"{owners}"
        )
    return matches[0].parents[1].name


def _selection_from_plan(
    plan_path: Path,
    *,
    scene_id: str,
    out_root: Path,
) -> FirePlanSelection:
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid FireWorld plan JSON: {plan_path}") from exc

    plan_id = str(payload.get("plan_id") or plan_path.stem)
    payload_scene = str(payload.get("scene_id") or scene_id)
    if payload_scene != scene_id:
        raise ValueError(
            f"FireWorld plan scene mismatch: expected {scene_id}, "
            f"found {payload_scene} in {plan_path}"
        )
    if plan_id != plan_path.stem:
        raise ValueError(
            f"FireWorld plan ID mismatch: filename={plan_path.stem}, "
            f"payload={plan_id}"
        )
    timeline_path = out_root / scene_id / plan_id / "timeline.npz"
    return FirePlanSelection(
        scene_id=scene_id,
        plan_id=plan_id,
        fire_type=str(payload.get("fire_type") or "unknown"),
        intensity=str(payload.get("intensity") or "unknown"),
        template_version=int(payload.get("template_version") or 0),
        seed=int(payload.get("seed") or 0),
        plan_path=plan_path,
        timeline_path=timeline_path,
    )


def select_fire_plan(
    scene_id: str,
    *,
    plan_id: Optional[str] = None,
    intensity: str = "medium",
    fire_type: str = "auto",
    scenes_root: str | Path = "scenes",
    out_root: str | Path = "outputs/fire_world",
) -> FirePlanSelection:
    """Select one runnable FireWorld plan for ``scene_id``.

    Auto mode first follows :data:`AUTO_FIRE_TYPE_PRIORITY`, then selects the
    highest template version within that type.  Lexical plan ID ordering is a
    stable final tie-breaker.  Explicit mode ignores type/intensity filters.
    """

    scene_id = str(scene_id)
    scenes_path = resolve_project_path(scenes_root)
    output_path = resolve_project_path(out_root)
    plans_dir = scenes_path / scene_id / "plans"

    if not is_auto_plan_id(plan_id):
        explicit_path = plans_dir / f"{plan_id}.json"
        if not explicit_path.exists():
            raise FileNotFoundError(
                f"FireWorld plan not found for scene={scene_id}: "
                f"{explicit_path}"
            )
        selection = _selection_from_plan(
            explicit_path,
            scene_id=scene_id,
            out_root=output_path,
        )
        if not selection.timeline_path.exists():
            raise FileNotFoundError(
                f"FireWorld timeline not found for explicit plan "
                f"{selection.plan_id}: {selection.timeline_path}"
            )
        return selection

    requested_intensity = str(intensity).strip().lower()
    requested_type = str(fire_type).strip().lower()
    if requested_type != "auto" and requested_type not in AUTO_FIRE_TYPE_PRIORITY:
        raise ValueError(
            f"unsupported FireWorld fire type {fire_type!r}; expected auto or "
            + ", ".join(AUTO_FIRE_TYPE_PRIORITY)
        )

    candidates = []
    for candidate_path in sorted(plans_dir.glob("*.json")):
        candidate = _selection_from_plan(
            candidate_path,
            scene_id=scene_id,
            out_root=output_path,
        )
        if candidate.intensity.lower() != requested_intensity:
            continue
        if requested_type != "auto" and candidate.fire_type.lower() != requested_type:
            continue
        if not candidate.timeline_path.exists():
            continue
        candidates.append(candidate)

    if not candidates:
        type_label = requested_type if requested_type != "auto" else "any"
        raise FileNotFoundError(
            "No runnable FireWorld plan for "
            f"scene={scene_id}, fire_type={type_label}, "
            f"intensity={requested_intensity}. Expected matching JSON under "
            f"{plans_dir} and timeline.npz under "
            f"{output_path / scene_id / '<plan_id>'}."
        )

    type_rank = {
        name: index for index, name in enumerate(AUTO_FIRE_TYPE_PRIORITY)
    }
    candidates.sort(
        key=lambda candidate: (
            type_rank.get(candidate.fire_type.lower(), len(type_rank)),
            -candidate.template_version,
            candidate.plan_id,
        )
    )
    return candidates[0]


def discover_runnable_fire_scenes(
    *,
    intensity: str = "medium",
    fire_type: str = "auto",
    scenes_root: str | Path = "scenes",
    out_root: str | Path = "outputs/fire_world",
    scene_ids: Optional[Iterable[str]] = None,
) -> Dict[str, FirePlanSelection]:
    """Return deterministic auto selections for every runnable scene."""

    scenes_path = resolve_project_path(scenes_root)
    if scene_ids is None:
        scene_ids = (
            path.parent.name
            for path in sorted(scenes_path.glob("*/plans"))
            if path.is_dir()
        )

    selections: Dict[str, FirePlanSelection] = {}
    for scene_id in sorted({str(value) for value in scene_ids}):
        try:
            selections[scene_id] = select_fire_plan(
                scene_id,
                intensity=intensity,
                fire_type=fire_type,
                scenes_root=scenes_path,
                out_root=out_root,
            )
        except FileNotFoundError:
            continue
    return selections
