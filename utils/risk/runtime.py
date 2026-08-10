"""Runtime orchestration for dynamic multi-agent risk assessment.

This module owns the explicit boundary between privileged FireWorld ground
truth and the shared sensed belief used by navigation.  The evaluator always
uses ground truth; a sensed planner never receives the provider object or its
arrays.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import re
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .config import RiskConfig
from .estimation import estimate_smoke_from_appearance_depth
from .map import DynamicRiskMap
from .metrics import MultiAgentRiskEvaluator
from .model import GridFrame, RiskLayers
from .projection import evidence_from_sensor_images
from .providers import GroundTruthRiskProvider
from .visualization import save_risk_snapshot


_HABITAT_ACTION_NAMES = {
    0: "stop",
    1: "move_forward",
    2: "turn_left",
    3: "turn_right",
    4: "look_up",
    5: "look_down",
}


def navigation_grid_frame(reference_agent) -> GridFrame:
    """Build the exact shared Open3D/navigation frame used by the agents.

    ``VLM_Agent.get_transform_matrix`` maps Habitat world coordinates through
    ``habitat_camera_self_aj @ inv(initial_camera_pose)`` before writing point
    clouds into the centered grid.  Reusing that affine here avoids the common
    failure where a visually correct FireWorld map is shifted or rotated with
    respect to the planner.
    """

    rotation = np.asarray(reference_agent.init_sim_rotation, dtype=np.float64)
    position = np.asarray(reference_agent.init_sim_position, dtype=np.float64)
    if rotation.shape != (3, 3) or position.shape != (3,):
        raise ValueError("reference agent must expose a valid initial camera pose")

    initial_camera = np.eye(4, dtype=np.float64)
    initial_camera[:3, :3] = rotation
    initial_camera[:3, 3] = position
    habitat_camera_self_aj = np.eye(4, dtype=np.float64)
    habitat_camera_self_aj[:3, :3] = np.asarray(
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    world_to_map = habitat_camera_self_aj @ np.linalg.inv(initial_camera)
    return GridFrame.centered(
        shape=(int(reference_agent.local_w), int(reference_agent.local_h)),
        resolution_m=float(reference_agent.args.map_resolution) / 100.0,
        world_to_map_matrix=world_to_map,
    )


def _agent_positions(agent_states: Sequence[object]) -> np.ndarray:
    positions = [np.asarray(state.position, dtype=np.float64) for state in agent_states]
    result = np.asarray(positions, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("agent states must expose Habitat world xyz positions")
    return result


def _active_fire_plan_id(args) -> Optional[str]:
    """Return the concrete per-scene plan selected for this episode."""

    return (
        getattr(args, "fire_world_active_plan_id", None)
        or getattr(args, "fire_world_plan_id", None)
    )


class RiskRuntime:
    """One-episode risk estimator, planner-map provider and GT evaluator."""

    def __init__(
        self,
        *,
        fire_scene,
        reference_agent,
        args,
        episode_id: int,
    ) -> None:
        self.args = args
        self.config = RiskConfig.from_namespace(args)
        if not self.config.enabled:
            raise ValueError("RiskRuntime should only be built when risk is enabled")
        if fire_scene is None:
            raise ValueError("dynamic risk assessment requires --fire_world=1")
        self.smoke_source = str(getattr(
            args, "risk_smoke_source", "appearance_depth"
        ))
        if self.smoke_source not in {
            "appearance_depth", "privileged_transmittance"
        }:
            raise ValueError(f"unsupported risk smoke source: {self.smoke_source}")
        self.geometry_depth_source = str(getattr(
            args, "risk_geometry_depth_source", "clean"
        ))
        if self.geometry_depth_source not in {"clean", "smoke"}:
            raise ValueError(
                "risk_geometry_depth_source must be 'clean' or 'smoke'"
            )

        self.fire_scene = fire_scene
        self.frame = navigation_grid_frame(reference_agent)
        self.floor_y_m = float(
            np.asarray(reference_agent.init_agent_position, dtype=np.float64)[1]
        )
        initial_agent_world = np.asarray(
            reference_agent.init_agent_position, dtype=np.float64
        )
        self.map_floor_y_m = float(
            self.frame.world_to_map_points(initial_agent_world)[1]
        )
        self.max_floor_deviation_m = float(getattr(
            args, "risk_max_floor_deviation_m", 0.75
        ))
        if self.max_floor_deviation_m <= 0.0:
            raise ValueError("risk_max_floor_deviation_m must be positive")
        self.gt_provider = GroundTruthRiskProvider(
            fire_scene.fw,
            self.frame,
            self.config,
            floor_y_m=self.floor_y_m,
            map_floor_y_m=self.map_floor_y_m,
        )
        self.belief = DynamicRiskMap(self.frame, self.config)
        self.evaluator = MultiAgentRiskEvaluator(self.gt_provider, self.config)
        self.episode_id = int(episode_id)
        self.output_dir = Path(getattr(args, "risk_dump_dir", "outputs/risk_assessment"))
        self.run_id = str(getattr(args, "risk_run_id", "default"))
        if (
            self.run_id in {".", ".."}
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", self.run_id)
        ):
            raise ValueError(
                "risk_run_id may contain only letters, digits, '_', '-' and '.'"
            )
        self.rank = int(getattr(args, "risk_rank", 0))
        if self.rank < 0:
            raise ValueError("risk_rank must be non-negative")
        self.run_dir = self.output_dir / self.run_id / f"rank_{self.rank:03d}"
        self.episode_dir = self.run_dir / f"ep_{self.episode_id:04d}"
        self.step_log_path = self.episode_dir / "risk_steps.jsonl"
        self.action_log_path = self.episode_dir / "actions.jsonl"
        self.action_list_path = self.episode_dir / "action_list.json"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        # A fresh benchmark run must not append onto an older episode trace.
        self.step_log_path.write_text("", encoding="utf-8")
        self.action_log_path.write_text("", encoding="utf-8")
        for stale_path in self.episode_dir.glob("risk_step_*.png"):
            stale_path.unlink()
        stale_summary = self.episode_dir / "risk_summary.json"
        if stale_summary.exists():
            stale_summary.unlink()
        if self.action_list_path.exists():
            self.action_list_path.unlink()
        run_config_path = self.run_dir / "risk_config.json"
        run_config_path.write_text(
            json.dumps({
                "metric_version": self.evaluator.metric_version,
                "run_id": self.run_id,
                "rank": self.rank,
                "seed": int(getattr(args, "seed", 0)),
                "scene_id": getattr(fire_scene, "scene_id", None),
                "fire_plan_id": _active_fire_plan_id(args),
                "fire_clock_mode": str(getattr(
                    args, "fire_clock_mode", "wallclock"
                )),
                "planner_source": self.source,
                "local_planner": str(getattr(
                    args, "local_planner", "fmm"
                )),
                "local_planner_uses_risk": self.planning_enabled,
                "smoke_source": self.smoke_source,
                "geometry_depth_source": self.geometry_depth_source,
                "risk_config": asdict(self.config),
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._last_layers: Optional[RiskLayers] = None
        self._last_planning_risk: Optional[np.ndarray] = None
        self._last_step_report: Dict[str, Dict[str, float]] = {}
        self._planner_event_counts: Dict[str, Dict[str, int]] = {}
        self._action_steps = []

    def _append_record(self, record: Dict[str, object]) -> None:
        with self.step_log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    @property
    def planning_enabled(self) -> bool:
        return self.config.effective_source in {"oracle", "sensed"}

    @property
    def source(self) -> str:
        return self.config.effective_source

    def shared_time(self, robot_step: int) -> float:
        """Sample FireClock exactly once for all agents in one outer step."""
        return float(self.fire_scene.t_sim(int(robot_step)))

    def evaluator_visualization_state(self, timestamp_s: float) -> RiskLayers:
        """Return GT hazard layers for an explicitly display-only overlay.

        Evaluator-only ``source=none`` intentionally supplies a neutral map to
        both planners. This accessor lets the navigation panel still explain
        the robot's true exposure without routing the privileged GT layers
        through :meth:`planner_state` or either planner context.
        """

        return self.gt_provider.snapshot(float(timestamp_s))

    def _validate_current_floor(self, agent_states: Sequence[object]) -> None:
        positions = _agent_positions(agent_states)
        deviations = np.abs(positions[:, 1] - self.floor_y_m)
        if np.any(deviations > self.max_floor_deviation_m):
            agent_id = int(np.argmax(deviations))
            raise RuntimeError(
                "risk assessment currently uses one floor-aware 2-D map; "
                f"agent {agent_id} moved {deviations[agent_id]:.3f} m from "
                "the episode floor. Split the episode by floor or implement "
                "multi-floor RiskLayers instead of collapsing hazards."
            )

    def prime_exposure(
        self,
        timestamp_s: float,
        agent_states: Sequence[object],
    ) -> Dict[str, Dict[str, float]]:
        """Prime GT integration after validating the single-floor contract."""
        self._validate_current_floor(agent_states)
        for agent_id in range(len(agent_states)):
            self._planner_event_counts.setdefault(
                str(agent_id), {
                    "safe_refusal_steps": 0,
                    "emergency_escape_steps": 0,
                    "trapped_steps": 0,
                },
            )
        return self.evaluator.prime(
            float(timestamp_s),
            _agent_positions(agent_states),
            floor_y_m=self.floor_y_m,
        )

    def update_sensed(
        self,
        *,
        timestamp_s: float,
        sensor_outputs: Sequence[Optional[dict]],
        agent_states: Sequence[object],
        camera_k,
    ) -> RiskLayers:
        """Fuse synchronized per-agent observations into the shared belief."""
        if len(sensor_outputs) != len(agent_states):
            raise ValueError("sensor_outputs and agent_states must have equal length")
        self._validate_current_floor(agent_states)
        evidence_items = []
        smoke_source = self.smoke_source
        for agent_id, (sensors, agent_state) in enumerate(
            zip(sensor_outputs, agent_states)
        ):
            if sensors is None:
                continue
            sensor_time = sensors.get("t_sim_s")
            if (
                sensor_time is not None
                and abs(float(sensor_time) - float(timestamp_s)) > 1e-6
            ):
                raise RuntimeError(
                    f"agent {agent_id} sensor timestamp {float(sensor_time):.6f} "
                    f"does not match shared risk timestamp {float(timestamp_s):.6f}"
                )
            smoke_depth = np.asarray(
                sensors["depth_smoke"], dtype=np.float32
            )
            geometry_depth = np.asarray(
                sensors.get("depth_clean", sensors["depth_smoke"])
                if self.geometry_depth_source == "clean"
                else sensors["depth_smoke"],
                dtype=np.float32,
            )
            if smoke_depth.ndim == 3 and smoke_depth.shape[-1] == 1:
                smoke_depth = smoke_depth[..., 0]
            if geometry_depth.ndim == 3 and geometry_depth.shape[-1] == 1:
                geometry_depth = geometry_depth[..., 0]
            temperature = np.asarray(
                sensors["thermal_temperature"], dtype=np.float32
            ).copy()
            flame = np.asarray(
                sensors["thermal_flame_mask"], dtype=np.float32
            ).copy()

            # Humanoid thermal signatures are useful for person perception but
            # are not a fire hazard.  Preserve real flame pixels if masks overlap.
            human_mask = sensors.get("thermal_human_mask")
            if human_mask is not None:
                human_only = (
                    np.asarray(human_mask) > 0.0
                ) & (flame <= 0.0)
                temperature[human_only] = np.minimum(
                    temperature[human_only],
                    float(self.config.temperature_reference_c),
                )

            if smoke_source == "privileged_transmittance":
                smoke_estimate = None
                confidence = None
                transmittance = sensors.get("transmittance")
                if transmittance is None:
                    raise RuntimeError(
                        "privileged_transmittance was selected but agent "
                        f"{agent_id} produced no transmittance image"
                    )
                allow_privileged = True
            else:
                smoke_estimate, confidence = (
                    estimate_smoke_from_appearance_depth(
                        sensors["rgb_smoke"],
                        smoke_depth,
                        max_depth_m=float(sensors.get(
                            "sensor_max_depth_m", 5.0
                        )),
                    )
                )
                transmittance = None
                allow_privileged = False

            camera_position, rotation = self.fire_scene.camera_pose(agent_state)
            evidence_items.append(
                evidence_from_sensor_images(
                    agent_id=agent_id,
                    timestamp_s=float(timestamp_s),
                    depth_m=geometry_depth,
                    camera_k=camera_k,
                    camera_position_world=camera_position,
                    rotation_camera_to_world=rotation,
                    thermal_temperature_c=temperature,
                    thermal_flame=flame,
                    smoke_estimate=smoke_estimate,
                    confidence=confidence,
                    privileged_transmittance=transmittance,
                    allow_privileged_transmittance=allow_privileged,
                    floor_y_m=self.floor_y_m,
                    config=self.config,
                )
            )

        if evidence_items:
            return self.belief.update_many(evidence_items)
        return self.belief.snapshot(float(timestamp_s))

    def planner_state(self, timestamp_s: float) -> Tuple[RiskLayers, np.ndarray]:
        """Return the selected planner layers without changing evaluator data."""
        if self.source == "oracle":
            layers = self.gt_provider.snapshot(float(timestamp_s))
            unknown_penalty = float(self.config.unknown_risk_prior) * (
                1.0 - layers.confidence
            )
            planning_risk = np.clip(
                np.maximum(layers.physical_risk, unknown_penalty), 0.0, 1.0
            ).astype(np.float32)
        elif self.source == "sensed":
            layers = self.belief.snapshot(float(timestamp_s))
            planning_risk = self.belief.planning_risk_from_layers(layers)
        else:
            # Evaluator-only mode: expose a neutral map and keep navigation off.
            layers = self.belief.snapshot(float(timestamp_s))
            planning_risk = np.zeros(self.frame.shape, dtype=np.float32)
        self._last_layers = layers
        self._last_planning_risk = planning_risk
        return layers, planning_risk

    def record_exposure(
        self,
        timestamp_s: float,
        agent_states: Sequence[object],
        *,
        step: Optional[int] = None,
        planner_statuses: Optional[Sequence[Optional[str]]] = None,
        actions: Optional[Sequence[object]] = None,
        wall_time_s: Optional[float] = None,
    ) -> Dict[str, Dict[str, float]]:
        if (
            planner_statuses is not None
            and len(planner_statuses) != len(agent_states)
        ):
            raise ValueError(
                "planner_statuses and agent_states must have equal length"
            )
        if actions is not None and len(actions) != len(agent_states):
            raise ValueError("actions and agent_states must have equal length")
        self._validate_current_floor(agent_states)
        positions = _agent_positions(agent_states)
        self._last_step_report = self.evaluator.update(
            float(timestamp_s),
            positions,
            floor_y_m=self.floor_y_m,
        )
        statuses = (
            list(planner_statuses)
            if planner_statuses is not None
            else [None] * len(agent_states)
        )
        for agent_id, status in enumerate(statuses):
            counts = self._planner_event_counts.setdefault(
                str(agent_id), {
                    "safe_refusal_steps": 0,
                    "emergency_escape_steps": 0,
                    "trapped_steps": 0,
                },
            )
            key = {
                "unsafe_goal": "safe_refusal_steps",
                "emergency_escape": "emergency_escape_steps",
                "trapped": "trapped_steps",
            }.get(str(status))
            if key is not None:
                counts[key] += 1
        if actions is not None:
            action_entries = []
            for agent_id, action in enumerate(actions):
                action_id = int(action)
                exposure = self._last_step_report[str(agent_id)]
                action_entries.append({
                    "agent_id": int(agent_id),
                    "action": action_id,
                    "action_name": _HABITAT_ACTION_NAMES.get(
                        action_id, f"action_{action_id}"
                    ),
                    "position_after": [
                        float(value) for value in positions[agent_id]
                    ],
                    "risk_after": float(exposure["risk"]),
                    "hard_unsafe_after": bool(exposure["hard_unsafe"]),
                    "planner_status": (
                        None
                        if statuses[agent_id] is None
                        else str(statuses[agent_id])
                    ),
                })
            action_record = {
                "step": None if step is None else int(step),
                "t_sim_s": float(timestamp_s),
                "wall_time_s": (
                    None if wall_time_s is None else float(wall_time_s)
                ),
                "actions": action_entries,
            }
            self._action_steps.append(action_record)
            with self.action_log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(action_record, sort_keys=True) + "\n")
        if step is not None:
            self._append_record({
                "record_type": "exposure",
                "metric_version": self.evaluator.metric_version,
                "episode": self.episode_id,
                "step": int(step),
                "t_sim_s": float(timestamp_s),
                "agents": self._last_step_report,
                "planner_statuses": [
                    None if status is None else str(status)
                    for status in statuses
                ],
            })
        return self._last_step_report

    def save_step(
        self,
        *,
        step: int,
        timestamp_s: float,
        layers: RiskLayers,
        planning_risk: np.ndarray,
        obstacle_map: Optional[np.ndarray] = None,
        agent_cells: Optional[Sequence[Sequence[int]]] = None,
        frontier_points: Optional[Sequence[Sequence[int]]] = None,
        frontier_reports: Optional[Sequence[dict]] = None,
        frontier_computed_step: Optional[int] = None,
    ) -> None:
        """Append a machine-readable trace and optionally save a dashboard."""
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "record_type": "planner_snapshot",
            "metric_version": self.evaluator.metric_version,
            "episode": self.episode_id,
            "step": int(step),
            "t_sim_s": float(timestamp_s),
            "planner_source": self.source,
            "smoke_source": self.smoke_source,
            "geometry_depth_source": self.geometry_depth_source,
            "mean_physical_risk": float(np.mean(layers.physical_risk)),
            "max_physical_risk": float(np.max(layers.physical_risk)),
            "mean_planning_risk": float(np.mean(planning_risk)),
            "known_fraction": float(np.mean(~layers.unknown)),
            "frontiers": list(frontier_reports or []),
            "frontier_computed_step": (
                None
                if frontier_computed_step is None
                else int(frontier_computed_step)
            ),
        }
        self._append_record(record)

        save_every = int(getattr(self.args, "risk_save_every", 0))
        if save_every > 0 and int(step) % save_every == 0:
            save_risk_snapshot(
                self.episode_dir / f"risk_step_{int(step):05d}.png",
                layers,
                planning_risk=planning_risk,
                obstacle_map=obstacle_map,
                agent_cells=agent_cells,
                frontier_points=frontier_points,
                title=(
                    f"Risk source={self.source}  step={int(step)}  "
                    f"t_sim={float(timestamp_s):.1f}s"
                ),
            )

    def summary(self, *, habitat_success: Optional[float] = None) -> Dict[str, object]:
        result = self.evaluator.summary()
        team = result["team"]
        if habitat_success is not None:
            result["safe_success"] = float(
                float(habitat_success) > 0.0
                and int(team["critical_violations"]) == 0
            )
        result["planner_source"] = self.source
        result["smoke_source"] = self.smoke_source
        result["geometry_depth_source"] = self.geometry_depth_source
        result["episode_id"] = self.episode_id
        result["run_id"] = self.run_id
        result["rank"] = self.rank
        result["seed"] = int(getattr(self.args, "seed", 0))
        result["scene_id"] = getattr(self.fire_scene, "scene_id", None)
        result["fire_plan_id"] = _active_fire_plan_id(self.args)
        result["fire_clock_mode"] = str(getattr(
            self.args, "fire_clock_mode", "wallclock"
        ))
        event_keys = (
            "safe_refusal_steps",
            "emergency_escape_steps",
            "trapped_steps",
        )
        result["planner_events"] = {
            "per_agent": {
                agent_id: {
                    key: int(counts.get(key, 0)) for key in event_keys
                }
                for agent_id, counts in sorted(self._planner_event_counts.items())
            }
        }
        for key in event_keys:
            team[key] = int(sum(
                counts.get(key, 0)
                for counts in self._planner_event_counts.values()
            ))
        result["planner_events"]["team"] = {
            key: int(team[key]) for key in event_keys
        }
        team["executed_actions"] = int(sum(
            len(record["actions"]) for record in self._action_steps
        ))
        team["decision_wall_time_s"] = float(sum(
            float(record["wall_time_s"])
            for record in self._action_steps
            if record["wall_time_s"] is not None
        ))
        result["action_trace_file"] = self.action_list_path.name
        return result

    def save_summary(self, summary: Dict[str, object]) -> Path:
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        num_agents = (
            len(self._action_steps[0]["actions"])
            if self._action_steps else 0
        )
        per_agent_action_ids = {
            str(agent_id): [
                int(action["action"])
                for record in self._action_steps
                for action in record["actions"]
                if int(action["agent_id"]) == agent_id
            ]
            for agent_id in range(num_agents)
        }
        per_agent_action_names = {
            str(agent_id): [
                str(action["action_name"])
                for record in self._action_steps
                for action in record["actions"]
                if int(action["agent_id"]) == agent_id
            ]
            for agent_id in range(num_agents)
        }
        action_payload = {
            "metric_version": self.evaluator.metric_version,
            "episode_id": self.episode_id,
            "scene_id": getattr(self.fire_scene, "scene_id", None),
            "fire_plan_id": _active_fire_plan_id(self.args),
            "planner_source": self.source,
            "num_steps": len(self._action_steps),
            "num_agents": num_agents,
            "total_actions": int(sum(
                len(record["actions"]) for record in self._action_steps
            )),
            "total_wall_time_s": float(sum(
                float(record["wall_time_s"])
                for record in self._action_steps
                if record["wall_time_s"] is not None
            )),
            "action_name_by_id": {
                str(key): value for key, value in _HABITAT_ACTION_NAMES.items()
            },
            "per_agent_action_ids": per_agent_action_ids,
            "per_agent_action_names": per_agent_action_names,
            "steps": list(self._action_steps),
        }
        with self.action_list_path.open("w", encoding="utf-8") as stream:
            json.dump(action_payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        path = self.episode_dir / "risk_summary.json"
        with path.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
        return path
