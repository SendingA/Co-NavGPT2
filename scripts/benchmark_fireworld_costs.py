#!/usr/bin/env python3
"""Measure FireWorld timeline baking and online observation costs.

The two costs intentionally use different evidence:

* offline baking is summarized from the real ``pipeline_runner`` manifest,
  whose per-plan ``elapsed_s`` includes propagation, compressed timeline
  writing, validation and publication;
* online latency is measured live at fixed Habitat poses.  Every sample first
  acquires the ordinary Habitat observations and then runs the same
  ``step_fire_observation`` path used by ``main.py`` for each agent.

No images, action lists, videos or raw sensor arrays are written.  The output
is one JSON report and one compact Markdown summary.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import platform
import re
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


DEFAULT_BAKE_RUN = Path(
    "outputs/fire_world/runs/"
    "multi_origin_medium_v12_seed42_36scenes_20260813"
)
DEFAULT_SCENE_ID = "Nfvxx8J5NCo"
DEFAULT_PLAN_ID = "Nfvxx8J5NCo_multi_origin_medium_24e63421b9fa"


def distribution_stats(values: Iterable[float]) -> Dict[str, float]:
    """Return population statistics for a non-empty finite sample."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("statistics require at least one sample")
    if not np.all(np.isfinite(array)):
        raise ValueError("statistics require finite samples")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std_population": float(np.std(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        records.append(value)
    return records


def summarize_bake_manifest(
    manifest_path: Path,
    summary_path: Path | None = None,
) -> Dict[str, Any]:
    """Summarize generated timeline records from a pipeline-run manifest."""

    records = [
        record
        for record in _load_jsonl(manifest_path)
        if record.get("stage") == "timeline"
        and record.get("status") == "generated"
        and isinstance(record.get("elapsed_s"), (int, float))
    ]
    if not records:
        raise ValueError(f"no generated timeline timing records in {manifest_path}")

    elapsed_s = [float(record["elapsed_s"]) for record in records]
    compressed_bytes = [
        int(record.get("timeline", {}).get("size_bytes", 0))
        for record in records
    ]
    uncompressed_bytes = [
        int(record.get("estimate", {}).get("timeline_uncompressed_bytes", 0))
        for record in records
    ]
    voxels = [
        int(record.get("estimate", {}).get("voxels", 0))
        for record in records
    ]
    duration_s = [
        float(record.get("timeline", {}).get("duration_s", 0.0))
        for record in records
    ]
    batch_summary: Mapping[str, Any] = {}
    if summary_path is not None and summary_path.exists():
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{summary_path} is not a JSON object")
        batch_summary = loaded

    task_sum_s = float(sum(elapsed_s))
    batch_wall_s = float(batch_summary.get("wall_time_s", task_sum_s))
    total_simulated_s = float(sum(duration_s))
    fastest = min(records, key=lambda item: float(item["elapsed_s"]))
    slowest = max(records, key=lambda item: float(item["elapsed_s"]))
    propagation_by_plan: Dict[str, float] = {}
    for record in records:
        log_path = Path(str(record.get("log_path", "")))
        if not log_path.is_file():
            continue
        match = re.search(
            r"\[propagation\].*?\bwall=([0-9]+(?:\.[0-9]+)?)s",
            log_path.read_text(encoding="utf-8", errors="replace"),
        )
        if match:
            propagation_by_plan[str(record.get("plan_id", ""))] = float(
                match.group(1)
            )
    correlation = float("nan")
    if len(records) > 1 and statistics.pstdev(voxels) > 0:
        correlation = float(np.corrcoef(elapsed_s, voxels)[0, 1])

    def compact_record(record: Mapping[str, Any]) -> Dict[str, Any]:
        timeline = record.get("timeline", {})
        return {
            "scene_id": str(record.get("scene_id", "")),
            "plan_id": str(record.get("plan_id", "")),
            "elapsed_s": float(record["elapsed_s"]),
            "propagation_only_s": propagation_by_plan.get(
                str(record.get("plan_id", ""))
            ),
            "timeline_shape": list(timeline.get("shape", [])),
            "compressed_size_bytes": int(timeline.get("size_bytes", 0)),
            "voxels_per_frame": int(
                record.get("estimate", {}).get("voxels", 0)
            ),
        }

    result = {
        "source": {
            "manifest": str(manifest_path),
            "summary": str(summary_path) if summary_path is not None else None,
            "timing_boundary": (
                "per-plan elapsed_s covers timeline generation, compressed "
                "write, validation and publication; batch wall includes "
                "scheduler and manifest overhead"
            ),
        },
        "generated_timeline_count": len(records),
        "per_plan_elapsed_s": distribution_stats(elapsed_s),
        "per_plan_elapsed_s_sum": task_sum_s,
        "batch_wall_time_s": batch_wall_s,
        "batch_overhead_s": batch_wall_s - task_sum_s,
        "total_simulated_fire_time_s": total_simulated_s,
        "simulated_seconds_per_batch_wall_second": (
            total_simulated_s / batch_wall_s if batch_wall_s > 0 else None
        ),
        "compressed_size_bytes": {
            **distribution_stats(compressed_bytes),
            "total": int(sum(compressed_bytes)),
        },
        "uncompressed_size_bytes_total": int(sum(uncompressed_bytes)),
        "elapsed_vs_voxels_pearson_r": correlation,
        "fastest": compact_record(fastest),
        "slowest": compact_record(slowest),
        "records": [compact_record(record) for record in records],
    }
    if propagation_by_plan:
        propagation_s = list(propagation_by_plan.values())
        matched_pipeline_s = sum(
            float(record["elapsed_s"])
            for record in records
            if str(record.get("plan_id", "")) in propagation_by_plan
        )
        propagation_sum_s = float(sum(propagation_s))
        non_propagation_s = matched_pipeline_s - propagation_sum_s
        result["propagation_only_elapsed_s"] = distribution_stats(
            propagation_s
        )
        result["propagation_only_elapsed_s_sum"] = propagation_sum_s
        result["pipeline_non_propagation_s_sum"] = non_propagation_s
        result["pipeline_non_propagation_fraction"] = (
            non_propagation_s / matched_pipeline_s
            if matched_pipeline_s > 0
            else None
        )
        result["propagation_log_count"] = len(propagation_s)
    return result


def summarize_observation_samples(
    *,
    habitat_joint_ms: Sequence[float],
    fireworld_agent_ms: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """Summarize matched joint Habitat and per-agent FireWorld samples."""

    base = np.asarray(habitat_joint_ms, dtype=np.float64)
    per_agent = [np.asarray(values, dtype=np.float64) for values in fireworld_agent_ms]
    if not per_agent:
        raise ValueError("at least one agent timing series is required")
    if any(values.shape != base.shape for values in per_agent):
        raise ValueError("all timing series must have the same sample count")
    joint_fire = np.sum(np.stack(per_agent, axis=0), axis=0)
    end_to_end = base + joint_fire
    all_agent_observations = np.concatenate(per_agent)
    base_stats = distribution_stats(base)
    joint_fire_stats = distribution_stats(joint_fire)
    end_stats = distribution_stats(end_to_end)
    return {
        "sample_count_joint_steps": int(base.size),
        "sample_count_agent_fire_observations": int(all_agent_observations.size),
        "habitat_only_joint_ms": base_stats,
        "fireworld_only_per_agent_ms": distribution_stats(all_agent_observations),
        "fireworld_only_by_agent_ms": [
            distribution_stats(values) for values in per_agent
        ],
        "fireworld_only_joint_ms": joint_fire_stats,
        "fire_end_to_end_joint_ms": end_stats,
        "mean_incremental_overhead_ms": joint_fire_stats["mean"],
        "mean_slowdown_ratio": end_stats["mean"] / base_stats["mean"],
        "mean_latency_increase_percent": (
            joint_fire_stats["mean"] / base_stats["mean"] * 100.0
        ),
        "habitat_only_joint_fps": 1000.0 / base_stats["mean"],
        "fire_end_to_end_joint_fps": 1000.0 / end_stats["mean"],
    }


def _synchronize_cuda(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    import torch

    torch.cuda.synchronize(device)


def _timed_ms(call: Callable[[], Any], device: str) -> Tuple[Any, float]:
    _synchronize_cuda(device)
    start = time.perf_counter()
    output = call()
    _synchronize_cuda(device)
    return output, (time.perf_counter() - start) * 1000.0


def _load_habitat_config(args: argparse.Namespace):
    from arguments import load_config

    namespace = argparse.Namespace(
        task_config=args.task_config.replace("configs/", ""),
        config=None,
        num_agents=int(args.num_agents),
        num_humans=0,
        gpu_id=int(args.gpu_id),
        seed=int(args.seed),
        robot_models_enabled=0,
        robot_profiles=None,
        robot_urdfs=None,
        dataset_path=None,
        scenes_dir=None,
        scene_dataset=None,
        frame_width=int(args.frame_width),
        frame_height=int(args.frame_height),
        hfov=float(args.hfov),
        turn_angle=30,
        local_planner="fmm",
    )
    return load_config(namespace)


def _yaw_quaternion_facing(
    source: Sequence[float], target: Sequence[float]
) -> List[float]:
    source_array = np.asarray(source, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    dx = float(target_array[0] - source_array[0])
    dz = float(target_array[2] - source_array[2])
    yaw = math.atan2(-dx, -dz)
    return [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)]


def _choose_camera_poses(
    pathfinder,
    ignitions: Sequence[Mapping[str, Any]],
    num_agents: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Pick navigable fixed poses around one low ignition source."""

    if not ignitions:
        raise ValueError("the selected fire plan has no ignitions")
    ignition = min(
        ignitions,
        key=lambda item: float(np.asarray(item["position"])[1]),
    )
    target = np.asarray(ignition["position"], dtype=np.float64)
    candidates: List[np.ndarray] = []
    seen = set()
    for radius in (1.6, 2.0, 2.4, 2.8, 3.2):
        for angle in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
            probe = np.asarray(
                [
                    target[0] + radius * math.cos(float(angle)),
                    target[1],
                    target[2] + radius * math.sin(float(angle)),
                ],
                dtype=np.float64,
            )
            snapped = np.asarray(pathfinder.snap_point(probe), dtype=np.float64)
            if snapped.shape != (3,) or not np.all(np.isfinite(snapped)):
                continue
            if abs(float(snapped[1] - target[1])) > 1.5:
                continue
            key = tuple(np.round(snapped, 2))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(snapped)
    if len(candidates) < num_agents:
        raise RuntimeError(
            f"only found {len(candidates)} navigable camera poses for "
            f"{num_agents} agents around ignition {target.tolist()}"
        )
    # Spread agents around the source instead of selecting adjacent poses.
    selected = [candidates[0]]
    while len(selected) < num_agents:
        best_index = max(
            range(len(candidates)),
            key=lambda index: min(
                float(
                    np.linalg.norm(
                        candidates[index][[0, 2]] - used[[0, 2]]
                    )
                )
                for used in selected
            ),
        )
        selected.append(candidates.pop(best_index))
    return selected, target


def _normalize_observations(value: Any, num_agents: int) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        observations = [dict(item) for item in value]
    elif num_agents == 1 and isinstance(value, Mapping):
        observations = [dict(value)]
    else:
        raise TypeError(
            f"unexpected Habitat observation type for {num_agents} agents: "
            f"{type(value).__name__}"
        )
    if len(observations) != num_agents:
        raise ValueError(
            f"Habitat returned {len(observations)} observations, "
            f"expected {num_agents}"
        )
    return observations


def _build_suite(
    *,
    mode: str,
    scene,
    camera_k: np.ndarray,
    max_depth_m: float,
    hfov_deg: float,
    device: str,
    seed: int,
):
    from arguments import voxel_smoke_kwargs
    from utils.fire_sensors import FireSensorConfig, FireSensorSuite
    from utils.fire_sensors.config import VoxelSmokeConfig

    fast = mode == "navigation_fast"
    render_args = SimpleNamespace(
        fire_fast=int(fast),
        fire_world_n_steps=24,
        fire_world_render_scale=0.5,
        fire_flame_noise=None,
        fire_world_smoke_k_ext=4.0,
        fire_render_backend="torch",
        fire_render_device=device,
        fire_render_dtype="float16",
        fire_render_max_sample_points=2_000_000,
    )
    voxel_kwargs = voxel_smoke_kwargs(render_args)
    config = FireSensorConfig(
        max_depth_m=float(max_depth_m),
        hfov_deg=float(hfov_deg),
        smoke_density=0.6,
        save_npz=False,
        voxel=VoxelSmokeConfig(**voxel_kwargs),
    )
    suite = FireSensorSuite(
        cfg=config,
        dump_dir="/tmp/fireworld_timing_no_outputs",
        save_every=0,
        seed=int(seed),
        scene=scene,
        camera_K=camera_k,
    )
    return suite, voxel_kwargs


def benchmark_observations(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the matched normal-versus-FireWorld live observation benchmark."""

    import habitat
    import torch
    from habitat import Env

    from utils.fire_pipeline import step_fire_observation
    from utils.fire_world.runtime import FireWorld
    from utils.fire_world.scene import FireClock, FireScene
    from utils.general_utils import get_camera_K

    plan_path = Path(args.scenes_root) / args.scene_id / "plans" / f"{args.plan_id}.json"
    timeline_path = (
        Path(args.fire_world_root) / args.scene_id / args.plan_id / "timeline.npz"
    )
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)
    if not timeline_path.exists():
        raise FileNotFoundError(timeline_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    config = _load_habitat_config(args)
    with habitat.config.read_write(config):
        config.habitat.dataset.content_scenes = [str(args.scene_id)]
    env = Env(config=config)
    mode_reports: Dict[str, Any] = {}
    try:
        env.reset()
        camera_positions, view_target = _choose_camera_poses(
            env.sim.pathfinder,
            plan.get("ignitions", []),
            int(args.num_agents),
        )
        rotation = _yaw_quaternion_facing(camera_positions[0], view_target)
        for agent_id, position in enumerate(camera_positions):
            # Each camera faces the same active source from a separated pose.
            rotation = _yaw_quaternion_facing(position, view_target)
            accepted = env.sim.set_agent_state(
                position, rotation, agent_id=agent_id
            )
            if accepted is False:
                raise RuntimeError(
                    f"Habitat rejected camera pose for agent {agent_id}: "
                    f"{position.tolist()}"
                )

        agent_name = config.habitat.simulator.agents_order[0]
        depth_config = (
            config.habitat.simulator.agents[agent_name]
            .sim_sensors.depth_sensor
        )
        rgb_config = (
            config.habitat.simulator.agents[agent_name]
            .sim_sensors.rgb_sensor
        )
        max_depth_m = float(depth_config.max_depth)
        hfov_deg = float(rgb_config.hfov)
        width = int(rgb_config.width)
        height = int(rgb_config.height)
        camera_k = get_camera_K(width, height, hfov_deg)

        fire_world = FireWorld.load(
            args.scene_id,
            args.plan_id,
            out_root=Path(args.fire_world_root),
        )
        fire_scene = FireScene(
            fw=fire_world,
            clock=FireClock(mode="step", base_t0_s=float(fire_world.times[0])),
        )
        pipeline_args = SimpleNamespace(
            depth_use_clean=-1,
            fire_apply_to_obs=1,
            use_thermal_perception=1,
            fire_save_every=0,
            fire_show_window=0,
            visualize=0,
            print_images=0,
            lidar_360=0,
        )

        def acquire_observations() -> List[Dict[str, Any]]:
            return _normalize_observations(
                env.sim.step(None), int(args.num_agents)
            )

        for mode in args.render_modes:
            suites_and_configs = [
                _build_suite(
                    mode=mode,
                    scene=fire_scene,
                    camera_k=camera_k,
                    max_depth_m=max_depth_m,
                    hfov_deg=hfov_deg,
                    device=args.device,
                    seed=args.seed + agent_id,
                )
                for agent_id in range(int(args.num_agents))
            ]
            suites = [item[0] for item in suites_and_configs]
            voxel_settings = suites_and_configs[0][1]

            def process_all(
                observations: List[Dict[str, Any]], t_sim_s: float
            ) -> Tuple[List[float], List[Mapping[str, Any]]]:
                timings: List[float] = []
                outputs: List[Mapping[str, Any]] = []
                for agent_id, (observation, suite) in enumerate(
                    zip(observations, suites)
                ):
                    sensors, elapsed_ms = _timed_ms(
                        lambda observation=observation, suite=suite, agent_id=agent_id: (
                            step_fire_observation(
                                observations=observation,
                                suite=suite,
                                agent_state=env.sim.get_agent_state(agent_id),
                                robot_step=0,
                                config=config,
                                args=pipeline_args,
                                t_sim_s=t_sim_s,
                            )
                        ),
                        args.device,
                    )
                    if sensors is None:
                        raise RuntimeError("FireSensorSuite unexpectedly returned None")
                    timings.append(elapsed_ms)
                    outputs.append(sensors)
                return timings, outputs

            cold_observations, cold_habitat_ms = _timed_ms(
                acquire_observations, args.device
            )
            cold_agent_ms, cold_outputs = process_all(
                cold_observations, float(args.fire_time_s) - 40.0
            )
            for warmup_index in range(int(args.warmup)):
                warm_observations = acquire_observations()
                process_all(
                    warm_observations,
                    float(args.fire_time_s) - 20.0
                    + (warmup_index // 5) * 2.0,
                )

            habitat_joint_ms: List[float] = []
            fireworld_agent_ms: List[List[float]] = [
                [] for _ in range(int(args.num_agents))
            ]
            renderer_backends = set()
            renderer_devices = set()
            for sample_index in range(int(args.samples)):
                observations, base_ms = _timed_ms(
                    acquire_observations, args.device
                )
                # Match the navigation step-clock: one new fire time every
                # five robot steps, advancing by two simulated seconds.
                t_sim_s = float(args.fire_time_s) + (sample_index // 5) * 2.0
                agent_ms, outputs = process_all(observations, t_sim_s)
                habitat_joint_ms.append(base_ms)
                for agent_id, elapsed_ms in enumerate(agent_ms):
                    fireworld_agent_ms[agent_id].append(elapsed_ms)
                renderer_backends.update(
                    str(output.get("fire_render_backend", "unknown"))
                    for output in outputs
                )
                renderer_devices.update(
                    str(output.get("fire_render_device", "unknown"))
                    for output in outputs
                )

            summary = summarize_observation_samples(
                habitat_joint_ms=habitat_joint_ms,
                fireworld_agent_ms=fireworld_agent_ms,
            )
            summary.update(
                {
                    "mode": mode,
                    "requested_settings": voxel_settings,
                    "actual_renderer_backends": sorted(renderer_backends),
                    "actual_renderer_devices": sorted(renderer_devices),
                    "cold_start": {
                        "habitat_joint_ms": cold_habitat_ms,
                        "fireworld_agent_ms": cold_agent_ms,
                        "fire_end_to_end_joint_ms": (
                            cold_habitat_ms + sum(cold_agent_ms)
                        ),
                        "actual_renderer_backends": sorted(
                            {
                                str(output.get("fire_render_backend", "unknown"))
                                for output in cold_outputs
                            }
                        ),
                    },
                    "raw_samples": {
                        "habitat_joint_ms": habitat_joint_ms,
                        "fireworld_agent_ms": fireworld_agent_ms,
                    },
                }
            )
            if renderer_backends != {"torch"} or not all(
                device.startswith("cuda") for device in renderer_devices
            ):
                raise RuntimeError(
                    "CUDA timing requested but renderer reported "
                    f"backends={renderer_backends}, devices={renderer_devices}"
                )
            mode_reports[mode] = summary

            # Release the mode-specific cached GPU volumes before building the
            # next quality mode. FireWorld's CPU timeline remains shared.
            del suites
            del suites_and_configs
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

        gpu = None
        if str(args.device).startswith("cuda"):
            gpu_index = torch.device(args.device).index or 0
            properties = torch.cuda.get_device_properties(gpu_index)
            gpu = {
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "torch_version": str(torch.__version__),
                "cuda_runtime": str(torch.version.cuda),
            }
        return {
            "scope": (
                "fixed-pose sensor observation only; excludes detector, "
                "mapping, frontier assignment and local/global planning"
            ),
            "dataset_config": str(args.task_config),
            "scene_id": str(args.scene_id),
            "plan_id": str(args.plan_id),
            "fire_type": str(plan.get("fire_type", "")),
            "intensity": str(plan.get("intensity", "")),
            "ignition_count": len(plan.get("ignitions", [])),
            "num_agents": int(args.num_agents),
            "resolution": [width, height],
            "hfov_deg": hfov_deg,
            "warmup_steps": int(args.warmup),
            "measured_joint_steps_per_mode": int(args.samples),
            "fire_clock_sampling": {
                "mode": "step",
                "base_time_s": float(args.fire_time_s),
                "steps_per_unit": 5,
                "seconds_per_unit": 2.0,
            },
            "camera_positions": [position.tolist() for position in camera_positions],
            "view_target": view_target.tolist(),
            "device_requested": str(args.device),
            "gpu": gpu,
            "modes": mode_reports,
        }
    finally:
        env.close()


def _format_seconds(value: float) -> str:
    if value >= 3600.0:
        return f"{value / 3600.0:.3f} h"
    if value >= 60.0:
        return f"{value / 60.0:.3f} min"
    return f"{value:.3f} s"


def render_markdown(report: Mapping[str, Any]) -> str:
    bake = report["timeline_baking"]
    lines = [
        "# FireWorld timing report",
        "",
        f"Generated: {report['created_at']}",
        "",
        "## Offline timeline baking",
        "",
        "| Quantity | Result |",
        "|---|---:|",
        f"| Generated timelines | {bake['generated_timeline_count']} |",
        f"| Complete batch wall time | {_format_seconds(bake['batch_wall_time_s'])} |",
        f"| Sum of per-plan elapsed time | {_format_seconds(bake['per_plan_elapsed_s_sum'])} |",
        f"| Per-plan mean | {bake['per_plan_elapsed_s']['mean']:.3f} s |",
        f"| Per-plan median | {bake['per_plan_elapsed_s']['median']:.3f} s |",
        f"| Per-plan P95 | {bake['per_plan_elapsed_s']['p95']:.3f} s |",
        f"| Per-plan min / max | {bake['per_plan_elapsed_s']['min']:.3f} / {bake['per_plan_elapsed_s']['max']:.3f} s |",
        f"| Compressed timelines total | {bake['compressed_size_bytes']['total'] / 2**30:.3f} GiB |",
        f"| Simulated-time throughput | {bake['simulated_seconds_per_batch_wall_second']:.3f}x real time |",
        "",
    ]
    if "propagation_only_elapsed_s" in bake:
        insertion = [
            f"| Propagation-only mean | {bake['propagation_only_elapsed_s']['mean']:.3f} s |",
            f"| Propagation-only sum | {_format_seconds(bake['propagation_only_elapsed_s_sum'])} |",
            f"| Compression, validation and publication share | {bake['pipeline_non_propagation_fraction'] * 100.0:.2f}% of per-plan elapsed |",
        ]
        # Insert before the table's terminating blank line.
        lines[-1:-1] = insertion
    observation = report.get("observation_benchmark")
    if observation:
        lines.extend(
            [
                "## Online observation latency",
                "",
                (
                    f"Matched {observation['num_agents']}-agent fixed-pose "
                    f"observations at {observation['resolution'][0]}x"
                    f"{observation['resolution'][1]}; detector, mapping and "
                    "planning are excluded."
                ),
                "",
                "| Mode | Habitat only joint mean / P50 / P95 / std | FireWorld only per-agent mean / P50 / P95 / std | FireWorld only joint mean / P50 / P95 / std | End-to-end joint mean / P50 / P95 / std | Slowdown | End-to-end FPS |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for mode in ("navigation_fast", "full_quality"):
            if mode not in observation["modes"]:
                continue
            value = observation["modes"][mode]
            base = value["habitat_only_joint_ms"]
            agent = value["fireworld_only_per_agent_ms"]
            joint = value["fireworld_only_joint_ms"]
            total = value["fire_end_to_end_joint_ms"]
            lines.append(
                f"| {mode} | {base['mean']:.3f} / {base['median']:.3f} / "
                f"{base['p95']:.3f} / {base['std_population']:.3f} ms "
                f"| {agent['mean']:.3f} / {agent['median']:.3f} / "
                f"{agent['p95']:.3f} / {agent['std_population']:.3f} ms "
                f"| {joint['mean']:.3f} / {joint['median']:.3f} / "
                f"{joint['p95']:.3f} / {joint['std_population']:.3f} ms "
                f"| {total['mean']:.3f} / {total['median']:.3f} / "
                f"{total['p95']:.3f} / {total['std_population']:.3f} ms "
                f"| {value['mean_slowdown_ratio']:.2f}x "
                f"| {value['fire_end_to_end_joint_fps']:.2f} |"
            )
        lines.extend(
            [
                "",
                "The navigation-fast mode is the current benchmark path "
                "(`fire_fast=1`, effective 10 ray steps and 0.35 render "
                "scale). Full-quality uses 24 steps, 0.5 scale and procedural "
                "flame/smoke texture. Warm samples exclude the separately "
                "reported first FireWorld call.",
                "",
            ]
        )
        matched_bake = next(
            (
                record
                for record in bake.get("records", [])
                if record.get("plan_id") == observation.get("plan_id")
            ),
            None,
        )
        if matched_bake is not None:
            propagation_text = (
                f", including {matched_bake['propagation_only_s']:.3f} s "
                "of propagation"
                if matched_bake.get("propagation_only_s") is not None
                else ""
            )
            lines.extend(
                [
                    (
                        f"The measured observation plan itself baked in "
                        f"{matched_bake['elapsed_s']:.3f} s{propagation_text}; "
                        f"its compressed timeline is "
                        f"{matched_bake['compressed_size_bytes'] / 2**20:.1f} MiB."
                    ),
                    "",
                ]
            )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bake-manifest",
        type=Path,
        default=DEFAULT_BAKE_RUN / "manifest.jsonl",
    )
    parser.add_argument(
        "--bake-summary",
        type=Path,
        default=DEFAULT_BAKE_RUN / "summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmarks/fireworld_timing"),
    )
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--task-config", default="person_objectnav_hm3d.yaml")
    parser.add_argument("--scenes-root", default="scenes")
    parser.add_argument("--fire-world-root", default="outputs/fire_world")
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--plan-id", default=DEFAULT_PLAN_ID)
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=79.0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--fire-time-s", type=float, default=300.0)
    parser.add_argument(
        "--render-modes",
        nargs="+",
        choices=("navigation_fast", "full_quality"),
        default=("navigation_fast", "full_quality"),
    )
    args = parser.parse_args(argv)
    if args.num_agents < 1:
        parser.error("--num-agents must be positive")
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "timeline_baking": summarize_bake_manifest(
            args.bake_manifest, args.bake_summary
        ),
    }
    if not args.skip_render:
        report["observation_benchmark"] = benchmark_observations(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "report.json"
    markdown_path = args.output_dir / "report.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"[fireworld-costs] JSON: {json_path}")
    print(f"[fireworld-costs] Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
