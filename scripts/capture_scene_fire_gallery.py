#!/usr/bin/env python3
"""One merged medium plan / timeline and four real FireSensor views per scene.

All derived inputs and products live in the output directory. Original plans,
timelines and navigation sensor defaults are never rewritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capture_fire_type_gallery import (
    FIRE_TYPE_LABELS, _candidate_positions, _make_suite, _save_observation,
    _write_image, balanced_visibility_score, make_montage,
)

TYPES = tuple(FIRE_TYPE_LABELS)
DEFAULT_OUTPUT = ROOT / "outputs/fire_gallery_medium_36_20260911"
RENDER_VERSION = 3
GALLERY_SMOKE_EXTINCTION = 3.0


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if not path.exists() or path.read_text() != text:
        temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temp.write_text(text)
        temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def merge_plans(plans):
    """Union exact initial objects, retaining multi-origin values on overlap."""
    base = plans["multi_origin"]
    merged = deepcopy(base)
    sources, memberships = {}, {}
    for fire_type in ("multi_origin", *TYPES[:3]):
        if fire_type not in plans:
            continue
        plan = plans[fire_type]
        if plan["scene_id"] != base["scene_id"] or plan["intensity"] != "medium":
            raise ValueError("all merged plans must belong to the same medium scene")
        for ignition in plan["ignitions"]:
            if ignition["ignite_time_s"] != 0 or ignition.get("ignition_role") != "initial":
                raise ValueError("gallery supports initial-only source plans")
            oid = int(ignition["object_id"])
            sources.setdefault(oid, deepcopy(ignition))
            memberships.setdefault(str(oid), []).append(fire_type)
    merged["ignitions"] = [sources[k] for k in sorted(sources)]
    merged["num_initial_ignitions"] = len(sources)
    merged["num_fire_sources"] = len(sources)
    merged["ignition_selection_mode"] = "gallery_union_of_medium_template_sources"
    merged["gallery_source_templates"] = memberships
    merged["gallery_base_plan_ids"] = {t: p["plan_id"] for t, p in plans.items()}
    merged["gallery_missing_types"] = [t for t in TYPES if t not in plans]
    merged["gallery_disclosure"] = (
        "Union of the available medium template initial-source sets, deduplicated "
        "by object ID. Medium multi-origin propagation rules and duration are "
        "preserved. Source count and area distribution are custom gallery settings, "
        "not the canonical area-stratified multi-origin sampling policy."
    )
    for key in ("plan_id", "plan_hash", "multi_origin_area_policy",
                "num_initial_ignitions_requested"):
        merged.pop(key, None)
    digest = hashlib.sha256(json.dumps(merged, sort_keys=True).encode()).hexdigest()[:12]
    merged["plan_hash"] = digest
    merged["plan_id"] = f"{base['scene_id']}_multi_origin_medium_gallery_union_{digest}"
    return merged


def prepare(output):
    records = []
    for shard in sorted((ROOT / "data/datasets/objectnav_hm3d_person_v1/val/content").glob("*.json.gz")):
        sid = shard.name.split(".")[0]
        plans, inputs = {}, []
        for path in sorted((ROOT / "scenes" / sid / "plans").glob("*.json")):
            plan = json.loads(path.read_text())
            if (plan.get("intensity") != "medium" or "gallery" in path.name
                    or "paper_spread" in path.name or plan.get("fire_type") not in TYPES):
                continue
            if plan["fire_type"] in plans:
                raise ValueError(f"ambiguous canonical plan: {sid} {plan['fire_type']}")
            plans[plan["fire_type"]] = plan
            inputs.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
        plan = merge_plans(plans)
        dest = output / sid / "plan.json"
        write_json(dest, plan)
        inventory = ROOT / "scenes" / sid / "inventory.json"
        with gzip.open(shard, "rt") as f:
            episode = json.load(f)["episodes"][0]
        records.append({
            "scene_id": sid, "plan_id": plan["plan_id"],
            "plan_path": str(dest), "plan_sha256": sha256(dest),
            "inventory_path": str(inventory), "inventory_sha256": sha256(inventory),
            "canonical_inputs": inputs, "source_count": len(plan["ignitions"]),
            "missing_types": plan["gallery_missing_types"],
            "scene_mesh": str(ROOT / "data/scene_datasets" / episode["scene_id"]),
        })
    if len(records) != 36:
        raise ValueError(f"expected 36 scenes, got {len(records)}")
    write_json(output / "inputs.json", {"scenes": records, "expected_views": 144})
    print(f"[prepare] {len(records)} merged plans; {sum(r['source_count'] for r in records)} sources", flush=True)
    return records


def bake_one(record, output):
    from utils.fire_world.propagation import run_propagation
    output = Path(output)
    dest = output / "timelines" / record["scene_id"] / record["plan_id"]
    provenance = dest / "provenance.json"
    expected = {k: record[k] for k in ("plan_sha256", "inventory_sha256")}
    expected.update(voxel_m=0.15, dt=0.5, save_dt=10.0, duration_s=600.0,
                    solver_sha256=sha256(ROOT / "utils/fire_world/propagation.py"))
    if provenance.exists() and json.loads(provenance.read_text()) == expected and (dest / "timeline.npz").exists():
        return record["scene_id"], "cached"
    start = time.monotonic()
    plan = json.loads(Path(record["plan_path"]).read_text())
    inv = json.loads(Path(record["inventory_path"]).read_text())
    run_propagation(inv, plan, voxel_m=.15, dt=.5, save_dt=10., out_dir=dest)
    with np.load(dest / "timeline.npz", allow_pickle=False) as z:
        assert len(z["times"]) == 61 and z["times"][-1] == 600
        assert np.max(z["flame"][0]) > 0
    write_json(provenance, expected)
    return record["scene_id"], round(time.monotonic() - start, 2)


def gray_haze(rgb, depth, transmittance, flame, density=.045):
    """Depth-aware gray presentation haze, suppressed in physical fire/smoke.

    Returns an RGB display and alpha without mutating any sensor/GT arrays.
    This aesthetic layer is explicitly excluded from physical transmittance.
    """
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    clear = np.clip((np.asarray(transmittance) - .65) / .35, 0, 1)
    clear = clear * clear * (3 - 2 * clear)
    protect = 1 - np.clip(np.asarray(flame) / .06, 0, 1)
    alpha = np.minimum(.22, 1 - np.exp(-density * np.clip(depth, 0, 12))) * clear * protect
    image = np.rint(np.asarray(rgb, dtype=np.float32) * (1-alpha[..., None]) + 140 * alpha[..., None])
    return np.clip(image, 0, 255).astype(np.uint8), alpha.astype(np.float32)


def create_sim(record, width, height):
    import habitat_sim
    from utils.fire_sensors.lidar_360 import LIDAR_DEPTH_YAW
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = record["scene_mesh"]
    cfg.scene_dataset_config_file = str(ROOT / "data/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json")
    cfg.gpu_device_id = 0
    cfg.enable_physics = False
    specs = []
    for uuid, sensor_type in (("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH), ("semantic", habitat_sim.SensorType.SEMANTIC)):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.sensor_type = uuid, sensor_type
        spec.resolution, spec.hfov, spec.position = [height, width], 79., [0., 1.3, 0.]
        specs.append(spec)
    for uuid, yaw in LIDAR_DEPTH_YAW.items():
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.sensor_type = uuid, habitat_sim.SensorType.DEPTH
        spec.resolution, spec.hfov = [256, 256], 90.
        spec.position, spec.orientation = [0., 1.3, 0.], [0., float(yaw), 0.]
        specs.append(spec)
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = specs
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))
    if not sim.pathfinder.is_loaded:
        navmesh = Path(record["scene_mesh"]).with_suffix("").with_suffix(".navmesh")
        if not sim.pathfinder.load_nav_mesh(str(navmesh)):
            raise RuntimeError(f"navmesh not loaded: {navmesh}")
    return sim


def observe(sim, position, source):
    import habitat_sim
    import quaternion
    from utils.fire_sensors.lidar_360 import LIDAR_DEPTH_UUIDS
    position = np.asarray(position, dtype=float)
    delta = np.asarray(source["position"], dtype=float) - position
    yaw = math.atan2(-delta[0], -delta[2])
    qyaw = quaternion.from_rotation_vector([0., yaw, 0.])
    state = habitat_sim.AgentState()
    state.position, state.rotation = position, qyaw
    agent = sim.get_agent(0)
    agent.set_state(state)
    state = agent.get_state()
    # Look slightly above the source center; retain a level body / 360 lidar.
    target_y = max(position[1] + .4, float(source["position"][1]) + .18)
    pitch = math.atan2(target_y - (position[1] + 1.3), max(.1, np.linalg.norm(delta[[0, 2]])))
    pitch = float(np.clip(pitch, -.55, .25))
    rotation = qyaw * quaternion.from_rotation_vector([pitch, 0., 0.])
    for uuid in ("rgb", "depth", "semantic"):
        state.sensor_states[uuid].rotation = rotation
    agent.set_state(state, infer_sensor_states=False)
    obs = sim.get_sensor_observations()
    obs["rgb"] = obs["rgb"][..., :3]
    obs["depth"] = np.clip(obs["depth"], 0., 12.).astype(np.float32)
    for uuid in LIDAR_DEPTH_UUIDS:
        obs[uuid] = np.clip(obs[uuid], 0., 12.).astype(np.float32) / 12.
    return obs, agent.get_state()


def make_suite(scene, width, height, final, seed):
    from utils.general_utils import get_camera_K
    k = get_camera_K(width, height, 79.)
    suite = _make_suite(scene=scene, camera_k=k, output_dir=Path("."), seed=seed,
        max_depth_m=12., hfov_deg=79., n_steps=96 if final else 24,
        render_scale=1. if final else .25, device="cuda:0", procedural=True,
        smoke_density=.6, smoke_noise_strength=.24, flame_noise_strength=.9,
        flame_edge_break=1.15, flame_color_jitter=.28, flame_glow_ksize=13,
        flame_glow_gain=.18, flame_surface_reveal=.08, flame_highlight_compression=.85)
    # Gallery-only optical preset; no changes to baked fields or nav defaults.
    suite.cfg.voxel.smoke_k_ext = GALLERY_SMOKE_EXTINCTION
    suite.cfg.voxel.flame_emission_gain = 8.0
    suite.cfg.voxel.flame_threshold = .015
    return suite


def source_groups(plan, inv):
    from utils.fire_world.templates import TEMPLATE_CATEGORY_GROUPS
    objects = {int(o.get("object_id", o.get("instance_id"))): o for o in inv.get("instances", inv.get("objects", []))}
    groups, used = [], set()
    for slot, kind in enumerate(TYPES):
        membership = plan["gallery_source_templates"]
        pool = [s for s in plan["ignitions"] if kind in membership[str(s["object_id"])]]
        fallback = not pool
        if fallback or kind == "multi_origin":
            pool = list(plan["ignitions"])
        primary = TEMPLATE_CATEGORY_GROUPS.get(kind, {}).get("primary", ())
        def score(source):
            oid = int(source["object_id"])
            ob = objects[oid]
            size = np.asarray(ob["aabb_max"]) - np.asarray(ob["aabb_min"])
            # Prefer recognizable complete objects over tiny textiles/electronics.
            category = source["category"]
            rank = (len(primary) - primary.index(category)) if category in primary else 0
            if fallback or kind == "multi_origin":
                # Missing template slots should add variety instead of
                # repeating the bedroom/electrical template's source set.
                tags = membership[str(oid)]
                rank = 100 if tags == ["multi_origin"] else (50 if "multi_origin" in tags else 0)
            return (oid not in used, rank, min(float(np.prod(np.maximum(size, .05))), 8.))
        pool.sort(key=score, reverse=True)
        groups.append({"slot": slot+1, "requested_type": kind,
                       "semantic_fallback": fallback, "sources": pool})
        used.add(int(pool[0]["object_id"]))
    return groups


def floor_aware_positions(pathfinder, source_position):
    """Avoid snapping a raised source (e.g. wall TV) onto the floor above it."""
    source = np.asarray(source_position, dtype=float)
    anchors, seen_floors = [], set()
    for drop in (0., .4, .8, 1.3, 2., 2.8):
        probe = source.copy()
        probe[1] -= drop
        anchor = np.asarray(pathfinder.snap_point(probe), dtype=float)
        if (not np.isfinite(anchor).all() or anchor[1] > source[1] + .2
                or source[1] - anchor[1] > 3.2):
            continue
        key = round(float(anchor[1]), 1)
        if key not in seen_floors:
            seen_floors.add(key)
            anchors.append(anchor)
    if not anchors:
        return _candidate_positions(pathfinder, source)
    positions, seen = [], set()
    for anchor in anchors:
        class FloorAnchor:
            def snap_point(self, point):
                if np.allclose(point, source):
                    return anchor
                return pathfinder.snap_point(point)
        for position in _candidate_positions(FloorAnchor(), source):
            key = tuple(np.round(position, 2))
            if key not in seen:
                positions.append(position)
                seen.add(key)
    return positions


def choose_view(sim, suite, group, used_sources, used_positions, used_regions):
    ranked = []
    # First reject occluded source views using the native semantic image.
    pool = [s for s in group["sources"] if s["object_id"] not in used_sources]
    pool = pool or group["sources"]
    if group["semantic_fallback"] or group["requested_type"] == "multi_origin":
        distinct = [s for s in pool if s.get("region_id") not in used_regions]
        pool = distinct or pool
        used_categories = {s["category"] for s in group["sources"] if s["object_id"] in used_sources}
        pool.sort(key=lambda s: s["category"] in used_categories)
    for source in pool[:4]:
        positions = floor_aware_positions(sim.pathfinder, source["position"])
        for position in positions:
            if any(np.linalg.norm(position - p) < .7 for p in used_positions):
                continue
            obs, _ = observe(sim, position, source)
            visible = float(np.mean(obs["semantic"] == source["object_id"]))
            obstruction = float(np.mean(obs["depth"] < .65))
            rank = 4 * min(visible, .18) - 2 * obstruction
            ranked.append((rank, position.copy(), source, visible))
    ranked.sort(key=lambda x: x[0], reverse=True)
    clear_source_views = [entry for entry in ranked if entry[3] >= .0005]
    if clear_source_views:
        ranked = clear_source_views
    if not ranked:
        raise RuntimeError("no distinct navigable pose for requested view")
    # Ensure more than one source survives the clean-view shortlist.
    shortlist, counts = [], {}
    for entry in ranked:
        oid = entry[2]["object_id"]
        if counts.get(oid, 0) < 6:
            shortlist.append(entry)
            counts[oid] = counts.get(oid, 0) + 1
        if len(shortlist) >= 18:
            break
    best, previews = None, []
    for _, position, source, visible in shortlist:
        obs, state = observe(sim, position, source)
        for t in (10., 30., 90.):
            out = suite.process(obs["rgb"], obs["depth"], obs=obs, agent_state=state,
                                t_sim_s=t, diagnostics=False)
            metrics = balanced_visibility_score(out)
            metrics["source_visible_fraction"] = visible
            score = metrics["score"] + 4 * min(visible/.03,1) - 20 * max(0., .65-metrics["central_flame_ratio"])
            entry = (score, position, source, t, metrics)
            if best is None or entry[0] > best[0]:
                best = entry
    assert best is not None
    for t in (10., 30., 60., 150., 240., 360.):
        obs, state = observe(sim, best[1], best[2])
        out = suite.process(obs["rgb"], obs["depth"], obs=obs, agent_state=state,
                            t_sim_s=t, diagnostics=False)
        metrics = balanced_visibility_score(out)
        visible = float(np.mean(obs["semantic"] == best[2]["object_id"]))
        metrics["source_visible_fraction"] = visible
        score = metrics["score"] + 4 * min(visible/.03,1) - 20 * max(0., .65-metrics["central_flame_ratio"])
        previews.append((out["rgb_smoke"].copy(), t, metrics))
        if score > best[0]:
            best = (score, best[1], best[2], t, metrics)
    if (used_regions and (group["semantic_fallback"] or group["requested_type"] == "multi_origin")
            and (best[4]["rgb_fire_fraction"] < .001
                 or best[4].get("source_visible_fraction", 0) < .0005
                 or best[4]["central_flame_ratio"] < .35)):
        # A distinct semantic region can be inaccessible or its one lamp can
        # be hidden inside geometry. Prefer a visible different source over
        # an empty fourth image, while retaining the distinct-camera rule.
        alternative, extra = choose_view(sim, suite, group, used_sources, used_positions, set())
        if alternative[0] > best[0]:
            alternative[4]["region_diversity_relaxed"] = True
            return alternative, extra
    return best, previews


def capture(record, output, width=960, height=720, force=False, reuse_views_from=None):
    from utils.fire_world.runtime import FireWorld
    from utils.fire_world.scene import FireScene, FireClock
    from scripts.keyboard_teleop_full import compose_view
    import quaternion
    import torch
    dest = output / record["scene_id"]
    manifest = dest / "manifest.json"
    if manifest.exists() and not force:
        old = json.loads(manifest.read_text())
        if old.get("complete") and old.get("plan_sha256") == record["plan_sha256"] and old.get("render_version") == RENDER_VERSION:
            print(f"[render] cached {record['scene_id']}", flush=True)
            return
    plan = json.loads(Path(record["plan_path"]).read_text())
    inv = json.loads(Path(record["inventory_path"]).read_text())
    fw = FireWorld.load(record["scene_id"], record["plan_id"], out_root=output / "timelines")
    scene = FireScene(fw=fw, clock=FireClock(mode="step"))
    # Camera search needs only a low-resolution native view. Keep the final
    # camera separate so selected captures still render natively at 960x720.
    sim = None if reuse_views_from else create_sim(record, 320, 240)
    final_sim = None
    previews = make_suite(scene, 320, 240, False, 17)
    previews.cfg.voxel.render_scale = .75
    final = make_suite(scene, width, height, True, 117)
    views, used_sources, used_positions, tiles, dashboards = [], set(), [], [], []
    used_regions = set()
    try:
        chosen = []
        if reuse_views_from:
            reference = json.loads((Path(reuse_views_from)/record['scene_id']/"manifest.json").read_text())
            if reference['plan_sha256'] != record['plan_sha256']:
                raise ValueError("Cannot reuse views from a different plan")
            for view in reference['views']:
                group = {k:view[k] for k in ('slot','requested_type','semantic_fallback')}
                best = (0., np.asarray(view['agent_position']), view['selected_source'],
                        view['fire_time_s'], view['preview_metrics'])
                chosen.append((group,best))
        else:
            for group in source_groups(plan, inv):
                best, search = choose_view(sim, previews, group, used_sources, used_positions, used_regions)
                chosen.append((group, best))
                used_sources.add(best[2]["object_id"])
                used_regions.add(best[2].get("region_id"))
                used_positions.append(best[1])
        # Habitat's windowless GL context is process-global. Never keep two
        # Simulator instances alive while creating/destroying their contexts.
        if sim is not None:
            sim.close()
        del sim
        sim = None
        gc.collect()
        final_sim = create_sim(record, width, height)
        from dataclasses import asdict
        write_json(output/"sensor_config.json", asdict(final.cfg))
        for group, best in chosen:
            _, position, source, fire_time, preview_metrics = best
            view_dir = dest / f"{group['slot']:02d}_{group['requested_type']}"
            obs, state = observe(final_sim, position, source)
            outputs = final.process(obs["rgb"], obs["depth"], obs=obs, agent_state=state,
                                    t_sim_s=fire_time, diagnostics=True)
            metrics = balanced_visibility_score(outputs)
            metrics["source_visible_fraction"] = float(np.mean(obs["semantic"] == source["object_id"]))
            physical_rgb = outputs["rgb_smoke"].copy()
            display, alpha = gray_haze(physical_rgb, outputs["depth_clean"], outputs["transmittance"], outputs["thermal_flame_mask"])
            outputs["rgb_smoke"] = display
            label = (FIRE_TYPE_LABELS[group["requested_type"]] if not group["semantic_fallback"]
                     else f"Additional origin: {source['category']}")
            status = [f"Scene: {record['scene_id']} | medium multi-origin | view {group['slot']}/4",
                      f"Source: {source['category']} #{source['object_id']} | region {source.get('region_id')} | t={fire_time:.0f}s",
                      "Shared merged plan | Native FireSensor modalities | RGB includes subtle presentation haze"]
            dashboard = compose_view(rgb_clean=outputs["rgb"], rgb_smoke=display,
                depth_clean=outputs["depth_clean"], depth_smoke=outputs["depth_smoke"],
                thermal=outputs["thermal_image"], lidar=outputs["lidar_image"],
                radar_bev=outputs["radar_image_bev"], radar_az=outputs["radar_image_az"],
                radar_el=outputs["radar_image_el"], status_lines=status, max_d=12.,
                dashboard_size=(2400, 1080), title=f"FireSensor Observation | {label}")
            files = _save_observation(output_dir=view_dir, outputs=outputs, dashboard=dashboard)
            _write_image(view_dir / "template_scene.png", display, rgb=True)
            _write_image(view_dir / "rgb_fire_smoke_physical.png", physical_rgb, rgb=True)
            _write_image(view_dir / "fire_sensor_observation.png", dashboard)
            _write_image(view_dir / "source_object_mask.png", (obs["semantic"] == source["object_id"]).astype(np.uint8)*255)
            np.savez_compressed(view_dir / "presentation_haze.npz", alpha=alpha,
                                rgb_fire_smoke_physical=physical_rgb)
            subtitle = f"{source['category']} | region {source.get('region_id')} | t={fire_time:.0f}s"
            tiles.append((cv2.cvtColor(display, cv2.COLOR_RGB2BGR), label, subtitle))
            dashboards.append((dashboard, label, subtitle))
            sensor_state = state.sensor_states["depth"]
            q = sensor_state.rotation
            item = {"slot":group["slot"], "requested_type":group["requested_type"],
                    "semantic_fallback":group["semantic_fallback"], "label":label,
                    "directory":view_dir.name, "plan_id":record["plan_id"],
                    "selected_source":source, "fire_time_s":fire_time,
                    "agent_position":position.tolist(),
                    "camera_position":np.asarray(sensor_state.position).tolist(),
                    "camera_rotation_xyzw":[float(q.x),float(q.y),float(q.z),float(q.w)],
                    "hfov_deg":79., "resolution":[width,height], "sensor_max_depth_m":12.,
                    "ray_samples":96, "render_scale":1., "smoke_k_ext":GALLERY_SMOKE_EXTINCTION, "preview_metrics":preview_metrics,
                    "reused_views_from":str(reuse_views_from) if reuse_views_from else None,
                    "render_version":RENDER_VERSION,
                    "preview_seed":17,"final_sensor_seed":117,
                    "preview_native_resolution":[320,240],"preview_ray_resolution":[240,180],
                    "physical_visibility_metrics":metrics, "lidar_is_360":bool(outputs["lidar_is_360"]),
                    "renderer":{"backend":outputs["fire_render_backend"],"device":outputs["fire_render_device"]},
                    "presentation_haze":{"density_per_m":.045,"gray_rgb":[140,140,140],"max_alpha":.22,
                        "physical_transmittance_unchanged":True},
                    "files":files+["template_scene.png","rgb_fire_smoke_physical.png","fire_sensor_observation.png","presentation_haze.npz","source_object_mask.png"]}
            write_json(view_dir / "manifest.json", item)
            views.append(item)
            print(f"[render] {record['scene_id']} {group['slot']}/4 {source['category']} t={fire_time} fire={metrics['rgb_fire_fraction']:.4f} smoke={metrics['smoke_fraction']:.4f}", flush=True)
        _write_image(dest / "four_templates.png", make_montage(tiles, tile_size=(960,720)))
        dashboard_height = round(dashboards[0][0].shape[0] * 1200 / dashboards[0][0].shape[1])
        _write_image(dest / "four_sensor_observations.jpg", make_montage(dashboards, tile_size=(1200,dashboard_height)))
        write_json(manifest, {"scene_id":record["scene_id"],"plan_id":record["plan_id"],
            "plan_sha256":record["plan_sha256"],"complete":True,"views":views,"render_version":RENDER_VERSION,
            "timeline":str((output/"timelines"/record["scene_id"]/record["plan_id"]/"timeline.npz").relative_to(output)),
            "disclosure":plan["gallery_disclosure"]})
    finally:
        if sim is not None:
            sim.close()
        if final_sim is not None:
            final_sim.close()
        cache = getattr(scene, "_fire_torch_volume_cache", None)
        if cache is not None: cache.clear()
        del sim, final_sim, previews, final, scene, fw
        gc.collect()
        torch.cuda.empty_cache()


def build_index(output, records):
    cards, completed = [], 0
    contact_tiles = []
    for r in records:
        sid = r["scene_id"]
        if not (output/sid/"manifest.json").exists(): continue
        completed += 1
        overview = cv2.imread(str(output/sid/"four_templates.png"))
        if overview is not None:
            tile = np.full((288, 320, 3), 235, np.uint8)
            cv2.putText(tile, sid, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (30,30,30), 1, cv2.LINE_AA)
            tile[24:] = cv2.resize(overview, (320,264), interpolation=cv2.INTER_AREA)
            contact_tiles.append(tile)
        scene_manifest = json.loads((output/sid/"manifest.json").read_text())
        view_links = []
        for view in scene_manifest["views"]:
            d = view["directory"]
            label = html.escape(view["label"])
            view_links.append(f'<article><h2>{view["slot"]}. {label}</h2><a href="{d}/template_scene.png"><img src="{d}/template_scene.png"></a><p><a href="{d}/fire_sensor_observation.png">Full sensor dashboard</a> · <a href="{d}/fire_sensor_arrays.npz">Raw sensor arrays</a> · <a href="{d}/rgb_fire_smoke_physical.png">RGB without presentation haze</a> · <a href="{d}/manifest.json">View metadata</a></p></article>')
        (output/sid/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>'+sid+'</title><style>body{background:#181b20;color:#eee;font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}a{color:#9acbff}img{max-width:100%}article{margin-bottom:40px}</style><a href="../index.html">All scenes</a><h1>'+sid+'</h1><p><a href="plan.json">Shared medium multi-origin plan</a></p>'+''.join(view_links))
        timeline_link = output/sid/"timeline"
        if not timeline_link.exists():
            timeline_link.symlink_to(Path("../timelines")/sid/r["plan_id"], target_is_directory=True)
        cards.append(f'<section><h2><a href="{sid}/index.html">{html.escape(sid)}</a></h2><a href="{sid}/four_templates.png"><img loading="lazy" src="{sid}/four_templates.png"></a><p><a href="{sid}/index.html">Individual views and arrays</a> · <a href="{sid}/four_sensor_observations.jpg">Sensor observations</a> · <a href="{sid}/manifest.json">Camera / time / provenance</a> · <a href="{sid}/plan.json">Shared medium plan</a></p></section>')
    (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>36-scene medium fire gallery</title><style>body{background:#181b20;color:#eee;font:16px system-ui;max-width:1400px;margin:40px auto;padding:20px}a{color:#9acbff}section{margin:40px 0}img{width:100%;height:auto}</style><h1>Medium multi-origin fire gallery</h1><p>One merged plan per scene; four actual Habitat / FireSensor views. RGB includes subtle depth-aware gray presentation haze; physical sensor arrays and an unhazed RGB reference are retained.</p>'+''.join(cards))
    if contact_tiles:
        while len(contact_tiles) % 6:
            contact_tiles.append(np.full_like(contact_tiles[0],235))
        contact = np.vstack([np.hstack(contact_tiles[i:i+6]) for i in range(0,len(contact_tiles),6)])
        _write_image(output/"all_scenes_contact_sheet.jpg", contact)
    write_json(output/"progress.json",{"completed_scenes":completed,"expected_scenes":36,"completed_views":4*completed,"updated_at":datetime.now(timezone.utc).isoformat()})


def validate(output, records, report_name="validation.json"):
    failures, warnings, count = [], [], 0
    for r in records:
        sid=r["scene_id"]; path=output/sid/"manifest.json"
        if not path.exists(): failures.append(f"{sid}: missing manifest");continue
        m=json.loads(path.read_text())
        if len(m["views"])!=4: failures.append(f"{sid}: not four views")
        if len({v["selected_source"]["object_id"] for v in m["views"]}) != 4:
            failures.append(f"{sid}: source objects are not distinct")
        timeline_path = output/m["timeline"]
        with np.load(timeline_path, allow_pickle=False) as timeline:
            times = timeline["times"]
            if len(times) != 61 or times[0] != 0 or times[-1] != 600:
                failures.append(f"{sid}: incorrect full timeline coverage")
        for view in m["views"]:
            d=output/sid/view["directory"]
            for name in view["files"]:
                p=d/name
                if not p.exists():failures.append(f"missing {p}")
                elif p.suffix in (".png",".jpg") and cv2.imread(str(p)) is None:failures.append(f"unreadable {p}")
            with np.load(d/"fire_sensor_arrays.npz",allow_pickle=False) as z:
                for key in z.files:
                    if not np.isfinite(z[key]).all():failures.append(f"{sid}/{view['slot']}: nonfinite {key}")
                if z["rgb_fire_smoke"].shape!=tuple(reversed(view["resolution"]))+(3,):failures.append(f"{sid}: wrong resolution")
                styled=cv2.cvtColor(cv2.imread(str(d/"template_scene.png")),cv2.COLOR_BGR2RGB)
                if not np.array_equal(styled,z["rgb_fire_smoke"]):failures.append(f"{sid}: RGB array/image mismatch")
                with np.load(d/"presentation_haze.npz",allow_pickle=False) as haze:
                    expected,alpha=gray_haze(haze["rgb_fire_smoke_physical"],z["depth_clean"],z["transmittance"],z["thermal_flame_mask"])
                    if not np.array_equal(expected,styled) or not np.array_equal(alpha,haze["alpha"]):
                        failures.append(f"{sid}: presentation haze provenance mismatch")
            metrics=view["physical_visibility_metrics"]
            if (metrics["rgb_fire_fraction"]<.001 or metrics["smoke_fraction"]<.01
                    or metrics.get("source_visible_fraction",0)<.0005
                    or metrics["central_flame_ratio"]<.35):
                warnings.append(f"{sid}/{view['slot']}: weak fire/smoke {metrics}")
            if not np.any(np.isclose(times,view["fire_time_s"])):
                failures.append(f"{sid}: capture time is not a baked frame")
            if view["plan_id"]!=m["plan_id"]:failures.append(f"{sid}: inconsistent shared plan")
            if not view["lidar_is_360"]:failures.append(f"{sid}: missing 360 lidar")
            count+=1
        for item in r["canonical_inputs"]:
            if sha256(ROOT/item["path"])!=item["sha256"]:failures.append(f"original input changed: {item['path']}")
    expected_views = 4 * len(records)
    report={"scenes":len(records),"views":count,"expected_views":expected_views,"failures":failures,"visibility_warnings":warnings,"passed":bool(records) and not failures and not warnings and count==expected_views}
    write_json(output/report_name,report)
    print(json.dumps(report,indent=2),flush=True)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--stage",choices=["prepare","bake","render","index","validate","all"],default="all")
    p.add_argument("--scene",action="append")
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--width",type=int,default=960)
    p.add_argument("--height",type=int,default=720)
    p.add_argument("--wait-for-timelines",type=int,default=0,
                   help="Wait at most this many seconds per scene for a concurrent bake")
    p.add_argument("--force-render",action="store_true",
                   help="Regenerate selected scene views while reusing their baked timeline")
    p.add_argument("--reuse-views-from",type=Path,
                   help="Use source IDs, camera poses and times from an existing gallery")
    args=p.parse_args(argv)
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.stage in ("prepare","all"): records=prepare(output)
    else:records=json.loads((output/"inputs.json").read_text())["scenes"]
    selected=[r for r in records if not args.scene or r["scene_id"] in args.scene]
    if args.stage in ("bake","all"):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(bake_one,r,str(output)) for r in selected]
            for f in as_completed(futures):print("[bake]",f.result(),flush=True)
    if args.stage in ("render","all"):
        pending = list(selected)
        deadline = time.monotonic() + args.wait_for_timelines
        while pending:
            ready = [r for r in pending if (output/"timelines"/r["scene_id"]/r["plan_id"]/"provenance.json").exists()]
            if not ready:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"baked timelines are not ready: {[r['scene_id'] for r in pending]}")
                time.sleep(5)
                continue
            r = ready[0]
            capture(r,output,args.width,args.height,force=args.force_render,reuse_views_from=args.reuse_views_from)
            build_index(output,records)
            pending.remove(r)
            deadline = time.monotonic() + args.wait_for_timelines
    if args.stage in ("index","all"):build_index(output,records)
    if args.stage in ("validate","all"):
        if not validate(output,records)["passed"]:
            return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
