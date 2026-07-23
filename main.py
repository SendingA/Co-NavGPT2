from collections import defaultdict
from typing import Dict, List, Optional
import os
import logging
import time
import threading
from multiprocessing import Queue

import torch
import numpy as np

# Habitat-Lab 0.3.3
import habitat
from habitat import Env

# Co-NavGPT2 modules
from utils.shortest_path_follower import ShortestPathFollowerCompat
from utils import chat_utils
import system_prompt
from utils.explored_map_utils import Global_Map_Proc

from agents.vlm_agents import VLM_Agent
import utils.visualization as vu
from arguments import get_args, load_config, humanoid_kwargs, robot_model_kwargs
from envs import RandomHumanoidWalker, RobotModelManager
from utils.fire_sensors import FireSensorSuite, FireSensorConfig, FireSensorViewer
from utils.person_objectnav import (
    objectnav_goal_debug_info,
    person_goal_positions,
    refresh_simulator_observations,
)

import cv2
import open3d as o3d
import open3d.visualization.gui as gui

from utils.vis_gui import ReconstructionWindow

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def _grid_line_cells(start, goal, shape) -> List[List[int]]:
    """Return a clipped one-cell-wide route between two grid positions."""
    canvas = np.zeros(shape, dtype=np.uint8)
    start_row = int(np.clip(round(float(start[0])), 0, shape[0] - 1))
    start_col = int(np.clip(round(float(start[1])), 0, shape[1] - 1))
    goal_row = int(np.clip(round(float(goal[0])), 0, shape[0] - 1))
    goal_col = int(np.clip(round(float(goal[1])), 0, shape[1] - 1))
    cv2.line(
        canvas,
        (start_col, start_row),
        (goal_col, goal_row),
        color=1,
        thickness=1,
    )
    return np.argwhere(canvas > 0).astype(int).tolist()


def _low_risk_fallback_goal(
    agent_cell,
    obstacle_map,
    explored_map,
    planning_risk,
    hard_unsafe,
) -> List[int]:
    """Select a nearby explored, navigable low-risk safety waypoint."""
    shape = np.asarray(planning_risk).shape
    obstacle = cv2.dilate(
        (np.asarray(obstacle_map) > 0.5).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ).astype(bool)
    explored = np.asarray(explored_map) > 0.0
    hard = np.asarray(hard_unsafe, dtype=bool)
    start = np.asarray(agent_cell[:2], dtype=np.float64)
    start_cell = (
        int(np.clip(round(start[0]), 0, shape[0] - 1)),
        int(np.clip(round(start[1]), 0, shape[1] - 1)),
    )
    free = ~obstacle & ~hard
    if not free.any():
        return [
            start_cell[0],
            start_cell[1],
        ]

    seed = start_cell
    if not free[seed]:
        free_cells = np.argwhere(free)
        seed = tuple(free_cells[int(np.argmin(
            np.linalg.norm(free_cells - start[None, :], axis=1)
        ))])
    _, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=8)
    reachable = labels == labels[seed]
    candidates = explored & reachable
    if not candidates.any():
        candidates = reachable

    cells = np.argwhere(candidates)
    distances = np.linalg.norm(cells - start[None, :], axis=1)
    # Prefer an actual waypoint over the current cell when one is available.
    nontrivial = distances >= 4.0
    if nontrivial.any():
        cells = cells[nontrivial]
        distances = distances[nontrivial]
    risk = np.asarray(planning_risk, dtype=np.float32)[cells[:, 0], cells[:, 1]]
    distance_scale = max(float(distances.max()), 1.0)
    score = risk + 0.08 * distances / distance_scale
    best = cells[int(np.argmin(score))]
    return [int(best[0]), int(best[1])]


def _risk_utility_weights(nav_mode: str, args):
    """Map legacy frontier policies onto a common safety-aware utility."""
    from utils.risk.frontier import UtilityWeights

    risk_weight = float(getattr(args, "risk_frontier_weight", 2.0))
    if nav_mode == "nearest":
        return UtilityWeights(
            information_gain=0.0,
            distance=1.0,
            risk=risk_weight,
            uncertainty=0.5,
            redundancy=0.0,
        )
    if nav_mode == "co_ut":
        return UtilityWeights(
            information_gain=0.15,
            distance=0.8,
            risk=risk_weight,
            uncertainty=0.5,
            redundancy=1.0,
        )
    if nav_mode == "fill":
        return UtilityWeights(
            information_gain=1.0,
            distance=0.25,
            risk=risk_weight,
            uncertainty=0.5,
            redundancy=0.75,
        )
    return UtilityWeights(
        information_gain=1.0,
        distance=0.35,
        risk=risk_weight,
        uncertainty=0.5,
        redundancy=0.75,
    )


def _flatten_risk_metrics(summary: Dict) -> Dict[str, float]:
    """Expose only the two primary risk benchmark metrics."""
    flat: Dict[str, float] = {
        "risk/che": float(summary.get("team", {}).get("CHE", 0.0)),
    }
    if "safe_success" in summary:
        flat["risk/safe_success"] = float(summary["safe_success"])
    return flat


def _find_scene_for_fire_plan(args):
    """Return the scene short-id that owns args.fire_world_plan_id.

    We look under ``scenes/<scene>/plans/<plan_id>.json`` — that's the
    layout the fire propagator writes. The scene short-id is used by
    the habitat dataset (``content_scenes: [<scene>]``) to restrict
    the episode iterator to matching scenes.
    """
    from pathlib import Path

    root = Path(getattr(args, "fire_world_scenes_root", "scenes"))
    if not root.is_absolute():
        root = Path(__file__).resolve().parent / root
    plan_id = args.fire_world_plan_id
    if not plan_id:
        return None
    for plan_file in root.glob(f"*/plans/{plan_id}.json"):
        return plan_file.parents[1].name
    return None


def main(args, send_queue, receive_queue):
    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_dir = "{}/logs/{}/".format(args.dump_location, args.nav_mode)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.nav_mode)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(dump_dir, exist_ok=True)
    logging.basicConfig(filename=log_dir + "output.log", level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    logging.info(args)

    agg_metrics: Dict = defaultdict(float)

    # ------------------------------------------------------------------
    # Config (Hydra / DictConfig, Habitat-Lab 0.3.3)
    # ------------------------------------------------------------------
    config = load_config(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from utils.risk.config import RiskConfig
    risk_config = RiskConfig.from_namespace(args)
    if risk_config.enabled and not int(getattr(args, "fire_world", 0)):
        raise ValueError(
            "--risk_enabled=1 requires --fire_world=1 and a valid "
            "--fire_world_plan_id"
        )
    if risk_config.enabled:
        print(
            "[risk] enabled "
            f"planner_source={risk_config.effective_source} "
            f"clock={args.fire_clock_mode} "
            f"weights=(T={risk_config.weights.temperature:.2f}, "
            f"S={risk_config.weights.smoke:.2f})"
        )
        if str(args.fire_clock_mode) != "step":
            print(
                "[risk] wallclock mode is a latency stress test; use "
                "--fire_clock_mode step for reproducible benchmark tables"
            )

    # Optional 360° LIDAR — installs 4 yaw-rotated depth sensors on every
    # navigation agent so utils.fire_sensors can stitch a 360° cloud.
    if int(getattr(args, "lidar_360", 0)) and int(getattr(args, "fire_world", 0)):
        from utils.fire_sensors.lidar_360 import (
            install_lidar_depth_sensors,
            LIDAR_DEPTH_UUIDS,
        )
        with habitat.config.read_write(config):
            install_lidar_depth_sensors(
                config,
                resolution=int(args.lidar_resolution),
                num_agents=args.num_agents,
            )
        print(f"[lidar_360] installed sensors: {LIDAR_DEPTH_UUIDS}")

    # When fire_world is on we can only render the scene the plan
    # was baked for. Filter the dataset to episodes whose scene_id
    # matches so env.reset() never loads an unrelated scene mid-run.
    if int(getattr(args, "fire_world", 0)) and args.fire_world_plan_id:
        _target_scene = _find_scene_for_fire_plan(args)
        if _target_scene:
            with habitat.config.read_write(config):
                config.habitat.dataset.content_scenes = [_target_scene]
            print(f"[fire_world] restricting dataset to scene {_target_scene}")

    # ------------------------------------------------------------------
    # Environment + agents (robot navigation policies)
    # ------------------------------------------------------------------
    env = Env(config=config)
    num_episodes = env.number_of_episodes
    assert num_episodes > 0, "num_episodes should be greater than 0"

    num_agents = int(config.conav.num_robots)
    agent = []
    for i in range(num_agents):
        follower = ShortestPathFollowerCompat(env.sim, 0.1, False, i)
        agent.append(VLM_Agent(args, i, follower, receive_queue))

    map_process = Global_Map_Proc(args)

    # ------------------------------------------------------------------
    # Humanoid pedestrians + visible robot URDF models (Habitat 3 only)
    # ------------------------------------------------------------------
    walker = RandomHumanoidWalker(
        sim=env.sim,
        **humanoid_kwargs(config, args.seed),
    )
    robot_models = RobotModelManager(
        sim=env.sim,
        **robot_model_kwargs(config, num_agents),
    )

    # ------------------------------------------------------------------
    # Fire scene + sensor suite (optional). The FireScene / FireSensorSuite
    # references ``config.habitat.simulator.scene``, which is only set to
    # the real episode scene AFTER ``env.reset()`` runs and calls
    # ``sim.reconfigure``. Env(config=...) initialises it to whatever the
    # dataset iterator's first episode is (often a different scene), so
    # we defer construction to the first iteration of the episode loop.
    # ------------------------------------------------------------------
    fire_scene = None
    fire_suites = None
    fire_viewers = None
    risk_runtime = None
    from utils.fire_pipeline import step_fire_observation  # noqa: E402

    def _build_fire_scene_and_suites():
        """Lazy construction; runs once, after the first env.reset()."""
        nonlocal fire_scene, fire_suites, fire_viewers
        if fire_scene is not None or fire_suites is not None:
            return
        if not int(getattr(args, "fire_world", 0)):
            return

        from utils.fire_world.scene import FireScene
        fire_scene = FireScene.from_args(args, config)
        print(f"[fire_world] {fire_scene.describe()}")

        from utils.general_utils import get_camera_K
        from utils.fire_sensors.config import VoxelSmokeConfig

        main_agent_name = config.habitat.simulator.agents_order[0]
        depth_cfg = (
            config.habitat.simulator.agents[main_agent_name]
            .sim_sensors.depth_sensor
        )

        from arguments import voxel_smoke_kwargs
        fire_cfg = FireSensorConfig(
            max_depth_m=float(depth_cfg.max_depth),
            hfov_deg=float(depth_cfg.hfov),
            smoke_density=float(args.smoke_density),
            save_npz=bool(args.fire_save_npz),
            voxel=VoxelSmokeConfig(**voxel_smoke_kwargs(args)),
        )
        K = get_camera_K(args.frame_width, args.frame_height, args.hfov)
        fire_suites = [
            FireSensorSuite(
                cfg=fire_cfg,
                dump_dir=os.path.join(args.fire_dump_dir, f"agent_{i}"),
                save_every=int(args.fire_save_every),
                seed=args.seed + i,
                scene=fire_scene,
                camera_K=K,
            )
            for i in range(num_agents)
        ]
        if int(getattr(args, "fire_show_window", 0)):
            fire_viewers = [
                FireSensorViewer.start(
                    window_name=f"Fire Sensors - agent {i}",
                    fps=10.0,
                    fallback_path=os.path.join(
                        args.fire_dump_dir, f"agent_{i}", "live.png"
                    ),
                )
                for i in range(num_agents)
            ]
        print(f"[fire_sensors] enabled (voxel RGB + Thermal), "
              f"density={args.smoke_density}, dump_dir={args.fire_dump_dir}")

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------
    count_episodes = 0
    goal_points = []
    target_edge_map = None
    target_score = None
    log_start = time.time()
    static_person_goal = bool(config.conav.get("static_person_goal", False))

    while count_episodes < num_episodes:
        observations = env.reset()

        # Fire construction is deferred until we know the real scene id.
        _build_fire_scene_and_suites()

        # Follow the original Habitat humanoid lifecycle: reset the task,
        # create/repose the articulated object, then render it. Static person
        # episodes only replace the random spawn point with the dataset goal.
        if static_person_goal:
            fixed_person_positions = person_goal_positions(
                env.current_episode
            )
            if not fixed_person_positions:
                raise RuntimeError(
                    "static_person_goal is enabled, but the current episode "
                    "has no valid person ObjectGoal positions"
                )
            walker.reset(fixed_person_positions, static=True)
        else:
            walker.reset()
        robot_models.reset()

        # This is the ObjectNav-safe equivalent of the original
        # ``env.sim.step(None)`` refresh: re-render RGB/depth after placement,
        # but merge them into env.reset() output so objectgoal/GPS/compass are
        # preserved and no task action is consumed.
        observations = refresh_simulator_observations(
            env.sim, observations, num_agents
        )

        map_process.reset()
        if fire_scene is not None:
            fire_scene.clock.start()

        agent_state = env.sim.get_agent_state(0)
        actions = []
        for i in range(num_agents):
            agent[i].reset(observations[i], agent_state)
            actions.append(0)
        # A detected object makes the frontier branch intentionally skip on
        # the first frame.  Keep a valid placeholder so act() never indexes an
        # empty/stale list; object-goal navigation replaces it internally.
        goal_points = [
            [int(a.map_size // 2), int(a.map_size // 2)] for a in agent
        ]

        risk_runtime = None
        risk_frontier_reports = []
        risk_frontier_computed_step = None
        if risk_config.enabled:
            from utils.risk.runtime import RiskRuntime
            risk_runtime = RiskRuntime(
                fire_scene=fire_scene,
                reference_agent=agent[0],
                args=args,
                episode_id=count_episodes,
            )
            initial_risk_t = risk_runtime.shared_time(agent[0].l_step)
            initial_agent_states = [
                env.sim.get_agent_state(i) for i in range(num_agents)
            ]
            risk_runtime.prime_exposure(
                initial_risk_t,
                initial_agent_states,
            )

        count_step = 0
        point_sum = o3d.geometry.PointCloud()

        while not env.episode_over:
            start = time.time()
            visited_vis = []
            pose_pred = []
            point_sum.clear()
            found_goal = False
            agent_states = [
                env.sim.get_agent_state(i) for i in range(num_agents)
            ]
            fire_sensor_outputs: List[Optional[dict]] = [None] * num_agents
            shared_risk_t = (
                risk_runtime.shared_time(agent[0].l_step)
                if risk_runtime is not None
                else None
            )
            navigation_step = int(agent[0].l_step)
            risk_layers = None
            planning_risk = None

            # ---------- Fire-scene perception ----------
            if fire_suites is not None:
                for i in range(num_agents):
                    sensors = step_fire_observation(
                        observations=observations[i],
                        suite=fire_suites[i],
                        agent_state=agent_states[i],
                        robot_step=int(getattr(agent[i], "l_step", 0)),
                        config=config,
                        args=args,
                        # Static-person ObjectNav must use the same visual
                        # perception chain as chair/bed/etc. Projecting the
                        # known goal position into thermal created an oracle
                        # person:0.95 box even when a wall occluded the model.
                        walker=(None if static_person_goal else walker),
                        t_sim_s=shared_risk_t,
                    )
                    fire_sensor_outputs[i] = sensors
                    if sensors is not None:
                        fire_suites[i].save_step(
                            sensors,
                            episode=count_episodes,
                            step=int(getattr(agent[i], "l_step", 0)),
                            agent_id=i,
                        )
                        if fire_viewers is not None:
                            fire_viewers[i].update(sensors.get("dashboard"))

            # ---------- Per-agent mapping ----------
            for i in range(num_agents):
                agent[i].mapping(observations[i], agent_states[i])
                point_sum += agent[i].point_sum
                visited_vis.append(agent[i].visited_vis)
                pose_pred.append([
                    agent[i].current_grid_pose[1],
                    int(agent[i].map_size) - agent[i].current_grid_pose[0],
                    np.deg2rad(agent[i].relative_angle),
                ])
                if agent[i].found_goal:
                    found_goal = True

            obstacle_map, explored_map, top_view_map = map_process.Map_Extraction(
                point_sum, agent[0].camera_position[1]
            )

            # ---------- Dynamic shared risk assessment ----------
            if risk_runtime is not None:
                if risk_runtime.source == "sensed":
                    risk_runtime.update_sensed(
                        timestamp_s=shared_risk_t,
                        sensor_outputs=fire_sensor_outputs,
                        agent_states=agent_states,
                        camera_k=agent[0].camera_K,
                    )
                risk_layers, planning_risk = risk_runtime.planner_state(
                    shared_risk_t
                )
                for i in range(num_agents):
                    agent[i].set_risk_map(
                        planning_risk,
                        risk_layers.hard_unsafe,
                        risk_alpha=float(args.risk_alpha),
                        enabled=risk_runtime.planning_enabled,
                    )

            # ---------- Global planner (frontier assignment) ----------
            if (agent[0].l_step % args.num_local_steps == args.num_local_steps - 1
                    or agent[0].l_step == 0) and not found_goal:
                goal_points.clear()
                target_score, target_edge_map, target_point_list = (
                    map_process.Frontier_Det(threshold_point=8)
                )

                if (
                    risk_runtime is not None
                    and risk_runtime.planning_enabled
                    and risk_layers is not None
                    and planning_risk is not None
                ):
                    from utils.risk.frontier import (
                        SeverityThresholds,
                        assign_frontiers,
                        build_frontier_risk_reports,
                        guard_frontier_assignments,
                        risk_context_payload,
                    )

                    risk_agent_cells = [
                        [int(a.current_grid_pose[0]), int(a.current_grid_pose[1])]
                        for a in agent
                    ]
                    route_cells = []
                    for frontier in target_point_list:
                        nearest_cell = min(
                            risk_agent_cells,
                            key=lambda cell: np.linalg.norm(
                                np.asarray(cell) - np.asarray(frontier)
                            ),
                        )
                        route_cells.append(
                            _grid_line_cells(
                                nearest_cell, frontier, planning_risk.shape
                            )
                        )

                    danger_threshold = float(risk_config.danger_threshold)
                    safe_threshold = min(0.25, danger_threshold)
                    moderate_threshold = max(
                        safe_threshold, danger_threshold
                    )
                    hard_threshold = float(np.clip(
                        args.risk_hard_frontier_threshold,
                        moderate_threshold,
                        1.0,
                    ))
                    thresholds = SeverityThresholds(
                        safe_max=safe_threshold,
                        moderate_max=moderate_threshold,
                        hard_max=hard_threshold,
                    )
                    risk_frontier_reports = build_frontier_risk_reports(
                        target_edge_map,
                        planning_risk,
                        risk_layers.confidence,
                        hard_unsafe_map=risk_layers.hard_unsafe,
                        frontier_points=target_point_list,
                        route_cells=route_cells,
                        route_is_proxy=True,
                        thresholds=thresholds,
                    )
                    risk_frontier_computed_step = navigation_step
                    deterministic_assignments = assign_frontiers(
                        risk_agent_cells,
                        risk_frontier_reports,
                        information_gain=target_score,
                        weights=_risk_utility_weights(args.nav_mode, args),
                        hard_risk_threshold=hard_threshold,
                        allow_shared=(
                            args.nav_mode != "co_ut"
                            or len(risk_frontier_reports) < num_agents
                        ),
                        redundancy_radius_cells=(
                            1.0 / (float(args.map_resolution) / 100.0)
                        ),
                    )

                    final_assignments = deterministic_assignments
                    if (
                        args.nav_mode == "gpt"
                        and len(target_point_list) > 0
                        and agent[0].l_step > 0
                    ):
                        candidate_map_list = chat_utils.get_all_candidate_maps(
                            target_edge_map, top_view_map, pose_pred
                        )
                        message = chat_utils.message_prepare(
                            system_prompt.risk_system_prompt,
                            candidate_map_list,
                            agent[0].goal_name,
                            risk_context=risk_context_payload(
                                risk_frontier_reports
                            ),
                        )
                        raw_assignments = chat_utils.chat_with_gpt4v(message)
                        guarded = guard_frontier_assignments(
                            raw_assignments,
                            risk_frontier_reports,
                            fallback_assignments=deterministic_assignments,
                            expected_robot_ids=range(num_agents),
                            hard_risk_threshold=hard_threshold,
                        )
                        final_assignments = guarded.assignments
                        if guarded.rejected:
                            logging.warning(
                                "risk guard replaced VLM frontier choices: %s",
                                guarded.rejected,
                            )

                    for i in range(num_agents):
                        frontier_id = final_assignments.get(i)
                        if (
                            frontier_id is not None
                            and 0 <= int(frontier_id) < len(target_point_list)
                        ):
                            goal_points.append(
                                target_point_list[int(frontier_id)]
                            )
                        else:
                            goal_points.append(
                                _low_risk_fallback_goal(
                                    risk_agent_cells[i],
                                    obstacle_map,
                                    explored_map,
                                    planning_risk,
                                    risk_layers.hard_unsafe,
                                )
                            )

                elif args.nav_mode == "gpt":
                    if len(target_point_list) > 0 and agent[0].l_step > 0:
                        candidate_map_list = chat_utils.get_all_candidate_maps(
                            target_edge_map, top_view_map, pose_pred
                        )
                        message = chat_utils.message_prepare(
                            system_prompt.system_prompt,
                            candidate_map_list,
                            agent[i].goal_name,
                        )
                        goal_frontiers = chat_utils.chat_with_gpt4v(message)
                        for i in range(num_agents):
                            goal_points.append(
                                target_point_list[
                                    int(goal_frontiers["robot_" + str(i)].split("_")[1])
                                ]
                            )
                    else:
                        for i in range(num_agents):
                            act_rand = np.random.rand(1, 2).squeeze() * (
                                obstacle_map.shape[0] - 1
                            )
                            goal_points.append([int(act_rand[0]), int(act_rand[1])])

                elif args.nav_mode == "nearest":
                    if len(target_point_list) > 0:
                        for i in range(num_agents):
                            distances = [
                                np.linalg.norm(
                                    np.array(target_point_list[j]) - np.array(pose_pred[i][:2])
                                )
                                for j in range(len(target_point_list))
                            ]
                            goal_points.append(target_point_list[np.argmin(distances)])
                    else:
                        for i in range(num_agents):
                            act_rand = np.random.rand(1, 2).squeeze() * (
                                obstacle_map.shape[0] - 1
                            )
                            goal_points.append([int(act_rand[0]), int(act_rand[1])])

                elif args.nav_mode == "co_ut":
                    if len(target_point_list) > 0:
                        assigned_frontiers = set()
                        for i in range(num_agents):
                            best_idx = -1
                            best_dist = float("inf")
                            for j, frontier in enumerate(target_point_list):
                                if j not in assigned_frontiers:
                                    dist = np.linalg.norm(
                                        np.array(frontier) - np.array(pose_pred[i][:2])
                                    )
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_idx = j
                            if best_idx != -1:
                                goal_points.append(target_point_list[best_idx])
                                assigned_frontiers.add(best_idx)
                            else:
                                distances = [
                                    np.linalg.norm(
                                        np.array(target_point_list[j])
                                        - np.array(pose_pred[i][:2])
                                    )
                                    for j in range(len(target_point_list))
                                ]
                                goal_points.append(
                                    target_point_list[np.argmin(distances)]
                                )
                    else:
                        for i in range(num_agents):
                            act_rand = np.random.rand(1, 2).squeeze() * (
                                obstacle_map.shape[0] - 1
                            )
                            goal_points.append([int(act_rand[0]), int(act_rand[1])])

                elif args.nav_mode == "fill":
                    if len(target_point_list) > 0:
                        for i in range(num_agents):
                            best_idx = 0
                            best_score = -1
                            for j, frontier in enumerate(target_point_list):
                                if target_score is not None and j < len(target_score):
                                    score = target_score[j]
                                else:
                                    score = 1.0 / (
                                        1.0
                                        + np.linalg.norm(
                                            np.array(frontier)
                                            - np.array(pose_pred[i][:2])
                                        )
                                    )
                                if score > best_score:
                                    best_score = score
                                    best_idx = j
                            goal_points.append(target_point_list[best_idx])
                    else:
                        for i in range(num_agents):
                            act_rand = np.random.rand(1, 2).squeeze() * (
                                obstacle_map.shape[0] - 1
                            )
                            goal_points.append([int(act_rand[0]), int(act_rand[1])])

                else:
                    for i in range(num_agents):
                        if len(target_point_list) > 0:
                            goal_points.append(
                                target_point_list[
                                    np.random.randint(0, len(target_point_list))
                                ]
                            )
                        else:
                            act_rand = np.random.rand(1, 2).squeeze() * (
                                obstacle_map.shape[0] - 1
                            )
                            goal_points.append([int(act_rand[0]), int(act_rand[1])])

            if risk_runtime is not None:
                risk_runtime.save_step(
                    step=navigation_step,
                    timestamp_s=shared_risk_t,
                    layers=risk_layers,
                    planning_risk=planning_risk,
                    obstacle_map=obstacle_map,
                    agent_cells=[a.current_grid_pose for a in agent],
                    frontier_points=[
                        report.point for report in risk_frontier_reports
                    ],
                    frontier_reports=[
                        report.to_dict() for report in risk_frontier_reports
                    ],
                    frontier_computed_step=risk_frontier_computed_step,
                )

            # ---------- Per-agent policy step ----------
            goal_map = []
            for i in range(num_agents):
                agent[i].obstacle_map = obstacle_map
                agent[i].explored_map = explored_map
                actions[i] = agent[i].act(goal_points[i])
                goal_map.append(agent[i].goal_map)

            if args.visualize or args.print_images:
                vu.Visualize(
                    args, agent[0].l_step,
                    pose_pred,
                    obstacle_map,
                    explored_map,
                    agent[0].goal_id,
                    visited_vis,
                    target_edge_map,
                    goal_map,
                    transform_rgb_bgr(top_view_map),
                    agent[0].episode_n,
                )

            # ---------- Advance humanoids + env ----------
            walker.step()
            observations = env.step(actions)
            robot_models.step()
            if not isinstance(observations, list):
                observations = [observations]

            if risk_runtime is not None:
                post_step_states = [
                    env.sim.get_agent_state(i) for i in range(num_agents)
                ]
                post_step_t = risk_runtime.shared_time(agent[0].l_step)
                risk_runtime.record_exposure(
                    post_step_t,
                    post_step_states,
                    step=navigation_step,
                    planner_statuses=[
                        getattr(a, "_risk_escape_reason", None) for a in agent
                    ],
                )

            step_end = time.time()

        count_episodes += 1
        count_step += agent[0].l_step

        # ---------- Logging ----------
        log_end = time.time()
        time_elapsed = time.gmtime(log_end - log_start)
        log = " ".join([
            "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
            "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
            "num timesteps {},".format(count_step),
            "FPS {},".format(int(count_step / max(1.0, log_end - log_start))),
        ]) + "\n"

        metrics = dict(env.get_metrics())
        if risk_runtime is not None:
            risk_summary = risk_runtime.summary(
                habitat_success=float(metrics.get("success", 0.0))
            )
            risk_runtime.save_summary(risk_summary)
            metrics.update(_flatten_risk_metrics(risk_summary))
            risk_team = risk_summary["team"]
            log += (
                "[risk] "
                f"source={risk_summary['planner_source']}  "
                f"CHE={risk_team['CHE']:.3f}  "
                f"critical={risk_team['critical_violations']}  "
                f"safe_refusal={risk_team['safe_refusal_steps']}  "
                f"escape={risk_team['emergency_escape_steps']}  "
                f"safe_success={risk_summary['safe_success']:.0f}\n"
            )

        # --- Debug: show why an episode was marked failed.
        try:
            ep = env.current_episode
            debug_info = objectnav_goal_debug_info(
                env.sim,
                ep,
                num_agents,
            )
            agent_position_fields = "  ".join(
                f"agent{agent_id}_xyz="
                f"{np.round(position, 2).tolist()}"
                for agent_id, position in enumerate(
                    debug_info["agent_positions"]
                )
            )
            log += (
                f"[dbg] target_class={getattr(agent[0], 'goal_name', '?')}  "
                f"{agent_position_fields}  "
                f"goal_index={debug_info['goal_index']}  "
                f"goal_xyz="
                f"{np.round(debug_info['goal_position'], 2).tolist()}  "
                f"nearest_agent={debug_info['nearest_agent_id']}  "
                f"nearest_goal_l2="
                f"{debug_info['nearest_goal_l2']:.3f}m  "
                f"success_thr={config.habitat.task.measurements.success.success_distance}m\n"
            )
        except Exception:
            logging.exception("failed to build ObjectNav episode debug info")

        for m, v in metrics.items():
            if isinstance(v, dict):
                for sub_m, sub_v in v.items():
                    agg_metrics[m + "/" + str(sub_m)] += sub_v
            else:
                agg_metrics[m] += v

        log += (
            ", ".join(
                k + ": {:.3f}".format(v / count_episodes)
                for k, v in agg_metrics.items()
            )
            + " ---({:.0f}/{:.0f})".format(count_episodes, num_episodes)
        )
        print(log)
        logging.info(log)

    avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

    if fire_viewers is not None:
        for v in fire_viewers:
            v.stop()

    env.close()
    return avg_metrics


def visualization_thread(send_queue, receive_queue):
    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    args = get_args()
    _ = ReconstructionWindow(args, mono, send_queue, receive_queue)
    app.run()


if __name__ == "__main__":
    args = get_args()

    send_queue = Queue()
    receive_queue = Queue()

    if args.visualize:
        visualization = threading.Thread(
            target=visualization_thread,
            args=(send_queue, receive_queue),
        )
        visualization.start()

    main(args, send_queue, receive_queue)
