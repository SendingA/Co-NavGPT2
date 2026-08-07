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
from utils.explored_map_utils import Global_Map_Proc
from utils.global_planners import (
    GlobalPlannerContext,
    RiskPlanningContext,
    create_global_planner,
)
from utils.evaluation_resume import (
    advance_episode_iterator,
    load_metric_resume,
    write_metric_resume,
)

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

import open3d as o3d
import open3d.visualization.gui as gui

from utils.vis_gui import ReconstructionWindow

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


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
    from utils.fire_world.plan_selection import find_scene_for_plan

    plan_id = getattr(args, "fire_world_plan_id", None)
    if not plan_id:
        return None
    return find_scene_for_plan(
        plan_id,
        scenes_root=getattr(args, "fire_world_scenes_root", "scenes"),
    )


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
    from utils.local_planners import validate_local_planner_config

    risk_config = RiskConfig.from_namespace(args)
    local_risk_awareness = validate_local_planner_config(args)
    if risk_config.enabled and not int(getattr(args, "fire_world", 0)):
        raise ValueError(
            "--risk_enabled=1 requires --fire_world=1 and a runnable "
            "FireWorld plan/timeline"
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
    print(
        "[local_planner] "
        f"name={args.local_planner} risk_aware={local_risk_awareness}"
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

    # FireWorld can only play scenes with a baked timeline. Explicit mode
    # keeps the historical one-plan/one-scene behavior. Auto mode discovers
    # every scene that has a runnable plan matching the requested defaults and
    # filters Habitat before Env creation, so a long run cannot fail midway on
    # a scene whose timeline is missing.
    if int(getattr(args, "fire_world", 0)):
        from utils.fire_world.plan_selection import (
            discover_runnable_fire_scenes,
            is_auto_plan_id,
            select_fire_plan,
        )

        if is_auto_plan_id(getattr(args, "fire_world_plan_id", None)):
            _fire_selections = discover_runnable_fire_scenes(
                intensity=args.fire_world_intensity,
                fire_type=args.fire_world_fire_type,
                scenes_root=args.fire_world_scenes_root,
                out_root=args.fire_world_out_root,
            )
            if not _fire_selections:
                raise FileNotFoundError(
                    "No runnable FireWorld scenes match "
                    f"fire_type={args.fire_world_fire_type}, "
                    f"intensity={args.fire_world_intensity}. Generate the "
                    "matching timeline.npz assets before navigation."
                )
            _fire_scene_ids = sorted(_fire_selections)
            with habitat.config.read_write(config):
                config.habitat.dataset.content_scenes = _fire_scene_ids
            print(
                "[fire_world] automatic plan selection "
                f"fire_type={args.fire_world_fire_type} "
                f"intensity={args.fire_world_intensity} "
                f"ready_scenes={len(_fire_scene_ids)}"
            )
            for _scene_id in _fire_scene_ids[:5]:
                print(
                    "[fire_world] ready "
                    f"scene={_scene_id} "
                    f"plan={_fire_selections[_scene_id].plan_id}"
                )
            if len(_fire_scene_ids) > 5:
                print(
                    "[fire_world] "
                    f"{len(_fire_scene_ids) - 5} additional ready scenes"
                )
        else:
            _target_scene = _find_scene_for_fire_plan(args)
            if _target_scene is None:
                raise FileNotFoundError(
                    "Explicit FireWorld plan was not found under "
                    f"{args.fire_world_scenes_root}: "
                    f"{args.fire_world_plan_id}"
                )
            # Fail before the expensive Habitat environment is created if the
            # matching timeline is absent or the plan metadata is malformed.
            select_fire_plan(
                _target_scene,
                plan_id=args.fire_world_plan_id,
                scenes_root=args.fire_world_scenes_root,
                out_root=args.fire_world_out_root,
            )
            with habitat.config.read_write(config):
                config.habitat.dataset.content_scenes = [_target_scene]
            print(f"[fire_world] restricting dataset to scene {_target_scene}")

    # ------------------------------------------------------------------
    # Environment + agents (robot navigation policies)
    # ------------------------------------------------------------------
    env = Env(config=config)
    available_episodes = int(env.number_of_episodes)
    episode_limit = int(getattr(args, "max_episodes", 0))
    if episode_limit < 0:
        raise ValueError("--max_episodes must be non-negative")
    num_episodes = (
        min(available_episodes, episode_limit)
        if episode_limit > 0
        else available_episodes
    )
    assert num_episodes > 0, "num_episodes should be greater than 0"
    start_episode = int(getattr(args, "start_episode", 1))
    if start_episode < 1 or start_episode > num_episodes:
        raise ValueError(
            "--start_episode must be between 1 and the planned episode count"
        )
    completed_before = start_episode - 1
    metric_precision = "exact"
    resume_state = None
    if completed_before:
        resume_path = getattr(args, "resume_metrics_path", None)
        if not resume_path:
            raise ValueError(
                "--start_episode above 1 requires --resume_metrics_path"
            )
        resume_state = load_metric_resume(resume_path)
        if resume_state.episodes_completed != completed_before:
            raise ValueError(
                "resume metrics completed count does not match "
                "--start_episode"
            )
        if resume_state.episodes_planned != num_episodes:
            raise ValueError(
                "resume metrics planned count does not match "
                "--max_episodes"
            )
        agg_metrics.update(resume_state.metric_sums)
        metric_precision = resume_state.precision
    print(
        "[evaluation] "
        f"episodes={num_episodes}/{available_episodes} "
        f"(max_episodes={episode_limit}, start_episode={start_episode})"
    )
    if completed_before:
        last_completed_episode = advance_episode_iterator(
            env,
            completed_before,
        )
        if (
            resume_state.last_episode_id is not None
            and str(last_completed_episode.episode_id)
            != resume_state.last_episode_id
        ):
            raise ValueError(
                "resume episode order does not match the current dataset"
            )
        print(
            "[evaluation] resuming after "
            f"{completed_before} episodes from "
            f"{args.resume_metrics_path} "
            f"(precision={metric_precision})"
        )

    num_agents = int(config.conav.num_robots)
    agent = []
    for i in range(num_agents):
        follower = ShortestPathFollowerCompat(env.sim, 0.1, False, i)
        agent.append(VLM_Agent(args, i, follower, receive_queue))

    map_process = Global_Map_Proc(args)
    global_planner = create_global_planner(
        args.nav_mode,
        cost_utility_lambda=args.cost_utility_lambda,
        random_seed=args.seed,
        random_goal_min_distance_m=args.random_goal_min_distance_m,
        map_resolution_cm=args.map_resolution,
    )

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
        """Build/reuse assets after reset for the active episode scene."""
        nonlocal fire_scene, fire_suites, fire_viewers
        if not int(getattr(args, "fire_world", 0)):
            return

        from utils.fire_world.scene import FireScene
        from utils.fire_world.plan_selection import scene_id_from_config

        current_scene_id = scene_id_from_config(config)
        if (
            fire_scene is not None
            and fire_suites is not None
            and fire_scene.scene_id == current_scene_id
        ):
            args.fire_world_active_plan_id = fire_scene.plan_id
            args.fire_world_active_scene_id = fire_scene.scene_id
            return

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
        if (
            int(getattr(args, "fire_show_window", 0))
            and fire_viewers is None
        ):
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
    count_episodes = completed_before
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

        reset_agent_states = [
            env.sim.get_agent_state(i) for i in range(num_agents)
        ]
        actions = []
        for i in range(num_agents):
            agent[i].reset(observations[i], reset_agent_states[i])
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
                        enabled=local_risk_awareness,
                    )

            # ---------- Global planner (frontier assignment) ----------
            pointnav_replan_requested = any(
                bool(getattr(a, "pointnav_replan_requested", False))
                for a in agent
            )
            if (
                agent[0].l_step % args.num_local_steps
                == args.num_local_steps - 1
                or agent[0].l_step == 0
                or pointnav_replan_requested
            ) and not found_goal:
                goal_points.clear()
                target_score, target_edge_map, target_point_list = (
                    map_process.Frontier_Det(threshold_point=8)
                )

                planner_risk = None
                if (
                    risk_runtime is not None
                    and risk_runtime.planning_enabled
                    and risk_layers is not None
                    and planning_risk is not None
                ):
                    planner_risk = RiskPlanningContext(
                        planning_risk=planning_risk,
                        confidence=risk_layers.confidence,
                        hard_unsafe=risk_layers.hard_unsafe,
                        danger_threshold=float(risk_config.danger_threshold),
                        hard_frontier_threshold=float(
                            args.risk_hard_frontier_threshold
                        ),
                        frontier_weight=float(args.risk_frontier_weight),
                        map_resolution_cm=float(args.map_resolution),
                    )

                planner_result = global_planner.plan(
                    GlobalPlannerContext(
                        target_score=target_score,
                        target_edge_map=target_edge_map,
                        target_points=target_point_list,
                        poses=pose_pred,
                        agent_cells=[
                            [
                                int(a.current_grid_pose[0]),
                                int(a.current_grid_pose[1]),
                            ]
                            for a in agent
                        ],
                        obstacle_map=obstacle_map,
                        explored_map=explored_map,
                        top_view_map=top_view_map,
                        goal_name=agent[0].goal_name,
                        local_step=int(agent[0].l_step),
                        navigation_step=navigation_step,
                        num_agents=num_agents,
                        risk=planner_risk,
                        episode_index=count_episodes,
                    )
                )
                goal_points.extend(planner_result.goal_points)
                risk_frontier_reports = planner_result.frontier_reports
                risk_frontier_computed_step = (
                    planner_result.frontier_computed_step
                )

                if pointnav_replan_requested:
                    for navigation_agent in agent:
                        navigation_agent.acknowledge_pointnav_replan()

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
        current_episode = env.current_episode
        write_metric_resume(
            os.path.join(
                args.dump_location,
                "metrics",
                "resume_state.json",
            ),
            episodes_completed=count_episodes,
            episodes_planned=num_episodes,
            metric_sums=agg_metrics,
            precision=metric_precision,
            last_episode_id=getattr(current_episode, "episode_id", None),
            last_scene_id=getattr(current_episode, "scene_id", None),
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
