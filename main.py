from collections import defaultdict
from typing import Dict
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

        count_step = 0
        point_sum = o3d.geometry.PointCloud()

        while not env.episode_over:
            start = time.time()
            visited_vis = []
            pose_pred = []
            point_sum.clear()
            found_goal = False

            # ---------- Fire-scene perception ----------
            if fire_suites is not None:
                for i in range(num_agents):
                    a_state = env.sim.get_agent_state(i)
                    sensors = step_fire_observation(
                        observations=observations[i],
                        suite=fire_suites[i],
                        agent_state=a_state,
                        robot_step=int(getattr(agent[i], "l_step", 0)),
                        config=config,
                        args=args,
                        # Static-person ObjectNav must use the same visual
                        # perception chain as chair/bed/etc. Projecting the
                        # known goal position into thermal created an oracle
                        # person:0.95 box even when a wall occluded the model.
                        walker=(None if static_person_goal else walker),
                    )
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
                agent_state = env.sim.get_agent_state(i)
                agent[i].mapping(observations[i], agent_state)
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

            # ---------- Global planner (frontier assignment) ----------
            if (agent[0].l_step % args.num_local_steps == args.num_local_steps - 1
                    or agent[0].l_step == 0) and not found_goal:
                goal_points.clear()
                target_score, target_edge_map, target_point_list = (
                    map_process.Frontier_Det(threshold_point=8)
                )

                if args.nav_mode == "gpt":
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

        metrics = env.get_metrics()

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
