#!/usr/bin/env python3
import math
import time
import os

import torch
import open3d as o3d
from multiprocessing import Process, Queue

# from habitat.core.agent import Agent
# from habitat.core.simulator import Observations
# from habitat.sims.habitat_simulator.actions import HabitatSimActions

from PIL import Image
import yaml
import quaternion
import logging

import numpy as np
import cv2
from skimage import measure
import skimage.morphology
from collections import Counter
from utils.general_utils import (
    get_camera_K
)
from utils.detection_segmentation import Object_Detection_and_Segmentation

from constants import color_palette, category_to_id #, category_to_id_replica
from utils.visualization import (
    draw_line,
    fit_image_to_panel,
    init_vis_image,
    vis_result_fast,
)
from utils.explored_map_utils import (
    build_full_scene_pcd,
    detect_frontier,
)
import utils.pose as pu
from utils.mapping import create_object_pcd, process_pcd
from utils.fmm_planner import FMMPlanner
from utils.local_planners import (
    AStarPathCache,
    create_local_planner,
    frontier_grid_to_world,
    shield_pointnav_action,
)

# Disable torch gradient computation
torch.set_grad_enabled(False)
   

FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
UP_KEY="q"
DOWN_KEY="e"
FINISH="f"


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


class VLM_Agent():
    def __init__(self, args, agent_id, follower=None, receive_queue=None) -> None:
        
        # ------------------------------------------------------------------
        ##### Initialize basic config
        # ------------------------------------------------------------------
        self.args = args
        self.agent_id = agent_id
        self.episode_n = 0
        print("init agent " + str(agent_id) )
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.receive_queue = receive_queue

        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        self.device = "cuda:{}".format(self.args.gpu_id)
        self.dump_dir = "{}/dump/{}/".format(args.dump_location, args.nav_mode)

        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        # ------------------------------------------------------------------
        ##### Initialize the perception model
        # ------------------------------------------------------------------
        self.classes = ["chair", "bed", "potted plant", "toilet", "tv_screen", "couch", "person", "fire"]

        self.obj_det_seg = Object_Detection_and_Segmentation(self.args, self.classes, self.device)
        
        self.annotated_image = None
        self.vis_map = None


        # 3D mapping
        self.camera_K = get_camera_K(
            self.args.frame_width, self.args.frame_height, self.args.hfov)

        self.init_map_and_pose()
        # ------------------------------------------------------------------
        
            
        # ------------------------------------------------------------------
        ##### Initialize navigation
        # ------------------------------------------------------------------
        if follower != None:
            self.follower = follower
            
        self.goal_name = None
        
        self.turn_angle = args.turn_angle
        self.init_map_and_navigation_param()
        self.pointnav_planner = None
        if str(getattr(args, 'local_planner', 'fmm')).lower() == 'pointnav':
            self.pointnav_planner = create_local_planner(
                'pointnav',
                pointnav_checkpoint=getattr(
                    args, 'pointnav_checkpoint', None
                ),
                pointnav_config=getattr(args, 'pointnav_config', None),
                pointnav_device=getattr(args, 'pointnav_device', 'cpu'),
                pointnav_deterministic=bool(int(getattr(
                    args, 'pointnav_deterministic', 1
                ))),
                pointnav_goal_tolerance=float(getattr(
                    args, 'pointnav_goal_tolerance', 0.05
                )),
                pointnav_env_action_map=getattr(
                    args, 'pointnav_env_action_map', None
                ),
                pointnav_observation_mode=getattr(
                    args, 'pointnav_observation_mode', 'auto'
                ),
            )
        # ------------------------------------------------------------------

    def reset(self, observations, agent_state) -> None:
        self.episode_n += 1
        self.init_map_and_pose()
        self.init_map_and_navigation_param()
        if self.pointnav_planner is not None:
            self.pointnav_planner.reset_robot(self.agent_id)
        
        # ------------------------------------------------------------------
        ##### At first step, get the object name and init the visualization
        # ------------------------------------------------------------------
        if self.l_step == 0:
            self.init_sim_position = agent_state.sensor_states["depth"].position
            self.init_agent_position = agent_state.position
            self.init_agent_rotation = agent_state.rotation
            self.init_sim_rotation = quaternion.as_rotation_matrix(agent_state.sensor_states["depth"].rotation)

            self.goal_id = int(observations['objectgoal'][0])
            self.goal_name = category_to_id[self.goal_id]
          
        # print("current position: ", agent_state.sensor_states["depth"].position)
        
    def init_map_and_pose(self):
        # local map
        self.map_size = self.args.map_size_cm // self.args.map_resolution
        self.map_real_halfsize  = self.args.map_size_cm / 100.0 / 2.0
        self.local_w, self.local_h = self.map_size, self.map_size
        
        self.explored_map = np.zeros((self.local_w, self.local_h))
        self.obstacle_map = np.zeros((self.local_w, self.local_h))
        self.visited_vis = np.zeros((self.local_w, self.local_h))
        self.goal_map = np.zeros((self.local_w, self.local_h))
        self.similarity_obj_map = np.zeros((self.local_w, self.local_h))
        self.similarity_img_map = np.zeros((self.local_w, self.local_h))
        self.collision_map = np.zeros((self.local_w, self.local_h))
        
        self.last_grid_pose = [self.map_size/2, self.map_size/2]
        self.last_real_pose = [0, 0]
        self.origins_grid = [self.map_size/2, self.map_size/2]
        self.origins_real = [0.0, 0.0]
        self.col_width = 1

        
    def move_map_and_pose(self, shift, axis):
        
        self.explored_map = pu.roll_array(self.explored_map, shift, axis)
        self.obstacle_map = pu.roll_array(self.obstacle_map, shift, axis)
        self.visited_vis = pu.roll_array(self.visited_vis, shift, axis)
        self.goal_map = pu.roll_array(self.goal_map, shift, axis)
        self.similarity_obj_map = pu.roll_array(self.similarity_obj_map, shift, axis)
        self.similarity_img_map = pu.roll_array(self.similarity_img_map, shift, axis)
        self.collision_map = pu.roll_array(self.collision_map, shift, axis)
        if getattr(self, 'risk_map', None) is not None:
            self.risk_map = pu.roll_array(self.risk_map, shift, axis)
        if getattr(self, 'hard_unsafe_mask', None) is not None:
            self.hard_unsafe_mask = pu.roll_array(
                self.hard_unsafe_mask, shift, axis
            ).astype(bool)
        
        self.last_grid_pose = pu.roll_pose(self.last_grid_pose, shift, axis)
        self.origins_grid = pu.roll_pose(self.origins_grid, shift, axis)
        self.origins_real = pu.roll_pose(self.origins_real, -shift * self.args.map_resolution / 100.0, axis)

        
    def init_map_and_navigation_param(self):
        
        # 3D mapping
        self.point_sum = o3d.geometry.PointCloud()
        self.object_pcd = o3d.geometry.PointCloud()
    
        self.init_sim_position = None
        self.init_sim_rotation = None
        self.init_agent_position = None
        self.init_agent_rotation = None
        self.Open3D_traj = []
        self.plan_path = []
        self.nearest_point = None
        self.current_grid_pose = None
        self.camera_position = None
        
        self.relative_angle = 0
        self.eve_angle = 0

        # navigation
        self.l_step = 0
        
        self.no_frontiers_count = 0
        self.curr_frontier_count = 0
        self.greedy_stop_count = 0
        self.replan_count = 0

        self.is_running = True
        self.found_goal = False
        self.last_action = 0
        self.last_goal = None
        self._latest_pointnav_observations = None
        self._latest_pointnav_agent_state = None
        self.pointnav_replan_requested = False
        self.pointnav_last_decision = None
        self._astar_path_cache = None

        # The shared risk map is supplied by the orchestration layer after
        # every dynamic fire-field update.  Keeping this state on the agent
        # makes replanning explicit while leaving legacy runs unchanged.
        self.risk_map = None
        self.hard_unsafe_mask = None
        self.risk_alpha = max(
            0.0, float(getattr(self.args, 'risk_alpha', 0.0))
        )
        # The CLI gate alone must never switch planners: only an orchestrator
        # that has supplied a synchronized, map-aligned risk state may enable
        # this path through set_risk_map(..., enabled=True).
        self.risk_navigation_enabled = False
        self._risk_escape_active = False
        self._risk_escape_reason = None
        
        self.upstair_flag = False
        self.downstair_flag = False
        self.another_floor = False
        self.clean_diff = True

    def set_risk_map(self, risk_map=None, hard_unsafe_mask=None,
                     risk_alpha=None, enabled=None):
        """Set the latest map-aligned risk state used by ``act``.

        Arrays must use the same grid frame as ``obstacle_map``.  They are
        copied because the shared mapper can update its buffers concurrently
        with agent planning.
        """
        expected_shape = (self.local_w, self.local_h)
        if risk_map is not None:
            risk_map = np.asarray(risk_map, dtype=np.float32)
            if risk_map.shape != expected_shape:
                raise ValueError(
                    "risk_map shape {} does not match agent map {}".format(
                        risk_map.shape, expected_shape
                    )
                )
            risk_map = np.nan_to_num(
                risk_map, nan=0.0, posinf=1.0, neginf=0.0
            )
            self.risk_map = np.clip(risk_map, 0.0, 1.0).copy()
        else:
            self.risk_map = None

        if hard_unsafe_mask is not None:
            hard_unsafe_mask = np.asarray(hard_unsafe_mask, dtype=bool)
            if hard_unsafe_mask.shape != expected_shape:
                raise ValueError(
                    "hard_unsafe_mask shape {} does not match agent map {}".format(
                        hard_unsafe_mask.shape, expected_shape
                    )
                )
            self.hard_unsafe_mask = hard_unsafe_mask.copy()
        else:
            self.hard_unsafe_mask = None

        if risk_alpha is not None:
            self.risk_alpha = max(0.0, float(risk_alpha))
        if enabled is not None:
            self.risk_navigation_enabled = bool(enabled)
        elif not hasattr(self.args, 'risk_enabled'):
            self.risk_navigation_enabled = (
                self.risk_map is not None or self.hard_unsafe_mask is not None
            )

        self._risk_escape_active = False
        self._risk_escape_reason = None

    def clear_risk_map(self):
        """Disable risk navigation and restore the historical planner path."""
        self.set_risk_map(enabled=False)


    def mapping(self, observations, agent_state):
        time_step_info = 'Mapping time (s): \n'
        # PointNav consumes the untouched Habitat RGB/depth tensors and the
        # live robot pose. Mapping preprocesses depth separately below.
        self._latest_pointnav_observations = observations
        self._latest_pointnav_agent_state = agent_state

        preprocess_s_time = time.time()

        # ------------------------------------------------------------------
        ##### Preprocess the observation
        # ------------------------------------------------------------------
        proc_time = time.time()
        image_rgb = observations['rgb']
        depth = observations['depth']
        image = transform_rgb_bgr(image_rgb)
        self.annotated_image = image

        depth = self._preprocess_depth(depth)

        camera_matrix_T = self.get_transform_matrix(agent_state)
        self.camera_position = camera_matrix_T[:3, 3]
        self.Open3D_traj.append(camera_matrix_T)
        self.relative_angle = round(np.arctan2(camera_matrix_T[2][0], camera_matrix_T[0][0])* 57.29577951308232 + 180)
        # print("self.relative_angle: ", self.relative_angle)


        thermal_mask = observations.get('thermal_flame_mask') \
            if int(getattr(self.args, 'use_thermal_perception', 0)) else None
        human_mask = observations.get('thermal_human_mask') \
            if int(getattr(self.args, 'use_thermal_perception', 0)) else None
        detections = self.obj_det_seg.detect(
            image,
            thermal_flame_mask=thermal_mask,
            thermal_human_mask=human_mask,
        )
        
        n_masks = len(detections.xyxy)
        for mask_idx in range(n_masks):
            if self.goal_id == detections.class_id[mask_idx] and (detections.confidence[mask_idx] > self.args.sem_threshold or ('plant' in self.goal_name and detections.confidence[mask_idx] > 0.5)):
                mask = detections.mask[mask_idx]

                # --- Debug: log every time we accept a goal detection.
                # Useful for chasing false positives that later drive the
                # follower to a wrong 3D point and cause STOP with
                # success=0. Comment out once tuned.
                try:
                    conf = float(detections.confidence[mask_idx])
                    logging.info(
                        "[agent %d step %d] accept detection: class=%s conf=%.3f "
                        "camera_pos=%s",
                        self.agent_id,
                        int(self.l_step),
                        self.goal_name,
                        conf,
                        np.round(self.camera_position, 2).tolist(),
                    )
                    print(
                        f"[agent {self.agent_id} step {int(self.l_step)}] "
                        f"detect {self.goal_name} conf={conf:.3f} "
                        f"cam_xyz={np.round(self.camera_position, 2).tolist()}"
                    )
                except Exception:
                    pass

                # make the pcd and color it
                camera_object_pcd = create_object_pcd(
                    depth,
                    mask,
                    self.camera_K,
                    image,
                    obj_color = None
                )

                if len(camera_object_pcd.points) < 10:
                    continue
                
                camera_object_pcd.transform(camera_matrix_T)
                # camera_object_pcd = process_pcd(camera_object_pcd)
                
                self.object_pcd += camera_object_pcd

        proc_end_time = time.time()
        # print('proc_time: %.3f秒'%(proc_end_time-proc_time))
        # ------------------------------------------------------------------

        
        # ------------------------------------------------------------------
        ##### 2D Obstacle Map
        # ------------------------------------------------------------------
        map_time = time.time()
        
        local_grid_pose = [self.camera_position[0]*100/self.args.map_resolution + int(self.origins_grid[0]), 
                      self.camera_position[2]*100/self.args.map_resolution + int(self.origins_grid[1])]
        pose_x = max(1, min(int(local_grid_pose[0]), self.map_size - 1))
        pose_y = max(1, min(int(local_grid_pose[1]), self.map_size - 1))
        
        # # Adjust the centriod of the map when the robot move to the edge of the map
        # if pose_x < 100:
        #     self.move_map_and_pose(shift = 100, axis=0)
        #     pose_x += 100
        # elif pose_x > self.map_size - 100:
        #     self.move_map_and_pose(shift = -100, axis=0)
        #     pose_x -= 100
        # elif pose_y < 100:
        #     self.move_map_and_pose(shift = 100, axis=1)
        #     pose_y += 100
        # elif pose_y > self.map_size - 100:
        #     self.move_map_and_pose(shift = -100, axis=1)
        #     pose_y -= 100
        
        self.current_grid_pose = [pose_x, pose_y]
        
        # visualize trajectory
        self.visited_vis = draw_line(self.last_grid_pose, self.current_grid_pose, self.visited_vis)
        self.last_grid_pose = self.current_grid_pose
        
        # Collision check
        full_scene_pcd = build_full_scene_pcd(depth, image_rgb, self.camera_K)
        
        
        # build 3D pc map
        full_scene_pcd.transform(camera_matrix_T)
        full_scene_pcd.voxel_down_sample(0.05)
        self.point_sum += self.remove_full_points_cell(full_scene_pcd, self.camera_position)
        
        # self.update_map(full_scene_pcd, self.camera_position[1], self.args.map_height_cm / 100.0 /2.0)
        if np.abs(self.eve_angle) > 10 or self.last_action == 4 or self.last_action == 5:
            self.clean_diff = False

        # target_score, target_edge_map, target_point_list = detect_frontier(self.explored_map, self.obstacle_map, self.current_grid_pose, threshold_point=8)
         
        map_end_time = time.time()
        # print('map_time: %.3f秒'%(map_end_time-map_time))
        
           
        if self.args.visualize or self.args.print_images:
            self.annotated_image  = vis_result_fast(image, detections, self.classes)
            
            self.vis_map = self._visualize(self.obstacle_map, self.explored_map, self.goal_map, self.goal_name)
        # ------------------------------------------------------------------
        
        if self.args.visualize:
            Open3d_goal_pose = []
            self.receive_queue.put([self.agent_id, 
                               image_rgb, 
                                depth, 
                                self.annotated_image , 
                                transform_rgb_bgr(self.vis_map),
                                np.asarray(self.point_sum.points), 
                                np.asarray(self.point_sum.colors), 
                                self.Open3D_traj,
                                self.episode_n,
                                self.plan_path,
                                Open3d_goal_pose,
                                time_step_info]
                                )   
        
    def act(self, goal_points: list):
        
        # ------------------------------------------------------------------
        ##### Goal selection
        # ------------------------------------------------------------------
        if np.array_equal(self.last_goal, goal_points):
            self.curr_frontier_count += 1
        else:
            self.curr_frontier_count = 0
                    
        act_time = time.time()
        has_navigation_goal = len(self.object_pcd.points) > 0
        if has_navigation_goal:
            goal_pcd = process_pcd(self.object_pcd)
            if self.found_goal == False:
                self.goal_map = np.zeros((self.local_w, self.local_h))
            self.goal_map[self.object_map_building(goal_pcd)] = 1
            self.nearest_point = self.find_nearest_point_cloud(
                goal_pcd, self.camera_position
            )
            x, y, z = self.nearest_point

        if has_navigation_goal:
            self.found_goal = True
        else:
            self.found_goal = False
            self.goal_map = np.zeros((self.local_w, self.local_h))
            self.goal_map[goal_points[0], goal_points[1]] = 1

            x = ((goal_points[0] - int(self.origins_grid[0]))
                 * self.args.map_resolution / 100.0)
            y = self.camera_position[1]
            z = ((goal_points[1] - int(self.origins_grid[1]))
                 * self.args.map_resolution / 100.0)
   
   
        Open3d_goal_pose = [x, y, z]
        
        Rx = np.array([[0, 0, -1],
                    [0, 1, 0],
                    [1, 0, 0]])
        R_habitat2open3d = self.init_sim_rotation @ Rx.T
        self.habitat_goal_pose = np.dot(R_habitat2open3d, Open3d_goal_pose) + self.init_agent_position
        habitat_final_pose = self.habitat_goal_pose.astype(np.float32)

        plan_path = []
        planner_name = str(
            getattr(self.args, 'local_planner', 'fmm')
        ).lower()
        if (
            planner_name == 'pointnav'
            and has_navigation_goal
            and self.pointnav_planner is not None
        ):
            self.pointnav_planner.clear_local_goal(self.agent_id)
            self.pointnav_replan_requested = False
        if planner_name == 'pointnav' and not has_navigation_goal:
            habitat_final_pose = frontier_grid_to_world(
                goal_points,
                origins_grid=self.origins_grid,
                map_resolution_cm=self.args.map_resolution,
                camera_local_y=self.camera_position[1],
                initial_agent_position=self.init_agent_position,
                initial_sensor_rotation=self.init_sim_rotation,
            )
            self.habitat_goal_pose = habitat_final_pose
            self.plan_path = np.asarray(
                [self.camera_position, Open3d_goal_pose],
                dtype=np.float32,
            )
            action = self._pointnav_frontier_act(habitat_final_pose)
        else:
            # PointNav is a frontier-local policy. Once an object is detected,
            # preserve the existing target-navigation and task STOP path.
            use_existing_object_path = (
                planner_name == 'pointnav' and has_navigation_goal
            )
            if planner_name == 'fmm' or use_existing_object_path:
                if not getattr(self, 'risk_navigation_enabled', False):
                    # Preserve the historical Habitat shortest-path-first
                    # behavior for FMM and detected-object completion.
                    plan_path = self.search_navigable_path(
                        habitat_final_pose
                    )

            if len(plan_path) > 1:
                self.plan_path = np.dot(
                    R_habitat2open3d.T,
                    (np.array(plan_path) - self.init_agent_position).T,
                ).T
                action = self.greedy_follower_act(self.plan_path)
            else:
                planner_override = (
                    'fmm' if use_existing_object_path else None
                )
                self.stg, self.stop, plan_path = self._get_stg(
                    self.obstacle_map,
                    self.current_grid_pose,
                    np.copy(self.goal_map),
                    planner_name_override=planner_override,
                )
                plan_path = np.array(plan_path)
                plan_path_x = (
                    plan_path[:, 0] - int(self.origins_grid[0])
                ) * self.args.map_resolution / 100.0
                plan_path_y = plan_path[:, 0] * 0
                plan_path_z = (
                    plan_path[:, 1] - int(self.origins_grid[1])
                ) * self.args.map_resolution / 100.0

                self.plan_path = np.stack(
                    (plan_path_x, plan_path_y, plan_path_z), axis=-1
                )
                action = self.ffm_act()

    
            
        # if self.args.visualize:
        #     receive_queue.put([self.agent_id, 
        #                        image_rgb, 
        #                         depth, 
        #                         self.annotated_image , 
        #                         transform_rgb_bgr(self.vis_map),
        #                         np.asarray(self.point_sum.points), 
        #                         np.asarray(self.point_sum.colors), 
        #                         self.Open3D_traj,
        #                         self.episode_n,
        #                         plan_path,
        #                         Open3d_goal_pose,
        #                         time_step_info]
                                # )    
        # action = self.keyboard_act()

        self.last_action = action
        self.last_goal = goal_points
        
        act_end_time = time.time()
        # print('act_time: %.3f秒'%(act_end_time - act_time)) 
        return action

    def _pointnav_frontier_act(self, goal_world):
        if self.pointnav_planner is None:
            raise RuntimeError("PointNav planner was not initialized")
        if (
            self._latest_pointnav_observations is None
            or self._latest_pointnav_agent_state is None
        ):
            raise RuntimeError(
                "mapping() must provide a fresh observation before PointNav"
            )
        decision = self.pointnav_planner.act(
            robot_id=self.agent_id,
            observations=self._latest_pointnav_observations,
            agent_state=self._latest_pointnav_agent_state,
            goal_world=goal_world,
            episode_start_position=self.init_agent_position,
            episode_start_rotation=self.init_agent_rotation,
        )
        self.pointnav_last_decision = decision
        if decision.request_global_replan:
            self.pointnav_replan_requested = True
            # Never forward local PointNav STOP to ObjectNav. Rotate for one
            # observation while main.py refreshes the global frontier.
            action = int(self.pointnav_planner.env_action_map['turn_left'])
        else:
            action = int(decision.action)
            if getattr(self, 'risk_navigation_enabled', False):
                shielded = shield_pointnav_action(
                    action,
                    env_action_map=self.pointnav_planner.env_action_map,
                    current_cell=self.current_grid_pose,
                    relative_angle_deg=self.relative_angle,
                    hard_unsafe_mask=self.hard_unsafe_mask,
                    risk_map=self.risk_map,
                    map_resolution_cm=self.args.map_resolution,
                    forward_step_size_m=self.pointnav_planner.spec.forward_step_size,
                    turn_angle_deg=self.pointnav_planner.spec.turn_angle,
                )
                if shielded != action:
                    action = shielded
                    self.pointnav_planner.record_executed_action(
                        self.agent_id, action
                    )
        self.l_step += 1
        return action

    def acknowledge_pointnav_replan(self):
        self.pointnav_replan_requested = False
    

    def search_navigable_path(self, original_point, offset = 0.1):
        
        plan_path = self.follower.get_path_points(
            original_point
        )
        
        if len(plan_path) > 1:
            return plan_path
  
        # Possible changes to each coordinate
        deltas = [-offset, offset]

        # Generate surrounding points using nested loops
        for dx in deltas:
            for dy in deltas:
                for dz in deltas:
                    new_point = (original_point[0] + dx, original_point[1] + dy, original_point[2] + dz)
                    plan_path = self.follower.get_path_points(
                        new_point
                    )
                    if len(plan_path) > 1:
                        self.habitat_goal_pose = new_point
                        
                        return plan_path
              
        return plan_path  

    # def keyboard_act(self):
    #     # ------------------------------------------------------------------
    #     ##### Update long-term goal if target object is found
    #     ##### Otherwise, use the LLM to select the goal
    #     # ------------------------------------------------------------------

    #     keystroke = cv2.waitKey(0)
    #     action = None
    #     if keystroke == ord(FORWARD_KEY):
    #         action = HabitatSimActions.MOVE_FORWARD
    #         print("action: FORWARD")
    #     elif keystroke == ord(LEFT_KEY):
    #         action = HabitatSimActions.TURN_LEFT
    #         print("action: LEFT")
    #     elif keystroke == ord(RIGHT_KEY):
    #         action = HabitatSimActions.TURN_RIGHT
    #         print("action: RIGHT")
    #     elif keystroke == ord(UP_KEY):
    #         action = HabitatSimActions.LOOK_UP
    #         print("action: UP")
    #         self.eve_angle += 30
    #     elif keystroke == ord(DOWN_KEY):
    #         action = HabitatSimActions.LOOK_DOWN
    #         print("action: DOWN")
    #         self.eve_angle -= 30
    #     elif keystroke == ord(FINISH):
    #         action = HabitatSimActions.STOP
    #         print("action: FINISH")
    #     else:
    #         print("INVALID KEY")
        
    #     self.l_step += 1
    #     return action


    
    def greedy_follower_act(self, plan_path):
        
        if self.is_running == False:
            return None   

        next_stg_x = np.floor((plan_path[1][0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        next_stg_y = np.floor((plan_path[1][2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])
        action = self.follower.get_next_action(
            self.habitat_goal_pose,
            self.current_grid_pose,
            self.relative_angle,
            next_stg_x, 
            next_stg_y
        )

        if not self.found_goal and action == 0:
            self.greedy_stop_count += 1
            action = 2
        else:
            self.greedy_stop_count = 0
        
        distance = np.linalg.norm(plan_path[0] - plan_path[1])
        high_diff = plan_path[1][1] - plan_path[0][1]
        angle_goal = math.degrees(math.asin(high_diff/distance))

        angle_agent = (360 - self.relative_angle) % 360.0
        eve_start_x = int(5 * math.sin(math.radians(angle_agent)) + self.current_grid_pose[0])
        eve_start_y = int(5 * math.cos(math.radians(angle_agent)) + self.current_grid_pose[1])
        eve_start_x = min(max(0, eve_start_x), self.map_size - 1)
        eve_start_y = min(max(0, eve_start_y), self.map_size - 1)
        
       
        if (self.explored_map[eve_start_x, eve_start_y] == 0 or (angle_goal - self.eve_angle) < -self.args.turn_angle/2 ) and self.eve_angle > -90:
            action = 5
            self.eve_angle -= 30
        elif self.explored_map[eve_start_x, eve_start_y] == 1 and (angle_goal - self.eve_angle) > self.args.turn_angle/2 and self.eve_angle < 0:
            action = 4
            self.eve_angle += 30
            
        action_e_time = time.time()

        # print('acton: %.3f秒'%(action_e_time - action_s_time)) 
        self.l_step += 1
        return action
    
    def ffm_act(self):
        if self.is_running == False:
            return None
        
        if self.stop and getattr(self, '_risk_escape_active', False):
            # Reaching an emergency waypoint is not task completion. Rotate
            # once so the next observation/act cycle can restore the original
            # goal without emitting Habitat STOP inside a hazard episode.
            action = 2
        elif self.stop and self.found_goal:
            action = 0
        else:
            (stg_x, stg_y) = self.stg
            angle_st_goal = math.degrees(math.atan2(stg_x - self.current_grid_pose[0],
                                                    stg_y - self.current_grid_pose[1]))
            angle_agent = (360 - self.relative_angle) % 360.0
            if angle_agent > 180:
                angle_agent -= 360
            # angle_agent = 360 - self.relative_angle
            relative_angle = angle_agent - angle_st_goal
            if relative_angle > 180:
                relative_angle -= 360
            if relative_angle < -180:
                relative_angle += 360

            eve_start_x = int(5 * math.sin(math.radians(self.relative_angle)) + self.current_grid_pose[0])
            eve_start_y = int(5 * math.cos(math.radians(self.relative_angle)) + self.current_grid_pose[1])
            eve_start_x = min(max(0, eve_start_x), self.map_size - 1)
            eve_start_y = min(max(0, eve_start_y), self.map_size - 1)
            # if eve_start_x >= self.map_size: eve_start_x = self.map_size-1
            # if eve_start_y >= self.map_size: eve_start_y = self.map_size-1 
            # if eve_start_x < 0: eve_start_x = 0 
            # if eve_start_y < 0: eve_start_y = 0 
            if self.explored_map[eve_start_x, eve_start_y] == 0 and self.eve_angle > -90:
                action = 5
                self.eve_angle -= 30
            elif self.explored_map[eve_start_x, eve_start_y] == 1 and self.eve_angle < 0:
                action = 4
                self.eve_angle += 30
            elif relative_angle > self.args.turn_angle:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle:
                action = 2  # Left
            # elif relative_angle > self.args.turn_angle / 2.:
            #     action = 7  # Right
            # elif relative_angle < -self.args.turn_angle / 2.:
            #     action = 6  # Left
            else:
                action = 1
        self.l_step += 1
        action_e_time = time.time()
        # print('action: %.3f秒'%(action_e_time - action_s_time)) 
        return action

    def _get_astar_stg_once(
        self,
        planner,
        state,
        *,
        x1,
        y1,
        coordinate_offset,
    ):
        """Derive control and visualization output from one A* search."""

        padded_start = planner._clip_cell(state)
        reusable_path = None
        astar_cache = getattr(self, '_astar_path_cache', None)
        if astar_cache is not None:
            astar_cache.restore_goal_distance(planner)
            reusable_path = astar_cache.reusable_suffix(
                planner,
                padded_start,
            )

        result = planner.plan(state, reusable_path=reusable_path)
        if result.path:
            self._astar_path_cache = AStarPathCache.capture(
                planner,
                result.path,
            )
            sampled = planner.sample_path(result.path, max_waypoints=10)
        else:
            # Keep the goal-distance transform even when the current map has
            # no route. The next step must search again, but it need not
            # recompute a goal-only heuristic if the goal is unchanged.
            self._astar_path_cache = AStarPathCache.capture(planner, ())
            sampled = [padded_start]

        if len(sampled) == 1:
            # Keep the historical visualization contract, which always
            # returned at least the current pose and one waypoint.
            sampled.append(sampled[0])
        path = [
            [
                cell[0] + x1 - coordinate_offset,
                cell[1] + y1 - coordinate_offset,
            ]
            for cell in sampled
        ]
        stg = (
            result.stg[0] + x1 - coordinate_offset,
            result.stg[1] + y1 - coordinate_offset,
        )

        if result.replan:
            self.replan_count += 1
        else:
            self.replan_count = 0
        return stg, result.stop, path

    def _get_stg(
        self, grid, start, goal, planner_name_override=None
    ):
        """Get short-term goal"""

        agent_args = getattr(self, 'args', None)
        planner_name = (
            planner_name_override
            if planner_name_override is not None
            else getattr(agent_args, 'local_planner', 'fmm')
        )
        risk_navigation_enabled = bool(
            getattr(self, 'risk_navigation_enabled', False)
        )

        [gx1, gx2, gy1, gy2] = [0, self.local_w, 0, self.local_h] 

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        # print("grid: ", grid.shape)

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        selem = skimage.morphology.disk(3)
        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            selem) != True
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        traversible[cv2.dilate(self.visited_vis[gx1:gx2, gy1:gy2][x1:x2, y1:y2], kernel) == 1] = 1
        traversible[cv2.dilate(self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2], kernel) == 1] = 0
        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)
        traversible[goal==1] = 1
        if risk_navigation_enabled or planner_name != 'fmm':
            # The padding exists for local-window arithmetic only; it is not
            # a navigable corridor around the outside of the scene map.
            traversible[[0, -1], :] = 0
            traversible[:, [0, -1]] = 0

        risk_map = None
        hard_unsafe_mask = None
        if risk_navigation_enabled:
            if getattr(self, 'risk_map', None) is not None:
                risk_map = add_boundary(
                    self.risk_map[x1:x2, y1:y2], value=0
                )
            if getattr(self, 'hard_unsafe_mask', None) is not None:
                hard_unsafe_mask = add_boundary(
                    self.hard_unsafe_mask[x1:x2, y1:y2], value=0
                ).astype(bool)

        if planner_name == 'fmm':
            # Keep the established FMM implementation and constructor path
            # unchanged for the default baseline.
            planner = FMMPlanner(
                traversible,
                risk_map=risk_map,
                risk_alpha=(
                    getattr(self, 'risk_alpha', 0.0)
                    if risk_navigation_enabled else 0.0
                ),
                hard_unsafe_mask=hard_unsafe_mask,
            )
        else:
            planner = create_local_planner(
                planner_name,
                traversible,
                risk_map=risk_map,
                risk_alpha=(
                    getattr(self, 'risk_alpha', 0.0)
                    if risk_navigation_enabled else 0.0
                ),
                hard_unsafe_mask=hard_unsafe_mask,
                rl_checkpoint=getattr(
                    agent_args, 'rl_local_checkpoint', None
                ),
                rl_device=getattr(agent_args, 'rl_local_device', 'cpu'),
                rl_deterministic=bool(int(getattr(
                    agent_args, 'rl_local_deterministic', 1
                ))),
                rl_crop_size=int(getattr(
                    agent_args, 'rl_local_crop_size', 31
                )),
                rl_rollout_steps=int(getattr(
                    agent_args, 'rl_local_rollout_steps', 5
                )),
                risk_aware=risk_navigation_enabled,
            )
        if ("plant" in self.goal_name or "tv" in self.goal_name) and \
            np.sum(self.goal_map) > 1:
            selem = skimage.morphology.disk(15)
        else:
            selem = skimage.morphology.disk(5)
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True
        goal = 1 - goal * 1.

        self._risk_escape_active = False
        self._risk_escape_reason = None
        coordinate_offset = 0
        if risk_navigation_enabled or planner_name != 'fmm':
            # Risk arrays follow the padded map exactly. The legacy branch
            # historically omitted this +1 offset, so correct it only for the
            # new planner to avoid changing risk-disabled trajectories.
            coordinate_offset = 1

            safe_goal = (
                (goal == 1)
                & (planner.traversible > 0)
                & ~planner.hard_unsafe_mask
            )
            if np.any(safe_goal):
                goal = safe_goal.astype(np.float32)
            else:
                # If every dilated goal cell is unsafe, stop at the closest
                # safe navigable cell instead of opening a path through fire.
                # It is a safety waypoint, not permission to emit task STOP.
                self._risk_escape_active = True
                self._risk_escape_reason = 'unsafe_goal'
                candidates = (
                    (planner.traversible > 0)
                    & ~planner.hard_unsafe_mask
                )
                if np.any(candidates):
                    distance_to_goal = cv2.distanceTransform(
                        (goal != 1).astype(np.uint8),
                        cv2.DIST_L2,
                        3,
                    )
                    candidate_cost = np.where(
                        candidates, distance_to_goal, np.inf
                    )
                    nearest_safe = np.unravel_index(
                        np.argmin(candidate_cost), candidate_cost.shape
                    )
                    goal = np.zeros_like(goal)
                    goal[nearest_safe] = 1
                else:
                    # No safe traversible cell exists. Hold the current cell
                    # as a non-terminal safety waypoint; ffm_act rotates and
                    # requests another observation instead of crossing the
                    # forbidden target or emitting Habitat STOP.
                    held = (
                        int(np.clip(start[0] + 1, 0, goal.shape[0] - 1)),
                        int(np.clip(start[1] + 1, 0, goal.shape[1] - 1)),
                    )
                    goal = np.zeros_like(goal)
                    goal[held] = 1
                    self._risk_escape_reason = 'trapped'

            padded_start = [
                start[0] - x1 + coordinate_offset,
                start[1] - y1 + coordinate_offset,
            ]
            escape_goal = planner.prepare_emergency_escape(padded_start)
            if escape_goal is not None:
                self._risk_escape_active = True
                if np.any(escape_goal):
                    goal = escape_goal
                    self._risk_escape_reason = 'emergency_escape'
                else:
                    goal = np.zeros_like(goal)
                    goal[tuple(map(int, padded_start))] = 1
                    self._risk_escape_reason = 'trapped'

        planner.set_multi_goal(goal)

        state = [
            start[0] - x1 + coordinate_offset,
            start[1] - y1 + coordinate_offset,
        ]
        if planner_name == 'astar':
            return self._get_astar_stg_once(
                planner,
                state,
                x1=x1,
                y1=y1,
                coordinate_offset=coordinate_offset,
            )

        path = []
        path.append(start)

        stg_x, stg_y, replan, stop_f = planner.get_short_term_goal(state)
        stg_x, stg_y = (
            stg_x + x1 - coordinate_offset,
            stg_y + y1 - coordinate_offset,
        )
        for i in range(10):
            state = [
                stg_x - x1 + coordinate_offset,
                stg_y - y1 + coordinate_offset,
            ]
            stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)
            stg_x, stg_y = (
                stg_x + x1 - coordinate_offset,
                stg_y + y1 - coordinate_offset,
            )
            
            path.append([stg_x, stg_y])
            if stop:
                break

        if replan:
            self.replan_count += 1
            # print("false: ", self.replan_count)
        else:
            self.replan_count = 0

        return (path[1][0], path[1][1]), stop_f, path
    


    def _preprocess_depth(self, depth, min_d=0.5, max_d=5.0):
        # print("depth origin: ", depth.shape)
        depth = depth[:, :, 0] * 1
        # print(np.max(depth))
        # print(np.min(depth))
        # for i in range(depth.shape[1]):
        #     depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        depth = depth * max_d 

        return depth

    def update_map(self, point_sum, camera_position_z, height_diff):

        explored_map = np.zeros((self.local_w, self.local_h))
        obstacle_map = np.zeros((self.local_w, self.local_h))

        # height range (z is down in Open3D)
        z_min = camera_position_z - height_diff
        z_max = camera_position_z + height_diff

        points = np.asarray(point_sum.points)
        
        common_mask = (
            (points[:, 0] >= self.origins_real[0] - self.map_real_halfsize) &
            (points[:, 0] <= self.origins_real[0] + self.map_real_halfsize) &
            (points[:, 2] >= self.origins_real[1] - self.map_real_halfsize) &
            (points[:, 2] <= self.origins_real[1] + self.map_real_halfsize)
        )

        mask_obstacle = common_mask & ((points[:, 1] >= z_min) & (points[:, 1] <= z_max))
        mask_explored = common_mask & (points[:, 1] <= z_max)

        points_obstacle = points[mask_obstacle]
        points_explored = points[mask_explored]

        # 计算二维地图的索引ww
        obs_i_values = np.floor((points_obstacle[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        obs_j_values = np.floor((points_obstacle[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])

        obstacle_map[obs_i_values, obs_j_values] = 1
        self.obstacle_map[obs_i_values, obs_j_values] = 1
        
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        self.obstacle_map[cv2.dilate(self.visited_vis, kernel) == 1] = 0

        exp_i_values = np.floor((points_explored[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        exp_j_values = np.floor((points_explored[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])

        explored_map[exp_i_values, exp_j_values] = 1
        self.explored_map[exp_i_values, exp_j_values] = 1

        diff_ob_ex = explored_map - obstacle_map

        if np.abs(self.eve_angle) < 10 and self.last_action != 4 and self.last_action != 5:
            self.obstacle_map[diff_ob_ex == 1] = 0
            

    
    def get_transform_matrix(self, agent_state):
        """
        transform the habitat-lab space to Open3D space (initial pose in habitat)
        habitat-lab space need to rotate camera from x,y,z to  x, -y, -z
        Returns Pose_diff, R_diff change of the agent relative to the initial timestep
        """
        camera_position = agent_state.sensor_states["depth"].position
        camera_rotation = quaternion.as_rotation_matrix(agent_state.sensor_states["depth"].rotation)

        h_camera_matrix = np.eye(4)
        h_camera_matrix[:3, :3] = camera_rotation
        h_camera_matrix[:3, 3] = camera_position

        habitat_camera_self = np.eye(4)
        habitat_camera_self[:3, :3] = np.array([[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]])
        habitat_camera_self_aj = np.eye(4)
        habitat_camera_self_aj[:3, :3] = np.array([[0, 0, -1],
                    [0, 1, 0],
                    [1, 0, 0]])
        
        R_habitat2open3d = np.eye(4)
        R_habitat2open3d[:3, :3] = self.init_sim_rotation
        R_habitat2open3d[:3, 3] = self.init_sim_position

        camera_pose = habitat_camera_self_aj @ np.linalg.inv(R_habitat2open3d) @ h_camera_matrix
        O_camera_matrix = habitat_camera_self_aj @ np.linalg.inv(R_habitat2open3d) @ h_camera_matrix @ habitat_camera_self


        return O_camera_matrix
    
    def find_nearest_point_cloud(self, point_cloud, target_point):
        # 创建 KDTree
        kdtree = o3d.geometry.KDTreeFlann(point_cloud)

        # 查找离目标点最近的点
        [k, idx, _] = kdtree.search_knn_vector_3d(target_point, 1)
        nearest_point = np.asarray(point_cloud.points)[idx[0]]
        
        return nearest_point
    
    
    def object_map_building(self, point_sum):

        points = np.asarray(point_sum.points)
        colors = np.asarray(point_sum.colors)

        mask = (points[:, 0] >= self.origins_real[0] - self.map_real_halfsize) & \
                (points[:, 0] <= self.origins_real[0] + self.map_real_halfsize) & \
                (points[:, 2] >= self.origins_real[1] - self.map_real_halfsize) & \
                (points[:, 2] <= self.origins_real[1] + self.map_real_halfsize)
                
        points_filtered = points[mask]
        colors_filtered = colors[mask]
 
        # 计算二维地图的索引ww
        i_values = np.floor((points_filtered[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        j_values = np.floor((points_filtered[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])
        
        return i_values, j_values

      
    def remove_full_points_cell(self, point_sum, camera_position):
        points = np.asarray(point_sum.points)
        colors = np.asarray(point_sum.colors)

        # 1) Drop points above the camera (ceiling / overhead clutter).
        mask = (points[:, 1] <= camera_position[1] + 0.5)

        # 2) Self-body exclusion: when a visible robot URDF is synced to
        #    the nav agent (or when the agent LOOKs_DOWN and sees the
        #    floor right under itself), the depth camera captures points
        #    belonging to the robot's own body. Those land inside the
        #    obstacle height band and get baked into obstacle_map at the
        #    agent's own cell, trapping the FMM planner. Discard any
        #    point whose XZ distance to the camera is below a small
        #    self-radius so the agent never treats itself as an obstacle.
        self_radius = float(getattr(self.args, "self_exclusion_radius",0))
        if self_radius > 0.0 and points.shape[0] > 0:
            dx = points[:, 0] - float(camera_position[0])
            dz = points[:, 2] - float(camera_position[2])
            xz_dist2 = dx * dx + dz * dz
            mask = mask & (xz_dist2 >= self_radius * self_radius)

        points_filtered = points[mask]
        colors_filtered = colors[mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_filtered)
        pcd.colors = o3d.utility.Vector3dVector(colors_filtered)

        return pcd
        

    def get_frontier_boundaries(self, frontier_loc, frontier_sizes, map_sizes):
        loc_r, loc_c = frontier_loc
        local_w, local_h = frontier_sizes
        full_w, full_h = map_sizes

        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
 
        return [int(gx1), int(gx2), int(gy1), int(gy2)]
    
    def save_rgbd_image(self, rgb_image, depth):
        vis_image_rgb = np.ones((480, 1280, 3)).astype(np.uint8) * 255
        vis_image_rgb[0:480, 0:640] = rgb_image 
        # Normalize the depth values to the range 0-255
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply a colormap (e.g., COLORMAP_JET)
        depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        vis_image_rgb[0:480, 640:1280] = depth_color
        
        ep_dir = '{}episodes/{}/eps_rgbd_{}/'.format(
            self.dump_dir, self.args.rank, self.episode_n)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)
        fn = ep_dir + 'Vis-{}.png'.format(self.l_step)
        cv2.imwrite(fn, vis_image_rgb)
        
    def save_similarity_map(self, map):
        depth_normalized = cv2.normalize(map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply a colormap (e.g., COLORMAP_JET)
        depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        ep_dir = '{}episodes/{}/eps_rgbd_{}/'.format(
            self.dump_dir, self.args.rank, self.episode_n)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)
        fn = ep_dir + 'Vis-simi-{}.png'.format(self.l_step)
        cv2.imwrite(fn, depth_color)
    
    def _visualize(self, map_pred, exp_pred, goal_map, text_queries):

        # start_x, start_y, start_o = pose

        sem_map = np.zeros((self.local_w, self.local_h))

        # no_cat_mask = sem_map == 20
        map_mask = map_pred == 1
        exp_mask = exp_pred == 1
        vis_mask = self.visited_vis == 1

        # sem_map[no_cat_mask] = 0
        # m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[exp_mask] = 2

        # m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[map_mask] = 1

        sem_map[vis_mask] = 3

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal_map, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
        # if np.sum(goal_map) == 1:
        #     f_pos = np.argwhere(goal_map == 1)
        #     # fmb = get_frontier_boundaries((f_pos[0][0], f_pos[0][1]))
        #     # goal_fmb = skimage.draw.circle_perimeter(int((fmb[0]+fmb[1])/2), int((fmb[2]+fmb[3])/2), 23)
        #     goal_fmb = skimage.draw.circle_perimeter(f_pos[0][0], f_pos[0][1], int(self.map_size/16 -1))
        #     goal_fmb[0][goal_fmb[0] > self.map_size-1] = self.map_size-1
        #     goal_fmb[1][goal_fmb[1] > self.map_size-1] = self.map_size-1
        #     goal_fmb[0][goal_fmb[0] < 0] = 0
        #     goal_fmb[1][goal_fmb[1] < 0] = 0
        #     # goal_fmb[goal_fmb < 0] =0
        #     goal_mask[goal_fmb[0], goal_fmb[1]] = 1
        #     sem_map[goal_mask] = 4


        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        vis_image = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)

       
        def get_contour_points(pos, origin, size=20):
            x, y, o = pos
            pt1 = (int(x) + origin[0],
                int(y) + origin[1])
            pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
                int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
            pt3 = (int(x + size * np.cos(o)) + origin[0],
                int(y + size * np.sin(o)) + origin[1])
            pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
                int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

            return np.array([pt1, pt2, pt3, pt4])

        pos = [self.last_grid_pose[1], int(self.map_size)-self.last_grid_pose[0], np.deg2rad(self.relative_angle)]
        agent_arrow = get_contour_points(pos, origin=(0, 0), size=10)
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(vis_image, [agent_arrow], 0, color, -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (20, 20, 20)  # BGR
        thickness = 2
        text = "Find {} ".format(text_queries)
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (480 - textsize[0]) // 2 + 30
        textY = (50 + textsize[1]) // 2
        vis_image_show = cv2.putText(vis_image, text, (textX, textY),
                                font, fontScale, color, thickness,
                                cv2.LINE_AA)

        vis_image_rgb = init_vis_image(text_queries, self.last_action)
        observation_panel = fit_image_to_panel(
            self.annotated_image,
            panel_width=640,
            panel_height=480,
        )
        vis_image_rgb[50:530, 15:655] = observation_panel
        vis_image_rgb[50:530, 670:1150] = vis_image
        
        if self.args.print_images:
            ep_dir = '{}episodes_multi/{}/eps_{}/'.format(
                self.dump_dir, self.args.rank, self.episode_n)
            if not os.path.exists(ep_dir):
                os.makedirs(ep_dir)
            fn = ep_dir + 'agent-{}-Vis-{}.png'.format(self.agent_id, self.l_step)
            cv2.imwrite(fn, vis_image_rgb)

        if self.args.visualize:
            cv2.imshow("episode_{}- agent_{}".format(self.episode_n, self.agent_id), vis_image_rgb)
            cv2.waitKey(1)

        return vis_image_show
