"""CLI for main.py / main_vec.py.

The flags below are grouped by what they actually affect at runtime;
unused legacy flags (``--exp_name``, ``--log_interval``, ``--agent``)
were removed in the 2026-06 cleanup since no caller read them.
"""
import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description='Multi-Agent-Semantic-Exploration')

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('-d', '--dump_location', type=str, default="./tmp",
                        help='where main.py writes logs / dumps. '
                             'output goes to <dump_location>/logs/<nav_mode>/'
                             ' and <dump_location>/dump/<nav_mode>/')
    parser.add_argument('-v', '--visualize', type=int, default=0,
                        help='1: render observations + predicted semantic '
                             'map; opens an Open3D GUI in main.py')
    parser.add_argument('--print_images', type=int, default=0,
                        help='1: persist visualization frames to disk')

    # ------------------------------------------------------------------
    # Camera + scene config
    # ------------------------------------------------------------------
    parser.add_argument('-fw', '--frame_width', type=int, default=640)
    parser.add_argument('-fh', '--frame_height', type=int, default=480)
    parser.add_argument("--task_config", type=str,
                        default="multi_objectnav_hm3d.yaml",
                        help="path to config yaml under configs/")
    parser.add_argument('--hfov', type=float, default=79.0,
                        help="horizontal field of view in degrees")

    # ------------------------------------------------------------------
    # Multi-agent / parallel run
    # ------------------------------------------------------------------
    parser.add_argument('--num_local_steps', type=int, default=25,
                        help='steps between two global re-plans')
    parser.add_argument('-n', '--num_processes', type=int, default=1,
                        help='only honored by main_vec.py')
    parser.add_argument('--rank', type=int, default=0,
                        help='set automatically by main_vec.py per worker; '
                             'main.py keeps the default 0')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Habitat-sim GPU device id')
    parser.add_argument('--num_agents', type=int, default=2,
                        help='number of agents in the simulator')

    # ------------------------------------------------------------------
    # Mapping / perception
    # ------------------------------------------------------------------
    parser.add_argument('--map_resolution', type=int, default=5,
                        help='cm per occupancy grid cell')
    parser.add_argument('--map_size_cm', type=int, default=2400,
                        help='occupancy map side length (cm)')
    parser.add_argument('--map_height_cm', type=int, default=130,
                        help='top-down map slice height (cm)')
    parser.add_argument('--sem_threshold', type=float, default=0.85,
                        help='semantic detection confidence above which '
                             'the goal is considered found')

    # ------------------------------------------------------------------
    # Global planner
    # ------------------------------------------------------------------
    parser.add_argument('--nav_mode', type=str, default="gpt",
                        choices=['nearest', 'co_ut', 'fill', 'gpt'],
                        help='global frontier policy. nearest=closest, '
                             'co_ut=cooperative assignment, fill=highest '
                             'frontier score, gpt=GPT-4o decision (calls '
                             'OpenAI; see --gpt_type)')
    parser.add_argument('--fill_mode', type=int, default=0,
                        help='1: when an agent revisits the same frontier, '
                             'mark its area as obstacle and re-detect')
    parser.add_argument('--gpt_type', type=int, default=2,
                        help='1: gpt-3.5-turbo  2: gpt-4o (default)  '
                             '3: gpt-4o-mini  (only used when nav_mode=gpt)')

    # ------------------------------------------------------------------
    # Fire-scene observation suite (Beer-Lambert RGB + noisy depth +
    # radar / lidar / thermal). The suite is auto-constructed whenever
    # --fire_world=1, even if --fire_sensors=0.
    # ------------------------------------------------------------------
    parser.add_argument('--fire_sensors', type=int, default=0,
                        help='1: enable the fire-scene observation suite '
                             '(Beer-Lambert RGB, smoke-degraded depth, '
                             'radar/lidar/thermal, dashboard). Implicitly '
                             'on when --fire_world=1.')
    parser.add_argument('--fire_apply_to_obs', type=int, default=1,
                        help='1: replace observations[\'rgb\'] with the '
                             'smoke-affected RGB before the agent sees it; '
                             '0: keep clean RGB for nav, only dump degraded '
                             'sensors to disk')
    parser.add_argument('--smoke_density', type=float, default=0.6,
                        help='[0,1] Beer-Lambert smoke density. Drives both '
                             'the noisy-depth visibility cutoff and the '
                             'optional --fire_world_compound_rgb pass.')
    parser.add_argument('--fire_dump_dir', type=str,
                        default='./outputs/fire_sensors',
                        help='per-step sensor image output directory')
    parser.add_argument('--fire_save_every', type=int, default=1,
                        help='save dumps every N steps (1 = every step)')
    parser.add_argument('--fire_save_npz', type=int, default=0,
                        help='1: also dump raw numpy arrays as .npz')
    parser.add_argument('--fire_show_window', type=int, default=0,
                        help='1: open a live OpenCV 2x4 dashboard window '
                             'per agent')
    parser.add_argument('--lidar_360', type=int, default=0,
                        help='1: install 4 yaw-rotated depth sensors '
                             '(front/left/back/right) so the LIDAR module '
                             'stitches a true 360 deg point cloud. Only '
                             'effective with --fire_sensors=1.')
    parser.add_argument('--lidar_resolution', type=int, default=320,
                        help='per-slice depth resolution for the 360 deg '
                             'LIDAR (square HxW). Lower = faster.')

    # ------------------------------------------------------------------
    # Smoke-scene perception switches
    # ------------------------------------------------------------------
    parser.add_argument('--depth_use_clean', type=int, default=0,
                        help='1: write Habitat\'s clean (un-degraded) depth '
                             'back into observations, even if the fire suite '
                             'computed a noisy version. Strongly recommended '
                             'when debugging / comparing nav under smoke.')
    parser.add_argument('--use_thermal_perception', type=int, default=1,
                        help='1: when the fire suite is on, inject the '
                             'thermal flame mask into observations and let '
                             'the detector source fire detections from '
                             'thermal instead of HSV-on-smoky-RGB.')
    parser.add_argument('--rgb_dehaze', type=int, default=0,
                        help='1: depth-aware inverse Beer-Lambert + CLAHE on '
                             'the smoky RGB before object detection. '
                             'Only meaningful when --depth_use_clean=1, '
                             'otherwise the inversion uses noisy depth and '
                             'amplifies artifacts.')

    # ------------------------------------------------------------------
    # FireWorld runtime: 3D voxel-driven RGB / Thermal, indexed by step
    # ------------------------------------------------------------------
    parser.add_argument('--fire_world', type=int, default=0,
                        help='1: load the precomputed FireWorld voxel '
                             'timeline and render RGB/Thermal from the '
                             'agent pose (overrides the Beer-Lambert RGB '
                             'inside the sensor suite).')
    parser.add_argument('--fire_world_plan_id', type=str, default=None,
                        help='Plan id (12-hex) under scenes/<scene>/plans/. '
                             'Required when --fire_world=1.')
    parser.add_argument('--fire_world_scenes_root', type=str, default='scenes',
                        help='where to find inventory.json + plan.json')
    parser.add_argument('--fire_world_out_root', type=str,
                        default='outputs/fire_world',
                        help='where to find timeline.npz '
                             '(out_root/<scene>/<plan_id>/timeline.npz)')
    parser.add_argument('--fire_steps_per_unit', type=int, default=5,
                        help='robot steps that elapse for every 1 unit of '
                             'fire-time (larger = slower fire vs the agent)')
    parser.add_argument('--fire_seconds_per_unit', type=float, default=2.0,
                        help='timeline seconds consumed per fire-time unit. '
                             'With defaults 5/2.0, 5 robot steps advance '
                             'the simulated fire by 2 s.')
    parser.add_argument('--fire_world_smoke_k_ext', type=float, default=4.0,
                        help='extinction coefficient multiplier on the '
                             'smoke voxel field (per metre)')
    parser.add_argument('--fire_world_n_steps', type=int, default=24,
                        help='ray-march samples per pixel inside the '
                             'voxel renderer')
    parser.add_argument('--fire_world_render_scale', type=float, default=0.5,
                        help='render the volume integrator at this fraction '
                             'of camera resolution (0.5 -> ~4x speedup; '
                             '1.0 -> full resolution)')
    parser.add_argument('--fire_world_compound_rgb', type=int, default=0,
                        help='1: stack a global Beer-Lambert pass on top of '
                             'the FireWorld voxel RGB so areas outside the '
                             'active fire room still feel smoky. Density '
                             'comes from --smoke_density. Flame pixels are '
                             'guarded so the second pass cannot erase them.')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    return args
