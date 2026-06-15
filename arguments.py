import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description='Multi-Agent-Semantic-Exploration')

    # General Arguments
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    # Logging, loading models, visualization
    parser.add_argument('--log_interval', type=int, default=10,
                        help="""log interval, one log per n updates
                                (default: 10) """)
    parser.add_argument('-d', '--dump_location', type=str, default="./tmp",
                        help='path to dump models and log (default: ./tmp/)')
    parser.add_argument('--exp_name', type=str, default="exp1",
                        help='experiment name (default: exp1)')
    parser.add_argument('-v', '--visualize', type=int, default=0,
                        help="""1: Render the observation and
                                   the predicted semantic map
                                (default: 0)""")
    parser.add_argument('--print_images', type=int, default=0,
                        help='1: save visualization as images')

    # Environment, dataset and episode specifications
    parser.add_argument('-fw', '--frame_width', type=int, default=640,
                        help='Frame width (default:160)')
    parser.add_argument('-fh', '--frame_height', type=int, default=480,
                        help='Frame height (default:120)')
    parser.add_argument("--task_config", type=str,
                        default="multi_objectnav_hm3d.yaml",
                        help="path to config yaml containing task information")
    parser.add_argument('--hfov', type=float, default=79.0,
                        help="horizontal field of view in degrees")

    # Model Hyperparameters
    parser.add_argument('--agent', type=str, default="sem_exp")
    parser.add_argument('--num_local_steps', type=int, default=25,
                        help="""Number of steps the local policy
                                between each global step""")
    parser.add_argument('-n', '--num_processes', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--map_size_cm', type=int, default=2400)
    parser.add_argument('--map_height_cm', type=int, default=130)
    parser.add_argument('--sem_threshold', type=float, default=0.85)
    parser.add_argument('--num_agents', type=int, default=2)
    
    
    # train_se_frontier
    parser.add_argument('--nav_mode', type=str, default="gpt",
                        choices=['nearest', 'co_ut', 'fill', "gpt"])
    parser.add_argument('--fill_mode', type=int, default=0)
    parser.add_argument('--gpt_type', type=int, default=2,
                        help="""0: text-davinci-003
                                1: gpt-3.5-turbo
                                2: gpt-4o
                                3: gpt-4o-mini
                                (default: 2)""")

    # ----------------------------------------------------------------------
    # Fire-scene multi-modal sensor simulator
    # ----------------------------------------------------------------------
    parser.add_argument('--fire_sensors', type=int, default=0,
                        help='1: enable smoke/depth-noise/radar/thermal sensor sim; '
                             'override RGB+Depth observations and dump per-step files')
    parser.add_argument('--fire_apply_to_obs', type=int, default=1,
                        help='1: feed degraded RGB+Depth back to mapping pipeline; '
                             '0: only save degraded sensors but keep clean obs for nav')
    parser.add_argument('--smoke_density', type=float, default=0.6,
                        help='[0,1] smoke optical thickness control. '
                             '0=clear, 1=visibility ~1m')
    parser.add_argument('--fire_dump_dir', type=str,
                        default='./outputs/fire_sensors',
                        help='output directory for per-step sensor images')
    parser.add_argument('--fire_save_every', type=int, default=1,
                        help='save every N steps (1 = save every step)')
    parser.add_argument('--fire_save_npz', type=int, default=0,
                        help='1: also dump raw numpy arrays as .npz alongside images')
    parser.add_argument('--fire_show_window', type=int, default=0,
                        help='1: open a live OpenCV 2x4 dashboard window')
    parser.add_argument('--lidar_360', type=int, default=0,
                        help='1: install 4 yaw-rotated depth sensors '
                             '(front/left/back/right) on each agent so the '
                             'LIDAR module can stitch a true 360° point cloud')
    parser.add_argument('--lidar_resolution', type=int, default=320,
                        help='per-slice depth resolution for the 360° LIDAR '
                             '(square HxW). Lower = faster.')

    # ----------------------------------------------------------------------
    # Smoke-scene perception switches: keep depth/thermal trustworthy and
    # let RGB degrade. Defaults preserve previous behaviour when off.
    # ----------------------------------------------------------------------
    parser.add_argument('--depth_use_clean', type=int, default=0,
                        help='1: keep the original (clean) Habitat depth for '
                             'mapping/navigation under smoke instead of the '
                             'sensor-simulated noisy depth. RGB is still the '
                             'smoke-attenuated version.')
    parser.add_argument('--use_thermal_perception', type=int, default=1,
                        help='1: when fire_sensors is enabled, inject the '
                             'thermal flame mask into observations and let the '
                             'detector source fire detections from thermal '
                             'instead of HSV on the smoky RGB.')
    parser.add_argument('--rgb_dehaze', type=int, default=0,
                        help='1: apply depth-aware inverse Beer-Lambert + CLAHE '
                             'on the smoky RGB before object detection. '
                             'Requires depth_use_clean=1 for best results.')

    # parse arguments
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    return args
