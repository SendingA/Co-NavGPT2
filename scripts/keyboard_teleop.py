#!/usr/bin/env python3

"""
Keyboard teleoperation for Habitat agents.
- Supports `--scene-id` to pick the first matching episode.
- Sends integer action indices to `env.step()` (multi-agent compatible).
- Separate RGB and Depth windows, positioned to avoid overlap.
- Graceful KeyboardInterrupt handling and traceback printing on errors.

Controls:
- W/A/D: move / turn
- S: stop
- Q: look down
- E: look up
- ESC: quit
"""

import argparse
import cv2
import numpy as np
import traceback
from habitat.config.default import get_config
from habitat import Env


def main():
    parser = argparse.ArgumentParser(description="Keyboard teleoperation for Habitat agents")
    parser.add_argument("--task-config", type=str, required=True,
                        help="Path to config yaml containing task information")
    parser.add_argument("--num-agents", type=int, default=1,
                        help="Number of agents")
    parser.add_argument("--agent-id", type=int, default=0,
                        help="Agent ID to control")
    parser.add_argument("--scene-id", type=str, default=None,
                        help="Scene ID to filter episodes (e.g., Nfvxx8J5NCo)")
    parser.add_argument("--show-depth", type=int, default=0,
                        help="Show depth sensor (1=yes, 0=no)")

    args = parser.parse_args()

    # Load config
    config = get_config(config_paths=[args.task_config])
    config.defrost()
    config.SIMULATOR.NUM_AGENTS = args.num_agents
    config.SIMULATOR.AGENTS = [f"AGENT_{i}" for i in range(args.num_agents)]
    config.freeze()

    # Build action mapping from config-defined possible actions
    action_list = list(config.TASK.POSSIBLE_ACTIONS)
    action_dict = {name: idx for idx, name in enumerate(action_list)}
    default_stop = action_dict.get("STOP", 0)

    # Create environment
    env = Env(config=config)

    # If scene-id provided, filter episodes and pick first match
    if args.scene_id is not None:
        episodes = env.episodes
        matching_episodes = [ep for ep in episodes if args.scene_id in ep.scene_id]
        if not matching_episodes:
            print(f"No episodes found for scene {args.scene_id}")
            return
        env.current_episode = matching_episodes[0]
        print(f"Selected episode: {env.current_episode.episode_id} in scene {args.scene_id}")

    # Reset environment and prepare single combined window
    observations = env.reset()
    window_name = "Habitat Teleop (RGB+Depth)" if args.show_depth else "Habitat Teleop (RGB)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Controls:")
    print("W/A/S/D: Move forward/left/turn right")
    print("Q/E: Look down/up")
    print("ESC: Quit")

    # Key -> action name mapping
    key_map = {
        ord('w'): "MOVE_FORWARD",
        ord('a'): "TURN_LEFT",
        ord('d'): "TURN_RIGHT",
        ord('s'): "STOP",
        ord('q'): "LOOK_DOWN",
        ord('e'): "LOOK_UP",
    }

    try:
        while True:
            obs = observations[args.agent_id]

            # Show RGB (convert from RGB->BGR for OpenCV display)
            rgb = obs["rgb"][:, :, [2, 1, 0]]

            # If depth requested, colorize and combine horizontally with RGB
            if args.show_depth and "depth" in obs:
                depth = obs["depth"]
                # squeeze to 2D if needed
                if depth.ndim == 3 and depth.shape[2] == 1:
                    depth_map = depth[:, :, 0]
                else:
                    depth_map = np.squeeze(depth)
                # normalize and colorize
                max_d = 5.0
                depth_normalized = np.clip((depth_map / max_d) * 255.0, 0, 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                # Ensure same height as RGB
                if depth_colored.shape[:2] != rgb.shape[:2]:
                    depth_colored = cv2.resize(depth_colored, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                combined = np.hstack((rgb, depth_colored))
                cv2.imshow(window_name, combined)
            else:
                cv2.imshow(window_name, rgb)

            # Wait for a keypress
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC
                break

            action_name = key_map.get(key, None)
            if action_name is None:
                # ignore unknown keys
                continue

            # Convert action name to integer index (if available)
            action_idx = action_dict.get(action_name)
            if action_idx is None:
                print(f"Action '{action_name}' not available in config.TASK.POSSIBLE_ACTIONS")
                continue

            # Build action list for environment (one per agent)
            actions = [default_stop for _ in range(args.num_agents)]
            actions[args.agent_id] = action_idx

            try:
                observations = env.step(actions)
            except Exception as e:
                traceback.print_exc()
                print(f"Error executing step with actions={actions}: {e}")
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
