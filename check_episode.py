#!/usr/bin/env python3
import argparse

from habitat.config.default import get_config
from habitat.datasets import make_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect episodes and scenes from a Habitat dataset config."
    )
    parser.add_argument(
        "--config",
        default="configs/multi_objectnav_hm3d.yaml",
        help="Path to the Habitat config used for loading the dataset.",
    )
    parser.add_argument(
        "--episode-idx",
        type=int,
        default=4,
        help="Zero-based episode index to inspect (default: 4 -> 第五个).",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="Show the first N episodes (default: 10).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = get_config(config_paths=[args.config])
    dataset = make_dataset(cfg.DATASET.TYPE, config=cfg.DATASET)
    episodes = dataset.episodes

    print(f"Config: {args.config}")
    print(f"Total episodes: {len(episodes)}")
    scene_ids = {ep.scene_id for ep in episodes}
    print(f"Unique scenes: {len(scene_ids)}")
    print()

    if args.episode_idx >= len(episodes):
        print(f"Requested episode_idx={args.episode_idx} but dataset has only {len(episodes)} episodes.")
        return

    ep = episodes[args.episode_idx]
    scene_name = ep.scene_id.split("/")[-1].replace(".glb", "").replace(".scene_instance.json", "")
    obj_cat = getattr(ep, "object_category", None)
    goals = getattr(ep, "goals", None)
    num_goals = len(goals) if goals is not None else 0

    print("Selected episode:")
    print(f"  episode_idx : {args.episode_idx}")
    print(f"  episode_id  : {ep.episode_id}")
    print(f"  scene_id    : {ep.scene_id}")
    print(f"  scene_name  : {scene_name}")
    print(f"  object_cat  : {obj_cat}")
    print(f"  start_pos   : {ep.start_position}")
    print(f"  start_rot   : {ep.start_rotation}")
    print(f"  goals_count : {num_goals}")
    print()

    print(f"First {min(args.preview, len(episodes))} episodes preview:")
    for i in range(min(args.preview, len(episodes))):
        ep_i = episodes[i]
        scene_name_i = ep_i.scene_id.split("/")[-1].replace(".glb", "").replace(".scene_instance.json", "")
        obj_cat_i = getattr(ep_i, "object_category", None)
        print(f"  #{i:03d} scene={scene_name_i} target={obj_cat_i} episode_id={ep_i.episode_id}")


if __name__ == "__main__":
    main()
