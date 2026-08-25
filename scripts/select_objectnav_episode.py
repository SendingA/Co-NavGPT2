#!/usr/bin/env python3
"""Package one ObjectNav shard episode as a runnable Habitat dataset."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, Sequence


def _read(path: Path) -> Dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _write(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"), sort_keys=True)


def package_episodes(
    source_shard: Path,
    episode_ids: Sequence[str],
    output_dir: Path,
    *,
    object_category: str | None = None,
    fire_plan_id: str | None = None,
) -> Path:
    source = _read(source_shard)
    selected_ids = [str(episode_id) for episode_id in episode_ids]
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("episode_ids must be a non-empty unique sequence")
    matches = [
        episode
        for episode in source.get("episodes", [])
        if str(episode.get("episode_id")) in set(selected_ids)
        and (
            object_category is None
            or str(episode.get("object_category")) == object_category
        )
    ]
    by_id = {str(episode.get("episode_id")): episode for episode in matches}
    if len(matches) != len(selected_ids) or len(by_id) != len(selected_ids):
        selector = f"episode_ids={selected_ids!r}"
        if object_category is not None:
            selector += f", object_category={object_category!r}"
        raise ValueError(
            f"expected one match for every {selector} in {source_shard}, "
            f"found {len(matches)}"
        )
    matches = [by_id[episode_id] for episode_id in selected_ids]
    episode = matches[0]
    scene_path = str(episode.get("scene_id", ""))
    scene_id = Path(scene_path).name.split(".", 1)[0]
    if not scene_id:
        raise ValueError("selected episode has no usable scene_id")

    scenario = {
        "schema_version": 1,
        "scene_id": scene_id,
        "episode_ids": selected_ids,
        "object_category": episode.get("object_category"),
        "fire_plan_id": fire_plan_id,
    }
    shard_payload = dict(source)
    shard_payload["episodes"] = matches
    shard_payload["fire_cost_scenario"] = scenario

    root_payload = {
        key: value
        for key, value in source.items()
        if key not in {"episodes", "goals_by_category"}
    }
    root_payload["episodes"] = []
    root_payload["content_scenes_path"] = (
        "{data_path}/content/{scene}.json.gz"
    )
    root_payload["fire_cost_scenario"] = scenario

    root_path = output_dir / "val.json.gz"
    _write(root_path, root_payload)
    _write(output_dir / "content" / f"{scene_id}.json.gz", shard_payload)
    return root_path


def package_episode(
    source_shard: Path,
    episode_id: str,
    output_dir: Path,
    *,
    object_category: str | None = None,
    fire_plan_id: str | None = None,
) -> Path:
    """Backward-compatible one-episode wrapper."""

    return package_episodes(
        source_shard,
        [episode_id],
        output_dir,
        object_category=object_category,
        fire_plan_id=fire_plan_id,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-shard", type=Path, required=True)
    parser.add_argument(
        "--episode-id",
        required=True,
        action="append",
        help="episode id to package; repeat the flag for a matched subset",
    )
    parser.add_argument(
        "--object-category",
        default=None,
        help="disambiguate shards that reuse episode ids across categories",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fire-plan-id", default=None)
    args = parser.parse_args()
    output = package_episodes(
        args.source_shard,
        args.episode_id,
        args.output_dir,
        object_category=args.object_category,
        fire_plan_id=args.fire_plan_id,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
