import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.select_objectnav_episode import package_episode, package_episodes


class SelectObjectNavEpisodeTests(unittest.TestCase):
    def test_packages_multiple_episodes_in_requested_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json.gz"
            payload = {
                "episodes": [
                    {
                        "episode_id": "5",
                        "scene_id": "data/scene.basis.glb",
                        "object_category": "chair",
                    }
                ],
                "goals_by_category": {},
                "category_to_task_category_id": {"chair": 0},
            }
            payload["episodes"].append(
                {
                    **payload["episodes"][0],
                    "episode_id": "7",
                }
            )
            with gzip.open(source, "wt", encoding="utf-8") as stream:
                json.dump(payload, stream)

            result = package_episodes(
                source,
                ["7", "5"],
                root / "subset",
                object_category="chair",
            )

            with gzip.open(
                result.parent / "content" / "scene.json.gz",
                "rt",
                encoding="utf-8",
            ) as stream:
                selected = json.load(stream)
            self.assertEqual(
                [episode["episode_id"] for episode in selected["episodes"]],
                ["7", "5"],
            )

    def test_object_category_disambiguates_reused_episode_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "scene.json.gz"
            payload = {
                "episodes": [
                    {
                        "episode_id": "5",
                        "scene_id": "data/SceneA.basis.glb",
                        "object_category": "bed",
                    },
                    {
                        "episode_id": "5",
                        "scene_id": "data/SceneA.basis.glb",
                        "object_category": "sofa",
                    },
                ],
                "goals_by_category": {},
                "category_to_task_category_id": {"bed": 1, "sofa": 5},
            }
            with gzip.open(source, "wt", encoding="utf-8") as stream:
                json.dump(payload, stream)

            with self.assertRaisesRegex(ValueError, "found 2"):
                package_episode(source, "5", root / "ambiguous")

            result = package_episode(
                source,
                "5",
                root / "selected",
                object_category="sofa",
                fire_plan_id="SceneA_multi_origin_medium_test",
            )
            with gzip.open(result, "rt", encoding="utf-8") as stream:
                root_payload = json.load(stream)
            with gzip.open(
                root / "selected/content/SceneA.json.gz",
                "rt",
                encoding="utf-8",
            ) as stream:
                shard_payload = json.load(stream)

            self.assertEqual([], root_payload["episodes"])
            self.assertEqual("sofa", shard_payload["episodes"][0]["object_category"])
            self.assertEqual(
                "SceneA_multi_origin_medium_test",
                shard_payload["fire_cost_scenario"]["fire_plan_id"],
            )


if __name__ == "__main__":
    unittest.main()
