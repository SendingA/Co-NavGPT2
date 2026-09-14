import unittest
from copy import deepcopy

import numpy as np

from scripts.capture_scene_fire_gallery import floor_aware_positions, gray_haze, merge_plans


class SceneFireGalleryTests(unittest.TestCase):
    def test_wall_tv_camera_stays_on_floor_below_source(self):
        class TwoFloors:
            def snap_point(self, point):
                p = np.array(point, dtype=float)
                p[1] = min((0., 2.6), key=lambda floor: abs(floor-p[1]))
                return p
        pf = TwoFloors()
        source = [0., 1.4, 0.]
        self.assertEqual(pf.snap_point(source)[1], 2.6)
        positions = floor_aware_positions(pf, source)
        self.assertTrue(positions)
        self.assertTrue(all(p[1] == 0. for p in positions))

    def plan(self, kind, ids):
        return {"scene_id":"sample", "plan_id":kind, "plan_hash":"old",
                "fire_type":kind, "intensity":"medium", "duration_s":600.,
                "propagation_rules":{"floor_max_spread_radius_m":2.2},
                "multi_origin_area_policy":{"old":True},
                "ignitions":[{"object_id":i, "ignite_time_s":0.,
                              "ignition_role":"initial", "source_temp_c":750.}
                             for i in ids]}

    def test_union_deduplicates_and_preserves_medium_physics(self):
        plans={"multi_origin":self.plan("multi_origin",[1,2]),
               "bedroom_textile":self.plan("bedroom_textile",[2,3])}
        before=deepcopy(plans)
        merged=merge_plans(plans)
        self.assertEqual([s["object_id"] for s in merged["ignitions"]],[1,2,3])
        self.assertEqual(merged["num_initial_ignitions"],3)
        self.assertEqual(merged["propagation_rules"],before["multi_origin"]["propagation_rules"])
        self.assertEqual(merged["gallery_source_templates"]["2"],["multi_origin","bedroom_textile"])
        self.assertEqual(merged["duration_s"],600.)
        self.assertNotIn("multi_origin_area_policy",merged)
        self.assertEqual(plans,before)
        self.assertEqual(merged,merge_plans(plans))
        self.assertEqual(merged["gallery_missing_types"],["kitchen_grease_fire","living_room_electric"])

    def test_reject_mixed_intensity_and_delayed_sources(self):
        base=self.plan("multi_origin",[1])
        other=self.plan("bedroom_textile",[2])
        other["intensity"]="severe"
        with self.assertRaises(ValueError):merge_plans({"multi_origin":base,"bedroom_textile":other})
        base["ignitions"][0]["ignite_time_s"]=5.
        with self.assertRaises(ValueError):merge_plans({"multi_origin":base})

    def test_haze_preserves_flame_smoke_and_inputs(self):
        rgb=np.full((2,3,3),40,np.uint8)
        depth=np.array([[0,2,8],[8,8,8]],np.float32)
        trans=np.array([[1,1,1],[1,.5,1]],np.float32)
        flame=np.array([[0,0,0],[1,0,0]],np.float32)
        originals=[x.copy() for x in (rgb,depth,trans,flame)]
        styled,alpha=gray_haze(rgb,depth,trans,flame)
        self.assertEqual(alpha[0,0],0)
        self.assertGreater(alpha[0,2],alpha[0,1])
        self.assertEqual(alpha[1,0],0)
        self.assertEqual(alpha[1,1],0)
        self.assertLessEqual(float(alpha.max()),.220001)
        self.assertGreater(int(styled[0,2,0]),40)
        for actual,old in zip((rgb,depth,trans,flame),originals):np.testing.assert_array_equal(actual,old)


if __name__=="__main__":unittest.main()
