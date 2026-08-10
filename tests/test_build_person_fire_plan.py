import unittest

import numpy as np

from scripts.build_person_fire_plan import (
    combined_surrogate,
    source_protection_metrics,
)


class BuildPersonFirePlanTests(unittest.TestCase):
    def test_source_protection_uses_person_and_all_viewpoints(self):
        sources = [{
            "instance_id": 7,
            "category": "table",
            "centroid": [0.0, 0.5, 0.0],
            "flammability": 0.55,
        }]
        result = source_protection_metrics(
            sources,
            person_centre=[3.0, 0.0, 4.0],
            person_viewpoints=np.asarray([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
            floor_y_m=0.0,
        )[0]
        self.assertAlmostEqual(result["person_centre_clearance_m"], 5.0)
        self.assertAlmostEqual(result["nearest_viewpoint_clearance_m"], 2.0)
        self.assertAlmostEqual(result["height_above_floor_m"], 0.5)

    def test_combined_surrogate_unions_hard_and_maxes_risk(self):
        risk, hard = combined_surrogate(
            (15, 15), [(4, 4), (10, 10)],
            resolution_m=0.2,
            core_radius_m=0.4,
            risk_radius_m=1.0,
        )
        self.assertEqual(risk.shape, (15, 15))
        self.assertEqual(float(risk[4, 4]), 1.0)
        self.assertEqual(float(risk[10, 10]), 1.0)
        self.assertTrue(hard[4, 4])
        self.assertTrue(hard[10, 10])
        self.assertFalse(hard[0, 14])


if __name__ == "__main__":
    unittest.main()
