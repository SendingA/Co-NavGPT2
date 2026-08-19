"""Regression tests for fire-sensor artifact controls."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from utils.fire_sensors.suite import FireSensorSuite


class FireSensorArtifactTests(unittest.TestCase):
    def test_zero_save_interval_disables_all_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "disabled"
            suite = FireSensorSuite(
                dump_dir=str(output_dir),
                save_every=0,
                seed=1,
            )

            suite.save_step({}, episode=0, step=0, agent_id=0)

            self.assertEqual(suite.save_every, 0)
            self.assertFalse(output_dir.exists())

    def test_negative_save_interval_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "non-negative"):
                FireSensorSuite(
                    dump_dir=str(Path(temporary) / "invalid"),
                    save_every=-1,
                )


if __name__ == "__main__":
    unittest.main()
