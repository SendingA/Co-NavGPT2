"""Check the privacy and portability boundaries of the dataset release."""
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.package_anonymous_dataset import package, source_files


class DatasetPackageTests(unittest.TestCase):
    def test_omits_git_and_materializes_internal_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            asset = root / "versioned_data" / "robot"
            (asset / ".git").mkdir(parents=True)
            (asset / ".git" / "config").write_text("private fixture")
            (asset / "mesh.txt").write_text("runtime asset")
            (asset / "val-files.json.gz").write_bytes(b"installer fixture")
            (root / "robot").symlink_to("versioned_data/robot", target_is_directory=True)
            output = Path(temporary) / "release.zip"
            result = package(root, output)
            self.assertEqual(len(result["files"]), 2)
            with zipfile.ZipFile(output) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(archive.read("data/robot/mesh.txt"), b"runtime asset")
                self.assertFalse(any(".git" in name or "-files.json.gz" in name for name in archive.namelist()))
                for info in archive.infolist():
                    self.assertEqual(info.date_time, (1980, 1, 1, 0, 0, 0))
                    self.assertFalse(info.comment)

    def test_rejects_external_broken_and_cyclic_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            root.mkdir()
            alias = root / "alias"
            alias.symlink_to("..", target_is_directory=True)
            with self.assertRaises(ValueError):
                list(source_files(root))
            alias.unlink()
            alias.symlink_to("missing")
            with self.assertRaises(FileNotFoundError):
                list(source_files(root))
            alias.unlink()
            alias.symlink_to(".", target_is_directory=True)
            with self.assertRaises(ValueError):
                list(source_files(root))


if __name__ == "__main__":
    unittest.main()
