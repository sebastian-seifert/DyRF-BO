import os
import unittest
import tempfile
import shutil

from scripts.archive_highdim_results import archive_highdim_results

class TestArchiveHighDimResults(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.target_dir = os.path.join(self.test_dir, "results/epistemic_ei_highdim")
        os.makedirs(self.target_dir, exist_ok=True)
        with open(os.path.join(self.target_dir, "dummy.txt"), "w") as f:
            f.write("test content\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_archive_highdim_results(self):
        tarball_path = os.path.join(self.test_dir, "epistemic_ei_highdim_archive_test.tar.gz")
        archive_path = archive_highdim_results(
            source_dir=self.target_dir,
            output_tarball=tarball_path
        )
        self.assertTrue(os.path.exists(archive_path))
        self.assertTrue(archive_path.endswith(".tar.gz"))

if __name__ == "__main__":
    unittest.main()
