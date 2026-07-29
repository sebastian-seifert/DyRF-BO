import os
import json
import unittest
import tempfile
import shutil

from scripts.create_highdim_manifest import generate_highdim_manifest

class TestCreateHighDimManifest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.manifest_path = os.path.join(self.test_dir, "EXPERIMENT_MANIFEST.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_generate_highdim_manifest(self):
        manifest = generate_highdim_manifest(output_path=self.manifest_path)
        self.assertTrue(os.path.exists(self.manifest_path))

        self.assertEqual(manifest["experiment_name"], "highdim_ei_epistemic_sweep")
        self.assertIn("git_commit", manifest)
        self.assertEqual(manifest["n_seeds"], 5)
        self.assertEqual(len(manifest["benchmark_tasks"]), 18)
        self.assertEqual(len(manifest["approaches"]), 9)

if __name__ == "__main__":
    unittest.main()
