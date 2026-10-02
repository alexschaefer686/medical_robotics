import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


class TestCompatibility(unittest.TestCase):
    def test_legacy_exports_are_identical(self):
        for suffix in ["", ".environment", ".geometry", ".evaluate", ".gui"]:
            old = importlib.import_module("vessel_phantom" + suffix)
            new = importlib.import_module("view_transition_rl.simulation" + suffix)
            for name in old.__all__:
                with self.subTest(module=suffix, name=name):
                    self.assertIs(getattr(old, name), getattr(new, name))

    def run_cli(self, cwd, *arguments):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
        return subprocess.run([sys.executable, *arguments], cwd=cwd, env=env,
                              check=True, capture_output=True, text=True).stdout

    def test_new_and_legacy_cli_outside_project(self):
        config = Path(__file__).resolve().parents[2] / "configs/simulation/default.json"
        with tempfile.TemporaryDirectory() as directory:
            new = json.loads(self.run_cli(directory, "-m", "view_transition_rl", "sim", "info"))
            old = json.loads(self.run_cli(directory, "-m", "vessel_phantom", "info"))
            self.assertEqual(new, old)
            for prefix, arguments in [("new", ["-m", "view_transition_rl", "sim"]),
                                      ("old", ["-m", "vessel_phantom"])]:
                self.run_cli(directory, *arguments, "--config", str(config),
                             "sample", "--frame", "95", "--output", prefix)
            with Image.open(Path(directory) / "new.png") as a, Image.open(Path(directory) / "old.png") as b:
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            self.assertEqual(json.loads((Path(directory) / "new.json").read_text()),
                             json.loads((Path(directory) / "old.json").read_text()))
            self.assertIn("gui", self.run_cli(directory, "-m", "view_transition_rl", "sim", "--help"))

    def test_project_and_reserved_packages_are_lightweight(self):
        code = """
import sys
import view_transition_rl
import view_transition_rl.envs
import view_transition_rl.agents
import view_transition_rl.training
import view_transition_rl.evaluation
import view_transition_rl.visual_servoing
assert not any(name in sys.modules for name in ['vtk', 'vtkmodules', 'pyvista', 'torch', 'gymnasium'])
"""
        with tempfile.TemporaryDirectory() as directory:
            self.run_cli(directory, "-c", code)


if __name__ == "__main__":
    unittest.main()
