import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from view_transition_rl.simulation import PhantomConfig, PhantomEnv
from view_transition_rl.simulation.evaluate import overlap_scores
from view_transition_rl.simulation.geometry import pose_from_euler, rotation_xyz


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = PhantomEnv()

    def pose(self):
        return np.array(self.env.get_state()["phantom_T_probe_mm"])

    def test_bundled_provenance(self):
        data = json.loads((self.env.assets / "provenance.json").read_text())
        self.assertEqual(len(self.env.reference_rows), 190)
        for record in data["files"]:
            digest = hashlib.sha256((self.env.assets / record["bundled_path"]).read_bytes()).hexdigest()
            self.assertEqual(digest, record["sha256"])

    def test_probe_model_is_a_bundled_closed_mesh(self):
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        metadata = json.loads((self.env.assets / "probe_model.json").read_text())
        self.assertEqual(metadata["length_unit"], "mm")
        self.assertEqual(metadata["probe_bottom_size_mm"], [40, 10])
        self.assertEqual(metadata["retained_height_mm"], 50)
        self.assertEqual(metadata["boundary_or_nonmanifold_edges"], 0)
        for key, expected in [("source_file", "source_sha256"),
                              ("processed_file", "processed_sha256")]:
            path = self.env.assets / metadata[key]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, metadata[expected])
        reader = vtk.vtkSTLReader()
        reader.SetFileName(str(self.env.assets / metadata["processed_file"]))
        reader.Update()
        bounds = reader.GetOutput().GetBounds()
        np.testing.assert_allclose(bounds, metadata["probe_bounds_mm"], atol=1e-5)
        points = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
        contact = points[points[:, 2] >= -2e-3]
        np.testing.assert_allclose([contact[:, :2].min(0), contact[:, :2].max(0)],
                                   [[-20, -5], [20, 5]], atol=2e-3)

    def test_world_transform_and_reference_units(self):
        np.testing.assert_allclose(self.env.world_T_phantom @ self.env.phantom_T_world, np.eye(4), atol=1e-12)
        row = self.env.reference_rows[0]
        world = self.env.world_T_phantom @ self.env.reference_pose(0)
        np.testing.assert_allclose(world[:3, 3], [float(row[k]) * 1000 for k in ["tx_m", "ty_m", "tz_m"]], atol=1e-10)
        self.assertLess(abs(self.env.reference_pose(0)[2, 3]), .05)

    def test_all_replay_frames_contact(self):
        for i in range(190):
            result = self.env.replay_frame(i)
            self.assertTrue(result.accepted, result.reason)
            self.assertEqual(self.pose()[2, 3], 0)

    def test_projected_translation_and_rotation_order(self):
        pose = pose_from_euler(0, 0, .25, -.2, .4)
        self.assertTrue(self.env.set_pose(pose).accepted)
        action = np.array([1.0, -2.0, .03, -.04, .02])
        self.assertTrue(self.env.step(action).accepted)
        axes = pose[:2, :2] / np.linalg.norm(pose[:2, :2], axis=0)
        actual = self.pose()
        np.testing.assert_allclose(actual[:2, 3], axes @ action[:2], atol=1e-12)
        np.testing.assert_allclose(actual[:3, :3], pose[:3, :3] @ rotation_xyz(*action[2:]), atol=1e-12)
        self.assertEqual(actual[2, 3], 0)

    def test_combined_motion_keeps_centre(self):
        self.env.set_pose(np.eye(4))
        rng = np.random.default_rng(31)
        for _ in range(200):
            action = rng.normal(size=5) * [.1, .1, .01, .01, .01]
            result = self.env.step(action)
            self.assertTrue(result.accepted, result.reason)
            self.assertEqual(self.pose()[2, 3], 0)
            np.testing.assert_allclose(self.pose()[:3, :3].T @ self.pose()[:3, :3], np.eye(3), atol=1e-10)

    def test_rejection_is_atomic(self):
        before = self.pose()
        mask = self.env.get_mask()
        actions = [[100, 0, .2, 0, 0], [0, 0, 2, 0, 0], [np.nan, 0, 0, 0, 0], [1, 2], "bad"]
        for action in actions:
            result = self.env.step(action)
            self.assertFalse(result.accepted)
            self.assertTrue(result.reason)
            np.testing.assert_array_equal(self.pose(), before)
            np.testing.assert_array_equal(self.env.get_mask(), mask)

    def test_invalid_absolute_pose(self):
        before = self.pose()
        for invalid in [np.eye(3), np.zeros((4, 4)), np.diag([-1, 1, 1, 1]),
                        pose_from_euler(30, 0), pose_from_euler(0, 0, np.deg2rad(61))]:
            self.assertFalse(self.env.set_pose(invalid).accepted)
            np.testing.assert_array_equal(self.pose(), before)
        off_plane = np.eye(4)
        off_plane[2, 3] = .01
        self.assertFalse(self.env.set_pose(off_plane).accepted)

    def test_footprint_boundary_and_tilt_limit(self):
        self.assertTrue(self.env.set_pose(pose_from_euler(15, 0)).accepted)
        self.assertFalse(self.env.step([.001, 0, 0, 0, 0]).accepted)
        self.assertTrue(self.env.set_pose(pose_from_euler(0, 0, np.deg2rad(60))).accepted)
        self.assertFalse(self.env.step([0, 0, .001, 0, 0]).accepted)

    def test_mask_copy_export_reset(self):
        mask = self.env.get_mask()
        self.assertGreater(mask.sum(), 0)
        mask[:] = 7
        self.assertNotIn(7, self.env.get_mask())
        original_pose = self.pose()
        self.env.step([1, 0, 0, 0, 0])
        self.env.reset()
        np.testing.assert_array_equal(self.pose(), original_pose)
        with tempfile.TemporaryDirectory() as directory:
            image_path, state_path = self.env.export(Path(directory) / "capture")
            with Image.open(image_path) as image:
                np.testing.assert_array_equal(np.asarray(image), self.env.get_mask())
            state = json.loads(state_path.read_text())
            np.testing.assert_allclose(state["phantom_T_probe_mm"], self.pose())

    def test_config_validation(self):
        for kwargs in [{"phantom_size_mm": (1, 1, -1)}, {"max_tilt_deg": 90},
                       {"image_shape": (480.2, 480)}, {"probe_size_mm": (0, 10)}]:
            with self.assertRaises(ValueError):
                replace(self.env.config, **kwargs)
        with self.assertRaises(ValueError):
            PhantomEnv(replace(self.env.config, phantom_size_mm=(70, 100, 5)))

    def test_custom_resolution(self):
        env = PhantomEnv(replace(self.env.config, image_shape=(120, 160)))
        self.assertEqual(env.get_mask().shape, (120, 160))

    def test_metrics_empty_policy(self):
        empty = np.zeros((10, 10), dtype=np.uint8)
        self.assertEqual(overlap_scores(empty, empty), (1, 1))
        self.assertEqual(overlap_scores(empty, np.ones_like(empty)), (0, 0))


if __name__ == "__main__":
    unittest.main()
