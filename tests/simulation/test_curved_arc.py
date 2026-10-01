import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from view_transition_rl.simulation import PhantomEnv
from view_transition_rl.simulation.__main__ import main
from view_transition_rl.simulation.curved_arc import (
    TARGET_RANGE_RAD, VESSEL_RADIUS_MM, centerline, roll_yaw_pose, section_frame,
    section_intersections,
)
from view_transition_rl.simulation.evaluate import evaluate
from view_transition_rl.simulation.geometry import pose_from_euler, read_stl


class TestCurvedArc(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = PhantomEnv(phantom='curved_arc')

    def setUp(self):
        self.env.reset()

    def test_closed_asset_and_centered_circle(self):
        env = self.env
        mesh_path = env.assets / env.definition['mesh']
        self.assertEqual(hashlib.sha256(mesh_path.read_bytes()).hexdigest(), env.calibration['sha256'])
        mesh = read_stl(mesh_path)
        self.assertEqual(mesh.GetNumberOfCells(), env.calibration['design']['triangle_count'])
        self.assertEqual(env.config.phantom_size_mm, (100, 100, 40))
        bounds = np.array(mesh.GetBounds()).reshape(3, 2)
        self.assertTrue(np.all(bounds[:, 0] > [-50, -50, 0]))
        self.assertTrue(np.all(bounds[:, 1] < [50, 50, 40]))
        mask = env.get_mask()
        self.assertEqual(mask.shape, (480, 480))
        self.assertEqual(set(np.unique(mask)), {0, 255})
        row, col = np.nonzero(mask)
        np.testing.assert_allclose([((col + .5 - 240) / 12).mean(), ((row + .5) / 12).mean()],
                                   [0, 20], atol=1 / 12)
        self.assertAlmostEqual(len(row) / 144, np.pi * VESSEL_RADIUS_MM**2, delta=.2)
        np.testing.assert_allclose([(col.max() - col.min() + 1) / 12,
                                   (row.max() - row.min() + 1) / 12],
                                   [2 * VESSEL_RADIUS_MM] * 2, atol=1 / 12)

    def test_yaw_failure_is_geometric_not_boundary(self):
        endpoints = centerline(np.asarray(TARGET_RANGE_RAD))
        bound = min(abs(endpoints[1, 0]), abs(endpoints[1, 1]))
        self.assertAlmostEqual(bound, 3.794492405743083)
        self.assertGreater(bound, VESSEL_RADIUS_MM)
        t = np.linspace(*TARGET_RANGE_RAD, 301)
        for yaw in np.arange(-180, 180, .25):
            pose = pose_from_euler(0, 0, yaw=np.deg2rad(yaw))
            self.assertTrue(self.env.set_pose(pose).accepted)
            self.assertFalse(section_intersections(pose, t).all())
            self.assertGreater(np.abs(endpoints @ pose[:3, 1]).max(), VESSEL_RADIUS_MM)
        # The tessellated mesh is contained in the convex hull of its nearby
        # section disks. Check its actual endpoint rings against the same bound.
        from vtk.util.numpy_support import vtk_to_numpy
        points = vtk_to_numpy(self.env.mesh.GetPoints().GetData())
        for yaw in [0, 30, 60, 90, 120, 150]:
            normal = pose_from_euler(0, 0, yaw=np.deg2rad(yaw))[:3, 1]
            distances = endpoints @ normal
            end = endpoints[np.abs(distances).argmax()]
            near = points[np.linalg.norm(points - end, axis=1) < VESSEL_RADIUS_MM + .01]
            self.assertGreater(len(near), 0)
            self.assertTrue(np.all(near @ normal > 0) or np.all(near @ normal < 0))

    def test_roll_and_yaw_reveal_target_without_translation(self):
        env = self.env
        pose = np.asarray(env.calibration['example_poses']['longitudinal'])
        self.assertTrue(env.set_pose(pose).accepted)
        self.assertEqual(pose[0, 3], 0)
        self.assertEqual(pose[1, 3], 0)
        self.assertAlmostEqual(np.rad2deg(roll_yaw_pose()), 4.956, places=2)
        self.assertAlmostEqual(env.get_state()['tilt_deg'], np.rad2deg(roll_yaw_pose()))
        t = np.linspace(*TARGET_RANGE_RAD, 301)
        target = centerline(t)
        self.assertTrue(section_intersections(pose, t).all())
        local = (target - pose[:3, 3]) @ pose[:3, :3]
        self.assertGreater(np.abs(local[:, 1]).max(), .5)
        self.assertLessEqual(np.abs(local[:, 1]).max(), VESSEL_RADIUS_MM)
        self.assertTrue(np.all(np.abs(local[:, 0]) + VESSEL_RADIUS_MM < 20))
        self.assertTrue(np.all(local[:, 2] + VESSEL_RADIUS_MM < 40))
        self.assertTrue(np.all(local[:, 2] - VESSEL_RADIUS_MM > 0))
        mask = env.get_mask()
        # Project the nearest point in each lumen disk onto the slice plane,
        # then confirm the rendered stencil includes all target samples.
        radial, binormal, _ = section_frame(t)
        normal = pose[:3, 1]
        ar, ab = radial @ normal, binormal @ normal
        distance = (target - pose[:3, 3]) @ normal
        shift = -distance[:, None] * (ar[:, None] * radial + ab[:, None] * binormal) / (
            ar[:, None]**2 + ab[:, None]**2)
        intersection = target + shift
        pixels = (intersection - pose[:3, 3]) @ pose[:3, :3]
        col = np.floor((pixels[:, 0] + 20) * 12).astype(int)
        row = np.floor(pixels[:, 2] * 12).astype(int)
        self.assertTrue(np.all(mask[row, col] == 255))
        with tempfile.TemporaryDirectory() as directory:
            _, state_path = env.export(Path(directory) / 'longitudinal')
            state = json.loads(state_path.read_text())
            self.assertEqual(state['phantom_id'], 'curved_arc')
            with contextlib.redirect_stdout(io.StringIO()):
                main(['--phantom', 'curved_arc', 'sample', '--pose-json', str(state_path),
                      '--output', str(Path(directory) / 'replay')])
            with self.assertRaisesRegex(ValueError, 'no reference'):
                evaluate(Path(directory) / 'evaluation', env)
            self.assertFalse((Path(directory) / 'evaluation').exists())
        env.reset()
        np.testing.assert_array_equal(env.get_state()['phantom_T_probe_mm'], np.eye(4))


if __name__ == '__main__':
    unittest.main()
