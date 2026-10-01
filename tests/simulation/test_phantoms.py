import contextlib
import io
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from view_transition_rl.simulation import PhantomEnv
from view_transition_rl.simulation.__main__ import main
from view_transition_rl.simulation.environment import load_config
from view_transition_rl.simulation.evaluate import evaluate


class TestPhantoms(unittest.TestCase):
    def test_placement_mask_motion_and_reset(self):
        env = PhantomEnv(phantom="vessel_no_hole")
        self.assertEqual(hashlib.sha256((env.assets / env.definition["mesh"]).read_bytes()).hexdigest(),
                         env.calibration["sha256"])
        points = vtk_to_numpy(env.mesh.GetPoints().GetData())
        self.assertAlmostEqual(float(points[:, 2].min()), 8, places=5)
        self.assertLess(float(points[:, 2].max()), 40)
        np.testing.assert_allclose([points.min(0), points.max(0)],
                                   env.calibration['vessel_bounds_phantom_mm'], atol=1e-5)
        # Source long axis must slope down towards phantom +Y, without scaling.
        long_axis = env.phantom_T_world[:3, 0]
        self.assertAlmostEqual(np.rad2deg(np.arctan2(long_axis[2], long_axis[1])), 8)
        shallow = points[points[:, 1] < -60, 2].mean()
        deep = points[points[:, 1] > 60, 2].mean()
        self.assertGreater(deep - shallow, 15)
        self.assertAlmostEqual(np.linalg.det(env.world_T_phantom[:3, :3]), 1)
        np.testing.assert_allclose(env.world_T_phantom @ env.phantom_T_world, np.eye(4), atol=1e-12)
        mask = env.get_mask()
        self.assertEqual(mask.shape, (480, 480))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(set(np.unique(mask)), {0, 255})
        self.assertTrue(env.step([0, 50, 0, 0, 0]).accepted)
        pose = env.get_state()['phantom_T_probe_mm']
        self.assertFalse(env.step([0, 100, 0, 0, 0]).accepted)
        self.assertEqual(env.get_state()['phantom_T_probe_mm'], pose)
        env.reset()
        np.testing.assert_array_equal(env.get_state()['phantom_T_probe_mm'], np.eye(4))
        np.testing.assert_array_equal(env.get_mask(), mask)

    def test_reference_operations_fail_without_outputs(self):
        env = PhantomEnv(phantom="vessel_no_hole")
        for operation in [lambda: env.reference_pose(0), lambda: env.replay_frame(0)]:
            with self.assertRaisesRegex(ValueError, "no reference"):
                operation()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'evaluation'
            with self.assertRaisesRegex(ValueError, 'no reference'):
                evaluate(output, env)
            self.assertFalse(output.exists())

    def test_config_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'image_shape': [120, 160]}))
            config = load_config(path, phantom='vessel_no_hole')
            self.assertEqual(config.phantom_size_mm, (70, 190, 40))
            env = PhantomEnv(config, phantom='vessel_no_hole')
            self.assertEqual(env.get_mask().shape, (120, 160))
            _, state = env.export(Path(directory) / 'sample')
            self.assertEqual(json.loads(state.read_text())['phantom_id'], 'vessel_no_hole')
        with self.assertRaises(ValueError):
            PhantomEnv(phantom='missing')
        with self.assertRaisesRegex(ValueError, 'contain'):
            PhantomEnv(phantom='vessel_no_hole', config_overrides={'phantom_size_mm': [70, 100, 40]})

    def test_cli_selection_and_pose_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'sample'
            with contextlib.redirect_stdout(io.StringIO()):
                main(['--phantom', 'vessel_no_hole', 'sample', '--output', str(prefix)])
            state_path = prefix.with_suffix('.json')
            self.assertEqual(json.loads(state_path.read_text())['phantom_id'], 'vessel_no_hole')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(['sample', '--pose-json', str(state_path), '--output', str(prefix)])
            self.assertEqual(error.exception.code, 2)
            state = json.loads(state_path.read_text())
            del state['phantom_id']
            state_path.write_text(json.dumps(state))
            with contextlib.redirect_stdout(io.StringIO()):
                main(['--phantom', 'vessel_no_hole', 'sample', '--pose-json', str(state_path), '--output', str(prefix)])
