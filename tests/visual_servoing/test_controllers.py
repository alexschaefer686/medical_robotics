import unittest

import numpy as np

from view_transition_rl.envs.metrics import longitudinal_metrics
from view_transition_rl.simulation import PhantomEnv
from view_transition_rl.simulation.geometry import pose_from_euler
from view_transition_rl.visual_servoing.controllers import SequentialController, action_to_pose
from view_transition_rl.visual_servoing.run import run


class TestActionToPose(unittest.TestCase):
    def test_step_reaches_target(self):
        rng = np.random.default_rng(1)
        env = PhantomEnv()
        for _ in range(50):
            start, target = (pose_from_euler(*rng.uniform(-5, 5, 2),
                                             *np.deg2rad(rng.uniform([-30, -30, -180], [30, 30, 180])))
                             for _ in range(2))
            self.assertTrue(env.set_pose(start).accepted)
            self.assertTrue(env.step(action_to_pose(start, target)).accepted)
            np.testing.assert_allclose(env.get_state()["phantom_T_probe_mm"], target, atol=1e-9)


class TestSequentialController(unittest.TestCase):
    def test_tracking_follows_user_moves_without_changing_x_y(self):
        env = PhantomEnv()
        env.step([8, 0, 0, 0, 0])
        controller = SequentialController()
        controller.reset()
        track_steps = 0
        for _ in range(200):
            action = controller(env.get_mask(), env.get_state()["phantom_T_probe_mm"])
            if controller.phase_name == "track":
                np.testing.assert_allclose(action[:2], 0, atol=1e-9)  # x and y belong to the user
                if track_steps == 2:
                    self.assertTrue(env.step([0, 3, 0, 0, 0]).accepted)  # user: 3 mm sideways (W)
                track_steps += 1
            self.assertTrue(env.step(action).accepted)
            if track_steps == 15:
                break
        pose = np.array(env.get_state()["phantom_T_probe_mm"])
        np.testing.assert_allclose(controller.plane.point[:2], pose[:2, 3], atol=1e-9)
        self.assertGreater(abs(np.rad2deg(controller.plane.pose_5dof[2])), 3)  # tilted to reach the vessel
        distance = np.abs((controller.cylinder.centreline(100) - pose[:3, 3]) @ pose[:3, 1])
        self.assertLess(distance.mean(), 1.0)
        self.assertGreater(longitudinal_metrics(env.get_mask())["column_run"], 0.95)

    def test_reaches_longitudinal_view_on_oblique_vessel(self):
        # vessel_no_hole runs at about 59 degrees yaw, so the fixed 90 degree rotation
        # leaves a yaw error that the alignment has to correct.
        env = PhantomEnv(phantom="vessel_no_hole")
        env.step([8, 0, 0, 0, 0])
        controller = SequentialController(track=False)
        trajectory = run(controller, env, max_steps=200)
        self.assertEqual(controller.phase_name, "done")
        self.assertFalse(any(row.get("rejected") for row in trajectory))
        self.assertGreater(longitudinal_metrics(env.get_mask())["column_run"], 0.95)
        pose = np.array(env.get_state()["phantom_T_probe_mm"])
        distance = np.abs((controller.cylinder.centreline(100) - pose[:3, 3]) @ pose[:3, 1])
        self.assertLess(distance.mean(), 1.0)


if __name__ == "__main__":
    unittest.main()
