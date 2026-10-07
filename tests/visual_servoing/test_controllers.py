import contextlib
import io
import unittest

import numpy as np

from view_transition_rl.envs.metrics import longitudinal_metrics
from view_transition_rl.simulation import PhantomEnv
from view_transition_rl.visual_servoing.controllers import (Controller, SequentialController, action_for_motion,
                                                            contact_motion, rotation_matrix)
from view_transition_rl.visual_servoing.run import move_probe, run, to_frame


class Observer(Controller):
    def act(self, mask, measures):
        return None


class TestProbeFrameBookkeeping(unittest.TestCase):
    def test_cloud_and_skin_normal_follow_the_probe(self):
        rng = np.random.default_rng(3)
        env = PhantomEnv(phantom="curved_arc")
        controller = Observer()
        controller.reset()
        controller(env.get_mask())
        seen_from = np.array(env.get_state()["phantom_T_probe_mm"])
        seen = to_frame(controller.cloud[0], seen_from)
        for _ in range(30):  # random relative moves, including tilts
            action = np.r_[rng.uniform(-2, 2, 2), np.deg2rad(rng.uniform(-6, 6, 3))]
            if move_probe(env, action).accepted:
                controller.moved(action)
        pose = np.array(env.get_state()["phantom_T_probe_mm"])
        self.assertGreater(np.rad2deg(np.arccos(pose[2, 2])), 5)  # the probe is tilted now
        # Placed with the true pose, the probe-frame cloud is still where it was seen.
        np.testing.assert_allclose(to_frame(controller.cloud[0], pose), seen, atol=1e-9)
        np.testing.assert_allclose(controller.surface_normal, pose[2, :3], atol=1e-12)

    def test_action_for_motion_inverts_contact_motion(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            up = rng.normal(size=3) + [0, 0, 4]
            up /= np.linalg.norm(up)
            translation = np.cross(up, rng.normal(size=3))  # in the skin plane
            rotation = rotation_matrix(rng.uniform(-0.4, 0.4, 3))
            executed = contact_motion(action_for_motion(translation, rotation, up), up)
            np.testing.assert_allclose(executed[:3, :3], rotation, atol=1e-12)
            np.testing.assert_allclose(executed[:3, 3], translation, atol=1e-12)


class TestSequentialController(unittest.TestCase):
    def test_tracking_follows_known_user_moves(self):
        env = PhantomEnv()
        env.step([8, 0, 0, 0, 0])
        controller = SequentialController(rotate_deg=90, align_steps=5)
        controller.reset()
        track_steps = 0
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(200):
                action = controller(env.get_mask())
                if controller.phase_name == "track":
                    np.testing.assert_allclose(action[:2], 0, atol=1e-9)  # x and y belong to the user
                    if track_steps == 2:  # user: 3 mm sideways (W), a known motion
                        self.assertTrue(move_probe(env, [0, 3, 0, 0, 0]).accepted)
                        controller.moved([0, 3, 0, 0, 0])
                    track_steps += 1
                self.assertTrue(move_probe(env, action).accepted)
                controller.moved(action)
                if track_steps == 15:
                    break
        self.assertGreater(abs(np.rad2deg(controller.plane.roll)), 3)  # tilted to reach the vessel
        centreline = controller.cylinder.centreline(100)  # probe frame: the image plane is y = 0
        self.assertLess(np.abs(centreline[:, 1]).mean(), 1.0)
        self.assertGreater(longitudinal_metrics(env.get_mask())["column_run"], 0.95)

    def test_reaches_longitudinal_view_on_oblique_vessel(self):
        # vessel_no_hole runs at about 59 degrees yaw, so the fixed 90 degree rotation
        # leaves a yaw error that the alignment has to correct.
        env = PhantomEnv(phantom="vessel_no_hole")
        env.step([8, 0, 0, 0, 0])
        controller = SequentialController(rotate_deg=90, align_steps=5, track=False)
        with contextlib.redirect_stdout(io.StringIO()):
            trajectory = run(controller, env, max_steps=200)
        self.assertEqual(controller.phase_name, "done")
        self.assertFalse(any(row.get("rejected") for row in trajectory))
        self.assertGreater(longitudinal_metrics(env.get_mask())["column_run"], 0.95)
        self.assertLess(np.abs(controller.cylinder.centreline(100)[:, 1]).mean(), 1.0)


if __name__ == "__main__":
    unittest.main()
