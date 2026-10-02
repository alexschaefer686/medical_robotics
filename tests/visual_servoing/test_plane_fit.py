import unittest

import numpy as np

from view_transition_rl.visual_servoing.plane_fit import fit_image_plane, vicinity_weights


def probe_rotation(roll, yaw):
    """Rz(yaw) @ Rx(roll): probe orientation with pitch = 0."""
    cr, sr, cy, sy = np.cos(roll), np.sin(roll), np.cos(yaw), np.sin(yaw)
    return np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])


class TestImagePlaneFit(unittest.TestCase):
    def test_straight_line_gives_vertical_plane_along_it(self):
        s = np.linspace(-15, 15, 50)[:, None]
        line = np.array([3.0, -4.0, 15.0]) + s * [np.cos(np.pi / 6), np.sin(np.pi / 6), 0]
        plane = fit_image_plane(line, yaw_hint=0.0)
        np.testing.assert_allclose(plane.pose_5dof, [3, -4, 0, 0, np.pi / 6], atol=1e-9)
        self.assertLess(plane.rms, 1e-9)

    def test_tilted_planar_curve(self):
        roll, yaw = np.deg2rad(20), np.deg2rad(60)
        rotation, contact = probe_rotation(roll, yaw), np.array([2.0, 1.0, 0.0])
        a = np.linspace(-15, 15, 60)
        curve = np.column_stack([a, np.zeros_like(a), 12 + 0.02 * a ** 2]) @ rotation.T + contact
        exact = fit_image_plane(curve, roll_weight=1e-6, yaw_hint=yaw)
        np.testing.assert_allclose(exact.pose_5dof, [2, 1, roll, 0, yaw], atol=1e-4)
        # The returned pose's image plane (probe y = 0) contains the curve.
        np.testing.assert_allclose(probe_rotation(exact.pose_5dof[2], exact.pose_5dof[4])[:, 1], exact.normal)
        self.assertLess(np.abs((curve - exact.point) @ exact.normal).max(), 1e-4)
        # A large roll weight trades distance for a nearly vertical plane.
        vertical = fit_image_plane(curve, roll_weight=100.0, yaw_hint=yaw)
        self.assertLess(abs(np.rad2deg(vertical.pose_5dof[2])), 1.0)
        self.assertGreater(vertical.rms, exact.rms)

    def test_yaw_hint_selects_equivalent_pose(self):
        roll, yaw = np.deg2rad(20), np.deg2rad(60)
        a = np.linspace(-15, 15, 60)
        curve = np.column_stack([a, np.zeros_like(a), 12 + 0.02 * a ** 2]) @ probe_rotation(roll, yaw).T
        flipped = fit_image_plane(curve, roll_weight=1e-6, yaw_hint=yaw + np.pi)
        np.testing.assert_allclose(flipped.pose_5dof[2:], [-roll, 0, yaw - np.pi], atol=1e-4)

    def test_plane_through_given_point(self):
        # Vessel along y at 15 mm depth; a probe 4 mm beside it must tilt by atan(4 / 15).
        line = np.array([0.0, 0.0, 15.0]) + np.linspace(-15, 15, 50)[:, None] * [0, 1, 0]
        plane = fit_image_plane(line, roll_weight=1e-6, yaw_hint=np.pi / 2, through=(4.0, 5.0, 0.0))
        np.testing.assert_allclose(plane.point, [4, 5, 0])
        np.testing.assert_allclose(plane.pose_5dof, [4, 5, -np.arctan2(4, 15), 0, np.pi / 2], atol=1e-6)
        self.assertLess(plane.rms, 1e-6)

    def test_vicinity_weights(self):
        points = np.array([[0, 0, 15], [5, 0, 15], [0, 10, 3], [30, 0, 15]], float)
        np.testing.assert_allclose(vicinity_weights(points, (0, 0), 10, "window"), [1, 1, 1, 0])
        np.testing.assert_allclose(vicinity_weights(points, (0, 0), 10, "linear"), [1, 0.5, 0, 0])
        np.testing.assert_allclose(vicinity_weights(points, (0, 0), 10, "quadratic"), [1, 0.75, 0, 0])
        np.testing.assert_allclose(vicinity_weights(points, (0, 0), 10, None), [1, 1, 1, 1])
        # Along a direction only: the point 10 mm along y has distance 0 along x.
        np.testing.assert_allclose(vicinity_weights(points, (0, 0), 10, "linear", direction=(1, 0, 0)),
                                   [1, 0.5, 1, 0])

    def test_weights_ignore_distant_bend(self):
        # Straight along y near the probe, then bending sideways far away (y > 25).
        y = np.linspace(-20, 60, 400)
        curve = np.column_stack([np.where(y > 25, 0.05 * (y - 25) ** 2, 0.0), y, np.full_like(y, 15.0)])
        weights = vicinity_weights(curve, (0, 0), 20, "linear")
        local = fit_image_plane(curve, roll_weight=1.0, yaw_hint=np.pi / 2, through=(0, 0, 0), weights=weights)
        whole = fit_image_plane(curve, roll_weight=1.0, yaw_hint=np.pi / 2, through=(0, 0, 0))
        np.testing.assert_allclose(local.pose_5dof[2:], [0, 0, np.pi / 2], atol=1e-9)  # the straight part's plane
        self.assertGreater(abs(whole.pose_5dof[4] - np.pi / 2), np.deg2rad(5))  # pulled by the far bend
        self.assertIsNone(fit_image_plane(curve, through=(0, 0, 0), weights=np.zeros(len(curve))))

    def test_too_few_points(self):
        self.assertIsNone(fit_image_plane(np.zeros((1, 3))))


if __name__ == "__main__":
    unittest.main()
