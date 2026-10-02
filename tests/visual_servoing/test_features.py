import unittest

import numpy as np

from view_transition_rl.visual_servoing.features import (area, boundary_nodes, boundary_points_3d, center_of_area,
                                                         elipsoid_fit)


def ellipse_mask(radius_a, radius_b, angle_deg, centre=(240, 240)):
    """Filled ellipse; radius_a lies along a direction angle_deg from image x."""
    rows, cols = np.mgrid[:480, :480]
    x, y = cols - centre[0], rows - centre[1]
    angle = np.deg2rad(angle_deg)
    u = x * np.cos(angle) + y * np.sin(angle)
    v = -x * np.sin(angle) + y * np.cos(angle)
    return np.where((u / radius_a) ** 2 + (v / radius_b) ** 2 <= 1, 255, 0).astype(np.uint8)


class TestFeatures(unittest.TestCase):
    def test_empty_mask(self):
        mask = np.zeros((480, 480), np.uint8)
        self.assertEqual(area(mask), 0)
        self.assertIsNone(center_of_area(mask))
        self.assertFalse(boundary_nodes(mask).any())

    def test_rectangle(self):
        mask = np.zeros((480, 480), np.uint8)
        mask[100:110, 200:240] = 255  # rows 100-109, cols 200-239
        self.assertEqual(area(mask), 400)
        self.assertEqual(center_of_area(mask), (219.5, 104.5))
        nodes = boundary_nodes(mask)
        self.assertEqual(nodes.shape, (479, 479))
        # Ring of nodes around the 10x40 block: 11x41 outer minus 9x39 interior.
        self.assertEqual(nodes.sum(), 11 * 41 - 9 * 39)
        rows, cols = np.nonzero(nodes)
        self.assertEqual((rows.min(), rows.max(), cols.min(), cols.max()), (99, 109, 199, 239))

    def test_elipsoid_fit_radii(self):
        for radius_a, radius_b, angle in [(60, 25, 0), (60, 25, 20), (30, 30, 0), (100, 8, 5)]:
            with self.subTest(radius_a=radius_a, radius_b=radius_b, angle=angle):
                np.testing.assert_allclose(elipsoid_fit(ellipse_mask(radius_a, radius_b, angle)),
                                           (radius_a, radius_b), atol=1.5)

    def test_elipsoid_fit_returns_axis_closer_to_x_first(self):
        # The 60 px axis points 80 degrees from x, so the 25 px axis is closer to x.
        np.testing.assert_allclose(elipsoid_fit(ellipse_mask(60, 25, 80)), (25, 60), atol=1)

    def test_elipsoid_fit_without_ellipse(self):
        self.assertIsNone(elipsoid_fit(np.zeros((480, 480), np.uint8)))
        band = np.zeros((480, 480), np.uint8)
        band[200:260, :] = 255  # two straight walls, no ellipse
        self.assertIsNone(elipsoid_fit(band))

    def test_boundary_points_3d(self):
        mask = np.zeros((480, 480), np.uint8)
        mask[100:110, 200:240] = 255  # boundary nodes: rows 99-109, cols 199-239
        nodes = boundary_nodes(mask)
        points = boundary_points_3d(nodes, np.eye(4))
        self.assertEqual(points.shape, (nodes.sum(), 3))
        mm = 40.0 / 480
        # Node col j -> x = (j + 1 - 240) px, node row i -> z = (i + 1) px; image plane y = 0.
        np.testing.assert_allclose(points.min(axis=0), [-40 * mm, 0, 100 * mm])
        np.testing.assert_allclose(points.max(axis=0), [0, 0, 110 * mm], atol=1e-12)
        # A probe yawed 90 degrees and moved to (5, 7) maps image x onto phantom y.
        pose = np.array([[0, -1, 0, 5], [1, 0, 0, 7], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        moved = boundary_points_3d(nodes, pose)
        np.testing.assert_allclose(moved, points[:, [1, 0, 2]] * [-1, 1, 1] + [5, 7, 0], atol=1e-12)

    def test_image_border_is_not_a_boundary(self):
        mask = np.zeros((480, 480), np.uint8)
        mask[100:110, :] = 255  # band crossing the full image width
        rows, _ = np.nonzero(boundary_nodes(mask))
        self.assertEqual(set(rows), {99, 109})
        self.assertFalse(boundary_nodes(np.full((480, 480), 255, np.uint8)).any())


if __name__ == "__main__":
    unittest.main()
