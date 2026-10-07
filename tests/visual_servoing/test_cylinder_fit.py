import unittest

import numpy as np

from view_transition_rl.visual_servoing.cylinder_fit import fit_bent_cylinder


def bent_tube(radius, n_rings=40, per_ring=60, noise=0.0, seed=0):
    """Points on rings around the cubic centreline (s, 0.02 s^2, 0.0008 s^3), rotated and shifted."""
    rng = np.random.default_rng(seed)
    s = np.linspace(-20, 20, n_rings)
    centre = np.column_stack([s, 0.02 * s ** 2, 0.0008 * s ** 3])
    tangent = np.column_stack([np.ones_like(s), 0.04 * s, 0.0024 * s ** 2])
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    n1 = np.cross(tangent, [0, 0, 1])
    n1 /= np.linalg.norm(n1, axis=1)[:, None]
    n2 = np.cross(tangent, n1)
    phi = np.linspace(0, 2 * np.pi, per_ring, endpoint=False)[None, :, None]
    points = (centre[:, None] + radius * (np.cos(phi) * n1[:, None] + np.sin(phi) * n2[:, None])).reshape(-1, 3)
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    shift = np.array([5.0, -3.0, 12.0])
    points = points @ rotation.T + shift + rng.normal(scale=noise, size=points.shape)
    return points, centre @ rotation.T + shift


class TestBentCylinderFit(unittest.TestCase):
    def test_recovers_radius_and_centreline(self):
        points, centre = bent_tube(2.5, noise=0.05)
        cylinder = fit_bent_cylinder(points, degree=5)
        self.assertAlmostEqual(cylinder.radius, 2.5, delta=0.01)
        self.assertLess(cylinder.rms, 0.06)
        fitted = cylinder.centreline(5000)  # dense, so sample spacing (~0.01 mm) does not dominate the error
        error = np.min(np.linalg.norm(centre[:, None] - fitted[None], axis=2), axis=1)
        self.assertLess(error[2:-2].max(), 0.05)  # ends are extrapolated by the fit's t_range
        self.assertAlmostEqual(np.abs(cylinder.surface_distances(points)).mean(), 0.04, delta=0.02)

    def test_used_index_lists_the_fitted_points(self):
        points, _ = bent_tube(2.5, n_rings=200, per_ring=60)  # 12,000 points
        cylinder = fit_bent_cylinder(points, max_points=5000)
        np.testing.assert_array_equal(cylinder.used_index, np.arange(0, 12000, 3))  # every 3rd point
        refit = fit_bent_cylinder(points[cylinder.used_index], max_points=5000)
        self.assertEqual(len(refit.used_index), len(cylinder.used_index))  # no further subsampling
        self.assertAlmostEqual(refit.radius, cylinder.radius, places=9)

    def test_needs_elongated_points(self):
        ring, _ = bent_tube(2.5, n_rings=1, per_ring=200)
        self.assertIsNone(fit_bent_cylinder(ring))
        self.assertIsNone(fit_bent_cylinder(np.zeros((5, 3))))


if __name__ == "__main__":
    unittest.main()
