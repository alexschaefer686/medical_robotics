import unittest

import numpy as np
import vtk

from view_transition_rl.simulation.geometry import MeshSlicer, inverse_pose, pose_from_euler, rotation_xyz, transform_mesh


def sphere(centre=(0, 0, 15), radius=5):
    source = vtk.vtkSphereSource()
    source.SetCenter(*centre)
    source.SetRadius(radius)
    source.SetThetaResolution(192)
    source.SetPhiResolution(192)
    source.Update()
    return source.GetOutput()


class TestSlicer(unittest.TestCase):
    def assert_circle_boundary(self, mask, slicer, cx, cz, radius):
        points = slicer.pixel_points
        distance = np.sqrt((points[..., 0] - cx) ** 2 + (points[..., 2] - cz) ** 2)
        tolerance = max(slicer.sx, slicer.sz)
        self.assertTrue(np.all(mask[distance < radius - tolerance] == 255))
        self.assertTrue(np.all(mask[distance > radius + tolerance] == 0))
        self.assertGreater(np.count_nonzero(mask), 0)

    def test_sphere_pixel_centres_and_direction(self):
        slicer = MeshSlicer(sphere((4, 0, 12), 3))
        mask = slicer.render(np.eye(4))
        self.assertEqual(mask.shape, (480, 480))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(set(np.unique(mask)), {0, 255})
        self.assert_circle_boundary(mask, slicer, 4, 12, 3)
        self.assertAlmostEqual(slicer.pixel_points[0, 0, 0], -20 + 40 / 960)
        self.assertAlmostEqual(slicer.pixel_points[0, 0, 2], 40 / 960)
        rows, columns = np.nonzero(mask)
        self.assertGreater(columns.mean(), 240)
        self.assertLess(rows.mean(), 240)

    def test_cylinder_circle(self):
        cylinder = vtk.vtkCylinderSource()
        cylinder.SetCenter(0, 0, 15)
        cylinder.SetRadius(4)
        cylinder.SetHeight(20)
        cylinder.SetResolution(360)
        cylinder.CappingOn()
        cylinder.Update()
        slicer = MeshSlicer(cylinder.GetOutput())
        self.assert_circle_boundary(slicer.render(np.eye(4)), slicer, 0, 15, 4)

    def test_oblique_sphere_intersection(self):
        centre = np.array([0, 0, 15.0])
        pose = pose_from_euler(0, 0, np.deg2rad(12), np.deg2rad(7), np.deg2rad(20))
        local_centre = inverse_pose(pose)[:3, :3] @ centre
        radius = np.sqrt(25 - local_centre[1] ** 2)
        slicer = MeshSlicer(sphere())
        self.assert_circle_boundary(slicer.render(pose), slicer, local_centre[0], local_centre[2], radius)

    def test_rigid_transform_invariance(self):
        slicer = MeshSlicer(sphere())
        pose = pose_from_euler(0, 0, .1, .1, .2)
        rigid = pose_from_euler(2, 3, .2, -.3, .4)
        rigid[2, 3] = 5
        transformed = MeshSlicer(transform_mesh(slicer.mesh, rigid))
        np.testing.assert_array_equal(slicer.render(pose), transformed.render(rigid @ pose))

    def test_hole_preservation(self):
        outer = sphere(radius=6)
        reverse = vtk.vtkReverseSense()
        reverse.SetInputData(sphere(radius=3))
        reverse.ReverseCellsOn()
        reverse.ReverseNormalsOn()
        reverse.Update()
        append = vtk.vtkAppendPolyData()
        append.AddInputData(outer)
        append.AddInputData(reverse.GetOutput())
        append.Update()
        slicer = MeshSlicer(append.GetOutput())
        mask = slicer.render(np.eye(4))
        points = slicer.pixel_points
        r = np.sqrt(points[..., 0] ** 2 + (points[..., 2] - 15) ** 2)
        self.assertTrue(np.all(mask[r < 2.9] == 0))
        self.assertTrue(np.all(mask[(r > 3.1) & (r < 5.9)] == 255))
        self.assertTrue(np.all(mask[r > 6.1] == 0))

    def test_disconnected_regions(self):
        append = vtk.vtkAppendPolyData()
        append.AddInputData(sphere((-7, 0, 10), 2))
        append.AddInputData(sphere((7, 0, 20), 2))
        append.Update()
        slicer = MeshSlicer(append.GetOutput())
        mask = slicer.render(np.eye(4))
        self.assertGreater(mask[:, :200].sum(), 0)
        self.assertGreater(mask[:, 280:].sum(), 0)
        self.assertEqual(mask[:, 200:280].sum(), 0)

    def test_empty_and_tangent_planes(self):
        slicer = MeshSlicer(sphere())
        pose = np.eye(4)
        pose[1, 3] = 20
        self.assertEqual(slicer.render(pose).sum(), 0)
        pose[1, 3] = 5
        self.assertEqual(slicer.render(pose).sum(), 0)

    def test_phantom_clipping(self):
        slicer = MeshSlicer(sphere())
        mask = slicer.render(np.eye(4), (40, 40, 15))
        self.assertGreater(mask[:180].sum(), 0)
        self.assertEqual(mask[180:].sum(), 0)

    def test_inverse(self):
        pose = pose_from_euler(3, 4, .2, -.3, 2)
        pose[2, 3] = 10
        np.testing.assert_allclose(inverse_pose(pose) @ pose, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(rotation_xyz(.1, .2, .3).T @ rotation_xyz(.1, .2, .3), np.eye(3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
