"""Millimetre geometry; no rendering context is created here."""

from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def rotation_xyz(rx=0.0, ry=0.0, rz=0.0):
    """Intrinsic incremental rotations: right multiply by Rx @ Ry @ Rz."""
    cx, cy, cz = np.cos([rx, ry, rz])
    sx, sy, sz = np.sin([rx, ry, rz])
    return (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))


def pose_from_euler(x, y, roll=0.0, pitch=0.0, yaw=0.0):
    """Absolute pose on the surface; R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    pose = np.eye(4)
    pose[:3, :3] = rotation_xyz(rz=yaw) @ rotation_xyz(ry=pitch) @ rotation_xyz(rx=roll)
    pose[:3, 3] = [x, y, 0]
    return pose


def euler_from_pose(pose):
    """Roll/pitch/yaw for Rz @ Ry @ Rx, away from pitch = +/-90 degrees."""
    r = pose[:3, :3]
    return np.array([np.arctan2(r[2, 1], r[2, 2]),
                     np.arcsin(np.clip(-r[2, 0], -1, 1)),
                     np.arctan2(r[1, 0], r[0, 0])])


def validate_pose(pose):
    a = np.asarray(pose, dtype=float)
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise ValueError("Pose must be a finite 4x4 matrix (translation in mm).")
    if not np.allclose(a[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError("Pose must have homogeneous last row [0,0,0,1].")
    r = a[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-7, rtol=0) or not np.isclose(np.linalg.det(r), 1, atol=1e-7):
        raise ValueError("Pose rotation must be right-handed and orthonormal.")
    return a.copy()


def inverse_pose(pose):
    result = np.eye(4)
    result[:3, :3] = pose[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ pose[:3, 3]
    return result


def transform_mesh(mesh, matrix):
    transform = vtk.vtkTransform()
    transform.SetMatrix(np.asarray(matrix, dtype=float).ravel())
    filt = vtk.vtkTransformPolyDataFilter()
    filt.SetInputData(mesh)
    filt.SetTransform(transform)
    filt.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(filt.GetOutput())
    return result


def read_stl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.Update()
    mesh = reader.GetOutput()
    if mesh.GetNumberOfCells() == 0:
        raise ValueError(f"Empty STL: {path}")
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(mesh)
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    if edges.GetOutput().GetNumberOfCells():
        raise ValueError("Vessel STL must be a closed, manifold surface.")
    result = vtk.vtkPolyData()
    result.DeepCopy(mesh)
    return result


class MeshSlicer:
    """Exact surface intersections rasterized at pixel centres (no voxel cache).

    VTK cuts the closed surface at image Z=0 and fills its contours using
    vtkPolyDataToImageStencil, preserving disconnected regions and holes.
    Image coordinates (X,Y,Z) correspond to probe (x,z,-y).
    """

    def __init__(self, mesh, shape=(480, 480), fov_mm=(40.0, 40.0)):
        self.mesh = mesh
        self.height, self.width = shape
        self.fov_width, self.fov_depth = fov_mm
        self.sx = self.fov_width / self.width
        self.sz = self.fov_depth / self.height
        x = (np.arange(self.width) + 0.5 - self.width / 2) * self.sx
        z = (np.arange(self.height) + 0.5) * self.sz
        xx, zz = np.meshgrid(x, z)
        self.pixel_points = np.stack([xx, np.zeros_like(xx), zz], axis=-1)
        self.probe_to_slice = np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                                        [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)

    def render(self, phantom_T_probe, phantom_size_mm=None):
        pose = validate_pose(phantom_T_probe)
        mesh = transform_mesh(self.mesh, self.probe_to_slice @ inverse_pose(pose))
        stencil = vtk.vtkPolyDataToImageStencil()
        stencil.SetInputData(mesh)
        stencil.SetOutputOrigin(-self.fov_width / 2 + self.sx / 2, self.sz / 2, 0)
        stencil.SetOutputSpacing(self.sx, self.sz, 1)
        stencil.SetOutputWholeExtent(0, self.width - 1, 0, self.height - 1, 0, 0)
        stencil.SetTolerance(1e-7)
        image = vtk.vtkImageStencilToImage()
        image.SetInputConnection(stencil.GetOutputPort())
        image.SetInsideValue(255)
        image.SetOutsideValue(0)
        image.SetOutputScalarTypeToUnsignedChar()
        image.Update()
        mask = vtk_to_numpy(image.GetOutput().GetPointData().GetScalars()).reshape(self.height, self.width).copy()
        if phantom_size_mm is not None:
            points = self.pixel_points @ pose[:3, :3].T + pose[:3, 3]
            width, length, depth = phantom_size_mm
            inside = ((np.abs(points[..., 0]) <= width / 2 + 1e-8)
                      & (np.abs(points[..., 1]) <= length / 2 + 1e-8)
                      & (points[..., 2] >= -1e-8) & (points[..., 2] <= depth + 1e-8))
            mask[~inside] = 0
        return mask
