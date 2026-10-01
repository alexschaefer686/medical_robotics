#!/usr/bin/env python3
"""Rebuild the deterministic, closed Phantom 3 mesh and its calibration."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

from view_transition_rl.simulation.curved_arc import (
    ARC_RANGE_RAD, CURVATURE_RADIUS_MM, TARGET_RANGE_RAD, VESSEL_RADIUS_MM,
    centerline, roll_yaw_pose, section_frame,
)
from view_transition_rl.simulation.environment import asset_dir
from view_transition_rl.simulation.geometry import pose_from_euler, read_stl


def generate(output_dir):
    output_dir = Path(output_dir)
    stations, sides = 401, 96
    t = np.linspace(*ARC_RANGE_RAD, stations)
    phi = np.arange(sides) * (2 * np.pi / sides)
    centres = centerline(t)
    radial, binormal, _ = section_frame(t)
    rings = centres[:, None, :] + VESSEL_RADIUS_MM * (
        np.cos(phi)[None, :, None] * radial[:, None, :]
        + np.sin(phi)[None, :, None] * binormal[:, None, :])
    points = np.vstack([rings.reshape(-1, 3), centres[[0, -1]]])
    triangles = []
    for i in range(stations - 1):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            c, d = a + sides, b + sides
            triangles.extend([(a, b, d), (a, d, c)])
    for j in range(sides):
        k = (j + 1) % sides
        triangles.append((stations * sides, k, j))
        triangles.append((stations * sides + 1, (stations - 1) * sides + j,
                          (stations - 1) * sides + k))
    triangles = np.asarray(triangles, dtype=np.int64)
    vertices = points[triangles]
    cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    if np.any(np.linalg.norm(cross, axis=1) < 1e-10):
        raise ValueError("Degenerate triangles in generated mesh.")
    volume = np.einsum('ij,ij->i', vertices[:, 0],
                       np.cross(vertices[:, 1], vertices[:, 2])).sum() / 6
    if volume <= 0:
        raise ValueError("Mesh winding is not outward.")
    # Each edge must occur exactly twice, with opposite directions.
    directed = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                   return_inverse=True, return_counts=True)
    balance = np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1))
    if np.any(counts != 2) or np.any(balance != 0):
        raise ValueError("Mesh must be closed and consistently oriented.")
    mesh = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points, deep=True))
    mesh.SetPoints(vtk_points)
    cells = vtk.vtkCellArray()
    cells.SetCells(len(triangles), numpy_to_vtkIdTypeArray(
        np.column_stack([np.full(len(triangles), 3), triangles]).ravel(), deep=True))
    mesh.SetPolys(cells)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = output_dir / 'curved_arc.stl'
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(mesh_path))
    writer.SetFileTypeToBinary()
    writer.SetHeader('View Transition RL - deterministic curved arc phantom')
    writer.SetInputData(mesh)
    if not writer.Write():
        raise OSError(f"Could not write {mesh_path}")
    loaded = read_stl(mesh_path)  # Validate closure after STL coordinate quantization.
    bounds = loaded.GetBounds()
    calibration = {
        'schema_version': 1, 'length_unit': 'mm',
        'world_T_phantom_mm': np.eye(4).tolist(), 'phantom_T_world_mm': np.eye(4).tolist(),
        'phantom_size_mm': [100, 100, 40], 'probe_size_mm': [40, 10],
        'image_shape': [480, 480], 'fov_mm': [40, 40], 'reference_frame_count': 0,
        'source_file': 'scripts/simulation/generate_curved_arc.py',
        'sha256': hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        'vessel_bounds_phantom_mm': [list(bounds[::2]), list(bounds[1::2])],
        'design': {
            'curvature_radius_mm': CURVATURE_RADIUS_MM, 'vessel_radius_mm': VESSEL_RADIUS_MM,
            'arc_range_rad': list(ARC_RANGE_RAD), 'target_range_rad': list(TARGET_RANGE_RAD),
            'arc_length_mm': 40, 'target_length_mm': 30, 'plane': 'z - x = 20 mm',
            'centerline': '[20*(1-cos(t))/sqrt(2), 20*sin(t), 20+20*(1-cos(t))/sqrt(2)]',
            'axial_samples': stations, 'circumferential_samples': sides,
            'cap_style': 'flat disks', 'triangle_count': len(triangles),
            'boundary_or_nonmanifold_edges': 0,
        },
        'example_poses': {
            'initial': np.eye(4).tolist(),
            'longitudinal': pose_from_euler(0, 0, roll_yaw_pose(), 0, np.deg2rad(90)).tolist(),
        },
        'assumptions': {
            'purpose': 'Yaw-only failure for the entire central 30 mm arc, not local longitudinal views.',
            'comparison': 'Complete target lumen is visible with Roll and Yaw while X=Y=0.',
            'surface': 'z=0; +z into phantom; synthetic geometry in mm.',
        },
    }
    calibration_path = output_dir / 'curved_arc_calibration.json'
    calibration_path.write_text(json.dumps(calibration, indent=2) + '\n')
    return mesh_path, calibration_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=asset_dir())
    args = parser.parse_args()
    print(*generate(args.output_dir), sep='\n')
