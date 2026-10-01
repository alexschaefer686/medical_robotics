#!/usr/bin/env python3
"""Align, scale and close the supplied probe STL as a packaged display asset."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def closed_edge_count(mesh):
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(mesh)
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    return edges.GetOutput().GetNumberOfCells()


def process(source_path, output_dir, *, bottom_width_mm=40.0,
            bottom_length_mm=10.0, head_height_mm=50.0):
    source_path = Path(source_path).resolve(strict=True)
    output_dir = Path(output_dir)
    if not source_path.is_file() or source_path.stat().st_size == 0:
        raise ValueError(f"Source is not a nonempty file: {source_path}")
    if any(not np.isfinite(v) or v <= 0 for v in
           [bottom_width_mm, bottom_length_mm, head_height_mm]):
        raise ValueError("Target dimensions must be finite positive millimetres.")

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(source_path))
    reader.Update()
    source = reader.GetOutput()
    if source.GetNumberOfCells() == 0 or closed_edge_count(source) != 0:
        raise ValueError("The input probe STL must be a nonempty closed surface.")
    points = vtk_to_numpy(source.GetPoints().GetData()).astype(float)
    lo, hi = points.min(axis=0), points.max(axis=0)
    extents = hi - lo
    if (extents <= 0).any():
        raise ValueError("The probe STL has a degenerate axis.")
    z_min = lo[2]
    bottom = points[points[:, 2] <= z_min + max(1e-5, extents[2] * 1e-7)]
    bottom_centre = (bottom[:, :2].min(axis=0) + bottom[:, :2].max(axis=0)) / 2
    bottom_extent = np.ptp(bottom[:, :2], axis=0)
    if (bottom_extent <= 0).any():
        raise ValueError("Could not find a nondegenerate planar probe bottom.")
    # Original X maps to probe +X. Original Y maps to probe -Y and original Z
    # maps to probe -Z; diag(+1,-1,-1) is a proper rotation (determinant +1).
    scales = np.array([bottom_width_mm / bottom_extent[0],
                       bottom_length_mm / bottom_extent[1], 1.0])
    transform = vtk.vtkTransform()
    matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            matrix.SetElement(i, j, 0)
    matrix.SetElement(0, 0, scales[0])
    matrix.SetElement(1, 1, -scales[1])
    matrix.SetElement(2, 2, -1)
    matrix.SetElement(0, 3, -scales[0] * bottom_centre[0])
    matrix.SetElement(1, 3, scales[1] * bottom_centre[1])
    matrix.SetElement(2, 3, z_min)
    matrix.SetElement(3, 3, 1)
    transform.SetMatrix(matrix)
    align = vtk.vtkTransformPolyDataFilter()
    align.SetInputData(source)
    align.SetTransform(transform)
    align.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(align.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()
    clip_plane = vtk.vtkPlane()
    clip_plane.SetOrigin(0, 0, -head_height_mm)
    clip_plane.SetNormal(0, 0, 1)
    planes = vtk.vtkPlaneCollection()
    planes.AddItem(clip_plane)
    clip = vtk.vtkClipClosedSurface()
    clip.SetInputConnection(normals.GetOutputPort())
    clip.SetClippingPlanes(planes)
    clip.SetGenerateFaces(1)
    clip.SetScalarModeToColors()
    clip.SetClipColor(0.72, 0.76, 0.80)
    clip.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(clip.GetOutputPort())
    clean.Update()
    closed_normals = vtk.vtkPolyDataNormals()
    closed_normals.SetInputConnection(clean.GetOutputPort())
    closed_normals.ConsistencyOn()
    closed_normals.AutoOrientNormalsOn()
    closed_normals.SplittingOff()
    closed_normals.Update()
    model = vtk.vtkPolyData()
    model.DeepCopy(closed_normals.GetOutput())
    if model.GetNumberOfCells() == 0 or closed_edge_count(model) != 0:
        raise ValueError("Cut probe is empty or did not produce a closed cap.")
    output_points = vtk_to_numpy(model.GetPoints().GetData())
    if not np.isfinite(output_points).all():
        raise ValueError("Processed probe contains non-finite points.")
    bounds = model.GetBounds()
    aligned_bottom = vtk_to_numpy(align.GetOutput().GetPoints().GetData())
    aligned_contact_face = aligned_bottom[aligned_bottom[:, 2] >= -2e-3]
    contact_bounds = [aligned_contact_face[:, :2].min(axis=0).tolist(),
                      aligned_contact_face[:, :2].max(axis=0).tolist()]
    if not np.allclose([contact_bounds[0][0], contact_bounds[1][0],
                        contact_bounds[0][1], contact_bounds[1][1]],
                       [-bottom_width_mm / 2, bottom_width_mm / 2,
                        -bottom_length_mm / 2, bottom_length_mm / 2], atol=2e-3):
        raise ValueError(f"Processed contact-face bounds are not aligned: {contact_bounds}")
    if not np.isclose(bounds[4], -head_height_mm, atol=2e-3) or abs(bounds[5]) > 2e-3:
        raise ValueError(f"Processed model is not centred/cropped to the requested height: {bounds}")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_copy = output_dir / "linear_transducer_source.STL"
    output_mesh = output_dir / "probe_model_50mm.stl"
    source_copy.write_bytes(source_path.read_bytes())
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(output_mesh))
    writer.SetFileTypeToBinary()
    writer.SetInputData(model)
    if writer.Write() != 1:
        raise OSError(f"Could not write {output_mesh}")
    # Read back the exact packaged mesh before recording provenance.
    packaged_reader = vtk.vtkSTLReader()
    packaged_reader.SetFileName(str(output_mesh))
    packaged_reader.Update()
    packaged = packaged_reader.GetOutput()
    if packaged.GetNumberOfCells() == 0 or closed_edge_count(packaged) != 0:
        raise ValueError("Written probe STL failed closed-surface validation.")
    metadata = {
        "schema_version": 1,
        "source_file": "linear_transducer_source.STL",
        "processed_file": "probe_model_50mm.stl",
        "length_unit": "mm",
        "source_units_assumed": "mm",
        "source_sha256": sha256(source_copy),
        "processed_sha256": sha256(output_mesh),
        "source_bounds": [float(x) for x in source.GetBounds()],
        "source_bottom_xy_bounds": [bottom[:, :2].min(axis=0).tolist(),
                                     bottom[:, :2].max(axis=0).tolist()],
        "source_bottom_center": [float(x) for x in bottom_centre],
        "source_bottom_extent": [float(x) for x in bottom_extent],
        "original_to_probe_transform": [[float(matrix.GetElement(i, j)) for j in range(4)] for i in range(4)],
        "scale_xyz": scales.tolist(),
        "probe_bounds_mm": [float(x) for x in packaged.GetBounds()],
        "contact_face_xy_bounds_mm": contact_bounds,
        "probe_bottom_size_mm": [bottom_width_mm, bottom_length_mm],
        "retained_height_mm": head_height_mm,
        "clip_plane_probe_local": {"origin": [0, 0, -head_height_mm], "normal": [0, 0, 1]},
        "triangles": packaged.GetNumberOfCells(),
        "boundary_or_nonmanifold_edges": closed_edge_count(packaged),
    }
    metadata_path = output_dir / "probe_model.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    project_root = Path(__file__).resolve().parents[2]
    default_output = project_root / "src/view_transition_rl/simulation/assets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--bottom-width-mm", type=float, default=40)
    parser.add_argument("--bottom-length-mm", type=float, default=10)
    parser.add_argument("--head-height-mm", type=float, default=50)
    args = parser.parse_args()
    print(json.dumps(process(args.source, args.output_dir,
                             bottom_width_mm=args.bottom_width_mm,
                             bottom_length_mm=args.bottom_length_mm,
                             head_height_mm=args.head_height_mm), indent=2))


if __name__ == "__main__":
    main()
