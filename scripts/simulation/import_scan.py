#!/usr/bin/env python3
"""One-time asset import. The installed simulator never invokes this script."""

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import vtk
import yaml
from PIL import Image
from vtk.util.numpy_support import vtk_to_numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from view_transition_rl.simulation.geometry import inverse_pose, read_stl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[2] / "src/view_transition_rl/simulation/assets")
    args = parser.parse_args()
    source, target = args.scan_dir.resolve(), args.output.resolve()
    if target.exists():
        raise SystemExit(f"Refusing to overwrite existing asset bundle: {target}")
    metadata = yaml.safe_load((source / "metadata.yaml").read_text())
    if metadata["transform_convention"] != "world_T_image" or metadata["translation_unit"] != "m":
        raise ValueError("Importer expects world_T_image with translations in metres.")
    with (source / "poses.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    poses = np.array([[[float(row[f"m{i}{j}"]) for j in range(4)] for i in range(4)] for row in rows])
    poses[:, :3, 3] *= 1000
    shape_set = {(int(row["height"]), int(row["width"])) for row in rows}
    if len(shape_set) != 1:
        raise ValueError("All reference images must have the same shape.")
    shape = next(iter(shape_set))
    z = poses[:, :3, 2].mean(axis=0)
    z /= np.linalg.norm(z)
    x = poses[:, :3, 0].mean(axis=0)
    x -= np.dot(x, z) * z
    x /= np.linalg.norm(x)
    basis = np.column_stack([x, np.cross(z, x), z])
    origin = poses[:, :3, 3].mean(axis=0)
    mesh = read_stl(source / "scan_baseline_smooth_mm.stl")
    vertices = vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)
    local_vertices = (vertices - origin) @ basis
    local_t = (poses[:, :3, 3] - origin) @ basis
    local_r = np.einsum("ij,njk->nik", basis.T, poses[:, :3, :3])
    corners = np.array([[-20, -5, 0], [-20, 5, 0], [20, -5, 0], [20, 5, 0]])
    footprint = np.einsum("nij,kj->nki", local_r, corners) + local_t[:, None, :]
    covered = np.concatenate([footprint.reshape(-1, 3), local_vertices])
    lo, hi = covered[:, :2].min(axis=0), covered[:, :2].max(axis=0)
    centre = (lo + hi) / 2
    dimensions = np.ceil((hi - lo + 20) / 10) * 10
    world_T_phantom = np.eye(4)
    world_T_phantom[:3, :3] = basis
    world_T_phantom[:3, 3] = origin + basis[:, :2] @ centre
    local_vertices[:, :2] -= centre
    calibration = {
        "schema_version": 1,
        "length_unit": "mm",
        "world_T_phantom_mm": world_T_phantom.tolist(),
        "phantom_T_world_mm": inverse_pose(world_T_phantom).tolist(),
        "phantom_size_mm": [*dimensions.tolist(), 40.0],
        "probe_size_mm": [40.0, 10.0],
        "image_shape": list(shape),
        "fov_mm": [metadata["image_physical_width_m"] * 1000,
                   metadata["image_physical_height_m"] * 1000],
        "reference_frame_count": len(rows),
        "surface_residual_rms_mm": float(np.sqrt(np.mean(local_t[:, 2] ** 2))),
        "surface_residual_max_mm": float(np.max(np.abs(local_t[:, 2]))),
        "vessel_bounds_phantom_mm": [local_vertices.min(axis=0).tolist(), local_vertices.max(axis=0).tolist()],
        "assumptions": {
            "surface": "Mean original image +z is inward normal; mean origin fixes plane offset.",
            "surface_x": "Mean original image +x, orthogonalized against inward normal.",
            "footprint": "Assumed 40x10 mm; not measured in source data.",
            "phantom_size": "Scan footprint and vessel bounds, 10 mm margin per side, rounded up to 10 mm; assumed 40 mm depth.",
            "origin": "Centre of inferred top rectangle; +z into phantom.",
            "contact": "Bottom centre on plane; penetration allowed; no deformation or acoustic coupling model.",
            "mesh_extent": "Source reconstruction closes scan ends at observed extent; unseen vessel is not extrapolated.",
        },
    }
    copies = [("scan_baseline_smooth_mm.stl", "vessel_world_mm.stl"),
              ("poses.csv", "reference/poses.csv"),
              ("metadata.yaml", "reference/metadata.yaml"),
              ("reconstruction_report.json", "source_reconstruction_report.json")]
    for row in rows:
        relative = Path(row["mask_file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source mask path.")
        with Image.open(source / relative) as image:
            if image.size != (shape[1], shape[0]):
                raise ValueError(f"Mask shape mismatch: {relative}")
        copies.append((str(relative), str(Path("reference") / relative)))
    # Validate all inputs before creating the output directory.
    for src, _ in copies:
        if not (source / src).is_file():
            raise FileNotFoundError(source / src)
    target.mkdir(parents=True)
    records = []
    for src, dst in copies:
        output = target / dst
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / src, output)
        records.append({"source_relative_path": src, "bundled_path": dst,
                        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                        "bytes": output.stat().st_size})
    (target / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    (target / "provenance.json").write_text(json.dumps({
        "schema_version": 1, "original_source_directory": str(source),
        "note": "Source path is provenance only; runtime uses packaged resources exclusively.",
        "files": records,
    }, indent=2) + "\n")
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()
