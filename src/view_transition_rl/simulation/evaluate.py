"""Original-pose reconstruction comparison, separate from constrained replay."""

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

from .environment import PhantomEnv


def overlap_scores(prediction, reference):
    a, b = prediction > 0, reference > 0
    intersection = np.count_nonzero(a & b)
    total = np.count_nonzero(a) + np.count_nonzero(b)
    union = np.count_nonzero(a | b)
    return (2 * intersection / total if total else 1.0,
            intersection / union if union else 1.0)


def evaluate(output_dir, env=None):
    env = env or PhantomEnv()
    env.require_reference()
    if tuple(env.config.image_shape) != tuple(env.calibration["image_shape"]) or not np.allclose(env.config.fov_mm, env.calibration["fov_mm"]):
        raise ValueError("Reference evaluation requires original image shape and field of view.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, previews = [], {}
    # Warm up VTK before measuring uncached slice-generation latency.
    env.slicer.render(env.reference_pose(0), env.config.phantom_size_mm)
    for index, reference_row in enumerate(env.reference_rows):
        pose = env.reference_pose(index)
        start = perf_counter()
        mask = env.slicer.render(pose, env.config.phantom_size_mm)
        duration_ms = (perf_counter() - start) * 1000
        with Image.open(env.assets / "reference" / reference_row["mask_file"]) as image:
            reference = np.asarray(image.convert("L"))
        dice, iou = overlap_scores(mask, reference)
        rows.append({"index": index, "dice": dice, "iou": iou,
                     "slice_ms": duration_ms, "reference_foreground_pixels": int((reference > 0).sum()),
                     "simulated_foreground_pixels": int((mask > 0).sum()),
                     "original_centre_surface_offset_mm": float(pose[2, 3])})
        previews[index] = (mask, reference)
    with (output / "frames.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "frames": len(rows),
        "pose_mode": "original acquisition poses; no contact projection",
        "length_unit": "mm", "image_shape": list(env.config.image_shape),
        "fov_mm": list(env.config.fov_mm),
        "phantom_size_mm": list(env.config.phantom_size_mm),
        "notes": ["STL smoothing and reconstruction change contours; masks are not expected to match exactly.",
                  "Both-empty Dice/IoU = 1; one-empty score = 0.",
                  "Unobserved vessel beyond closed reconstructed scan ends is not extrapolated.",
                  "Latency includes uncached mesh transform, stencil rasterization and phantom clipping; excludes PNG I/O."],
    }
    for key in ["dice", "iou", "slice_ms"]:
        values = np.array([r[key] for r in rows])
        summary[key] = {"mean": float(values.mean()), "median": float(np.median(values)),
                        "min": float(values.min()), "max": float(values.max()),
                        "p95": float(np.percentile(values, 95))}
    valid = [r for r in rows if r["reference_foreground_pixels"] > 0]
    summary["nonempty_reference_frames"] = len(valid)
    summary["nonempty_reference_dice_mean"] = float(np.mean([r["dice"] for r in valid])) if valid else None
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    selected = sorted({0, len(rows) // 4, len(rows) // 2, 3 * len(rows) // 4,
                       len(rows) - 1, min(rows, key=lambda r: r["dice"])["index"]})
    tile = 240
    panel = Image.new("RGB", (tile * 3, (tile + 32) * len(selected)), "#16202c")
    draw = ImageDraw.Draw(panel)
    for j, index in enumerate(selected):
        mask, reference = previews[index]
        overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
        overlay[..., 0] = reference
        overlay[..., 1] = mask
        for k, image in enumerate([Image.fromarray(reference).convert("RGB"),
                                   Image.fromarray(mask).convert("RGB"), Image.fromarray(overlay)]):
            resampling = getattr(Image, "Resampling", Image)
            panel.paste(image.resize((tile, tile), resampling.NEAREST), (k * tile, j * (tile + 32) + 32))
        for k, label in enumerate([f"Frame {index}: reference", "Simulated STL slice",
                                   f"Overlay Dice={rows[index]['dice']:.3f}"]):
            draw.text((k * tile + 8, j * (tile + 32) + 9), label, fill="white")
    panel.save(output / "comparison.png")
    (output / "README.md").write_text(
        "# Bundled scan reconstruction evaluation\n\n"
        f"Frames: {len(rows)}; original poses (no contact projection).\n\n"
        f"Mean Dice: {summary['dice']['mean']:.4f}; mean IoU: {summary['iou']['mean']:.4f}. "
        f"Uncached slice median: {summary['slice_ms']['median']:.2f} ms; p95: {summary['slice_ms']['p95']:.2f} ms.\n\n"
        "See `frames.csv` for per-frame values and `summary.json` for aggregate results. "
        "In comparison.png: red = reference only, green = simulation only, yellow = overlap. "
        "STL smoothing and reconstruction prevent exact agreement with source segmentation.\n"
    )
    return summary
