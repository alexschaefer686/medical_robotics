"""Mask-only view metrics; pure NumPy so they can be reused without Gymnasium."""

import numpy as np


def longitudinal_metrics(mask, fov_mm=(40.0, 40.0)):
    """Describe how well a binary vessel mask forms a longitudinal view.

    column_run: longest contiguous run of non-empty columns / image width.
    coverage: fraction of non-empty columns (runs need not be contiguous).
    thickness_mm: mean vessel height over non-empty columns.
    """
    mask = np.asarray(mask) > 0
    height, width = mask.shape
    columns = mask.any(axis=0)
    if not columns.any():
        return {"column_run": 0.0, "coverage": 0.0, "thickness_mm": 0.0}
    padded = np.concatenate([[0], columns.astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded))
    run = int((edges[1::2] - edges[::2]).max())
    thickness = mask.sum(axis=0)[columns].mean() * fov_mm[1] / height
    return {"column_run": run / width, "coverage": float(columns.mean()), "thickness_mm": float(thickness)}


def is_longitudinal(metrics, min_column_run=0.95, min_thickness_mm=4.5):
    return metrics["column_run"] >= min_column_run and metrics["thickness_mm"] >= min_thickness_mm
