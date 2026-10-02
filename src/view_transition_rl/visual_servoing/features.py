"""Image features of a vessel mask for visual servoing; pure NumPy.

Masks are (H, W) arrays where nonzero = vessel. Pixel coordinates are
(col, row): col along probe +x, row along depth (probe +z).
"""

import numpy as np

FOV_MM = (40.0, 40.0)  # image width, depth in mm (all phantoms)
MASK_SHAPE = (480, 480)  # image shape in pixels (all phantoms)

# features

def area(mask):
    """Number of vessel pixels."""
    return int(np.count_nonzero(mask))


def boundary_nodes(mask):
    """Corner nodes between pixels whose surrounding 2x2 block is mixed vessel/background.

    Returns a bool array of shape (H-1, W-1); node (i, j) lies at pixel
    coordinate (row i + 0.5, col j + 0.5). The image border is never a
    boundary: a vessel cut off by the field of view has no nodes there.
    """
    vessel = np.asarray(mask) > 0
    a, b, c, d = vessel[:-1, :-1], vessel[:-1, 1:], vessel[1:, :-1], vessel[1:, 1:]
    return (a | b | c | d) & ~(a & b & c & d)

def boundary_points_3d(boundary, pose, fov_mm=FOV_MM):
    """Boundary nodes as 3D points (N, 3) in mm, expressed in the frame of pose.

    boundary: bool node array (H-1, W-1) from boundary_nodes(); pose: 4x4 probe pose.
    The image plane is probe y = 0 with columns along probe +x (centred on the
    probe) and rows along probe +z, matching simulation.geometry.MeshSlicer.
    """
    h, w = boundary.shape[0] + 1, boundary.shape[1] + 1
    rows, cols = np.nonzero(boundary)
    # Pixel (r, c) centre lies at x = (c + 0.5 - w/2) * sx, z = (r + 0.5) * sz;
    # node (i, j) lies at pixel (i + 0.5, j + 0.5).
    x = (cols + 1 - w / 2) * fov_mm[0] / w
    z = (rows + 1) * fov_mm[1] / h
    local = np.column_stack([x, np.zeros_like(x), z])
    return local @ pose[:3, :3].T + pose[:3, 3]


def elipsoid_fit(mask):
    """Fit an ellipse to the boundary nodes and return its two radii in pixels.

    Returns (radius along the principal axis closer to image x, other radius),
    or None if there are fewer than 5 boundary nodes or no ellipse fits (e.g.
    two straight vessel walls in a longitudinal view).
    Direct least-squares ellipse fit (Fitzgibbon et al. 1999, in the
    numerically stable form of Halir & Flusser 1998).
    """
    rows, cols = np.nonzero(boundary_nodes(mask))
    if rows.size < 5:
        return None
    # Node (i, j) lies at pixel (row i + 0.5, col j + 0.5). Centre and scale the
    # points for a well-conditioned fit; radii are scaled back at the end.
    x, y = cols + 0.5, rows + 0.5
    scale = max(x.std(), y.std())
    if scale == 0:
        return None
    x, y = (x - x.mean()) / scale, (y - y.mean()) / scale

    # Conic A x^2 + B xy + C y^2 + D x + E y + F = 0, constrained to 4AC - B^2 = 1.
    quadratic = np.column_stack([x * x, x * y, y * y])
    linear = np.column_stack([x, y, np.ones_like(x)])
    s1, s2, s3 = quadratic.T @ quadratic, quadratic.T @ linear, linear.T @ linear
    try:
        t = -np.linalg.solve(s3, s2.T)
    except np.linalg.LinAlgError:
        return None
    m = s1 + s2 @ t
    m = np.array([m[2] / 2, -m[1], m[0] / 2])  # C^-1 @ m for the constraint matrix C
    _, vectors = np.linalg.eig(m)
    vectors = vectors.real
    is_ellipse = 4 * vectors[0] * vectors[2] - vectors[1] ** 2 > 0
    if is_ellipse.sum() != 1:
        return None
    a, b, c = vectors[:, is_ellipse][:, 0]
    d, e, f = t @ (a, b, c)

    # Centre, then radii from the eigen-decomposition of the quadratic form.
    q = np.array([[a, b / 2], [b / 2, c]])
    centre = np.linalg.solve(2 * q, [-d, -e])
    f0 = f + (d * centre[0] + e * centre[1]) / 2  # conic value at the centre
    eigenvalues, axes = np.linalg.eigh(q)
    squared = -f0 / eigenvalues
    if np.any(squared <= 0):
        return None
    radii = np.sqrt(squared) * scale
    first = int(np.argmax(np.abs(axes[0])))  # axis with the larger x component
    return float(radii[first]), float(radii[1 - first])


def center_of_area(mask):
    """Vessel centroid (col, row) in pixels, or None if the mask is empty."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    return float(cols.mean()), float(rows.mean())

def mask_measures(mask):
    """Return a dict of features for the given mask."""
    return {
        "area": area(mask),
        "center_of_area": center_of_area(mask),
        "boundary_nodes": boundary_nodes(mask),
        "radii": elipsoid_fit(mask),
    }

# controller

def center_controller(metrics, gain=0.3):
    """Controller that moves the probe to center the vessel in the image."""
    coa = metrics[-1].get("center_of_area")  # last step's center of area
    if coa is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0]  # [dx_mm, dy_mm, roll_rad, pitch_rad, yaw_rad]
    # Example: move the probe to the center of area
    dx = (coa[0] - MASK_SHAPE[1] / 2) * FOV_MM[0] / MASK_SHAPE[1]
    return [dx * gain, 0.0, 0.0, 0.0, 0.0]

def longitudinal_controller(metrics, gain=0.3):
    """Controller that moves the probe to maximize longitudinal view metrics."""
    if not metrics:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    # Example: move the probe to maximize column coverage
    dx = (metrics[-1]["column_run"] - 1.0) * FOV_MM[0] / MASK_SHAPE[1]
    return [dx * gain, 0.0, 0.0, 0.0, 0.0]