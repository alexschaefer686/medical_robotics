"""Probe image plane closest to a curve (e.g. a fitted vessel centreline), preferring roll = 0.

The image plane is the probe's local y = 0 plane through its contact point on
the surface (z = 0). Pitch rotates the image within that plane, so a plane fixes
only x, y, roll and yaw. With pitch = 0 the plane normal (probe y axis) is
    Rz(yaw) Rx(roll) e_y = (-sin(yaw) cos(roll), cos(yaw) cos(roll), sin(roll)),
so roll = 0 is a vertical plane (n_z = 0).

Minimises  mean_k (n . (c_k - q))^2 + roll_weight * n_z^2  over unit normals n,
for planes through q: the smallest eigenvector of mean_k (c_k - q)(c_k - q)^T +
roll_weight * e_z e_z^T. Without a given point, q is the curve's centroid (the
best plane passes through it); with one, e.g. the probe's contact point, the
plane is constrained to contain it. roll_weight is in mm^2: the mean squared
distance worth accepting to avoid sin^2(roll) = 1. A nearly straight curve lies
in many planes; the roll term then picks the most vertical one.
With point weights w_k (e.g. vicinity_weights() around the probe) the means
become weighted means, so only the curve near the probe matters.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ImagePlane:
    normal: np.ndarray     # unit normal = probe y axis; normal[2] = sin(roll)
    point: np.ndarray      # probe contact point on the surface (z = 0), in the plane
    rms: float             # (weighted) RMS distance of the curve points to the plane, mm
    pose_5dof: np.ndarray  # probe pose [x_mm, y_mm, roll_rad, pitch_rad = 0, yaw_rad] with this image plane

    def probe_axes(self):
        """(3, 3) rows: probe x, y, z axes for pose_5dof (x horizontal, z into the phantom)."""
        _, _, roll, _, yaw = self.pose_5dof
        x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        y = self.normal
        return np.array([x, y, np.cross(x, y)])

    def patch(self, points, depth_mm=40.0, margin_mm=5.0):
        """(4, 3) corners of the plane region below the contact point covering points along probe x."""
        x, _, z = self.probe_axes()
        along = (np.asarray(points) - self.point) @ x
        a = (along.min() - margin_mm, along.max() + margin_mm)
        return np.array([self.point + a[0] * x, self.point + a[1] * x,
                         self.point + a[1] * x + depth_mm * z, self.point + a[0] * x + depth_mm * z])


def vicinity_weights(points, centre_xy, radius_mm, profile="quadratic", direction=None):
    """Weights of (N, 3) points by their horizontal distance d to centre_xy, zero beyond radius_mm:

    "window": 1 for d <= radius   "linear": 1 - d / radius   "quadratic": 1 - (d / radius)^2
    profile None weights all points equally. d is the radial x-y distance, or with a
    horizontal direction (e.g. the probe x axis, the image's width direction) the
    distance along it only, so "window" with radius = half the image width keeps
    exactly the points within the image's width, however far the image plane is tilted.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if profile is None:
        return np.ones(len(points))
    offset = points[:, :2] - np.asarray(centre_xy, dtype=float)[:2]
    if direction is None:
        d = np.linalg.norm(offset, axis=1) / radius_mm
    else:
        direction = np.asarray(direction, dtype=float)[:2]
        d = np.abs(offset @ (direction / np.linalg.norm(direction))) / radius_mm
    if profile == "window":
        weights = (d <= 1).astype(float)
    elif profile == "linear":
        weights = 1 - d
    elif profile == "quadratic":
        weights = 1 - d ** 2
    else:
        raise ValueError(f"Unknown weighting profile: {profile!r}")
    return np.clip(weights, 0.0, None)


def fit_image_plane(points, roll_weight=1.0, yaw_hint=0.0, through=None, weights=None):
    """Fit an ImagePlane to (N, 3) curve points in mm; None if fewer than 2 points carry weight.

    yaw_hint (rad): of the two equivalent probe poses (yaw 180 degrees apart,
    opposite roll), the one with yaw closer to yaw_hint is returned.
    through: optional (3,) point the plane must contain, e.g. the probe's
    contact point (x, y, 0); the returned contact point is then that point.
    weights: optional (N,) non-negative point weights, e.g. vicinity_weights().
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    weights = np.ones(len(points)) if weights is None else np.asarray(weights, dtype=float)
    if np.count_nonzero(weights > 0) < 2:
        return None
    weights = weights / weights.sum()
    centre = weights @ points if through is None else np.asarray(through, dtype=float)
    centred = points - centre
    scatter = (centred * weights[:, None]).T @ centred
    normal = np.linalg.eigh(scatter + roll_weight * np.diag([0.0, 0.0, 1.0]))[1][:, 0]
    rms = float(np.sqrt(weights @ (centred @ normal) ** 2))  # weighted RMS distance

    # n and -n describe the same plane; pick the probe orientation closest to yaw_hint.
    candidates = []
    for n in (normal, -normal):
        yaw = np.arctan2(-n[0], n[1])
        candidates.append((abs(np.angle(np.exp(1j * (yaw - yaw_hint)))), n, yaw))
    _, normal, yaw = min(candidates, key=lambda c: c[0])
    roll = np.arcsin(np.clip(normal[2], -1.0, 1.0))

    # Contact point: on the plane's intersection with the surface, nearest above the centre
    # (the centre itself when it lies on the surface, as a given contact point does).
    horizontal = normal[:2]
    if np.linalg.norm(horizontal) < 1e-9:
        return None  # horizontal plane, no probe can image it
    point_xy = centre[:2] + normal[2] * centre[2] / (horizontal @ horizontal) * horizontal
    point = np.array([*point_xy, 0.0])
    return ImagePlane(normal=normal, point=point, rms=rms,
                      pose_5dof=np.array([point[0], point[1], roll, 0.0, yaw]))
