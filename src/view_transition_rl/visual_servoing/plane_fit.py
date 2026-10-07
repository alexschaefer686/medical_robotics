"""Probe image plane closest to a curve (e.g. a fitted vessel centreline), preferring roll = 0.

Works in any frame in which the skin surface is the plane through the origin
with unit normal `up` (pointing into the tissue): the phantom frame (up = +z) or
the current probe frame (origin = contact point, up = the skin normal there).

The image plane is the probe's local y = 0 plane through its contact point on
the surface. Pitch rotates the image within that plane, so a plane fixes only the
contact point, roll and yaw. Roll is the tilt of the plane from the vertical:
sin(roll) = n . up for the plane normal n (probe y axis), so roll = 0 is a plane
containing `up`.

Minimises  mean_k (n . (c_k - q))^2 + roll_weight * (n . up)^2  over unit normals n,
for planes through q: the smallest eigenvector of mean_k (c_k - q)(c_k - q)^T +
roll_weight * up up^T. Without a given point, q is the curve's centroid (the
best plane passes through it); with one, e.g. the probe's contact point, the
plane is constrained to contain it. roll_weight is in mm^2: the mean squared
distance worth accepting to avoid sin^2(roll) = 1. A nearly straight curve lies
in many planes; the roll term then picks the most vertical one.
With point weights w_k (e.g. vicinity_weights() around the probe) the means
become weighted means, so only the curve near the probe matters.
"""

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class ImagePlane:
    normal: np.ndarray     # unit normal = target probe y axis; normal . up = sin(roll)
    point: np.ndarray      # target probe contact point, on the surface and in the plane
    rms: float             # (weighted) RMS distance of the curve points to the plane, mm
    roll: float            # tilt of the plane from the vertical (rad)
    frame: np.ndarray      # (4, 4) target probe frame (pitch 0: x horizontal, z into the tissue)
    pose_5dof: np.ndarray  # [x_mm, y_mm, roll_rad, 0, yaw_rad] of frame; x, y, yaw assume up = +z

    def probe_axes(self):
        """(3, 3) rows: target probe x, y, z axes."""
        return self.frame[:3, :3].T

    def transformed(self, matrix):
        """The same plane expressed in another frame: points p -> matrix @ p (rigid 4x4)."""
        frame = np.asarray(matrix, dtype=float) @ self.frame
        x_axis = frame[:3, 0]
        return replace(self, normal=frame[:3, 1], point=frame[:3, 3], frame=frame,
                       pose_5dof=np.array([frame[0, 3], frame[1, 3], self.roll, 0.0, np.arctan2(x_axis[1], x_axis[0])]))

    def patch(self, points, depth_mm=40.0, margin_mm=5.0):
        """(4, 3) corners of the plane region below the contact point covering points along probe x."""
        x, _, z = self.probe_axes()
        along = (np.asarray(points) - self.point) @ x
        a = (along.min() - margin_mm, along.max() + margin_mm)
        return np.array([self.point + a[0] * x, self.point + a[1] * x,
                         self.point + a[1] * x + depth_mm * z, self.point + a[0] * x + depth_mm * z])


def vicinity_weights(points, centre, radius_mm, profile="quadratic", direction=None, up=(0.0, 0.0, 1.0)):
    """Weights of (N, 3) points by their horizontal distance d to centre, zero beyond radius_mm:

    "window": 1 for d <= radius   "linear": 1 - d / radius   "quadratic": 1 - (d / radius)^2
    profile None weights all points equally. Horizontal means perpendicular to up;
    centre may be given as (x, y) when up = +z. d is the radial horizontal distance,
    or with a horizontal direction (e.g. the probe x axis, the image's width
    direction) the distance along it only, so "window" with radius = half the image
    width keeps exactly the points within the image's width.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if profile is None:
        return np.ones(len(points))
    centre_3d = np.zeros(3)
    centre_3d[:len(centre)] = centre
    up = np.asarray(up, dtype=float) / np.linalg.norm(up)
    offset = points - centre_3d
    offset -= np.outer(offset @ up, up)
    if direction is None:
        d = np.linalg.norm(offset, axis=1) / radius_mm
    else:
        direction = np.asarray(direction, dtype=float)
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


def fit_image_plane(points, roll_weight=1.0, yaw_hint=0.0, through=None, weights=None, up=(0.0, 0.0, 1.0)):
    """Fit an ImagePlane to (N, 3) curve points in mm; None if fewer than 2 points carry weight.

    yaw_hint (rad): a plane is imaged by two probe orientations (yaw 180 degrees
    apart, opposite roll); the one whose y axis is closer to (-sin, cos, 0)(yaw_hint)
    is returned. In the probe frame, yaw_hint = 0 keeps the current orientation's side.
    through: optional (3,) point the plane must contain, e.g. the probe's contact point.
    weights: optional (N,) non-negative point weights, e.g. vicinity_weights().
    up: unit skin normal into the tissue (see the module docstring).
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    weights = np.ones(len(points)) if weights is None else np.asarray(weights, dtype=float)
    if np.count_nonzero(weights > 0) < 2:
        return None
    up = np.asarray(up, dtype=float) / np.linalg.norm(up)
    weights = weights / weights.sum()
    centre = weights @ points if through is None else np.asarray(through, dtype=float)
    centred = points - centre
    scatter = (centred * weights[:, None]).T @ centred
    normal = np.linalg.eigh(scatter + roll_weight * np.outer(up, up))[1][:, 0]
    rms = float(np.sqrt(weights @ (centred @ normal) ** 2))  # weighted RMS distance

    # n and -n describe the same plane; take the probe orientation on the yaw_hint side.
    if normal @ [-np.sin(yaw_hint), np.cos(yaw_hint), 0.0] < 0:
        normal = -normal
    horizontal = normal - (normal @ up) * up
    if np.linalg.norm(horizontal) < 1e-9:
        return None  # plane parallel to the skin, no probe can image it
    x_axis = np.cross(normal, up)
    x_axis /= np.linalg.norm(x_axis)
    roll = float(np.arcsin(np.clip(normal @ up, -1.0, 1.0)))

    # Contact point: on the plane's intersection with the surface, nearest above the centre
    # (the centre itself when it lies on the surface, as a given contact point does).
    height = centre @ up
    point = centre - height * up + (normal @ up) * height / (horizontal @ horizontal) * horizontal
    frame = np.eye(4)
    frame[:3, :3] = np.column_stack([x_axis, normal, np.cross(x_axis, normal)])
    frame[:3, 3] = point
    return ImagePlane(normal=normal, point=point, rms=rms, roll=roll, frame=frame,
                      pose_5dof=np.array([point[0], point[1], roll, 0.0, np.arctan2(x_axis[1], x_axis[0])]))
