"""Analytic geometry of Phantom 3, in phantom millimetres.

The counterexample concerns the entire central 30 mm arc, not a local tangent
view. Roll plus yaw exposes the complete target at the initial X/Y location.
"""

import numpy as np


CURVATURE_RADIUS_MM = 20.0
VESSEL_RADIUS_MM = 2.0
ARC_RANGE_RAD = (-1.0, 1.0)
TARGET_RANGE_RAD = (-0.75, 0.75)


def centerline(t):
    t = np.asarray(t, dtype=float)
    offset = CURVATURE_RADIUS_MM * (1 - np.cos(t)) / np.sqrt(2)
    return np.stack([offset, CURVATURE_RADIUS_MM * np.sin(t), 20 + offset], axis=-1)


def section_frame(t):
    """Return unit radial normal, binormal and tangent for circular sections."""
    t = np.asarray(t, dtype=float)
    radial = np.stack([np.cos(t) / np.sqrt(2), -np.sin(t),
                       np.cos(t) / np.sqrt(2)], axis=-1)
    tangent = np.stack([np.sin(t) / np.sqrt(2), np.cos(t),
                        np.sin(t) / np.sqrt(2)], axis=-1)
    binormal = np.broadcast_to([1 / np.sqrt(2), 0, -1 / np.sqrt(2)], radial.shape)
    return radial, binormal, tangent


def section_intersections(pose, t):
    """Whether the infinite image plane intersects each analytic lumen disk.

    This measures geometry only, deliberately independent of pixels and FOV.
    """
    pose = np.asarray(pose)
    normal = pose[:3, 1]
    radial, binormal, _ = section_frame(t)
    support = VESSEL_RADIUS_MM * np.hypot(radial @ normal, binormal @ normal)
    distance = np.abs((centerline(t) - pose[:3, 3]) @ normal)
    return distance <= support + 1e-9


def roll_yaw_pose():
    """Pose that covers the target arc without translating X or Y."""
    endpoint = centerline(np.asarray(TARGET_RANGE_RAD))[1]
    sagitta = endpoint[0]
    # Equalize the centre and end plane offsets; both fit inside the lumen.
    roll = np.arctan(sagitta / (40 + sagitta))
    return roll
