"""Compatibility forwarding for geometric utilities."""

from view_transition_rl.simulation.geometry import (
    MeshSlicer, euler_from_pose, inverse_pose, pose_from_euler, read_stl,
    rotation_xyz, transform_mesh, validate_pose,
)

__all__ = ["MeshSlicer", "euler_from_pose", "inverse_pose", "pose_from_euler",
           "read_stl", "rotation_xyz", "transform_mesh", "validate_pose"]
