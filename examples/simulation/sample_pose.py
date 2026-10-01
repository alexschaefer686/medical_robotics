"""Run from project root: PYTHONPATH=src python3 examples/simulation/sample_pose.py"""
import numpy as np

from view_transition_rl.simulation import PhantomEnv
from view_transition_rl.simulation.geometry import pose_from_euler

env = PhantomEnv()
result = env.set_pose(pose_from_euler(0, 0, roll=np.deg2rad(15)))
assert result.accepted, result.reason
result = env.step([1, 0, 0, np.deg2rad(2), 0])
assert result.accepted, result.reason
print("Mask:", env.get_mask().shape, env.get_mask().dtype)
print("Export:", env.export("outputs/simulation/example"))
