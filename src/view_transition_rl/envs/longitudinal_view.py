"""Phantom 1 task: rotate the probe from a cross-sectional to a longitudinal view.

The probe centre stays fixed; only absolute Roll and Yaw change, so Pitch stays 0.
Actions are applied to the absolute Euler angles rather than through
PhantomEnv.step(), whose local increments do not commute and would accumulate Pitch.
"""

import json
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image

from ..simulation import PhantomEnv
from ..simulation.geometry import pose_from_euler
from .metrics import is_longitudinal, longitudinal_metrics

ENV_ID = "ViewTransition/LongitudinalPhantom1-v0"  # see envs.register_envs()

DEFAULT_TASK_CONFIG = {
    "phantom": "reference",
    "initial_pose": {"x_mm": 0.0, "y_mm": 0.0, "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0},
    "roll_step_deg": 2.0,
    "yaw_step_deg": 5.0,
    "max_roll_deg": 20.0,
    "success": {"min_column_run": 0.95, "min_thickness_mm": 4.5},
    "reward": {
        "column_run_weight": 0.7,
        "thickness_weight": 0.3,
        "potential_scale": 10.0,
        "step_penalty": 0.05,
        "rejected_penalty": 0.2,
        "success_bonus": 10.0,
    },
    "observation_size": 84,
    "max_episode_steps": 60,
}

# Action index -> (roll steps, yaw steps).
ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_NAMES = ("roll-", "roll+", "yaw-", "yaw+")


def load_task_config(path=None, **overrides):
    """DEFAULT_TASK_CONFIG updated by an optional JSON file and keyword overrides."""
    config = deepcopy(DEFAULT_TASK_CONFIG)
    updates = json.loads(Path(path).read_text()) if path is not None else {}
    updates.update(overrides)
    for key, value in updates.items():
        if key not in config:
            raise ValueError(f"Unknown task config field: {key}")
        if isinstance(config[key], dict):
            unknown = set(value) - set(config[key])
            if unknown:
                raise ValueError(f"Unknown {key} fields: {sorted(unknown)}")
            config[key].update(value)
        else:
            config[key] = value
    return config


class LongitudinalViewEnv(gym.Env):
    """Discrete Roll/Yaw control on a fixed grid around the initial pose.

    Observation: {"mask": (1, S, S) uint8 0/255, "pose": [roll/max_roll, sin(yaw), cos(yaw)]}.
    Rendered masks and metrics are cached per grid cell and per instance.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, config=None, config_path=None, render_mode=None):
        if config is not None and config_path is not None:
            raise ValueError("Use config or config_path, not both.")
        self.config = load_task_config(config_path) if config is None else load_task_config(**config)
        self.render_mode = render_mode
        self.sim = PhantomEnv(phantom=self.config["phantom"])
        init = self.config["initial_pose"]
        self._x, self._y = init["x_mm"], init["y_mm"]
        self._pitch = np.deg2rad(init["pitch_deg"])
        self._roll0, self._yaw0 = init["roll_deg"], init["yaw_deg"]
        self._roll_step, self._yaw_step = self.config["roll_step_deg"], self.config["yaw_step_deg"]
        self._max_roll = float(self.config["max_roll_deg"])
        self._yaw_cells = int(round(360.0 / self._yaw_step))
        if not np.isclose(self._yaw_cells * self._yaw_step, 360.0):
            raise ValueError("yaw_step_deg must divide 360.")
        size = int(self.config["observation_size"])
        self.observation_space = spaces.Dict({
            "mask": spaces.Box(0, 255, (1, size, size), np.uint8),
            "pose": spaces.Box(-1.0, 1.0, (3,), np.float32),
        })
        self.action_space = spaces.Discrete(len(ACTIONS))
        self._cache = {}
        self._cell = (0, 0)
        self._steps = 0
        self._potential = 0.0

    # -- grid helpers -------------------------------------------------------
    def angles_deg(self, cell=None):
        roll_idx, yaw_idx = self._cell if cell is None else cell
        yaw = (self._yaw0 + yaw_idx * self._yaw_step + 180.0) % 360.0 - 180.0
        return self._roll0 + roll_idx * self._roll_step, yaw

    def _pose(self, cell):
        roll, yaw = self.angles_deg(cell)
        return pose_from_euler(self._x, self._y, roll=np.deg2rad(roll), pitch=self._pitch, yaw=np.deg2rad(yaw))

    def _evaluate(self, cell):
        """Render and score a grid cell; None if the simulator rejects the pose."""
        if cell not in self._cache:
            result = self.sim.set_pose(self._pose(cell))
            if not result.accepted:
                self._cache[cell] = None
            else:
                mask = self.sim.get_mask()
                metrics = longitudinal_metrics(mask, self.sim.config.fov_mm)
                size = int(self.config["observation_size"])
                small = np.asarray(Image.fromarray(mask).resize((size, size), Image.BOX))
                self._cache[cell] = {
                    "mask": mask,
                    "obs_mask": np.where(small >= 128, 255, 0).astype(np.uint8)[None],
                    "metrics": metrics,
                    "success": is_longitudinal(metrics, **self.config["success"]),
                }
        return self._cache[cell]

    def _potential_of(self, entry):
        weights, metrics = self.config["reward"], entry["metrics"]
        thickness = min(metrics["thickness_mm"] / self.config["success"]["min_thickness_mm"], 1.0)
        return weights["column_run_weight"] * metrics["column_run"] + weights["thickness_weight"] * thickness

    def _observation(self, entry):
        roll, yaw = self.angles_deg()
        pose = np.array([roll / self._max_roll, np.sin(np.deg2rad(yaw)), np.cos(np.deg2rad(yaw))], dtype=np.float32)
        return {"mask": entry["obs_mask"].copy(), "pose": np.clip(pose, -1.0, 1.0)}

    def _info(self, entry, rejected=False):
        roll, yaw = self.angles_deg()
        return {"roll_deg": roll, "yaw_deg": yaw, "rejected": rejected, "success": entry["success"],
                "steps": self._steps, **entry["metrics"]}

    # -- Gymnasium API ------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._cell, self._steps = (0, 0), 0
        entry = self._evaluate(self._cell)
        if entry is None:
            raise ValueError("Initial task pose violates the simulator constraints.")
        self._potential = self._potential_of(entry)
        return self._observation(entry), self._info(entry)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action!r}")
        d_roll, d_yaw = ACTIONS[int(action)]
        cell = (self._cell[0] + d_roll, (self._cell[1] + d_yaw) % self._yaw_cells)
        rewards = self.config["reward"]
        self._steps += 1
        candidate = None
        if abs(self.angles_deg(cell)[0]) <= self._max_roll + 1e-9:
            candidate = self._evaluate(cell)
        rejected = candidate is None
        if not rejected:
            self._cell = cell
        entry = self._evaluate(self._cell)
        potential = self._potential_of(entry)
        reward = rewards["potential_scale"] * (potential - self._potential) - rewards["step_penalty"]
        if rejected:
            reward -= rewards["rejected_penalty"]
        terminated = bool(entry["success"])
        if terminated:
            reward += rewards["success_bonus"]
        self._potential = potential
        truncated = not terminated and self._steps >= self.config["max_episode_steps"]
        return self._observation(entry), float(reward), terminated, truncated, self._info(entry, rejected)

    def current_mask(self):
        """Full-resolution mask of the current state."""
        return self._evaluate(self._cell)["mask"].copy()

    def current_pose(self):
        return self._pose(self._cell)

    def render(self):
        if self.render_mode == "rgb_array":
            return np.repeat(self.current_mask()[..., None], 3, axis=-1)
        return None
