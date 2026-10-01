"""Five-DOF probe motion constrained to a planar phantom surface."""

import csv
import json
from dataclasses import asdict, dataclass, replace
from importlib.resources import files
from pathlib import Path

import numpy as np
from PIL import Image
from vtk.util.numpy_support import vtk_to_numpy

from .geometry import MeshSlicer, inverse_pose, read_stl, rotation_xyz, transform_mesh, validate_pose


PHANTOMS = {
    "reference": {"name": "Reference", "mesh": "vessel_world_mm.stl", "calibration": "calibration.json"},
    "vessel_no_hole": {"name": "Vessel no hole", "mesh": "vessel_no_hole.stl", "calibration": "vessel_no_hole_calibration.json"},
    "curved_arc": {"name": "Curved arc", "mesh": "curved_arc.stl", "calibration": "curved_arc_calibration.json"},
}


def asset_dir():
    return Path(str(files("view_transition_rl.simulation").joinpath("assets")))


@dataclass(frozen=True)
class PhantomConfig:
    phantom_size_mm: tuple[float, float, float]
    probe_size_mm: tuple[float, float] = (40.0, 10.0)
    image_shape: tuple[int, int] = (480, 480)
    fov_mm: tuple[float, float] = (40.0, 40.0)
    max_tilt_deg: float = 60.0

    def __post_init__(self):
        for name, count in [("phantom_size_mm", 3), ("probe_size_mm", 2), ("fov_mm", 2), ("image_shape", 2)]:
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (count,) or not np.isfinite(value).all() or (value <= 0).any():
                raise ValueError(f"{name} requires {count} finite positive values.")
        if any(int(v) != v for v in self.image_shape):
            raise ValueError("image_shape must contain integers.")
        object.__setattr__(self, "image_shape", tuple(int(v) for v in self.image_shape))
        for name in ["phantom_size_mm", "probe_size_mm", "fov_mm"]:
            object.__setattr__(self, name, tuple(float(v) for v in getattr(self, name)))
        if not np.isfinite(self.max_tilt_deg) or not 0 < self.max_tilt_deg < 90:
            raise ValueError("max_tilt_deg must be between 0 and 90 (exclusive).")


@dataclass
class StepResult:
    accepted: bool
    reason: str
    state: dict


class PhantomEnv:
    """Standalone geometric environment; API lengths are mm and angles radians.

    No Gym reward/termination semantics. Each rejected operation is atomic.
    get_mask() returns a new uint8 array (0/255), rows increasing along probe +z.
    Instances are intended for a single thread; use one instance per worker.
    phantom selects a PHANTOMS entry. config supplies a complete configuration;
    config_overrides instead overrides only explicit fields on model defaults.
    """

    def __init__(self, config: PhantomConfig | None = None, *, phantom="reference", config_overrides=None):
        self.assets = asset_dir()
        if phantom not in PHANTOMS:
            raise ValueError(f"Unknown phantom: {phantom}")
        self.phantom_id = phantom
        self.definition = PHANTOMS[phantom]
        self.calibration = json.loads((self.assets / self.definition["calibration"]).read_text())
        if config is not None and config_overrides is not None:
            raise ValueError("Use config or config_overrides, not both.")
        self.config_overrides = asdict(config) if config is not None else dict(config_overrides or {})
        self.config = PhantomConfig(
            phantom_size_mm=tuple(self.calibration["phantom_size_mm"]),
            probe_size_mm=tuple(self.calibration["probe_size_mm"]),
            image_shape=tuple(self.calibration["image_shape"]),
            fov_mm=tuple(self.calibration["fov_mm"]),
        )
        self.config = replace(self.config, **self.config_overrides)
        self.world_T_phantom = np.array(self.calibration["world_T_phantom_mm"], dtype=float)
        self.phantom_T_world = inverse_pose(self.world_T_phantom)
        self.mesh = transform_mesh(read_stl(self.assets / self.definition["mesh"]), self.phantom_T_world)
        points = vtk_to_numpy(self.mesh.GetPoints().GetData())
        width, length, depth = self.config.phantom_size_mm
        if (np.any(np.abs(points[:, 0]) > width / 2) or np.any(np.abs(points[:, 1]) > length / 2)
                or points[:, 2].min() < 0 or points[:, 2].max() > depth):
            raise ValueError("Configured phantom does not contain the vessel.")
        self.slicer = MeshSlicer(self.mesh, self.config.image_shape, self.config.fov_mm)
        self.reference_rows = []
        if self.calibration["reference_frame_count"]:
            with (self.assets / "reference/poses.csv").open() as handle:
                self.reference_rows = list(csv.DictReader(handle))
        self._pose = np.eye(4)
        self._mask = None
        self._last_accepted = True
        self._last_reason = ""
        result = self.reset()
        if not result.accepted:
            raise ValueError(f"Configured phantom cannot accommodate the initial probe: {result.reason}")

    def _constraint_reason(self, pose):
        if abs(pose[2, 3]) > 1e-7:
            return "Probe bottom centre must remain on the surface (z=0 mm)."
        if pose[2, 2] < np.cos(np.deg2rad(self.config.max_tilt_deg)) - 1e-9:
            return f"Probe tilt exceeds {self.config.max_tilt_deg:g} degrees."
        lx, ly = np.array(self.config.probe_size_mm) / 2
        corners = np.array([[-lx, -ly, 0], [-lx, ly, 0], [lx, -ly, 0], [lx, ly, 0]])
        corners = corners @ pose[:3, :3].T + pose[:3, 3]
        if np.any(np.abs(corners[:, :2]) > np.array(self.config.phantom_size_mm[:2]) / 2 + 1e-8):
            return "Projected probe footprint would leave the phantom top surface."
        return ""

    def _result(self, accepted, reason=""):
        self._last_accepted, self._last_reason = accepted, reason
        return StepResult(accepted, reason, self.get_state())

    def set_pose(self, phantom_T_probe):
        """Set an absolute rigid pose. Invalid inputs return a rejected result."""
        try:
            pose = validate_pose(phantom_T_probe)
        except (ValueError, TypeError) as exc:
            return self._result(False, str(exc))
        reason = self._constraint_reason(pose)
        if reason:
            return self._result(False, reason)
        pose[2, 3] = 0.0
        self._pose = pose
        self._mask = None
        return self._result(True)

    def step(self, action):
        """[dx_mm, dy_mm, rx_rad, ry_rad, rz_rad].

        Translation uses normalized projections of the PRE-ACTION local x/y
        axes onto the top surface (the two projections need not be orthogonal).
        Then R_new = R_old @ Rx(rx) @ Ry(ry) @ Rz(rz).
        """
        try:
            action = np.asarray(action, dtype=float)
        except (TypeError, ValueError):
            return self._result(False, "Action must contain five finite numbers.")
        if action.shape != (5,) or not np.isfinite(action).all():
            return self._result(False, "Action must contain five finite numbers.")
        pose = self._pose.copy()
        directions = pose[:2, :2].copy()
        norms = np.linalg.norm(directions, axis=0)
        if (norms < 1e-9).any():
            return self._result(False, "Local translation direction has no surface projection.")
        pose[:2, 3] += (directions / norms) @ action[:2]
        pose[:3, :3] = pose[:3, :3] @ rotation_xyz(*action[2:])
        return self.set_pose(pose)

    def reset(self):
        return self.replay_frame(len(self.reference_rows) // 2) if self.reference_rows else self.set_pose(np.eye(4))

    def reference_pose(self, index, *, project_to_surface=False):
        """Original image pose expressed in phantom coordinates, in mm.

        Unprojected poses are exclusively for reconstruction evaluation.
        """
        self.require_reference()
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.reference_rows):
            raise IndexError("Reference frame index out of range.")
        row = self.reference_rows[index]
        world_pose = np.array([[float(row[f"m{i}{j}"]) for j in range(4)] for i in range(4)])
        world_pose[:3, 3] *= 1000.0
        pose = self.phantom_T_world @ world_pose
        if project_to_surface:
            pose[2, 3] = 0.0
        return pose

    def require_reference(self):
        if not self.reference_rows:
            raise ValueError(f"Phantom {self.phantom_id!r} has no reference acquisition data.")

    def replay_frame(self, index):
        return self.set_pose(self.reference_pose(index, project_to_surface=True))

    def get_mask(self):
        if self._mask is None:
            self._mask = self.slicer.render(self._pose, self.config.phantom_size_mm)
        return self._mask.copy()

    def get_state(self):
        return {
            "phantom_id": self.phantom_id,
            "phantom_T_probe_mm": self._pose.tolist(),
            "world_T_probe_mm": (self.world_T_phantom @ self._pose).tolist(),
            "tilt_deg": float(np.rad2deg(np.arccos(np.clip(self._pose[2, 2], -1, 1)))),
            "contact_model": "bottom_centre_on_plane_allow_penetration",
            "last_action_accepted": self._last_accepted,
            "constraint_reason": self._last_reason,
            "phantom_size_mm": list(self.config.phantom_size_mm),
            "probe_size_mm": list(self.config.probe_size_mm),
            "image_shape": list(self.config.image_shape),
            "fov_mm": list(self.config.fov_mm),
            "max_tilt_deg": self.config.max_tilt_deg,
            "pixel_axes": "columns: probe +x; rows: probe +z; image plane: probe y=0",
        }

    def export(self, prefix):
        """Export PREFIX.png and PREFIX.json; paths are returned to the caller."""
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        image_path, pose_path = Path(str(prefix) + ".png"), Path(str(prefix) + ".json")
        Image.fromarray(self.get_mask()).save(image_path)
        pose_path.write_text(json.dumps(self.get_state(), indent=2) + "\n")
        return image_path, pose_path


def load_config(path, *, phantom="reference"):
    """Optional JSON overrides for PhantomConfig fields."""
    base = json.loads((asset_dir() / PHANTOMS[phantom]["calibration"]).read_text())
    config = PhantomConfig(**{key: tuple(base[key]) for key in
                              ("phantom_size_mm", "probe_size_mm", "image_shape", "fov_mm")})
    return replace(config, **json.loads(Path(path).read_text()))
