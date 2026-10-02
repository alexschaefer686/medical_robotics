"""Controllers: current mask in, probe movement out.

A movement is the PhantomEnv.step() action [dx_mm, dy_mm, roll_rad, pitch_rad, yaw_rad]
in the probe's local frame. Return None to stop.

Add your algorithm as a Controller subclass that implements act() and
register it in CONTROLLERS.
"""

from collections import deque

import numpy as np

from ..envs.metrics import is_longitudinal, longitudinal_metrics
from ..simulation.geometry import euler_from_pose, pose_from_euler
from .cylinder_fit import fit_bent_cylinder
from .features import FOV_MM, MASK_SHAPE, boundary_points_3d, center_controller, mask_measures  # noqa: F401 (for your controller)
from .plane_fit import fit_image_plane, vicinity_weights


def action_to_pose(current, target):
    """PhantomEnv.step() action that moves the probe from pose current to pose target (4x4, mm).

    step() first translates along the normalised surface projections of the
    current local x and y axes, then rotates R <- R @ Rx(rx) @ Ry(ry) @ Rz(rz).
    Local roll and yaw increments do not simply add once roll != 0, so the
    rotation is solved exactly from R_current^T @ R_target.
    """
    current, target = np.asarray(current, dtype=float), np.asarray(target, dtype=float)
    directions = current[:2, :2] / np.linalg.norm(current[:2, :2], axis=0)
    dx, dy = np.linalg.solve(directions, target[:2, 3] - current[:2, 3])
    m = current[:3, :3].T @ target[:3, :3]  # = Rx(rx) @ Ry(ry) @ Rz(rz)
    rx = np.arctan2(-m[1, 2], m[2, 2])
    ry = np.arcsin(np.clip(m[0, 2], -1.0, 1.0))
    rz = np.arctan2(-m[0, 1], m[0, 0])
    return np.array([dx, dy, rx, ry, rz])


class Controller:
    """Computes mask_measures() each step and keeps the last history_length of them.

    Each history entry is the measures dict of one step: self.history[-1] is
    the current step, self.history[-2] the previous one, ...
    Besides the mask features, each entry holds the probe pose:
      "pose":      4x4 matrix, translation in mm
      "pose_5dof": [x_mm, y_mm, roll_rad, pitch_rad, yaw_rad], R = Rz(yaw) Ry(pitch) Rx(roll)
      "boundary_points": (N, 3) boundary nodes in 3D, mm, same frame as "pose"
    The boundary points of all steps (not only the last history_length) are
    kept in self.cloud; self.point_cloud stacks them.
    Subclasses implement act(); a subclass reset() must call super().reset().
    """

    phase_name = ""  # shown in the 3D view; set by controllers with phases

    def __init__(self, history_length=10):
        self.history_length = history_length
        self.reset()

    def reset(self):
        """Called before the loop starts; clears the history and the point cloud."""
        self.history = deque(maxlen=self.history_length)
        self.cloud = []

    @property
    def point_cloud(self):
        """All boundary points so far, (N, 3) mm."""
        return np.vstack(self.cloud) if self.cloud else np.empty((0, 3))

    def __call__(self, mask, pose):
        """mask: (480, 480) uint8, 0/255; columns = probe +x, rows = depth (probe +z).
        pose: 4x4 probe pose in the robot frame (see run.probe_pose)."""
        measures = mask_measures(mask)
        pose = np.array(pose, dtype=float)
        measures["pose"] = pose
        measures["pose_5dof"] = np.array([pose[0, 3], pose[1, 3], *euler_from_pose(pose)])
        measures["boundary_points"] = points = boundary_points_3d(measures["boundary_nodes"], pose)
        last = self.cloud[-1] if self.cloud else None
        if len(points) and not (last is not None and last.shape == points.shape and np.allclose(last, points)):
            self.cloud.append(points)  # an unchanged frame would only add duplicate points
        self.history.append(measures)
        return self.act(mask, measures)

    def act(self, mask, measures):
        """Return the movement for this step; measures == self.history[-1]."""
        raise NotImplementedError


class MyController(Controller):
    """Template for your visual servoing algorithm."""

    def act(self, mask, measures):
        # 1. Extract image features from the mask, e.g. centroid, orientation, extent.
        # 2. Compare with the desired features (longitudinal view).
        # 3. Map the feature error to a probe movement.
        # Earlier steps: self.history[-2] (previous), ..., self.history[0] (oldest kept).
        move = center_controller(self.history, gain=1)  # movement in probe x direction to center the vessel in the image
        
        move = np.asarray(move) + [0.0, 0.0, 0.0, 0.0, np.pi/20.0]  # element-wise: [dx_mm, dy_mm, roll_rad, pitch_rad, yaw_rad]
        return move  # [dx_mm, dy_mm, roll_rad, pitch_rad, yaw_rad] or None to stop


class SequentialController(Controller):
    """Phases, each running until done, then handing over in the same step:

    1. centre:       move along probe x until the vessel is centred in the image
    2. rotate:       rotate yaw by rotate_deg in rotate_steps equal steps, collecting boundary points
    3. align:        fit a bent cylinder to all points and the closest image plane to its
                     centreline (plane_fit), then move into that plane over align_steps:
                     sideways along the probe's y direction, roll and (align_yaw) yaw.
                     y and roll alone cannot reach the plane unless the yaw already matches.
    4. centre_again: as phase 1
    5. track:        (if track) never ends: x and y are left to the user (W/A/S/D). Each step
                     the cylinder is refitted to all points so far, the image plane through the
                     current contact point that fits its centreline best is fitted (plane_fit,
                     through=...), and the probe turns towards it by track_gain using only
                     roll and yaw (pitch 0).
    Both plane fits weight the centreline by horizontal distance to the probe's contact
    point (plane_fit.vicinity_weights): plane_weighting "window", "linear", "quadratic"
    (zero beyond plane_radius_mm) or None for the whole centreline. plane_distance "xy"
    measures that distance radially, "image_x" only along the probe's x axis (the image's
    width direction), so "window" + "image_x" keeps exactly the centreline within the
    image's width. In simulated tracking, "linear" + "xy" kept the centreline near the
    probe closest to the image plane (about half the distance of the unweighted fit).
    Returns None (stop) after the last phase.
    self.cylinder and self.plane hold the latest fits (None before the align phase);
    self.plane_curve and self.plane_weights the centreline samples and weights used.
    """

    def __init__(self, rotate_deg=90.0, rotate_steps=18, align_steps=5, align_yaw=True, gain=0.5,
                 centre_tol_mm=0.1, max_centre_steps=30, fit_degree=3, roll_weight=1.0, track=True,
                 track_gain=0.5, plane_weighting="linear", plane_radius_mm=FOV_MM[0] / 2,
                 plane_distance="xy", **kwargs):
        self.rotate_step = np.deg2rad(rotate_deg) / rotate_steps
        self.rotate_steps, self.align_steps, self.align_yaw = rotate_steps, align_steps, align_yaw
        self.gain, self.centre_tol_mm, self.max_centre_steps = gain, centre_tol_mm, max_centre_steps
        self.fit_degree, self.roll_weight, self.track_gain = fit_degree, roll_weight, track_gain
        self.plane_weighting, self.plane_radius_mm, self.plane_distance = plane_weighting, plane_radius_mm, plane_distance
        self.PHASES = ("centre", "rotate", "align", "centre_again") + (("track",) if track else ())
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.phase, self.phase_step = 0, 0
        self.align_targets = self.cylinder = self.plane = None
        self.plane_curve = self.plane_weights = None

    def _fit_plane(self, measures, through=None):
        """Image plane fitted to the cylinder centreline near the probe (see class docstring)."""
        self.plane_curve = self.cylinder.centreline(200)
        direction = measures["pose"][:3, 0] if self.plane_distance == "image_x" else None
        self.plane_weights = vicinity_weights(self.plane_curve, measures["pose"][:2, 3],
                                              self.plane_radius_mm, self.plane_weighting, direction)
        return fit_image_plane(self.plane_curve, self.roll_weight, yaw_hint=measures["pose_5dof"][4],
                               through=through, weights=self.plane_weights)

    @property
    def phase_name(self):
        return self.PHASES[self.phase] if self.phase < len(self.PHASES) else "done"

    def act(self, mask, measures):
        while self.phase < len(self.PHASES):
            action = getattr(self, "_" + self.phase_name)(measures)
            if action is not None:
                self.phase_step += 1
                return action
            self.phase, self.phase_step = self.phase + 1, 0
            print(f"phase -> {self.phase_name}")
        return None

    def _centre(self, measures):
        coa = measures["center_of_area"]
        if coa is None or self.phase_step >= self.max_centre_steps:
            return None  # vessel not visible, or not converging
        error_mm = (coa[0] - MASK_SHAPE[1] / 2) * FOV_MM[0] / MASK_SHAPE[1]
        if abs(error_mm) < self.centre_tol_mm:
            return None
        return [self.gain * error_mm, 0.0, 0.0, 0.0, 0.0]

    _centre_again = _centre

    def _rotate(self, measures):
        if self.phase_step >= self.rotate_steps:
            return None
        return [0.0, 0.0, 0.0, 0.0, self.rotate_step]

    def _align(self, measures):
        if self.phase_step == 0:
            self.align_targets = self._plan_alignment(measures)
        if self.align_targets is None or self.phase_step >= len(self.align_targets):
            return None
        # Closed loop in pose: from the measured pose to the next intermediate target.
        return action_to_pose(measures["pose"], self.align_targets[self.phase_step])

    def _plan_alignment(self, measures):
        """Intermediate poses (pitch 0) from the current pose into the closest plane; None if impossible."""
        self.cylinder = fit_bent_cylinder(self.point_cloud, self.fit_degree)
        if self.cylinder is None:
            print("align: cylinder fit failed (points not elongated enough), skipping")
            return None
        x, y, roll, _, yaw = measures["pose_5dof"]
        self.plane = plane = self._fit_plane(measures)
        if plane is None:
            print(f"align: no centreline within {self.plane_radius_mm:g} mm of the probe, skipping")
            return None
        roll_t = plane.pose_5dof[2]
        yaw_t = plane.pose_5dof[4] if self.align_yaw else yaw
        # Shift the contact point along the target's y direction (horizontal, normal to its image
        # plane) into the plane; with pitch 0 that direction does not depend on roll.
        y_dir = np.array([-np.sin(yaw_t), np.cos(yaw_t), 0.0])
        along = plane.normal @ y_dir
        if abs(along) < 0.2:
            print("align: plane is nearly parallel to the probe's y direction, skipping")
            return None
        shift = plane.normal @ (plane.point - measures["pose"][:3, 3]) / along
        droll = np.angle(np.exp(1j * (roll_t - roll)))
        dyaw = np.angle(np.exp(1j * (yaw_t - yaw)))
        print(f"align: cylinder radius {self.cylinder.radius:.2f} mm; plane roll "
              f"{np.rad2deg(roll_t):+.1f} yaw {np.rad2deg(plane.pose_5dof[4]):+.1f} deg; "
              f"moving sideways {shift:+.2f} mm, roll {np.rad2deg(droll):+.2f} deg, yaw {np.rad2deg(dyaw):+.2f} deg"
              + ("" if self.align_yaw else
                 f"; yaw difference {np.rad2deg(np.angle(np.exp(1j * (plane.pose_5dof[4] - yaw)))):+.1f} deg left"))
        fractions = np.arange(1, self.align_steps + 1) / self.align_steps
        return [pose_from_euler(x + f * shift * y_dir[0], y + f * shift * y_dir[1],
                                roll=roll + f * droll, pitch=0.0, yaw=yaw + f * dyaw) for f in fractions]

    def _track(self, measures):
        cylinder = fit_bent_cylinder(self.point_cloud, self.fit_degree)
        if cylinder is not None:
            self.cylinder = cylinder  # otherwise keep the last good fit
        if self.cylinder is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        x, y, roll, _, yaw = measures["pose_5dof"]
        plane = self._fit_plane(measures, through=measures["pose"][:3, 3])
        if plane is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0]  # no centreline near the probe: hold orientation
        self.plane = plane
        droll = np.angle(np.exp(1j * (plane.pose_5dof[2] - roll)))
        dyaw = np.angle(np.exp(1j * (plane.pose_5dof[4] - yaw)))
        target = pose_from_euler(x, y, roll=roll + self.track_gain * droll, pitch=0.0,
                                 yaw=yaw + self.track_gain * dyaw)
        return action_to_pose(measures["pose"], target)  # x and y unchanged: dx = dy = 0


class YawSweepController(Controller):
    """Example: rotate about the probe axis until the mask looks longitudinal.

    Turns toward whichever direction increased column coverage last step.
    """

    def __init__(self, step_deg=5.0, **kwargs):
        self.step = np.deg2rad(step_deg)
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.direction, self.previous = 1.0, None

    def act(self, mask, measures):
        metrics = longitudinal_metrics(mask, FOV_MM)  # RL task metrics, not the measures dict
        if is_longitudinal(metrics):
            return None
        if self.previous is not None and metrics["coverage"] < self.previous:
            self.direction = -self.direction
        self.previous = metrics["coverage"]
        return [0.0, 0.0, 0.0, 0.0, self.direction * self.step]


CONTROLLERS = {"mine": MyController, "sequential": SequentialController}
