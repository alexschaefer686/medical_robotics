"""Controllers: current mask in, relative probe movement out. No probe pose is used.

An action [dx_mm, dy_mm, rx_rad, ry_rad, rz_rad] is a motion relative to the probe:
move by (dx, dy) along the probe's own x and y axes, then rotate by
R = Rx(rx) @ Ry(ry) @ Rz(rz) about the new contact point (rz turns about the probe
axis). The robot, or the simulation (run.move_probe), keeps skin contact by sliding
along the new probe z axis back onto the skin; contact_motion() gives the motion
actually executed. On a robot this is one relative Twist in the probe frame.
Return None to stop.

Controllers know the probe only through their own frame: after every executed
motion, the caller reports it with controller.moved(action), so the controller can
express what it has seen in the new probe frame.

Add your algorithm as a Controller subclass that implements act() and
register it in CONTROLLERS.
"""

from collections import deque

import numpy as np

from ..envs.metrics import is_longitudinal, longitudinal_metrics
from ..simulation.geometry import rotation_xyz
from .cylinder_fit import fit_bent_cylinder
from .features import FOV_MM, MASK_SHAPE, boundary_points_3d, center_controller, mask_measures  # noqa: F401 (for your controller)
from .plane_fit import fit_image_plane, vicinity_weights


def rotation_vector(rotation):
    """Axis * angle (rad) of a 3x3 rotation matrix (angles below 180 degrees)."""
    angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]]) / (2 * np.sin(angle))
    return axis * angle


def rotation_matrix(vector):
    """3x3 rotation matrix of a rotation vector (Rodrigues)."""
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    kx, ky, kz = np.asarray(vector) / angle
    k = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def xyz_angles(rotation):
    """(rx, ry, rz) with rotation = Rx(rx) @ Ry(ry) @ Rz(rz)."""
    return np.array([np.arctan2(-rotation[1, 2], rotation[2, 2]),
                     np.arcsin(np.clip(rotation[0, 2], -1.0, 1.0)),
                     np.arctan2(-rotation[0, 1], rotation[0, 0])])


def relative_motion(action):
    """Commanded motion of an action as a 4x4 transform in the probe frame before the move."""
    delta = np.eye(4)
    delta[:3, :3] = rotation_xyz(*np.asarray(action, dtype=float)[2:])
    delta[:2, 3] = action[:2]
    return delta


def contact_motion(action, up):
    """Executed motion (4x4, probe frame before the move) when the probe keeps skin contact.

    up: unit skin normal into the tissue, in the probe frame before the move; the
    skin is the plane through the contact point (origin) with that normal. After the
    commanded motion the probe slides along its new z axis back onto the skin.
    """
    up = np.asarray(up, dtype=float)
    delta = relative_motion(action)
    z_new, t = delta[:3, 2], delta[:3, 3]
    delta[:3, 3] = t - (t @ up) / (z_new @ up) * z_new
    return delta


def action_for_motion(translation, rotation, up):
    """Action whose executed motion (contact_motion) is the given rotation and translation.

    translation must lie in the skin plane (perpendicular to up); the part along the
    new probe z axis is then exactly what the contact sliding supplies.
    """
    rotation = np.asarray(rotation, dtype=float)
    tx, ty, _ = np.linalg.solve(np.column_stack([[1.0, 0, 0], [0, 1.0, 0], rotation[:, 2]]), translation)
    return np.array([tx, ty, *xyz_angles(rotation)])


class Controller:
    """Computes mask_measures() each step and keeps the last history_length of them.

    Each history entry is the measures dict of one step: self.history[-1] is
    the current step, self.history[-2] the previous one, ...
    Besides the mask features, each entry holds
      "boundary_points": (N, 3) boundary nodes in 3D, mm, in the probe frame of that step.
    The boundary points of all steps (not only the last history_length) are kept
    in self.cloud, always expressed in the CURRENT probe frame (origin = contact
    point, x/y/z = probe axes); self.point_cloud stacks them. self.surface_normal
    is the skin normal ("down", into the tissue) in the current probe frame,
    assuming the probe started perpendicular to the skin. Both are updated only
    from the motions reported with moved(); there is no probe pose.
    fov_mm: physical (width, depth) of the mask image; any mask size works.
    Subclasses implement act(); a subclass reset() must call super().reset().
    """

    phase_name = ""  # shown in the 3D view; set by controllers with phases

    def __init__(self, history_length=10, fov_mm=FOV_MM):
        self.history_length = history_length
        self.fov_mm = tuple(fov_mm)
        self.mask_shape = MASK_SHAPE
        self.reset()

    def reset(self):
        """Called before the loop starts; clears the history and the point cloud."""
        self.history = deque(maxlen=self.history_length)
        self.cloud = []
        self.surface_normal = np.array([0.0, 0.0, 1.0])

    @property
    def point_cloud(self):
        """All boundary points so far in the current probe frame, (N, 3) mm."""
        return np.vstack(self.cloud) if self.cloud else np.empty((0, 3))

    def __call__(self, mask):
        """mask: (H, W) uint8, nonzero = vessel; columns = probe +x, rows = depth (probe +z),
        covering fov_mm."""
        self.mask_shape = np.shape(mask)
        measures = mask_measures(mask)
        measures["boundary_points"] = points = boundary_points_3d(measures["boundary_nodes"], np.eye(4), self.fov_mm)
        last = self.cloud[-1] if self.cloud else None
        if len(points) and not (last is not None and last.shape == points.shape and np.allclose(last, points)):
            self.cloud.append(points)  # an unchanged frame would only add duplicate points
        self.history.append(measures)
        return self.act(mask, measures)

    def moved(self, action):
        """Report that the probe executed the relative motion `action` (from this controller
        or anyone else, e.g. the user); moves the point cloud and skin normal into the new
        probe frame. Unreported motion (e.g. the patient moving) is not accounted for."""
        return self.apply_motion(contact_motion(action, self.surface_normal))

    def apply_motion(self, delta):
        """Report an executed rigid motion (4x4, probe frame before the move) as is, without the
        skin-contact sliding of moved(), e.g. a deliberate move along the probe axis. Returns delta;
        old-frame points p become inv(delta) @ p."""
        delta = np.asarray(delta, dtype=float)
        rotation, translation = delta[:3, :3], delta[:3, 3]
        self.cloud = [(chunk - translation) @ rotation for chunk in self.cloud]  # R^T (p - t) per point
        self.surface_normal = rotation.T @ self.surface_normal
        return delta

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
    2. rotate:       rotate about the probe axis by rotate_deg in rotate_steps equal steps,
                     collecting boundary points
    3. align:        fit a bent cylinder to all points and the closest image plane to its
                     centreline (plane_fit); the target is the probe frame in that plane
                     reached by sliding sideways along the skin and turning. Over align_steps
                     steps, each covers 1 / (steps left) of the remaining way; the target is
                     kept in the current probe frame through moved().
    4. centre_again: as phase 1
    5. track:        (if track) never ends: x and y are left to the user (W/A/S/D, reported
                     with moved()). Each step the cylinder is refitted to all points so far,
                     the image plane through the current contact point that fits its
                     centreline best is fitted (plane_fit, through=...), and the probe turns
                     towards it by track_gain without moving (pitch stays 0).
    Everything is computed in the current probe frame; "roll = 0" refers to
    self.surface_normal (see Controller).
    Both plane fits weight the centreline by horizontal distance to the probe's contact
    point (plane_fit.vicinity_weights): plane_weighting "window", "linear", "quadratic"
    (zero beyond plane_radius_mm) or None for the whole centreline. plane_distance "xy"
    measures that distance radially, "image_x" only along the probe's x axis (the image's
    width direction), so "window" + "image_x" keeps exactly the centreline within the
    image's width. In simulated tracking, "linear" + "xy" kept the centreline near the
    probe closest to the image plane (about half the distance of the unweighted fit).
    Returns None (stop) after the last phase.
    self.cylinder and self.plane hold the latest fits in the probe frame of their step
    (None before the align phase); self.plane_curve and self.plane_weights the centreline
    samples and weights used.
    """

    def __init__(self, rotate_deg=180.0, rotate_steps=18, align_steps=1, gain=0.5,
                 centre_tol_mm=0.1, max_centre_steps=30, fit_degree=3, roll_weight=1.0, track=True,
                 track_gain=0.5, plane_weighting="linear", plane_radius_mm=FOV_MM[0] / 2,
                 plane_distance="xy", **kwargs):
        self.rotate_step = np.deg2rad(rotate_deg) / rotate_steps
        self.rotate_steps, self.align_steps = rotate_steps, align_steps
        self.gain, self.centre_tol_mm, self.max_centre_steps = gain, centre_tol_mm, max_centre_steps
        self.fit_degree, self.roll_weight, self.track_gain = fit_degree, roll_weight, track_gain
        self.plane_weighting, self.plane_radius_mm, self.plane_distance = plane_weighting, plane_radius_mm, plane_distance
        self.PHASES = ("centre", "rotate", "align", "centre_again") + (("track",) if track else ())
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.phase, self.phase_step = 0, 0
        self.cylinder = self.plane = self.align_target = None
        self.plane_curve = self.plane_weights = None

    def apply_motion(self, delta):
        """As Controller.apply_motion() (also used by moved()), and keeps the fits and the
        alignment target in the current probe frame."""
        to_new_frame = np.linalg.inv(super().apply_motion(delta))
        if self.cylinder is not None:
            self.cylinder = self.cylinder.transformed(to_new_frame)
        if self.plane is not None:
            self.plane = self.plane.transformed(to_new_frame)
        if self.plane_curve is not None:
            self.plane_curve = self.plane_curve @ to_new_frame[:3, :3].T + to_new_frame[:3, 3]
        if self.align_target is not None:
            self.align_target = to_new_frame @ self.align_target
        return np.linalg.inv(to_new_frame)

    def _refit_cylinder(self):
        """Refit to all points so far; keeps the last good fit (None at first) if this one fails."""
        cylinder = fit_bent_cylinder(self.point_cloud, self.fit_degree)
        if cylinder is not None:
            self.cylinder = cylinder
        return cylinder

    def _fit_plane(self, through=None):
        """Image plane fitted to the cylinder centreline near the probe (see class docstring)."""
        self.plane_curve = self.cylinder.centreline(200)
        direction = (1.0, 0.0, 0.0) if self.plane_distance == "image_x" else None
        self.plane_weights = vicinity_weights(self.plane_curve, np.zeros(3), self.plane_radius_mm,
                                              self.plane_weighting, direction, up=self.surface_normal)
        return fit_image_plane(self.plane_curve, self.roll_weight, yaw_hint=0.0, through=through,
                               weights=self.plane_weights, up=self.surface_normal)

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
        width = self.mask_shape[1]
        error_mm = (coa[0] - width / 2) * self.fov_mm[0] / width
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
            self.align_target = self._plan_alignment()
        steps_left = self.align_steps - self.phase_step
        if self.align_target is None or steps_left <= 0:
            return None
        # Remaining motion = the target in the current probe frame; cover 1 / steps_left of it.
        fraction = 1.0 / steps_left
        turn = rotation_matrix(fraction * rotation_vector(self.align_target[:3, :3]))
        return action_for_motion(fraction * self.align_target[:3, 3], turn, self.surface_normal)

    def _plan_alignment(self):
        """Target probe frame (4x4, current probe frame) in the closest image plane; None if impossible."""
        if self._refit_cylinder() is None:
            print("align: cylinder fit failed (points not elongated enough), skipping")
            return None
        plane = self._fit_plane()
        if plane is None:
            print(f"align: no centreline within {self.plane_radius_mm:g} mm of the probe, skipping")
            return None
        self.plane = plane
        # Slide the contact point (origin) along the skin, in the horizontal direction closest to
        # the plane normal, until it lies in the plane; the probe axes become plane.frame's.
        up, normal = self.surface_normal, plane.normal
        side = normal - (normal @ up) * up
        side /= np.linalg.norm(side)
        if abs(normal @ side) < 0.2:
            print("align: plane is nearly parallel to the skin, skipping")
            return None
        shift = (normal @ plane.point) / (normal @ side)
        target = plane.frame.copy()
        target[:3, 3] = shift * side
        print(f"align: cylinder radius {self.cylinder.radius:.2f} mm; plane tilt {np.rad2deg(plane.roll):+.1f} deg; "
              f"sliding {shift:+.2f} mm sideways and turning "
              f"{np.rad2deg(np.linalg.norm(rotation_vector(target[:3, :3]))):.1f} deg")
        return target

    def _track(self, measures):
        self._refit_cylinder()
        if self.cylinder is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        plane = self._fit_plane(through=np.zeros(3))
        if plane is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0]  # no centreline near the probe: hold orientation
        self.plane = plane
        turn = rotation_matrix(self.track_gain * rotation_vector(plane.frame[:3, :3]))
        return action_for_motion(np.zeros(3), turn, self.surface_normal)  # turn about the contact point


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
