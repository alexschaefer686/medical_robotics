"""Closed loop: mask -> controller -> relative probe movement, with a live 3D view.

python -m view_transition_rl.visual_servoing.run --offset-x 8 --delay 0.5
python -m view_transition_rl.visual_servoing.run --no-view --output outputs/visual_servoing/run

The controller only gets the mask; its relative motions are executed like a robot
that keeps skin contact (move_probe) and reported back with controller.moved().
The true pose is used for logging and for drawing only.
Can also be started directly (e.g. the VS Code play button), hence absolute imports.
"""

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np

from view_transition_rl.envs.metrics import longitudinal_metrics
from view_transition_rl.simulation import PHANTOMS, PhantomEnv
from view_transition_rl.simulation.geometry import euler_from_pose
from view_transition_rl.visual_servoing.controllers import CONTROLLERS, contact_motion
from view_transition_rl.visual_servoing.cylinder_fit import fit_bent_cylinder
from view_transition_rl.visual_servoing.plane_fit import fit_image_plane
from view_transition_rl.visual_servoing.features import boundary_nodes


# Arrow key -> body displacement direction along phantom (x, y); unknown to the controller.
BODY_KEYS = {"Left": (-1, 0), "Right": (1, 0), "Up": (0, 1), "Down": (0, -1)}
# WASD -> relative probe movement along its own x / y, executed like a controller move and
# reported to the controller (known motion).
PROBE_KEYS = {"a": (-1, 0), "d": (1, 0), "w": (0, 1), "s": (0, -1)}


def move_probe(env, action):
    """Execute a relative probe motion like a robot that keeps skin contact (controllers.contact_motion)."""
    pose = np.array(env.get_state()["phantom_T_probe_mm"])
    skin_normal = pose[2, :3]  # phantom +z (into the tissue) in the probe frame
    return env.set_pose(pose @ contact_motion(action, skin_normal))


def to_frame(points, transform):
    """Apply a 4x4 transform to (N, 3) points."""
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def move_body(env, dx_mm, dy_mm):
    """Shift the body (phantom) under the probe by (dx, dy) mm along phantom x/y.

    Only the relative pose matters for the image, so this is the opposite probe
    translation in the phantom frame (not the probe's local frame). A rejected
    shift (probe footprint would leave the phantom) leaves the pose unchanged.
    """
    pose = np.array(env.get_state()["phantom_T_probe_mm"])
    pose[:2, 3] -= (dx_mm, dy_mm)
    return env.set_pose(pose)


def probe_pose(env, body):
    """Probe pose in the robot (world) frame, 4x4 in mm.

    The simulator stores the pose relative to the phantom; adding the body
    displacement gives what a robot knows from its own kinematics, so body
    motion does not show up in it. Equals the phantom pose while body == 0.
    """
    pose = np.array(env.get_state()["phantom_T_probe_mm"])
    pose[:2, 3] += body
    return pose


def observe(env, step, body):
    mask = env.get_mask()
    pose = np.array(env.get_state()["phantom_T_probe_mm"])
    metrics = longitudinal_metrics(mask, env.config.fov_mm)
    roll, pitch, yaw = np.rad2deg(euler_from_pose(pose))
    row = {"step": step, "x_mm": pose[0, 3], "y_mm": pose[1, 3], "roll_deg": roll, "pitch_deg": pitch,
           "yaw_deg": yaw, "body_x_mm": body[0], "body_y_mm": body[1], **metrics}
    return mask, row


def status_text(row, notice="", phase=""):
    return ((f"phase: {phase}\n" if phase else "") +
            f"step {row['step']}   roll {row['roll_deg']:+.1f}  pitch {row['pitch_deg']:+.1f}  "
            f"yaw {row['yaw_deg']:+.1f} deg\n"
            f"column run {row['column_run']:.2f}   thickness {row['thickness_mm']:.2f} mm\n"
            f"body x {row['body_x_mm']:+.1f}  y {row['body_y_mm']:+.1f} mm{notice}")


def show_fits(cloud_view, cylinder, plane, title, curve=None, weights=None, transform=np.eye(4)):
    """Draw a fitted cylinder and image plane (either may be None) in the point cloud window.
    curve and weights mark the centreline part the plane was fitted to; transform maps the
    fits' frame (e.g. the controller's probe frame) to the window's frame."""
    if cylinder is not None and curve is None:
        curve = cylinder.centreline(200)
    cloud_view.set_fit(cylinder, label="cylinder fit", weights=weights,
                       curve=None if curve is None else to_frame(curve, transform))
    if plane is None or cylinder is None:
        cloud_view.set_plane(None)
        return
    cloud_view.set_plane(to_frame(plane.patch(cylinder.centreline(100)), transform),
                         f"{title}\ntilt from vertical {np.rad2deg(plane.roll):+.1f} deg\n"
                         f"centreline rms to plane {plane.rms:.2f} mm")


def show_controller_fits(cloud_view, controller, transform=np.eye(4)):
    """Draw the controller's cylinder, image plane and the points its fit used.
    transform maps the controller's (current) probe frame to the window's frame.
    Returns False (drawing nothing) while the controller has no fit of its own."""
    if getattr(controller, "cylinder", None) is None:
        return False
    title = "image plane through probe x, y" if controller.phase_name == "track" else "closest image plane"
    weighting = getattr(controller, "plane_weighting", None)
    if weighting is not None:
        title += f"\nfitted to centreline within {controller.plane_radius_mm:g} mm ({weighting})"
    show_fits(cloud_view, controller.cylinder, controller.plane, title,
              getattr(controller, "plane_curve", None), getattr(controller, "plane_weights", None), transform)
    # Only the points its least squares used; the cloud only grows, so the
    # indices stay valid even when the fit is from an earlier step.
    frames = np.repeat(np.arange(len(controller.cloud)), [len(c) for c in controller.cloud])
    used = controller.cylinder.used_index
    cloud_view.draw_points(to_frame(controller.point_cloud[used], transform), frames[used], total=len(frames))
    return True


def run(controller, env, max_steps=None, view=None, output_dir=None, delay=0.0, body_step=1.0,
        cloud_view=None, fit_degree=3, roll_weight=1.0, probe_step=1.0):
    """Loop until the controller returns None, max_steps is reached (None = no limit)
    or the view is closed. delay: seconds per step, only to make the view watchable.

    The controller gets only the mask; each action is executed with move_probe and,
    if the simulator accepts it, reported with controller.moved(). With a view, the
    arrow keys move the body by body_step mm (see move_body), which the controller
    only notices through the mask. W/A/S/D move the probe by probe_step mm along its
    own y/x axes as known motions (also reported with moved()). cloud_view
    (PointCloudView) shows the controller's probe-frame reconstruction placed with the
    true pose (for display only), or, before the controller has a fit, the window's own
    bent cylinder with a polynomial centreline of fit_degree (0 = no fit) and the image
    plane closest to it, pulled towards roll = 0 by roll_weight (see plane_fit).
    """
    controller.reset()
    trajectory = []
    body = np.zeros(2)  # accumulated body displacement along phantom x/y, mm
    step = 0

    def redraw(notice=""):
        mask, row = observe(env, step, body)
        view.update(status_text(row, notice, controller.phase_name), boundary=boundary_nodes(mask))

    def push(dx, dy):
        result = move_body(env, dx * body_step, dy * body_step)
        if result.accepted:
            body[:] += (dx * body_step, dy * body_step)
            view.body_offset = body.copy()
            if cloud_view is not None and cloud_view.is_open:
                cloud_view.set_body_offset(body)
        redraw("" if result.accepted else f"   body move rejected: {result.reason}")

    def nudge(dx, dy):
        action = [dx * probe_step, dy * probe_step, 0.0, 0.0, 0.0]
        result = move_probe(env, action)
        if result.accepted:
            controller.moved(action)  # a commanded motion, so the controller knows about it
        redraw("" if result.accepted else f"   probe move rejected: {result.reason}")

    if view is not None:
        for key, (dx, dy) in BODY_KEYS.items():
            view.on_key(key, lambda dx=dx, dy=dy: push(dx, dy))
        for key, (dx, dy) in PROBE_KEYS.items():
            view.on_key(key, lambda dx=dx, dy=dy: nudge(dx, dy))
        view.set_help(f"W/A/S/D: move probe {probe_step:g} mm along its own y (W/S) and x (A/D)\n"
                      f"Arrow keys: move body {body_step:g} mm   (left/right: x, up/down: y)")

    for step in itertools.count():
        if view is not None:
            if not view.is_open:
                break
            redraw()
            view.process_events()
            if output_dir is not None:
                view.screenshot(Path(output_dir) / f"frame_{step:04d}.png")
            end = time.monotonic() + delay
            while time.monotonic() < end and view.is_open:  # keep the windows responsive while waiting
                view.process_events()
                if cloud_view is not None:
                    cloud_view.process_events()
                time.sleep(0.01)
        # Observe after the wait so the controller sees body moves made meanwhile.
        mask, row = observe(env, step, body)
        seen_from = probe_pose(env, body)  # true pose, for drawing only
        called = max_steps is None or step < max_steps
        action = None
        if called:
            action = controller(mask)
            row["phase"] = controller.phase_name
        row["action"] = "" if action is None else json.dumps([round(float(a), 6) for a in action])
        if action is not None:
            result = move_probe(env, action)
            row["rejected"] = "" if result.accepted else result.reason
            if result.accepted:
                controller.moved(action)
        if called and cloud_view is not None and cloud_view.is_open:
            added = cloud_view.add(to_frame(controller.history[-1]["boundary_points"], seen_from), step)
            if show_controller_fits(cloud_view, controller, transform=probe_pose(env, body)):
                pass  # the controller's own fits, refreshed every step while it tracks
            elif added:
                cylinder = fit_bent_cylinder(cloud_view.points, fit_degree) if fit_degree > 0 else None
                if fit_degree > 0:
                    plane = None if cylinder is None else fit_image_plane(
                        cylinder.centreline(100), roll_weight, euler_from_pose(seen_from)[2])
                    show_fits(cloud_view, cylinder, plane, f"closest image plane (roll weight {roll_weight:g} mm^2)")
                if cylinder is None:
                    cloud_view.draw_points()  # no fit yet: all recorded points
                else:
                    used = cylinder.used_index
                    cloud_view.draw_points(cloud_view.points[used], cloud_view.order[used],
                                           total=len(cloud_view.points))
        trajectory.append(row)
        if action is None:
            break
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "trajectory.csv").open("w", newline="") as handle:
            # The last row may lack "rejected" (no action); dict.fromkeys keeps order without duplicates.
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys([*trajectory[0], "rejected"])))
            writer.writeheader()
            writer.writerows(trajectory)
        env.export(output / "final")
    return trajectory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--controller", choices=sorted(CONTROLLERS), default="sequential")
    parser.add_argument("--phantom", choices=sorted(PHANTOMS), default="curved_arc") #reference, vessel_no_hole or curved_arc
    parser.add_argument("--max-steps", type=int,
                        help="Stop after this many controller steps; default: no limit with the view, 100 without")
    parser.add_argument("--start-at-zero", action="store_true",
                        help="Start at the zero pose instead of the phantom's reset pose")
    parser.add_argument("--offset-x", type=float, default=8.0,
                        help="Initial shift along probe x in mm, moving the vessel off the image centre")
    parser.add_argument("--offset-y", type=float, default=0.0, help="Initial shift along probe y in mm")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds per step in the 3D view")
    parser.add_argument("--body-step", type=float, default=1.0, help="Body displacement per arrow key press, mm")
    parser.add_argument("--probe-step", type=float, default=1.0, help="Probe movement per W/A/S/D key press, mm")
    parser.add_argument("--no-view", action="store_true", help="Run without the 3D windows")
    parser.add_argument("--no-cloud", action="store_true", help="Do not open the boundary point cloud window")
    parser.add_argument("--fit-degree", type=int, default=3,
                        help="Polynomial degree of the fitted cylinder centreline; 0 disables the fit")
    parser.add_argument("--roll-weight", type=float, default=1.0,
                        help="Pull of the fitted image plane towards roll = 0, in mm^2 (see plane_fit)")
    parser.add_argument("--output", help="Save trajectory.csv, final mask and (with view) frames")
    args = parser.parse_args(argv)

    env = PhantomEnv(phantom=args.phantom)
    if args.start_at_zero:
        env.set_pose(np.eye(4))
    start = env.step([args.offset_x, args.offset_y, 0.0, 0.0, 0.0])
    if not start.accepted:
        parser.error(f"Start offset rejected: {start.reason}")
    view = cloud_view = None
    if not args.no_view:
        from view_transition_rl.simulation.scene import PointCloudView, SceneView
        view = SceneView(env)
        view.show()
        if not args.no_cloud:
            cloud_view = PointCloudView(env)
            cloud_view.show()
    max_steps = args.max_steps if args.max_steps is not None or view is not None else 100
    try:
        trajectory = run(CONTROLLERS[args.controller](), env, max_steps, view, args.output,
                         delay=0 if view is None else args.delay, body_step=args.body_step,
                         probe_step=args.probe_step,
                         cloud_view=cloud_view, fit_degree=args.fit_degree, roll_weight=args.roll_weight)
        last = trajectory[-1]
        print(f"Stopped after {last['step']} steps: column run {last['column_run']:.2f}, "
              f"thickness {last['thickness_mm']:.2f} mm")
    except KeyboardInterrupt:
        print("Interrupted.")
    if view is not None and view.is_open:
        print("Close the window to exit.")
        view.show(block=True)


if __name__ == "__main__":
    main()
