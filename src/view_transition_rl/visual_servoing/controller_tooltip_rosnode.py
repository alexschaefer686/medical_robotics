#!/usr/bin/env python3
"""Run a sequential controller on the real robot over ROS1, reconstructing from the measured tooltip pose.

Same loop, keys and parameters as controller_rosnode.py (read its docstring first),
except for how the reconstruction follows the probe. controller_rosnode.py reports
every command with controller.moved(), assuming the robot executes it exactly and
keeps skin contact by sliding along the probe axis; dropped or clamped commands,
tracking errors and any motion this node did not command end up as errors in the
point cloud. Here the probe motion is MEASURED on ~pose_topic instead:

  1. Each mask is stored with the latest tooltip pose received before it. It is only
     used if that pose is at most ~max_pose_age_sec old and the probe had been at rest
     (within ~still_mm / ~still_deg) for ~still_sec before the mask arrived, so that
     the image, which reaches this node with some latency, and the pose belong
     together. ~still_sec must exceed that latency.
  2. Before the controller sees the mask, it gets the measured motion since its last
     step, inv(pose_last) @ pose_now (probe frame before the move), through
     controller.apply_motion(): its point cloud, fits, alignment target and skin
     normal move into the probe frame the mask was taken in.
  3. Motions (controller and keys) are published but not reported to the controller;
     the measured pose at its next step accounts for whatever the robot actually did.
     Controller actions are sent as the motion the controller assumes (contact_motion,
     including the slide along the new probe axis back onto the skin), and
     Twist.angular as ZYX angles (rx, ry, rz) with R = Rz(rz) Ry(ry) Rx(rx), which is how
     probe_scan_motion_node reads it (not as a rotation vector).

~controller_type selects the controller (controllers.py): "no_increments" makes every
motion ONE command, never a series of increments or corrections: the levelling, the
centring (the whole offset at once), the turn about the probe axis (turn_deg, 90 by
default; set it with ~controller, e.g. {turn_deg: 60}) and the move into the plane; it
has no tracking. After each move the node logs how far off it ended up, without
correcting. "sequential" (default) works in steps, e.g. rotate_steps separate turns
and repeated corrections. The no_increments turn is driven at constant probe-frame
angular velocity on ~cruise_topic. During it, steps 1 and 2 are replaced by streaming:
every mask is given to the controller as it arrives, each with the probe pose at the
mask's header stamp (the time its frame was grabbed; the receipt time if the stamp is
0), interpolated from the poses received around it. The velocity is set back to zero
when the controller has measured the requested turn, or if the probe stops short of it.

When started (first Enter), the node first levels the probe, before the controller
runs: it turns the probe about its contact point until the tooltip z axis points
straight down (pose frame -z), i.e. tooltip roll +-pi (whichever is closer: the same
orientation, reached by the shortest turn) and pitch 0, yaw kept, within
~level_tol_deg. This is closed loop on ~pose_topic, one command (within
~max_rotation_deg) per ~command_period_sec once the probe is at rest; if a command does
not reduce the remaining turn, the node pauses. With no_increments it is one command;
the controller starts once the probe is at rest after it. With ~dry_run the first
correction is only logged and the controller starts right after.

With ~stop_in_plane (default), the node takes over once the controller has fitted the
cylinder and the closest image plane (the start of its align phase): it moves the probe
straight to the planned probe pose in that plane in ~plane_steps commands, each covering
1 / (commands left) of the remaining motion measured on ~pose_topic (as the controller's
own align phase does), then single corrections until within ~plane_tol_mm /
~plane_tol_deg, and then stops
(no centre_again or track). With no_increments it is one command for the whole motion
and no corrections. If no plane could be fitted it stops right away; if a command does
not reduce the remaining motion, it pauses.

The controller's probe frame (origin = contact point, x = image columns, z = depth)
must be the frame probe_scan_motion_node executes the Twist commands in. The tooltip
pose only adds probe_length along the flange z axis, so the node composes it with the
rest of that node's calibration, Rot_z(probe_z_rotation_deg) * Trans_y(probe_y_offset)
(ultrasound_probe_toolkit's config/probe_calibration.yaml, plus any launch overrides),
read from the parameter server under ~calibration_ns; nothing is hardcoded, and the
node does not start without it. A 7.75 mm offset alone puts frames taken 180 degrees
apart about the probe axis up to 15.5 mm off.

The controller's skin normal (its "roll = 0") is the probe axis at its first step,
right after levelling, so the skin under the probe should be about horizontal. Body
(patient) motion is still not accounted for.

The reconstruction window shows the probe frame at the first tooltip pose received,
fixed: the reconstruction stays in place and the probe, drawn live from ~pose_topic,
moves through it.

Parameters (private), in addition to those of controller_rosnode.py:
  ~pose_topic                  measured tooltip pose, lwr_controllers/PoseRPY with position in m and
                               orientation R = Rz(yaw) Ry(pitch) Rx(roll) in rad
                               (/cartesian_position/pose/tooltip)
  ~controller_type             "sequential" (default) or "no_increments"
  ~controller                  keyword arguments of that controller, e.g. {turn_deg: 60}
  ~cruise_topic                Twist command used for the no_increments constant-speed turn
                               (/probe_cruise_velocity)
  ~turn_speed_deg_s            magnitude of that probe-frame angular Z velocity (10.0)
  ~turn_command_period_sec     resend period while turning, so a command rejected during the
                               preceding relative step is accepted once the motion node is idle (0.5)
  ~calibration_ns              namespace holding probe_y_offset (m) and probe_z_rotation_deg
                               (/probe_scan_motion_node, where probe_scan_motion.launch loads
                               probe_calibration.yaml; start that node first)
  ~max_pose_age_sec            maximum age of the pose stored with a mask (0.5)
  ~still_sec                   how long the probe must have been at rest before a mask (0.5)
  ~still_mm, ~still_deg        motion that still counts as at rest (0.2, 0.2)
  ~level_tol_deg               remaining turn at which the levelling ends (0.5)
  ~stop_in_plane               move into the fitted plane and stop there (True); False: run the
                               controller's own align, centre_again and track phases
  ~plane_steps                 commands for the move into the plane (the controller's align_steps, 5;
                               always 1 with no_increments)
  ~plane_tol_mm, ~plane_tol_deg  remaining motion at which the probe is in the plane (0.5, 0.5)
  ~max_step_m, ~max_rotation_deg  as in controller_rosnode.py, but no limit by default (for
                               simulation); set them (0.005, 10.0) on the real robot
  ~image_x_sign                as in controller_rosnode.py, but -1 by default (the simulation
                               mirrors the image)

Requires lwr_controllers (for PoseRPY) in addition to the requirements of controller_rosnode.py.
"""

import bisect
import sys
import threading
from collections import deque

import numpy as np

# ROS Kinetic's rospy still calls the Python 2-era Thread.isAlive() during
# shutdown; Python 3.9+ renamed it to is_alive().
if not hasattr(threading.Thread, "isAlive"):
    threading.Thread.isAlive = threading.Thread.is_alive

import rospy
from geometry_msgs.msg import Twist
from lwr_controllers.msg import PoseRPY
from sensor_msgs.msg import Image

from view_transition_rl.simulation.geometry import euler_from_pose, rotation_xyz
from view_transition_rl.visual_servoing.controller_rosnode import DEPTH_KEYS, PROBE_KEYS, START_KEYS, mask_from_msg
from view_transition_rl.visual_servoing.controllers import (NoIncrementsController, SequentialController,
                                                            contact_motion, rotation_matrix, rotation_vector)

CONTROLLER_TYPES = {"no_increments": NoIncrementsController, "sequential": SequentialController}


def pose_matrix(xyz_mm, roll, pitch, yaw):
    """4x4 pose from a position (mm) and R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (rad), as in PoseRPY."""
    pose = np.eye(4)
    pose[:3, :3] = rotation_xyz(rz=yaw) @ rotation_xyz(ry=pitch) @ rotation_xyz(rx=roll)
    pose[:3, 3] = xyz_mm
    return pose


def pose_from_msg(msg):
    """lwr_controllers/PoseRPY (m, rad) -> 4x4 pose in mm."""
    position, orientation = msg.position, msg.orientation
    return pose_matrix(1000.0 * np.array([position.x, position.y, position.z]),
                       orientation.roll, orientation.pitch, orientation.yaw)


def tooltip_to_probe_from_calibration(ns):
    """Probe frame in the tooltip frame from ultrasound_probe_toolkit's calibration parameters under ns:
    Rot_z(probe_z_rotation_deg) * Trans_y(probe_y_offset), i.e. its T_E_P without probe_length, which
    the tooltip pose already includes."""
    try:
        y_offset_mm = 1000.0 * rospy.get_param(ns + '/probe_y_offset')
        z_rotation = np.deg2rad(rospy.get_param(ns + '/probe_z_rotation_deg'))
    except KeyError as missing:
        raise RuntimeError(f"tooltip_servo: no probe calibration parameter {missing} under {ns}; start "
                           f"probe_scan_motion_node (probe_calibration.yaml) first or set ~calibration_ns") from None
    return pose_matrix([0.0, 0.0, 0.0], 0.0, 0.0, z_rotation) @ pose_matrix([0.0, y_offset_mm, 0.0], 0.0, 0.0, 0.0)


def motion_size(delta):
    """Translation (mm) and rotation angle (rad) of a 4x4 relative motion."""
    return np.linalg.norm(delta[:3, 3]), np.linalg.norm(rotation_vector(delta[:3, :3]))


class TooltipServoNode(object):
    def __init__(self):
        self.mask_topic = rospy.get_param('~mask_topic', '/vessel_segmentation_node/vessel_mask')
        self.action_topic = rospy.get_param('~action_topic', '/probe_action')
        self.cruise_topic = rospy.get_param('~cruise_topic', '/probe_cruise_velocity')
        self.pose_topic = rospy.get_param('~pose_topic', '/cartesian_position/pose/tooltip')
        self.command_period_sec = rospy.get_param('~command_period_sec', 4.0)
        self.settle_sec = rospy.get_param('~settle_sec', 3.0)
        self.max_pose_age_sec = rospy.get_param('~max_pose_age_sec', 0.5)
        self.still_sec = rospy.get_param('~still_sec', 0.5)
        self.still_mm = rospy.get_param('~still_mm', 0.2)
        self.still_rad = np.deg2rad(rospy.get_param('~still_deg', 0.2))
        self.level_tol = np.deg2rad(rospy.get_param('~level_tol_deg', 0.5))
        self.stop_in_plane = rospy.get_param('~stop_in_plane', True)
        self.plane_tol_mm = rospy.get_param('~plane_tol_mm', 0.5)
        self.plane_tol = np.deg2rad(rospy.get_param('~plane_tol_deg', 0.5))
        self.min_vessel_pixels = rospy.get_param('~min_vessel_pixels', 10)
        self.fov_mm = (rospy.get_param('~fov_width_mm', 40.0), rospy.get_param('~fov_depth_mm', 40.0))
        self.image_x_sign = rospy.get_param('~image_x_sign', -1.0)
        # No limits by default, for simulation; on the real robot set e.g. ~max_step_m:=0.005, ~max_rotation_deg:=10.
        self.max_step_mm = 1000.0 * rospy.get_param('~max_step_m', np.inf)
        self.max_rotation = np.deg2rad(rospy.get_param('~max_rotation_deg', np.inf))
        self.probe_step_mm = rospy.get_param('~probe_step_mm', 1.0)
        self.z_step_mm = rospy.get_param('~z_step_mm', 1.0)
        self.dry_run = rospy.get_param('~dry_run', False)
        self.turn_speed = np.deg2rad(rospy.get_param('~turn_speed_deg_s', 10.0))
        if not np.isfinite(self.turn_speed) or self.turn_speed <= 0.0:
            raise ValueError("~turn_speed_deg_s must be finite and greater than zero")
        self.turn_command_period_sec = rospy.get_param('~turn_command_period_sec', 0.5)
        if self.turn_command_period_sec <= 0.0:
            raise ValueError("~turn_command_period_sec must be greater than zero")
        self.calibration_ns = rospy.get_param('~calibration_ns', '/probe_scan_motion_node')
        self.tooltip_to_probe = tooltip_to_probe_from_calibration(self.calibration_ns)
        controller_type = rospy.get_param('~controller_type', 'sequential')
        controller_kwargs = dict(rospy.get_param('~controller', {}))
        controller_kwargs.setdefault('fov_mm', self.fov_mm)

        self.controller = CONTROLLER_TYPES[controller_type](**controller_kwargs)
        self.controller.reset()
        # no_increments: every motion is one command, without corrections.
        self.single_moves = isinstance(self.controller, NoIncrementsController)
        self.plane_steps = 1 if self.single_moves else rospy.get_param('~plane_steps', self.controller.align_steps)
        self._turning = False  # a continuous turn is running: masks are streamed to the controller
        self._turn_stopping = False
        self._turn_velocity = 0.0
        self._last_turn_publish_time = -np.inf
        self._post_turn_action = None
        self._plot_time = -np.inf
        self.running = False  # the controller only runs after Enter
        self.levelled = False  # the controller only runs after levelling
        self._level_error = None  # tooltip roll error before the last levelling command
        self._level_moves = 0  # levelling commands sent so far
        self._plane_target = None  # probe pose (mm) in the fitted image plane, with ~stop_in_plane
        self._plane_error = None  # remaining motion into the plane before the last command
        self._plane_moves = 0  # commands sent towards the plane so far
        self.done = False
        self._start_requested = False
        self._last_command_time = -np.inf
        self._frame_pose = None  # probe pose (mm) of the controller's current frame, set at its first step

        self.view = None
        self._probe_drawn_time = -np.inf
        if rospy.get_param('~show_plot', True):
            from view_transition_rl.simulation.scene import PointCloudView
            self.view = PointCloudView(title="Boundary reconstruction (measured tooltip pose)")
            self.view.set_probe(np.eye(4), self.fov_mm)  # until the first pose arrives
            for key, (dx, dy) in PROBE_KEYS.items():
                self.view.on_key(key, lambda dx=dx, dy=dy: self._user_move(dx, dy))
            for key, direction in DEPTH_KEYS.items():
                self.view.on_key(key, lambda direction=direction: self._user_depth(direction))
            for key in START_KEYS:
                self.view.on_key(key, self._request_toggle)
            self._update_help()
            self.view.show()
        if sys.stdin is not None and sys.stdin.isatty():  # Enter in the terminal works too (rosrun, not roslaunch)
            threading.Thread(target=self._read_terminal, daemon=True).start()

        self._lock = threading.Lock()
        self._pose, self._pose_time = None, -np.inf
        self._rest_pose, self._still_since = None, -np.inf
        self._anchor = None  # first pose received: the window's frame
        self._latest = None  # (mask msg, time, pose, pose time, at rest since) of the latest mask
        self._masks = deque(maxlen=300)  # (mask msg, stamp) of every mask, consumed during a turn
        self._pose_history = deque(maxlen=5000)  # (time, pose) for the pose at a mask's stamp
        self.action_pub = rospy.Publisher(self.action_topic, Twist, queue_size=1)
        self.cruise_pub = rospy.Publisher(self.cruise_topic, Twist, queue_size=1)
        rospy.on_shutdown(self._stop_turn)
        rospy.Subscriber(self.pose_topic, PoseRPY, self._on_pose, queue_size=10)
        rospy.Subscriber(self.mask_topic, Image, self._on_mask, queue_size=1)

        rospy.loginfo(
            "tooltip_servo: %s + %s -> %s every %.1fs (fov %.1fx%.1f mm, x sign %+.0f, max step %.1f mm / %.1f deg%s)",
            self.mask_topic, self.pose_topic, self.action_topic, self.command_period_sec, self.fov_mm[0],
            self.fov_mm[1], self.image_x_sign, self.max_step_mm, np.rad2deg(self.max_rotation),
            ", DRY RUN" if self.dry_run else ""
        )
        rospy.loginfo("tooltip_servo: paused; position the probe (W/A/S/D, R/F), then press Enter to start")

    def _on_pose(self, msg):
        pose = pose_from_msg(msg) @ self.tooltip_to_probe
        now = rospy.get_time()
        with self._lock:
            if self._anchor is None:
                self._anchor = pose
            moved_mm, turned = (np.inf, np.inf) if self._rest_pose is None else \
                motion_size(np.linalg.inv(self._rest_pose) @ pose)
            if moved_mm > self.still_mm or turned > self.still_rad:
                self._rest_pose, self._still_since = pose, now
            self._pose, self._pose_time = pose, now
            self._pose_history.append((now, pose))

    def _on_mask(self, msg):
        now = rospy.get_time()
        with self._lock:
            self._latest = (msg, now, self._pose, self._pose_time, self._still_since)
            self._masks.append((msg, msg.header.stamp.to_sec() or now))

    def _pose_at(self, t):
        """Probe pose at time t, interpolated between the received poses; None if they do not cover t."""
        with self._lock:
            history = list(self._pose_history)
        if not history or t < history[0][0] or t > history[-1][0] + self.max_pose_age_sec:
            return None
        i = bisect.bisect_right([time for time, _ in history], t)
        if i == len(history):
            return history[-1][1]
        (t0, p0), (t1, p1) = history[i - 1], history[i]
        fraction = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        delta = np.linalg.inv(p0) @ p1
        part = np.eye(4)
        part[:3, :3] = rotation_matrix(fraction * rotation_vector(delta[:3, :3]))
        part[:3, 3] = fraction * delta[:3, 3]
        return p0 @ part

    def _read_terminal(self):
        while not rospy.is_shutdown():
            if sys.stdin.readline() == "":
                return  # stdin closed
            self._request_toggle()

    def _request_toggle(self):
        """Enter: start or pause the controller (applied by the main loop)."""
        self._start_requested = True

    def _update_help(self):
        if self.view is None:
            return
        state = ("FINISHED" if self.done else "PAUSED - Enter: start the controller" if not self.running
                 else "LEVELLING - Enter: pause" if not self.levelled
                 else "TURNING, recording every mask - Enter: pause" if self._turning
                 else "MOVING INTO THE FITTED PLANE - Enter: pause" if self._plane_target is not None
                 else "RUNNING - Enter: pause")
        self.view.set_help(f"{state}\n"
                           f"W/A/S/D: move probe {self.probe_step_mm:g} mm along its own y (W/S) and x (A/D)\n"
                           f"R/F: lift / press probe {self.z_step_mm:g} mm along its axis")
        self.view.plotter.render()

    def spin(self):
        """Main thread: controller steps at the command period while running; the plot window
        (its keys and the live probe pose) stays responsive in between."""
        rate = rospy.Rate(30)
        next_command = rospy.get_time()
        while not rospy.is_shutdown():
            if self._start_requested:
                self._start_requested = False
                if not self.done:
                    self.running = not self.running
                    if self._turning:
                        velocity = self._turn_velocity if self.running and not self._turn_stopping else 0.0
                        self._set_turn_velocity(velocity)
                    self._level_error = self._plane_error = None  # the probe may have been moved meanwhile
                    next_command = rospy.get_time()  # first step right away
                    rospy.loginfo("tooltip_servo: controller %s", "started" if self.running else "paused")
                    self._update_help()
            if self.running and not self.done and self._turning:
                stepped = self.record_turn()  # None while the turn runs
                if stepped is not None:
                    next_command = rospy.get_time() + (self.command_period_sec if stepped else 0.2)
            elif self.running and not self.done and rospy.get_time() >= next_command:
                # Retry soon if there was no usable mask or pose yet, otherwise wait a full period.
                stepped = (self.level() if not self.levelled else
                           self.move_to_plane() if self._plane_target is not None else self.step())
                next_command = rospy.get_time() + (self.command_period_sec if stepped else 0.2)
            if self.view is not None and self.view.is_open:
                self._draw_probe()
                self.view.process_events()
            rate.sleep()

    def _pose_at_rest(self, what):
        """Latest pose, or None (logged) while it is not current or the probe is not yet at rest
        after the last command."""
        now = rospy.get_time()
        with self._lock:
            pose, pose_time, still_since = self._pose, self._pose_time, self._still_since
        if pose is None or now - pose_time > self.max_pose_age_sec:
            rospy.logwarn_throttle(10.0, "tooltip_servo: no current pose on %s" % self.pose_topic)
            return None
        if now < self._last_command_time + self.settle_sec or now < still_since + self.still_sec:
            rospy.loginfo_throttle(5.0, "tooltip_servo: %s: waiting for the probe to come to rest" % what)
            return None
        return pose

    def level(self):
        """First step: one command turning the probe towards tooltip roll +-pi and pitch 0 (z axis
        straight down), yaw kept; returns False if it sent nothing (waiting, or levelled)."""
        pose = self._pose_at_rest("levelling")
        if pose is None:
            return False
        tooltip = pose @ np.linalg.inv(self.tooltip_to_probe)
        roll, pitch, yaw = euler_from_pose(tooltip)
        # Rx(pi) == Rx(-pi): the shortest turn to the target goes to whichever of +-pi is closer.
        target = pose_matrix(tooltip[:3, 3], np.pi, 0.0, yaw)
        turn = rotation_vector(tooltip[:3, :3].T @ target[:3, :3])  # tooltip frame
        error = np.linalg.norm(turn)
        rospy.loginfo("tooltip_servo: levelling: tooltip roll %+.1f, pitch %+.1f deg; z axis %.1f deg off straight down",
                      np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(np.arccos(np.clip(-tooltip[2, 2], -1.0, 1.0))))
        if error <= self.level_tol or (self.single_moves and self._level_moves > 0):
            self.levelled = True
            rospy.loginfo("tooltip_servo: levelled (%.2f deg off); starting the controller", np.rad2deg(error))
            self._update_help()
            return False  # controller step right away
        if self._level_error is not None and error >= self._level_error:
            self.running, self._level_error = False, None
            rospy.logerr("tooltip_servo: levelling: remaining turn %.1f deg did not shrink after the last command; "
                         "paused (check ~pose_topic and the probe calibration), Enter to retry", np.rad2deg(error))
            self._update_help()
            return False
        self._level_error = error
        self._level_moves += 1
        # The command is executed in the probe frame.
        self._publish(np.zeros(3), self.tooltip_to_probe[:3, :3].T @ turn, "level")
        if self.dry_run:
            self.levelled = True  # the probe does not move; let the controller run
            self._update_help()
        return True

    def step(self):
        """One controller step; returns False if it had to wait for a usable mask."""
        with self._lock:
            latest = self._latest
        if latest is None:
            rospy.logwarn_throttle(10.0, "tooltip_servo: no mask received yet")
            return False
        mask_msg, mask_time, pose, pose_time, still_since = latest
        if mask_time < self._last_command_time + self.settle_sec:
            rospy.loginfo_throttle(5.0, "tooltip_servo: waiting for a mask taken after the last move")
            return False
        if pose is None or mask_time - pose_time > self.max_pose_age_sec:
            rospy.logwarn_throttle(10.0, "tooltip_servo: no current pose on %s with the mask" % self.pose_topic)
            return False
        if mask_time < still_since + self.still_sec:
            rospy.loginfo_throttle(5.0, "tooltip_servo: waiting for a mask taken with the probe at rest")
            return False

        mask = self._mask_array(mask_msg)
        vessel_pixels = int(np.count_nonzero(mask))
        if vessel_pixels < self.min_vessel_pixels:
            rospy.logwarn("tooltip_servo: vessel not visible enough (%d px); holding", vessel_pixels)
            return True

        self._move_controller_to(pose)
        action = self.controller(mask)
        self._update_plot()
        return self._use(action)

    def record_turn(self):
        """During a continuous turn: give the controller every new mask, each with the probe pose at its
        stamp. Returns None while the turn runs, then what _use() returns for the controller's next action."""
        if self._turn_stopping:
            if self._pose_at_rest("turn stop") is None:
                return None
            self._turning = self._turn_stopping = False
            action, self._post_turn_action = self._post_turn_action, None
            rospy.loginfo("tooltip_servo: turn stopped; %d frames recorded", len(self.controller.cloud))
            self._update_help()
            return self._use(action)

        # probe_scan_motion_node rejects cruise commands while a relative step is
        # STEPPING/SETTLING. Keep sending the requested velocity until it becomes
        # idle; this also makes the cruise command robust to a dropped ROS message.
        if rospy.get_time() >= self._last_turn_publish_time + self.turn_command_period_sec:
            self._set_turn_velocity(self._turn_velocity, heartbeat=True)

        with self._lock:
            masks = list(self._masks)
            self._masks.clear()
        for mask_msg, stamp in masks:
            pose = self._pose_at(stamp)
            if pose is None:
                rospy.logwarn_throttle(5.0, "tooltip_servo: turn: no pose around a mask's stamp; mask skipped")
                continue
            mask = self._mask_array(mask_msg)
            if np.count_nonzero(mask) < self.min_vessel_pixels:
                rospy.logwarn_throttle(5.0, "tooltip_servo: turn: vessel not visible enough; mask skipped")
                continue
            self._move_controller_to(pose, quiet=True)
            action = self.controller(mask)
            if self.controller.phase_name != "turn":  # the controller ended the turn and handed over
                self._post_turn_action = action
                self._turn_stopping = True
                self._set_turn_velocity(0.0)
                rospy.loginfo("tooltip_servo: requested turn reached; stopping angular cruise")
                self._update_plot()
                self._update_help()
                return None
        if rospy.get_time() > self._plot_time + 0.5:
            self._update_plot()
            self._plot_time = rospy.get_time()
        return None

    def _mask_array(self, mask_msg):
        mask = mask_from_msg(mask_msg)
        return mask[:, ::-1] if self.image_x_sign < 0 else mask  # columns along probe +X, as the controller expects

    def _use(self, action):
        """Act on the controller's action; returns False if the next step should follow right away."""
        if self.controller.phase_name == "turn" and not self._turning and action is not None:
            # Stream a constant probe-frame angular velocity; measured poses determine
            # when the requested turn has been reached.
            with self._lock:
                self._masks.clear()
            self._turn_velocity = np.copysign(self.turn_speed, action[4])
            self._turning = True
            self._turn_stopping = False
            self._post_turn_action = None
            self._set_turn_velocity(self._turn_velocity)
            self._update_help()
            return True
        if self.stop_in_plane and self.controller.phase_name not in ("centre", "rotate", "turn"):
            # The controller has just fitted the cylinder and planned its alignment (or failed to):
            # move straight into the fitted plane instead of its align, centre_again and track phases.
            if self.controller.align_target is None:
                self.done = True
                rospy.logwarn("tooltip_servo: no fitted plane to move into (see the controller's align output); "
                              "stopping")
            else:
                self._plane_target = self._frame_pose @ self.controller.align_target
                self._plane_moves = 0
                rospy.loginfo("tooltip_servo: plane fitted; moving into it")
            self._update_help()
            return False  # first move right away
        if action is None:
            self.done = True
            rospy.loginfo("tooltip_servo: sequence finished")
            self._update_help()
        elif np.any(action):  # a zero action has nothing to send
            self._send(action, self.controller.phase_name)
        return True

    def _set_turn_velocity(self, angular_z, heartbeat=False):
        """Command probe-frame angular velocity on the motion node's cruise channel."""
        msg = Twist()
        msg.angular.z = float(angular_z)
        if not self.dry_run:
            self.cruise_pub.publish(msg)
        now = rospy.get_time()
        self._last_turn_publish_time = now
        if not heartbeat:
            self._last_command_time = now
            rospy.loginfo("tooltip_servo [turn]: %s angular velocity %.2f deg/s",
                          "logged" if self.dry_run else "sent", np.rad2deg(angular_z))

    def _stop_turn(self):
        if getattr(self, '_turning', False) and not self.dry_run:
            msg = Twist()
            self.cruise_pub.publish(msg)

    def move_to_plane(self):
        """After the fit (~stop_in_plane): one command moving the probe towards the planned pose in the
        fitted image plane, closed loop on the measured pose; returns False if it sent nothing."""
        pose = self._pose_at_rest("moving into the plane")
        if pose is None:
            return False
        remaining = np.linalg.inv(pose) @ self._plane_target  # probe frame
        moved_mm, turned = motion_size(remaining)
        error = max(moved_mm / self.plane_tol_mm, turned / self.plane_tol)  # <= 1: within the tolerances
        rospy.loginfo("tooltip_servo: plane: %.2f mm and %.2f deg to go", moved_mm, np.rad2deg(turned))
        if error <= 1.0 or (self.single_moves and self._plane_moves > 0):
            self.done = True
            rospy.loginfo("tooltip_servo: in the fitted plane (%.2f mm, %.2f deg off); finished",
                          moved_mm, np.rad2deg(turned))
            self._update_help()
            return True
        if self._plane_error is not None and error >= self._plane_error:
            self.running, self._plane_error = False, None
            rospy.logerr("tooltip_servo: moving into the plane: no progress after the last command; paused "
                         "(check ~pose_topic and the probe calibration), Enter to retry")
            self._update_help()
            return False
        self._plane_error = error
        # Like the controller's align phase: 1 / (commands left) of the remaining motion; the last
        # planned command, and any correction after it, covers all of it.
        fraction = 1.0 / max(1, self.plane_steps - self._plane_moves)
        self._plane_moves += 1
        self._publish(fraction * remaining[:3, 3], fraction * rotation_vector(remaining[:3, :3]), "plane")
        if self.dry_run:
            self.done = True  # the probe does not move
            self._update_help()
        return True

    def _move_controller_to(self, pose, quiet=False):
        """Report the measured motion since the controller's last step; its reconstruction then
        lives in the probe frame at pose. quiet: log at debug level (masks streamed during a turn)."""
        if self._frame_pose is not None:
            delta = np.linalg.inv(self._frame_pose) @ pose  # probe frame before the move
            self.controller.apply_motion(delta)
            moved_mm, turned = motion_size(delta)
            (rospy.logdebug if quiet else rospy.loginfo)(
                "tooltip_servo: measured motion since the last step: %.2f mm, %.2f deg", moved_mm, np.rad2deg(turned))
        self._frame_pose = pose

    def _robot_busy(self):
        if rospy.get_time() < self._last_command_time + self.settle_sec:
            rospy.loginfo("tooltip_servo: robot still busy with the last move; key ignored")
            return True
        return False

    def _user_move(self, dx, dy):
        if not self._robot_busy():
            self._send([dx * self.probe_step_mm, dy * self.probe_step_mm, 0.0, 0.0, 0.0], "user")

    def _user_depth(self, direction):
        """Move along the probe axis."""
        if not self._robot_busy():
            self._publish(np.array([0.0, 0.0, direction * self.z_step_mm]), np.zeros(3), "user")

    def _send(self, action, source):
        """Publish a controller-style action as the motion the controller assumes (with the slide back
        onto the skin), within the limits. Not reported to the controller: the measured pose at its
        next step accounts for it."""
        delta = contact_motion(action, self.controller.surface_normal)  # probe frame, mm
        self._publish(delta[:3, 3], rotation_vector(delta[:3, :3]), source)

    def _publish(self, translation, rotation, source):
        """Publish one relative probe-frame motion (translation in mm, rotation vector in rad),
        scaled down to the limits; returns what was sent. Twist.angular gets ZYX angles (rx, ry, rz),
        R = Rz(rz) @ Ry(ry) @ Rx(rx), as probe_scan_motion_node reads them."""
        scale = min(1.0, self.max_step_mm / max(np.linalg.norm(translation), 1e-12),
                    self.max_rotation / max(np.linalg.norm(rotation), 1e-12))
        translation, rotation = scale * translation, scale * rotation
        angles = euler_from_pose(rotation_matrix(rotation))  # (rx, ry, rz)
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = (translation / 1000.0).tolist()
        twist.angular.x, twist.angular.y, twist.angular.z = angles.tolist()
        if not self.dry_run:
            self.action_pub.publish(twist)
        self._last_command_time = rospy.get_time()
        rospy.loginfo(
            "tooltip_servo [%s]: %s linear (%.2f, %.2f, %.2f) mm, angular ZYX (%.2f, %.2f, %.2f) deg%s",
            source, "logged" if self.dry_run else "sent", *translation,
            *np.rad2deg(angles), "" if scale == 1.0 else f" (scaled to {scale:.2f} by the limits)"
        )
        return translation, rotation

    def _draw_probe(self):
        """Draw the latest tooltip pose in the window frame, at most 10 times a second."""
        with self._lock:
            pose, pose_time, anchor = self._pose, self._pose_time, self._anchor
        if pose is None or pose_time < self._probe_drawn_time + 0.1:
            return
        self.view.set_probe(np.linalg.inv(anchor) @ pose, self.fov_mm)
        self._probe_drawn_time = pose_time

    def _update_plot(self):
        if self.view is None or not self.view.is_open:
            return
        from view_transition_rl.visual_servoing.run import show_controller_fits, to_frame
        transform = np.linalg.inv(self._anchor) @ self._frame_pose  # controller's probe frame -> window frame
        if not show_controller_fits(self.view, self.controller, transform):  # no fit yet: all recorded points
            cloud = self.controller.cloud
            self.view.draw_points(to_frame(self.controller.point_cloud, transform),
                                  np.repeat(np.arange(len(cloud)), [len(c) for c in cloud]))


def main():
    rospy.init_node('tooltip_servo_node')
    TooltipServoNode().spin()


if __name__ == '__main__':
    main()
