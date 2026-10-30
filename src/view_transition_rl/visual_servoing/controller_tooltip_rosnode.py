#!/usr/bin/env python3
"""Run the SequentialController on the real robot over ROS1, reconstructing from the measured tooltip pose.

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
  3. Motions (controller and keys) are published as before but not reported to the
     controller; the measured pose at its next step accounts for whatever the robot
     actually did.

When started (first Enter), the node first levels the probe, before the controller
runs: it turns the probe about its contact point until the tooltip z axis points
straight down (pose frame -z), i.e. tooltip roll +-pi (whichever is closer: the same
orientation, reached by the shortest turn) and pitch 0, yaw kept, within
~level_tol_deg. This is closed loop on ~pose_topic, one command (within
~max_rotation_deg) per ~command_period_sec once the probe is at rest; if a command does
not reduce the remaining turn, the node pauses. With ~dry_run the first correction is
only logged and the controller starts right after.

The controller's probe frame (origin = contact point, x = image columns, z = depth,
which is also the frame probe_scan_motion_node executes the Twist commands in) is the
tooltip frame composed with ~tooltip_to_probe, identity by default. Set it if the
tooltip frame differs from that frame, e.g. by the z rotation and y offset of
ultrasound_probe_toolkit's probe_calibration.yaml: a 7.75 mm offset alone puts
frames taken 180 degrees apart about the probe axis up to 15.5 mm off.

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
  ~tooltip_to_probe            the controller's probe frame in the tooltip frame,
                               {xyz_mm: [x, y, z], rpy_deg: [roll, pitch, yaw]} (identity)
  ~max_pose_age_sec            maximum age of the pose stored with a mask (0.5)
  ~still_sec                   how long the probe must have been at rest before a mask (0.5)
  ~still_mm, ~still_deg        motion that still counts as at rest (0.2, 0.2)
  ~level_tol_deg               remaining turn at which the levelling ends (0.5)
  ~max_step_m, ~max_rotation_deg  as in controller_rosnode.py, but no limit by default (for
                               simulation); set them (0.005, 10.0) on the real robot

Requires lwr_controllers (for PoseRPY) in addition to the requirements of controller_rosnode.py.
"""

import sys
import threading

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from lwr_controllers.msg import PoseRPY
from sensor_msgs.msg import Image

from view_transition_rl.simulation.geometry import euler_from_pose, rotation_xyz
from view_transition_rl.visual_servoing.controller_rosnode import DEPTH_KEYS, PROBE_KEYS, START_KEYS, mask_from_msg
from view_transition_rl.visual_servoing.controllers import SequentialController, relative_motion, rotation_vector


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


def motion_size(delta):
    """Translation (mm) and rotation angle (rad) of a 4x4 relative motion."""
    return np.linalg.norm(delta[:3, 3]), np.linalg.norm(rotation_vector(delta[:3, :3]))


class TooltipServoNode(object):
    def __init__(self):
        self.mask_topic = rospy.get_param('~mask_topic', '/vessel_segmentation_node/vessel_mask')
        self.action_topic = rospy.get_param('~action_topic', '/probe_action')
        self.pose_topic = rospy.get_param('~pose_topic', '/cartesian_position/pose/tooltip')
        self.command_period_sec = rospy.get_param('~command_period_sec', 4.0)
        self.settle_sec = rospy.get_param('~settle_sec', 3.0)
        self.max_pose_age_sec = rospy.get_param('~max_pose_age_sec', 0.5)
        self.still_sec = rospy.get_param('~still_sec', 0.5)
        self.still_mm = rospy.get_param('~still_mm', 0.2)
        self.still_rad = np.deg2rad(rospy.get_param('~still_deg', 0.2))
        self.level_tol = np.deg2rad(rospy.get_param('~level_tol_deg', 0.5))
        self.min_vessel_pixels = rospy.get_param('~min_vessel_pixels', 10)
        self.fov_mm = (rospy.get_param('~fov_width_mm', 40.0), rospy.get_param('~fov_depth_mm', 40.0))
        self.image_x_sign = rospy.get_param('~image_x_sign', 1.0)
        # No limits by default, for simulation; on the real robot set e.g. ~max_step_m:=0.005, ~max_rotation_deg:=10.
        self.max_step_mm = 1000.0 * rospy.get_param('~max_step_m', np.inf)
        self.max_rotation = np.deg2rad(rospy.get_param('~max_rotation_deg', np.inf))
        self.probe_step_mm = rospy.get_param('~probe_step_mm', 1.0)
        self.z_step_mm = rospy.get_param('~z_step_mm', 1.0)
        self.dry_run = rospy.get_param('~dry_run', False)
        offset = dict(rospy.get_param('~tooltip_to_probe', {}))
        self.tooltip_to_probe = pose_matrix(offset.get('xyz_mm', [0.0, 0.0, 0.0]),
                                            *np.deg2rad(offset.get('rpy_deg', [0.0, 0.0, 0.0])))
        controller_kwargs = dict(rospy.get_param('~controller', {}))
        controller_kwargs.setdefault('fov_mm', self.fov_mm)

        self.controller = SequentialController(**controller_kwargs)
        self.controller.reset()
        self.running = False  # the controller only runs after Enter
        self.levelled = False  # the controller only runs after levelling
        self._level_error = None  # tooltip roll error before the last levelling command
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
        self.action_pub = rospy.Publisher(self.action_topic, Twist, queue_size=1)
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

    def _on_mask(self, msg):
        with self._lock:
            self._latest = (msg, rospy.get_time(), self._pose, self._pose_time, self._still_since)

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
        state = ("FINISHED" if self.done else "LEVELLING - Enter: pause" if self.running and not self.levelled
                 else "RUNNING - Enter: pause" if self.running else "PAUSED - Enter: start the controller")
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
                    self._level_error = None  # the probe may have been moved meanwhile
                    next_command = rospy.get_time()  # first step right away
                    rospy.loginfo("tooltip_servo: controller %s", "started" if self.running else "paused")
                    self._update_help()
            if self.running and not self.done and rospy.get_time() >= next_command:
                # Retry soon if there was no usable mask or pose yet, otherwise wait a full period.
                stepped = self.step() if self.levelled else self.level()
                next_command = rospy.get_time() + (self.command_period_sec if stepped else 0.2)
            if self.view is not None and self.view.is_open:
                self._draw_probe()
                self.view.process_events()
            rate.sleep()

    def level(self):
        """First step: one command turning the probe towards tooltip roll +-pi and pitch 0 (z axis
        straight down), yaw kept; returns False if it sent nothing (waiting, or levelled)."""
        now = rospy.get_time()
        with self._lock:
            pose, pose_time, still_since = self._pose, self._pose_time, self._still_since
        if pose is None or now - pose_time > self.max_pose_age_sec:
            rospy.logwarn_throttle(10.0, "tooltip_servo: no current pose on %s", self.pose_topic)
            return False
        if now < self._last_command_time + self.settle_sec or now < still_since + self.still_sec:
            rospy.loginfo_throttle(5.0, "tooltip_servo: levelling: waiting for the probe to come to rest")
            return False
        tooltip = pose @ np.linalg.inv(self.tooltip_to_probe)
        roll, pitch, yaw = euler_from_pose(tooltip)
        # Rx(pi) == Rx(-pi): the shortest turn to the target goes to whichever of +-pi is closer.
        target = pose_matrix(tooltip[:3, 3], np.pi, 0.0, yaw)
        turn = rotation_vector(tooltip[:3, :3].T @ target[:3, :3])  # tooltip frame
        error = np.linalg.norm(turn)
        rospy.loginfo("tooltip_servo: levelling: tooltip roll %+.1f, pitch %+.1f deg; z axis %.1f deg off straight down",
                      np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(np.arccos(np.clip(-tooltip[2, 2], -1.0, 1.0))))
        if error <= self.level_tol:
            self.levelled = True
            rospy.loginfo("tooltip_servo: levelled; starting the controller")
            self._update_help()
            return False  # controller step right away
        if self._level_error is not None and error >= self._level_error:
            self.running, self._level_error = False, None
            rospy.logerr("tooltip_servo: levelling: remaining turn %.1f deg did not shrink after the last command; "
                         "paused (check ~pose_topic and ~tooltip_to_probe), Enter to retry", np.rad2deg(error))
            self._update_help()
            return False
        self._level_error = error
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
            rospy.logwarn_throttle(10.0, "tooltip_servo: no current pose on %s with the mask", self.pose_topic)
            return False
        if mask_time < still_since + self.still_sec:
            rospy.loginfo_throttle(5.0, "tooltip_servo: waiting for a mask taken with the probe at rest")
            return False

        mask = mask_from_msg(mask_msg)
        if self.image_x_sign < 0:
            mask = mask[:, ::-1]  # so that columns increase along probe +X, as the controller expects
        vessel_pixels = int(np.count_nonzero(mask))
        if vessel_pixels < self.min_vessel_pixels:
            rospy.logwarn("tooltip_servo: vessel not visible enough (%d px); holding", vessel_pixels)
            return True

        self._move_controller_to(pose)
        action = self.controller(mask)
        self._update_plot()
        if action is None:
            self.done = True
            rospy.loginfo("tooltip_servo: sequence finished")
            self._update_help()
        else:
            self._send(action, self.controller.phase_name)
        return True

    def _move_controller_to(self, pose):
        """Report the measured motion since the controller's last step; its reconstruction then
        lives in the probe frame at pose."""
        if self._frame_pose is not None:
            delta = np.linalg.inv(self._frame_pose) @ pose  # probe frame before the move
            self.controller.apply_motion(delta)
            moved_mm, turned = motion_size(delta)
            rospy.loginfo("tooltip_servo: measured motion since the last step: %.2f mm, %.2f deg",
                          moved_mm, np.rad2deg(turned))
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
        """Publish a controller-style action (within the limits). Not reported to the controller:
        the measured pose at its next step accounts for it."""
        translation = np.array([action[0], action[1], 0.0])  # mm, probe frame
        self._publish(translation, rotation_vector(relative_motion(action)[:3, :3]), source)

    def _publish(self, translation, rotation, source):
        """Publish one relative probe-frame motion (translation in mm, rotation vector in rad),
        scaled down to the limits; returns what was sent."""
        scale = min(1.0, self.max_step_mm / max(np.linalg.norm(translation), 1e-12),
                    self.max_rotation / max(np.linalg.norm(rotation), 1e-12))
        translation, rotation = scale * translation, scale * rotation
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = (translation / 1000.0).tolist()
        twist.angular.x, twist.angular.y, twist.angular.z = rotation.tolist()
        if not self.dry_run:
            self.action_pub.publish(twist)
        self._last_command_time = rospy.get_time()
        rospy.loginfo(
            "tooltip_servo [%s]: %s linear (%.2f, %.2f, %.2f) mm, angular (%.2f, %.2f, %.2f) deg%s",
            source, "logged" if self.dry_run else "sent", *translation,
            *np.rad2deg(rotation), "" if scale == 1.0 else f" (scaled to {scale:.2f} by the limits)"
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
