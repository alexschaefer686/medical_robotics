#!/usr/bin/env python3
"""Run the SequentialController on the real robot over ROS1, with a live reconstruction plot.

Bridges vessel_segmentation and probe_scan_motion_node: read the vessel mask, let
visual_servoing.controllers.SequentialController decide the next move (centre ->
rotate -> align into the fitted plane -> centre again -> track), and publish it as
one relative probe-frame motion on /probe_action. No simulator and no probe pose
are used.

Once every ~command_period_sec (deliberately paced slower than
probe_scan_motion.launch's step_duration, since probe_scan_motion_node ignores
new /probe_action commands while a step is still executing and gives the sender
no feedback about that):

  1. Take the most recent mask (sensor_msgs/Image, mono8), received at least
     ~settle_sec after the previous command so the probe has stopped moving.
  2. Give it to the controller, which returns a relative motion
     [dx_mm, dy_mm, rx, ry, rz]: move along the probe's own x/y, then rotate by
     Rx(rx) Ry(ry) Rz(rz) (see controllers.py).
  3. Publish it as a geometry_msgs/Twist in the probe frame: linear = (dx, dy, 0)
     in metres, angular = the rotation as a rotation vector (axis * angle) in
     radians. Columns of the mask map to the probe's local +X axis and rows to
     +Z (into the tissue), per ROBOTIC_3D_VESSEL_RECONSTRUCTION_CRUISE.md.
  4. Report the sent motion to the controller (controller.moved), which keeps its
     point cloud, cylinder and plane fits in the current probe frame.

There is no pose estimate: the reconstruction lives in the probe frame and is
moved by each reported motion. The controller assumes the robot keeps skin
contact by sliding along the probe's own z axis after each motion, and tracks
the skin normal in the probe frame from its own rotations (the probe is assumed
perpendicular to the skin at node start; that defines "roll = 0"). The
reconstruction window shows the probe frame: the probe stays at the origin and
the reconstruction moves around it.

W/A/S/D in the reconstruction window send user moves of ~probe_step_mm along the
probe's y (W/S) and x (A/D) through the same path (limits apply, and they are
reported to the controller, so they are known motions). Keys pressed within
~settle_sec of the previous command are ignored, as the robot would drop them.

IMPORTANT: there is no real image-plane calibration (probe_to_image is an
uncalibrated placeholder elsewhere in the source workspace too), so ~fov_width_mm,
~fov_depth_mm and ~image_x_sign below are unverified starting guesses. Before
trusting this near real tissue: start with ~dry_run, then with small limits
(~max_step_m, ~max_rotation_deg), watch the first centring moves, and flip
~image_x_sign if they move the vessel further off-centre instead of recentring it.
Check that probe_scan_motion_node reads Twist.angular as a rotation vector in the
probe frame. Motions larger than the limits are scaled down (direction kept) and
logged; the controller is told what was actually sent. There is no completion
feedback and no obstacle/limit awareness beyond probe_scan_motion_node's own clamps.

Parameters (private):
  ~mask_topic, ~action_topic   as before
  ~command_period_sec          seconds between commands (4.0)
  ~settle_sec                  minimum mask age after the previous command (3.0)
  ~min_vessel_pixels           hold while fewer vessel pixels are visible (50)
  ~fov_width_mm, ~fov_depth_mm physical size of the mask image (40.0, 40.0), uncalibrated
  ~image_x_sign                +1: mask columns increase along probe +X; -1: mirrored mask
  ~max_step_m                  max translation per command (0.005)
  ~max_rotation_deg            max rotation per command (10.0)
  ~probe_step_mm               W/A/S/D step (1.0)
  ~controller                  dict of SequentialController keyword arguments, e.g.
                               {rotate_deg: 90, rotate_steps: 18, align_steps: 5, track: true}
  ~show_plot                   open the reconstruction window (True; needed for W/A/S/D)
  ~dry_run                     log the motions instead of publishing them (False)

Requires numpy and this package (view_transition_rl, Python >= 3.10) with its "gui"
extra for the plot, in addition to the normal rospy deps.
"""

import threading

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

from view_transition_rl.visual_servoing.controllers import (SequentialController, relative_motion, rotation_matrix,
                                                            rotation_vector, xyz_angles)

# Key -> user move direction along the probe's own (x, y).
PROBE_KEYS = {"a": (-1, 0), "d": (1, 0), "w": (0, 1), "s": (0, -1)}


def mask_from_msg(msg):
    """sensor_msgs/Image (mono8) -> (H, W) uint8 array, honouring the row stride."""
    if msg.encoding not in ("mono8", "8UC1"):
        rospy.logwarn_throttle(10.0, "sequential_servo: expected a mono8 mask, got %s", msg.encoding)
    return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width]


class SequentialServoNode(object):
    def __init__(self):
        self.mask_topic = rospy.get_param('~mask_topic', '/vessel_segmentation_node/vessel_mask')
        self.action_topic = rospy.get_param('~action_topic', '/probe_action')
        self.command_period_sec = rospy.get_param('~command_period_sec', 4.0)
        self.settle_sec = rospy.get_param('~settle_sec', 3.0)
        self.min_vessel_pixels = rospy.get_param('~min_vessel_pixels', 50)
        self.fov_mm = (rospy.get_param('~fov_width_mm', 40.0), rospy.get_param('~fov_depth_mm', 40.0))
        self.image_x_sign = rospy.get_param('~image_x_sign', 1.0)
        self.max_step_mm = 1000.0 * rospy.get_param('~max_step_m', 0.005)
        self.max_rotation = np.deg2rad(rospy.get_param('~max_rotation_deg', 10.0))
        self.probe_step_mm = rospy.get_param('~probe_step_mm', 1.0)
        self.dry_run = rospy.get_param('~dry_run', False)
        controller_kwargs = dict(rospy.get_param('~controller', {}))
        controller_kwargs.setdefault('fov_mm', self.fov_mm)

        self.controller = SequentialController(**controller_kwargs)
        self.controller.reset()
        self.done = False
        self._last_command_time = -np.inf

        self.view = None
        if rospy.get_param('~show_plot', True):
            from view_transition_rl.simulation.scene import PointCloudView
            self.view = PointCloudView(title="Boundary reconstruction (probe frame)")
            self.view.set_probe(np.eye(4), self.fov_mm)  # the probe is always at the origin of its frame
            for key, (dx, dy) in PROBE_KEYS.items():
                self.view.on_key(key, lambda dx=dx, dy=dy: self._user_move(dx, dy))
            self.view.set_help(f"W/A/S/D: move probe {self.probe_step_mm:g} mm along its own y (W/S) and x (A/D)")
            self.view.show()

        self._lock = threading.Lock()
        self._latest_mask, self._mask_time = None, -np.inf
        self.action_pub = rospy.Publisher(self.action_topic, Twist, queue_size=1)
        rospy.Subscriber(self.mask_topic, Image, self._on_mask, queue_size=1)

        rospy.loginfo(
            "sequential_servo: %s -> %s every %.1fs (fov %.1fx%.1f mm, x sign %+.0f, max step %.1f mm / %.1f deg%s)",
            self.mask_topic, self.action_topic, self.command_period_sec, self.fov_mm[0], self.fov_mm[1],
            self.image_x_sign, self.max_step_mm, np.rad2deg(self.max_rotation), ", DRY RUN" if self.dry_run else ""
        )

    def _on_mask(self, msg):
        with self._lock:
            self._latest_mask, self._mask_time = msg, rospy.get_time()

    def spin(self):
        """Main thread: controller steps at the command period; the plot window (and its keys)
        stays responsive in between."""
        rate = rospy.Rate(30)
        next_command = rospy.get_time()
        while not rospy.is_shutdown():
            if not self.done and rospy.get_time() >= next_command:
                # Retry soon if there was no usable mask yet, otherwise wait a full period.
                next_command = rospy.get_time() + (self.command_period_sec if self.step() else 0.2)
            if self.view is not None and self.view.is_open:
                self.view.process_events()
            rate.sleep()

    def step(self):
        """One controller step; returns False if it had to wait for a usable mask."""
        with self._lock:
            mask_msg, mask_time = self._latest_mask, self._mask_time
        if mask_msg is None:
            rospy.logwarn_throttle(10.0, "sequential_servo: no mask received yet")
            return False
        if mask_time < self._last_command_time + self.settle_sec:
            rospy.loginfo_throttle(5.0, "sequential_servo: waiting for a mask taken after the last move")
            return False

        mask = mask_from_msg(mask_msg)
        if self.image_x_sign < 0:
            mask = mask[:, ::-1]  # so that columns increase along probe +X, as the controller expects
        vessel_pixels = int(np.count_nonzero(mask))
        if vessel_pixels < self.min_vessel_pixels:
            rospy.logwarn("sequential_servo: vessel not visible enough (%d px); holding", vessel_pixels)
            return True

        action = self.controller(mask)
        if action is None:
            self.done = True
            rospy.loginfo("sequential_servo: sequence finished")
        else:
            self._send(action, self.controller.phase_name)
        return True

    def _user_move(self, dx, dy):
        if rospy.get_time() < self._last_command_time + self.settle_sec:
            rospy.loginfo("sequential_servo: robot still busy with the last move; key ignored")
            return
        self._send([dx * self.probe_step_mm, dy * self.probe_step_mm, 0.0, 0.0, 0.0], "user")

    def _send(self, action, source):
        """Publish a relative probe motion (within the limits), report it to the controller, redraw."""
        translation = np.array([action[0], action[1], 0.0])  # mm, probe frame
        rotation = rotation_vector(relative_motion(action)[:3, :3])
        scale = min(1.0, self.max_step_mm / max(np.linalg.norm(translation), 1e-12),
                    self.max_rotation / max(np.linalg.norm(rotation), 1e-12))
        translation, rotation = scale * translation, scale * rotation

        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = (translation / 1000.0).tolist()
        twist.angular.x, twist.angular.y, twist.angular.z = rotation.tolist()
        if not self.dry_run:
            self.action_pub.publish(twist)
        self._last_command_time = rospy.get_time()
        self.controller.moved([translation[0], translation[1], *xyz_angles(rotation_matrix(rotation))])

        rospy.loginfo(
            "sequential_servo [%s]: %s linear (%.2f, %.2f) mm, angular (%.2f, %.2f, %.2f) deg%s",
            source, "logged" if self.dry_run else "sent", translation[0], translation[1],
            *np.rad2deg(rotation), "" if scale == 1.0 else f" (scaled to {scale:.2f} by the limits)"
        )
        self._update_plot()

    def _update_plot(self):
        if self.view is None or not self.view.is_open:
            return
        from view_transition_rl.visual_servoing.run import show_controller_fits
        if not show_controller_fits(self.view, self.controller):  # no fit yet: all recorded points
            cloud = self.controller.cloud
            self.view.draw_points(self.controller.point_cloud, np.repeat(np.arange(len(cloud)), [len(c) for c in cloud]))


def main():
    rospy.init_node('sequential_servo_node')
    SequentialServoNode().spin()


if __name__ == '__main__':
    main()
