#!/usr/bin/env python
"""Simple mask-based vessel centring and longitudinal-view control.

Every ``~command_period_sec`` the node applies one direct rule:

1. If the vessel is not horizontally centred, move along probe X.
2. Otherwise, measure the vessel mask's bounding-box width / image width.
3. Rotate by a large Z step when the ratio is below ``~far_width_ratio``.
4. Rotate by a small Z step until ``~longitudinal_width_ratio`` is reached.

The node starts controlling automatically as soon as it receives a valid mask.
"""

from __future__ import print_function

import math
import threading

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image


def mask_from_msg(msg):
    """Convert a mono8 Image to a 2-D array while honouring row padding."""
    if msg.height <= 0 or msg.width <= 0 or msg.step < msg.width:
        raise ValueError("invalid image dimensions or row stride")
    required = int(msg.height) * int(msg.step)
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if data.size < required:
        raise ValueError("image data is shorter than height * step")
    return data[:required].reshape(int(msg.height), int(msg.step))[:, :int(msg.width)]


def vessel_measurements(mask):
    """Return vessel pixel count, horizontal centroid, and bounding-box width ratio."""
    _rows, cols = np.nonzero(np.asarray(mask) != 0)
    if cols.size == 0:
        return 0, None, 0.0
    bounding_width = int(cols.max()) - int(cols.min()) + 1
    return int(cols.size), float(cols.mean()), float(bounding_width) / float(mask.shape[1])

class VesselTrackingExample(object):
    def __init__(self):
        self.mask_topic = rospy.get_param('~mask_topic', '/vessel_segmentation_node/vessel_mask')
        self.action_topic = rospy.get_param('~action_topic', '/probe_action')

        self.gain_m_per_px = float(rospy.get_param('~gain_m_per_px', 0.0000833333))
        self.direction_sign = float(rospy.get_param('~direction_sign', -1.0))
        self.max_step_m = float(rospy.get_param('~max_step_m', 0.005))
        self.deadband_px = float(rospy.get_param('~deadband_px', 10.0))
        self.min_vessel_pixels = int(rospy.get_param('~min_vessel_pixels', 50))
        self.command_period_sec = float(rospy.get_param('~command_period_sec', 4.0))
        self.mask_timeout_sec = float(rospy.get_param('~mask_timeout_sec', 1.0))

        self.angular_direction = float(rospy.get_param('~angular_direction', 1.0))
        self.far_width_ratio = float(rospy.get_param('~far_width_ratio', 0.60))
        self.longitudinal_width_ratio = float(rospy.get_param('~longitudinal_width_ratio', 0.95))
        self.large_yaw_step = math.radians(float(rospy.get_param('~large_yaw_step_deg', 10.0)))
        self.small_yaw_step = math.radians(float(rospy.get_param('~small_yaw_step_deg', 2.0)))

        self._validate_parameters()

        self._lock = threading.Lock()
        self._latest_mask = None
        self._mask_time = None

        self.action_pub = rospy.Publisher(self.action_topic, Twist, queue_size=1)
        rospy.Subscriber(self.mask_topic, Image, self._on_mask, queue_size=1)
        rospy.Timer(rospy.Duration(self.command_period_sec), self._on_timer)

        rospy.loginfo(
            "vessel_tracking_example_2: automatic control %s -> %s every %.1fs; "
            "width ratios far=%.2f longitudinal=%.2f; yaw steps %.1f/%.1f deg; direction=%+.0f",
            self.mask_topic, self.action_topic, self.command_period_sec,
            self.far_width_ratio, self.longitudinal_width_ratio,
            math.degrees(self.large_yaw_step), math.degrees(self.small_yaw_step),
            self.angular_direction)

    def _validate_parameters(self):
        positive = {
            'gain_m_per_px': self.gain_m_per_px,
            'max_step_m': self.max_step_m,
            'deadband_px': self.deadband_px,
            'command_period_sec': self.command_period_sec,
            'mask_timeout_sec': self.mask_timeout_sec,
            'large_yaw_step_deg': self.large_yaw_step,
            'small_yaw_step_deg': self.small_yaw_step,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("~%s must be finite and positive" % name)
        if self.direction_sign not in (-1.0, 1.0):
            raise ValueError("~direction_sign must be -1 or +1")
        if self.angular_direction not in (-1.0, 1.0):
            raise ValueError("~angular_direction must be -1 or +1")
        if self.min_vessel_pixels <= 0:
            raise ValueError("~min_vessel_pixels must be positive")
        if not (0.0 < self.far_width_ratio < self.longitudinal_width_ratio <= 1.0):
            raise ValueError(
                "width ratios must satisfy 0 < ~far_width_ratio < ~longitudinal_width_ratio <= 1")
        if self.small_yaw_step >= self.large_yaw_step:
            raise ValueError("~small_yaw_step_deg must be smaller than ~large_yaw_step_deg")

    def _on_mask(self, msg):
        with self._lock:
            self._latest_mask = msg
            self._mask_time = rospy.get_time()

    def _on_timer(self, _event):
        now = rospy.get_time()
        with self._lock:
            mask_msg = self._latest_mask
            mask_time = self._mask_time

        if mask_msg is None or mask_time is None:
            rospy.logwarn_throttle(10.0, "vessel_tracking_example_2: no mask received; holding")
            return
        if now - mask_time > self.mask_timeout_sec:
            rospy.logwarn_throttle(5.0, "vessel_tracking_example_2: latest mask is stale; holding")
            return
        if mask_msg.encoding not in ('mono8', '8UC1'):
            rospy.logwarn_throttle(
                10.0, "vessel_tracking_example_2: expected mono8/8UC1, got %s; holding",
                mask_msg.encoding)
            return

        try:
            mask = mask_from_msg(mask_msg)
        except (TypeError, ValueError) as error:
            rospy.logwarn_throttle(10.0, "vessel_tracking_example_2: malformed mask (%s); holding", error)
            return

        pixel_count, centroid_x, width_ratio = vessel_measurements(mask)
        if pixel_count < self.min_vessel_pixels:
            rospy.logwarn_throttle(
                5.0, "vessel_tracking_example_2: vessel not visible enough (%d px); holding",
                pixel_count)
            return

        offset_px = centroid_x - mask.shape[1] / 2.0
        if abs(offset_px) >= self.deadband_px:
            step_m = self.direction_sign * self.gain_m_per_px * offset_px
            #step_m = max(-self.max_step_m, min(self.max_step_m, step_m))
            self._publish(step_m, 0.0)
            rospy.loginfo(
                "vessel_tracking_example_2: centring offset=%+.1f px -> linear.x=%+.5f m",
                offset_px, step_m)
            return

        if width_ratio >= self.longitudinal_width_ratio:
            rospy.loginfo(
                "vessel_tracking_example_2: centred longitudinal view reached (width ratio=%.3f); holding",
                width_ratio)
            return

        if width_ratio < self.far_width_ratio:
            yaw_step = self.large_yaw_step
            description = "large"
        else:
            yaw_step = self.small_yaw_step
            description = "small"

        angular_z = self.angular_direction * yaw_step
        self._publish(0.0, angular_z)
        rospy.loginfo(
            "vessel_tracking_example_2: centred, width ratio=%.3f -> %s angular.z=%+.1f deg",
            width_ratio, description, math.degrees(angular_z))

    def _publish(self, linear_x, angular_z):
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.action_pub.publish(twist)


def main():
    rospy.init_node('vessel_tracking_example_node')
    try:
        VesselTrackingExample()
        rospy.spin()
    except (ValueError, rospy.ROSException) as error:
        rospy.logfatal("vessel_tracking_example_2: startup failed: %s", error)


if __name__ == '__main__':
    main()
