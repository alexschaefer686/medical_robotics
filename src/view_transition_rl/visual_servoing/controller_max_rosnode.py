#!/usr/bin/env python3
"""Minimal controller template: the vessel areas of the last N masks in, one Twist out.

Every ~command_period_sec the vessel area of the latest mask is added to the history (one
entry per step, at most ~history_length) and compute_twist(areas) decides the motion from
that one array, oldest first, the current area last. Write your controller there: return
a geometry_msgs/Twist to publish on ~action_topic, or None to send nothing this step.

probe_scan_motion_node executes the Twist as ONE relative step in the probe frame:
linear in m (x = image columns, y = out of the image plane, z = depth into the tissue),
angular as ZYX angles in rad (R = Rz(angular.z) Ry(angular.y) Rx(angular.x)); it ignores
new steps while one is still running. A step only uses a mask received at least
~settle_sec after the last Twist, so that the probe has stopped.

Masks are (H, W) uint8 arrays, nonzero = vessel, rows along the depth; with
~image_x_sign -1 they are mirrored first so that columns increase along probe +x (the
simulation needs this).

Parameters (private):
  ~mask_topic          vessel mask, sensor_msgs/Image mono8 (/vessel_segmentation_node/vessel_mask)
  ~action_topic        relative probe step, geometry_msgs/Twist (/probe_action)
  ~history_length      N, areas kept (10)
  ~command_period_sec  seconds between steps (4.0)
  ~settle_sec          minimum mask age after the last Twist (3.0)
  ~image_x_sign        +1, or -1 to mirror the mask (-1.0)
  ~dry_run             log the Twists instead of publishing them (False)
"""

import threading
from collections import deque

import numpy as np

# ROS Kinetic's rospy still calls the Python 2-era Thread.isAlive() during
# shutdown; Python 3.9+ renamed it to is_alive().
if not hasattr(threading.Thread, "isAlive"):
    threading.Thread.isAlive = threading.Thread.is_alive

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

def vessel_area(mask):
    """Vessel area of a mask: vessel pixels (nonzero, the mask is 0/255)."""
    return np.count_nonzero(mask)


def compute_twist(areas):
    """Your controller. areas: vessel area (pixels) of the masks of the last steps, oldest first;
    areas[-1] is the current mask's, areas[-2] the previous step's. Return a Twist, or None."""
    rospy.loginfo("controller_max: vessel areas, oldest first: %s", areas)

    # Example, replace it: turn 5 deg about the probe axis per step as long as the area grows.
    if len(areas) > 1 and areas[-1] <= areas[-2]:
        return None
    twist = Twist()
    twist.angular.z = np.deg2rad(5.0)
    return twist


class MaxControllerNode(object):
    def __init__(self):
        mask_topic = rospy.get_param('~mask_topic', '/vessel_segmentation_node/vessel_mask')
        self.action_topic = rospy.get_param('~action_topic', '/probe_action')
        self.command_period_sec = rospy.get_param('~command_period_sec', 4.0)
        self.settle_sec = rospy.get_param('~settle_sec', 3.0)
        self.image_x_sign = rospy.get_param('~image_x_sign', -1.0)
        self.dry_run = rospy.get_param('~dry_run', False)
        self.history = deque(maxlen=int(rospy.get_param('~history_length', 10)))

        self._lock = threading.Lock()
        self._latest, self._latest_time = None, -np.inf
        self._last_twist_time = -np.inf
        self.action_pub = rospy.Publisher(self.action_topic, Twist, queue_size=1)
        rospy.Subscriber(mask_topic, Image, self._on_mask, queue_size=1)
        rospy.loginfo("controller_max: %s -> %s every %.1fs, history of %d masks%s", mask_topic, self.action_topic,
                      self.command_period_sec, self.history.maxlen, ", DRY RUN" if self.dry_run else "")

    def _on_mask(self, msg):
        with self._lock:
            self._latest, self._latest_time = msg, rospy.get_time()

    def spin(self):
        rate = rospy.Rate(10)
        next_step = rospy.get_time()
        while not rospy.is_shutdown():
            if rospy.get_time() >= next_step and self.step():
                next_step = rospy.get_time() + self.command_period_sec
            rate.sleep()

    def step(self):
        """One controller step; returns False while there is no mask taken after the last Twist."""
        with self._lock:
            msg, received = self._latest, self._latest_time
        if msg is None or received < self._last_twist_time + self.settle_sec:
            return False
        mask = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
        if self.image_x_sign < 0:
            mask = mask[:, ::-1]
        self.history.append(vessel_area(mask))
        twist = compute_twist(np.array(self.history))
        if twist is not None:
            if not self.dry_run:
                self.action_pub.publish(twist)
            self._last_twist_time = rospy.get_time()
            rospy.loginfo("controller_max: %s linear (%.4f, %.4f, %.4f) m, angular (%.3f, %.3f, %.3f) rad",
                          "logged" if self.dry_run else "sent", twist.linear.x, twist.linear.y, twist.linear.z,
                          twist.angular.x, twist.angular.y, twist.angular.z)
        return True


def main():
    rospy.init_node('controller_max_node')
    try:
        MaxControllerNode().spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
