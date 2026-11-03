#!/usr/bin/env python3
"""Minimal controller template: the scaled vessel area changes of the last N steps in, one Twist out.

Every ~command_period_sec the node measures the vessel area of the latest mask and adds
its change since the previous step's area, divided by the turn that caused it (pixels per
rad), to the history (one entry per step with a turn, at most ~history_length). Unlike the
raw change, this scaled change does not grow with the size of the turn, and its sign says
which way the area grows whichever way the probe turned. compute_twist(scaled_area_changes)
decides the motion from that one array, oldest first, the current one last. Write your
controller there: return a geometry_msgs/Twist to publish on ~action_topic, or None to
send nothing this step.

Starting procedure, so that compute_twist always gets a full history: the area at the
start pose is only measured (it has no previous area, so no change is recorded); then,
until the history holds N changes, the node turns the probe about its axis instead, N
turns of ~start_turn_deg / N, each followed by one area and its change. The step that
records the N-th change calls compute_twist.

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
  ~history_length      N, scaled area changes kept (10)
  ~start_turn_deg      total turn about the probe axis of the starting procedure (20.0)
  ~gain                turn (rad) per weighted sum of scaled area changes (pixels per rad) (6e-6)
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




class MaxControllerNode(object):
    def __init__(self):
        mask_topic = rospy.get_param('~mask_topic', '/vessel_segmentation_node/vessel_mask')
        self.action_topic = rospy.get_param('~action_topic', '/probe_action')
        self.command_period_sec = rospy.get_param('~command_period_sec', 4.0)
        self.settle_sec = rospy.get_param('~settle_sec', 3.0)
        self.image_x_sign = rospy.get_param('~image_x_sign', -1.0)
        self.gain = rospy.get_param('~gain', 6e-6)  # rad per (pixels per rad), for the weighted sum over N = 20
        self.dry_run = rospy.get_param('~dry_run', False)
        self.hist_N = int(rospy.get_param('~history_length', 20))
        self.history = deque(maxlen=self.hist_N)
        self.start_turn = np.deg2rad(rospy.get_param('~start_turn_deg', 20.0))
        self._previous_area = None  # area of the previous step's mask
        self._last_turn = 0.0  # turn sent at the previous step (rad)

        self._lock = threading.Lock()
        self._latest, self._latest_time = None, -np.inf
        self._last_twist_time = -np.inf
        self.action_pub = rospy.Publisher(self.action_topic, Twist, queue_size=1)
        rospy.Subscriber(mask_topic, Image, self._on_mask, queue_size=1)
        rospy.loginfo("controller_max: %s -> %s every %.1fs, history of %d area changes%s", mask_topic, self.action_topic,
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
        area = vessel_area(mask)
        # Area change scaled by the turn that caused it (pixels per rad). The start pose's area has no
        # previous one, and without a turn there is nothing to scale by.
        if self._previous_area is not None and self._last_turn != 0.0:
            self.history.append((area - self._previous_area) / self._last_turn)
        self._previous_area = area
        if len(self.history) < self.hist_N:
            # Starting procedure: fill the history, N turns of start_turn_deg / N.
            source = "start %d/%d" % (len(self.history) + 1, self.hist_N)
            twist = Twist()
            twist.angular.z = self.start_turn / self.hist_N
        else:
            source, twist = "controller", self.compute_twist(np.array(self.history))
        self._last_turn = twist.angular.z if twist is not None else 0.0
        if twist is not None:
            if not self.dry_run:
                self.action_pub.publish(twist)
            self._last_twist_time = rospy.get_time()
            rospy.loginfo("controller_max [%s]: %s linear (%.4f, %.4f, %.4f) m, angular (%.3f, %.3f, %.3f) rad",
                          source, "logged" if self.dry_run else "sent", twist.linear.x, twist.linear.y,
                          twist.linear.z, twist.angular.x, twist.angular.y, twist.angular.z)
        return True

    def compute_twist(self, scaled_area_changes):
        """Your controller. scaled_area_changes: change of the vessel area from each step to the next,
        divided by the turn that caused it (pixels per rad), over the last steps, oldest first;
        scaled_area_changes[-1] = (current area - previous step's area) / last turn. Return a Twist, or None."""
        rospy.loginfo("controller_max: scaled vessel area changes (px/rad), oldest first: %s", scaled_area_changes)

        # Example, replace it: turn 5 deg about the probe axis per step as long as the area grows.
        weights = np.hamming(2 * self.hist_N)[:self.hist_N]  # rising half: newest weighs most
        w_change = np.sum(weights * scaled_area_changes)/np.sum(weights)  # rising half: newest weighs most

        if abs(w_change) < 40: #pixels per rad
            rospy.loginfo("controller_max: area changes ~0, max found, no twist sent")
            return None

        rospy.loginfo("controller_max: weighted area change %.3f, sending twist", w_change)
        
        weighted_change = self.gain * w_change
        
        twist = Twist()
        twist.angular.z = weighted_change
        return twist

def main():
    rospy.init_node('controller_max_node')
    try:
        MaxControllerNode().spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
