#!/usr/bin/env python
# coding:utf-8

import rospy
import cv2
import numpy as np
import os
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from common import *
from time import sleep

class LaneFollower:
    def __init__(self):
        rospy.init_node('lane_follower', anonymous=False)
        rospy.on_shutdown(self.cancel)
        self.r = rospy.Rate(20)
        self.bridge = CvBridge()
        self.ros_ctrl = ROSCtrl()
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)

        self.linear = 0.1  # Normal speed
        self.fast_linear = 0.1  # Faster speed (optional for broken lines)
        self.P = 0.1  # Stronger turning reaction
        self.I = 0.0
        self.D = 0.1
        self.angular_pid = simplePID(0, self.P, self.I, self.D)
        self.prev_angular_z = 0.0

    def cancel(self):
        self.ros_ctrl.pub_cmdVel.publish(Twist())
        self.ros_ctrl.cancel()
        self.image_sub.unregister()
        rospy.loginfo("Shutting down lane follower node.")

    def image_callback(self, msg):
        if self.ros_ctrl.Joy_active:
            self.ros_ctrl.pub_cmdVel.publish(Twist())
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        processed_image, twist_cmd = self.process_lane(cv_image)

        self.ros_ctrl.pub_cmdVel.publish(twist_cmd)

        if "DISPLAY" in os.environ:
            cv2.imshow('Lane Detection', processed_image)
            cv2.waitKey(1)

        self.r.sleep()

    def process_lane(self, image):
        height, width, _ = image.shape
        image_center_x = width / 2

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        roi_mask = np.zeros_like(mask)
        polygon = np.array([[
            (0, height), (0, int(height * 0.6)),
            (width, int(height * 0.6)), (width, height)
        ]], np.int32)
        cv2.fillPoly(roi_mask, polygon, 255)
        cropped_mask = cv2.bitwise_and(mask, roi_mask)

        hsv_lines = cv2.HoughLinesP(cropped_mask, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=150)

        debug_image = image.copy()
        slopes = []

        if hsv_lines is not None:
            for line in hsv_lines:
                x1, y1, x2, y2 = line[0]
                if y1 > height * 0.75 or y2 > height * 0.75:
                    if x2 - x1 != 0:
                        slope = (y2 - y1) / float(x2 - x1)
                        if abs(slope) > 0.2:  # Filter horizontal noisy lines
                            slopes.append(slope)
                            cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        else:
                            # Optional: visualize ignored noisy lines
                            cv2.line(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 1)

        mean_slope = np.mean(slopes) if slopes else 0.0

        # PID control based on mean_slope
        twist = Twist()
        pid_output = self.angular_pid.update(mean_slope)

        twist.linear.x = self.linear
        twist.angular.z = pid_output

        # Step 2: Draw center line (optional visual aid)
        center_x = int(image_center_x)
        cv2.line(debug_image, (center_x, height), (center_x, int(height * 0.6)), (255, 0, 0), 2)

        # Display information
        cv2.putText(debug_image, 'Mean Slope: %.3f' % mean_slope, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(debug_image, 'PID Output: %.3f' % pid_output, (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return debug_image, twist

if __name__ == '__main__':
    LaneFollower()
    rospy.spin()

