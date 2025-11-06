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

        self.linear = 0.1
        self.fast_linear = 0.1
        self.P = 0.004
        self.I = 0.0
        self.D = 0.002
        self.angular_pid = simplePID(0, self.P, self.I, self.D)
        self.prev_angular_z = 0.0

        self.left_turn_counter = 0
        self.right_turn_counter = 0
        self.turn_threshold_frames = 3  # frames needed
        self.avg_y_threshold = 0.85  # 85% down the image

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

        left_lines = []
        right_lines = []
        all_y_coords = []
        broken_lines_detected = False

        if hsv_lines is not None:
            for line in hsv_lines:
                x1, y1, x2, y2 = line[0]
                if y1 > height * 0.75 or y2 > height * 0.75:
                    angle = np.arctan2((y2 - y1), (x2 - x1)) * 180.0 / np.pi

                    if -90 < angle < -25:  # tighter window for left
                        left_lines.append((x1, y1, x2, y2))
                        all_y_coords.append(y1)
                        all_y_coords.append(y2)
                    elif 25 < angle < 90:  # tighter window for right
                        right_lines.append((x1, y1, x2, y2))
                        all_y_coords.append(y1)
                        all_y_coords.append(y2)

            if len(hsv_lines) < 8:
                broken_lines_detected = True

        total_lines = len(left_lines) + len(right_lines)

        # Calculate average Y position
        avg_y = np.mean(all_y_coords) if all_y_coords else 0

        allow_turn = avg_y > self.avg_y_threshold * height

        command_text = "Move Forward"

        if total_lines > 0:
            if len(right_lines) / float(total_lines) >= 0.75:
                self.left_turn_counter += 1
                self.right_turn_counter = 0
            elif len(left_lines) / float(total_lines) >= 0.75:
                self.right_turn_counter += 1
                self.left_turn_counter = 0
            else:
                self.left_turn_counter = 0
                self.right_turn_counter = 0
        else:
            self.left_turn_counter = 0
            self.right_turn_counter = 0

        if allow_turn:
            if self.left_turn_counter >= 0.8*total_lines:
                command_text = "Turn Left"
            elif self.right_turn_counter >= 0.8*total_lines:
                command_text = "Turn Right"

        # Step 2: Control logic
        center_x = image_center_x
        if left_lines and right_lines:
            left_x = np.mean([(x1 + x2) / 2 for (x1, y1, x2, y2) in left_lines])
            right_x = np.mean([(x1 + x2) / 2 for (x1, y1, x2, y2) in right_lines])
            center_x = (left_x + right_x) / 2
        elif left_lines:
            left_x = np.mean([(x1 + x2) / 2 for (x1, y1, x2, y2) in left_lines])
            center_x = left_x + 200
        elif right_lines:
            right_x = np.mean([(x1 + x2) / 2 for (x1, y1, x2, y2) in right_lines])
            center_x = right_x - 200

        twist = Twist()

        lateral_error = center_x - image_center_x

        if command_text == "Turn Left":
            twist.linear.x = self.linear
            twist.angular.z = +0.8
        elif command_text == "Turn Right":
            twist.linear.x = self.linear
            twist.angular.z = -0.8
        elif command_text == "Move Forward":
            twist.linear.x = self.linear
            angular_z = self.angular_pid.update(lateral_error)
            self.prev_angular_z = angular_z
            twist.angular.z = angular_z
        else:
            twist.linear.x = 0
            twist.angular.z = 0

        # Draw debug info
        if left_lines:
            for (x1, y1, x2, y2) in left_lines:
                cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if right_lines:
            for (x1, y1, x2, y2) in right_lines:
                cv2.line(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.line(debug_image, (int(center_x), height), (int(center_x), int(height * 0.6)), (255, 0, 0), 2)
        cv2.putText(debug_image, 'Command: %s' % command_text, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(debug_image, 'Left: %d Right: %d AvgY: %.1f' % (len(left_lines), len(right_lines), avg_y), (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return debug_image, twist

if __name__ == '__main__':
    LaneFollower()
    rospy.spin()

