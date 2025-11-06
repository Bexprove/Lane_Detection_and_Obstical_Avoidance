#!/usr/bin/env python
# coding:utf-8

import rospy
import cv2
import numpy as np
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

        cv2.imshow('Lane Detection', processed_image)
        cv2.waitKey(1)
        self.r.sleep()

    def process_lane(self, image):
        height, width, _ = image.shape
        image_center_x = width / 2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height), (0, int(height * 0.6)),
            (width, int(height * 0.6)), (width, height)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(cropped_edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=150)

        left_lines = []
        right_lines = []
        broken_lines_detected = False

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2((y2 - y1), (x2 - x1)) * 180.0 / np.pi

                if -90 < angle < -20:
                    left_lines.append(line[0])
                elif 20 < angle < 90:
                    right_lines.append(line[0])

            if len(lines) < 8:
                broken_lines_detected = True

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
        else:
            twist = Twist()
            twist.linear.x = 0
            twist.angular.z = self.prev_angular_z
            return image, twist

        lateral_error = center_x - image_center_x

        angular_z = self.angular_pid.update(lateral_error)
        self.prev_angular_z = angular_z

        linear_x = self.fast_linear if broken_lines_detected else self.linear

        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z

        debug_image = image.copy()
        if left_lines:
            for (x1, y1, x2, y2) in left_lines:
                cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if right_lines:
            for (x1, y1, x2, y2) in right_lines:
                cv2.line(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.line(debug_image, (int(center_x), height), (int(center_x), int(height * 0.6)), (255, 0, 0), 2)
        cv2.putText(debug_image, 'Error: %.2f' % lateral_error, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return debug_image, twist

if __name__ == '__main__':
    LaneFollower()
    rospy.spin()

