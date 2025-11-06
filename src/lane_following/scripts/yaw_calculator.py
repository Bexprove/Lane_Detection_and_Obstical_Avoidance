#!/usr/bin/env python
# coding:utf-8
import rospy
import cv2
import numpy as np
import os
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from collections import deque, Counter
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
        self.P = 0.004
        self.I = 0.0
        self.D = 0.002
        self.angular_pid = simplePID(0, self.P, self.I, self.D)
        self.prev_angular_z = 0.0

        self.turn_history = deque(maxlen=3)
        self.avg_y_threshold = 0.7

    def cancel(self):
        self.ros_ctrl.pub_cmdVel.publish(Twist())
        self.ros_ctrl.cancel()
        self.image_sub.unregister()
        rospy.loginfo("Shutting down lane follower node.")

    def get_edges_and_roi(self, frame):
        frame = cv2.resize(frame, (480, 360))
        height = frame.shape[0]
        cropped = frame[int(height * 0.2):, :]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 70, 150)
        return cropped, edges

    def detect_lane_lines(self, edges):
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=70, minLineLength=50, maxLineGap=100)
        left_lines = []
        right_lines = []
        height, width = edges.shape
        midpoint = width // 2

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length < 40 or abs(slope) < 0.25 or abs(slope) > 4:
                    continue
                x_avg = (x1 + x2) // 2
                if slope < 0 and x_avg < midpoint:
                    left_lines.append([x1, y1, x2, y2])
                elif slope > 0 and x_avg > midpoint:
                    right_lines.append([x1, y1, x2, y2])
        return left_lines, right_lines

    def average_lane_line(self, lines, height):
        if len(lines) == 0:
            return None
        x_coords = []
        y_coords = []
        for x1, y1, x2, y2 in lines:
            x_coords += [x1, x2]
            y_coords += [y1, y2]
        poly = np.polyfit(y_coords, x_coords, 1)
        y1 = height
        y2 = int(height * 0.3)
        x1 = int(np.polyval(poly, y1))
        x2 = int(np.polyval(poly, y2))
        return [(x1, y1), (x2, y2)]

    def get_center_line_points(self, left_line, right_line, height, num_points=20):
        if not left_line or not right_line:
            return []

        lx_poly = np.polyfit([left_line[0][1], left_line[1][1]], [left_line[0][0], left_line[1][0]], 1)
        rx_poly = np.polyfit([right_line[0][1], right_line[1][1]], [right_line[0][0], right_line[1][0]], 1)

        center_points = []
        for i in range(num_points):
            y = int(height * (1.0 - 0.4 * i / num_points))
            lx = int(np.polyval(lx_poly, y))
            rx = int(np.polyval(rx_poly, y))
            cx = int((lx + rx) / 2)
            center_points.append((cx, y))

        return center_points

    def calculate_yaw_command(self, center_points, frame_width, left_line, right_line):
        if center_points:
            bottom_center_x = center_points[0][0]
            image_center_x = frame_width // 2
            offset_x = bottom_center_x - image_center_x

            normalized_offset = offset_x / (frame_width / 2)
            aggressive_offset = normalized_offset * 2.0

            max_yaw_rate_rad = 0.8

            yaw_command = aggressive_offset * max_yaw_rate_rad
            return np.clip(yaw_command, -max_yaw_rate_rad, max_yaw_rate_rad)

        if left_line:
            lx1, ly1 = left_line[0]
            lx2, ly2 = left_line[1]
            slope = (ly2 - ly1) / (lx2 - lx1 + 1e-6)
            yaw_command = -slope * 0.4
            return np.clip(yaw_command, -0.8, 0.8)

        if right_line:
            rx1, ry1 = right_line[0]
            rx2, ry2 = right_line[1]
            slope = (ry2 - ry1) / (rx2 - rx1 + 1e-6)
            yaw_command = -slope * 0.4
            return np.clip(yaw_command, -0.8, 0.8)

        return 0.0

    def predict_turn_from_centerline(self, center_points, left_line, right_line):
        if len(center_points) >= 3:
            x_coords = [p[0] for p in center_points]
            y_coords = [p[1] for p in center_points]

            poly = np.polyfit(y_coords, x_coords, 2)
            curvature = poly[0]

            if curvature > 0.0005:
                return "Left Turn"
            elif curvature < -0.0005:
                return "Right Turn"
            else:
                return "Straight"

        if left_line:
            lx1, ly1 = left_line[0]
            lx2, ly2 = left_line[1]
            slope = (ly2 - ly1) / (lx2 - lx1 + 1e-6)
            if slope > 0.2:
                return "Left Turn"
            elif slope < -0.2:
                return "Right Turn"
            else:
                return "Straight"

        if right_line:
            rx1, ry1 = right_line[0]
            rx2, ry2 = right_line[1]
            slope = (ry2 - ry1) / (rx2 - rx1 + 1e-6)
            if slope > 0.2:
                return "Left Turn"
            elif slope < -0.2:
                return "Right Turn"
            else:
                return "Straight"

        return "Unknown"

    def smooth_turn(self, current_turn):
        self.turn_history.append(current_turn)
        most_common = Counter(self.turn_history).most_common(1)[0][0]
        return most_common

    def image_callback(self, msg):
        if self.ros_ctrl.Joy_active:
            self.ros_ctrl.pub_cmdVel.publish(Twist())
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cropped, edges = self.get_edges_and_roi(frame)
        left_lines, right_lines = self.detect_lane_lines(edges)

        height, width, _ = cropped.shape

        left_avg = self.average_lane_line(left_lines, height)
        right_avg = self.average_lane_line(right_lines, height)

        center_points = self.get_center_line_points(left_avg, right_avg, height)

        avg_y = np.mean([p[1] for p in center_points]) if center_points else 0
        allow_turn = avg_y > self.avg_y_threshold * height

        current_turn = self.predict_turn_from_centerline(center_points, left_avg, right_avg)
        smooth_turn_prediction = self.smooth_turn(current_turn)
        yaw_command = self.calculate_yaw_command(center_points, width, left_avg, right_avg)

        twist = Twist()
        twist.linear.x = self.linear * (1 - min(abs(yaw_command)*0.7, 1.0))  # reduce speed when turning
        twist.angular.z = yaw_command if allow_turn else 0.0

        self.ros_ctrl.pub_cmdVel.publish(twist)

        debug_image = cropped.copy()
        if left_avg:
            cv2.line(debug_image, left_avg[0], left_avg[1], (0, 255, 0), 2)
        if right_avg:
            cv2.line(debug_image, right_avg[0], right_avg[1], (0, 0, 255), 2)
        for pt in center_points:
            cv2.circle(debug_image, pt, 3, (255, 100, 100), -1)
        for i in range(len(center_points) - 1):
            cv2.line(debug_image, center_points[i], center_points[i + 1], (255, 0, 255), 2)

        cv2.putText(debug_image, "Yaw: {:.2f}".format(yaw_command), (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if "DISPLAY" in os.environ:
            cv2.imshow("Lane Detection", debug_image)
            cv2.waitKey(1)

        self.r.sleep()

if __name__ == '__main__':
    LaneFollower()
    rospy.spin()
