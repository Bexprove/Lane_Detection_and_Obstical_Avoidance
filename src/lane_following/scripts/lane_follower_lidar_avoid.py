#!/usr/bin/env python
# coding:utf-8

import rospy
import cv2
import numpy as np
import os
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from common import *
from time import sleep

clicked_points = []

def click_event(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print("Clicked: {}, {}".format(x, y))

def mask_white(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([35, 40, 80])
    upper_white = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result

def warp_perspective(frame, src_points):
    height, width = frame.shape[:2]
    dst_points = np.float32([
        [width * 0.25, 0],
        [width * 0.75, 0],
        [width * 0.25, height],
        [width * 0.75, height]
    ])
    M = cv2.getPerspectiveTransform(np.float32(src_points), dst_points)
    warped = cv2.warpPerspective(frame, M, (width, height))
    crop_h = int(height * 0.9)
    warped = warped[:crop_h, :]
    return warped

class LaneFollower:
    def __init__(self):
        rospy.init_node('lane_follower', anonymous=False)
        rospy.on_shutdown(self.cancel)
        self.r = rospy.Rate(20)
        self.bridge = CvBridge()
        self.ros_ctrl = ROSCtrl()
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)

        self.linear = 0.23
        self.P = 0.0014
        self.I = 0.0
        self.D = 0.0023
        self.angular_pid = simplePID(0, self.P, self.I, self.D)

        self.avg_y_threshold = 0.85
        self.got_clicks = False

        self.avoiding = False
        self.avoid_stage = 0
        self.avoid_start_time = None

    def cancel(self):
        self.ros_ctrl.pub_cmdVel.publish(Twist())
        self.ros_ctrl.cancel()
        self.image_sub.unregister()
        self.scan_sub.unregister()
        rospy.loginfo("Shutting down lane follower node.")

    def lidar_callback(self, scan):
        center_idx = len(scan.ranges) // 2
        min_range = min(scan.ranges[center_idx - 5:center_idx + 5])
        if 0.05 < min_range < 0.2 and not self.avoiding:
            self.avoiding = True
            self.avoid_stage = 0
            self.avoid_start_time = rospy.Time.now()
            rospy.loginfo("Obstacle detected, starting avoidance arc.")

    def detect_lane_lines(self, warped):
        height, width = warped.shape[:2]
        region_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(region_mask, (int(width * 0.1), 0), (int(width * 0.9), height), 255, -1)
        masked = cv2.bitwise_and(warped, warped, mask=region_mask)

        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, binary = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        edges = cv2.Canny(morph, 30, 90)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=70, minLineLength=40, maxLineGap=75)
        filtered = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > 50:
                    filtered.append(line)
        lines = filtered

        left_lines, right_lines, left_perp, right_perp = [], [], [], []
        all_y_coords = []
        mid_x = width // 2

        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            x_avg = (x1 + x2) // 2

            if abs(slope) > 50:
                if slope < 0 and x_avg < mid_x:
                    left_perp.append((x1, y1, x2, y2))
                    all_y_coords.extend([y1, y2])
                elif slope > 0 and x_avg > mid_x:
                    right_perp.append((x1, y1, x2, y2))
                    all_y_coords.extend([y1, y2])
            else:
                if slope < 0:
                    left_lines.append((x1, y1, x2, y2))
                    all_y_coords.extend([y1, y2])
                elif slope > 0:
                    right_lines.append((x1, y1, x2, y2))
                    all_y_coords.extend([y1, y2])

        return left_lines, right_lines, left_perp, right_perp, all_y_coords

    def image_callback(self, msg):
        global clicked_points
        twist = Twist()

        if self.avoiding:
            now = rospy.Time.now()
            dt = (now - self.avoid_start_time).to_sec()

            if self.avoid_stage == 0:
                twist.linear.x = 0.1
                twist.angular.z = -0.5
                if dt > 1.2:
                    self.avoid_stage = 1
                    self.avoid_start_time = now
            elif self.avoid_stage == 1:
                twist.linear.x = 0.2
                twist.angular.z = 0.0
                if dt > 1.4:
                    self.avoid_stage = 2
                    self.avoid_start_time = now
            elif self.avoid_stage == 2:
                twist.linear.x = 0.1
                twist.angular.z = 0.5
                if dt > 1.2:
                    self.avoid_stage = 3
                    self.avoid_start_time = now
            elif self.avoid_stage == 3:
                twist.linear.x = 0.2
                twist.angular.z = 0.0
                if dt > 1.0:
                    self.avoiding = False
                    rospy.loginfo("Avoidance complete. Resuming lane following.")

            self.ros_ctrl.pub_cmdVel.publish(twist)

            if "DISPLAY" in os.environ:
                dbg = np.zeros((100, 400, 3), dtype=np.uint8)
                text = "Avoid Stage {}".format(self.avoid_stage)
                cv2.putText(dbg, text, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("Lane Detection", dbg)
                cv2.waitKey(1)
            return

        if self.ros_ctrl.Joy_active:
            self.ros_ctrl.pub_cmdVel.publish(Twist())
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width, _ = cv_image.shape
        image_center_x = width / 2

        if not self.got_clicks:
            clicked_points[:] = [(261, 289), (484, 291), (164, 419), (619, 404)]
            print("Selected Points (default): {}".format(clicked_points))
            self.got_clicks = True

        filtered_white = mask_white(cv_image)
        warped = warp_perspective(filtered_white, clicked_points)

        left_lines, right_lines, left_perp, right_perp, all_y_coords = self.detect_lane_lines(warped)
        debug_image = warped.copy()

        if not (left_lines or right_lines or left_perp or right_perp):
            rospy.loginfo("No lines detected. Robot stopping.")
            self.ros_ctrl.pub_cmdVel.publish(Twist())
            return

        if left_lines and right_lines:
            left_x = np.mean([(x1 + x2) / 2 for x1, _, x2, _ in left_lines])
            right_x = np.mean([(x1 + x2) / 2 for x1, _, x2, _ in right_lines])
            center_x = (left_x + right_x) / 2
        elif left_perp and right_perp:
            left_x = np.mean([(x1 + x2) / 2 for x1, _, x2, _ in left_perp])
            right_x = np.mean([(x1 + x2) / 2 for x1, _, x2, _ in right_perp])
            center_x = (left_x + right_x) / 2
        elif left_lines:
            center_x = np.mean([(x1 + x2) / 2 for x1, _, x2, _ in left_lines]) + 200
        elif right_lines:
            center_x = np.mean([(x1 + x2) / 2 for x1, _, x2, _ in right_lines]) - 200
        else:
            center_x = image_center_x

        lateral_error = center_x - image_center_x
        avg_y = np.mean(all_y_coords) if all_y_coords else 0
        allow_turn = avg_y > self.avg_y_threshold * warped.shape[0]

        twist.linear.x = self.linear
        twist.angular.z = self.angular_pid.update(lateral_error)

        for (x1, y1, x2, y2) in left_lines + right_lines + left_perp + right_perp:
            cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(debug_image, (int(center_x), warped.shape[0]), (int(center_x), int(warped.shape[0]*0.6)), (255, 0, 0), 2)
        cv2.putText(debug_image, "Following Lane", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if "DISPLAY" in os.environ:
            cv2.imshow('Lane Detection', debug_image)
            cv2.waitKey(1)

        self.ros_ctrl.pub_cmdVel.publish(twist)
        self.r.sleep()

if __name__ == '__main__':
    LaneFollower()
    rospy.spin()

