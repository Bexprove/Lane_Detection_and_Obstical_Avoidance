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

        self.linear = 0.23
        self.P = 0.00157 #0.004
        self.I = 0.0 #0001
        self.D = 0.0023 #.00007 #0.0023
        self.angular_pid = simplePID(0, self.P, self.I, self.D)
        self.prev_angular_z = 0.0

        self.left_turn_counter = 0
        self.right_turn_counter = 0
        self.turn_threshold_frames = 3
        self.avg_y_threshold = 0.85

        self.got_clicks = False

    def cancel(self):
        self.ros_ctrl.pub_cmdVel.publish(Twist())
        self.ros_ctrl.cancel()
        self.image_sub.unregister()
        rospy.loginfo("Shutting down lane follower node.")

    def detect_lane_lines(self, warped):
        height, width = warped.shape[:2]
        region_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(region_mask, (int(width * 0.1), 0), (int(width * 0.9), height), 255, -1)
        masked = cv2.bitwise_and(warped, warped, mask=region_mask)

        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=70, minLineLength=40, maxLineGap=75)

        left_lines, right_lines = [], []
        left_perp, right_perp = [], []
        all_y_coords = []
        mid_x = width // 2

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                line_length = np.hypot(x2 - x1, y2 - y1)
                #if line_length < 40 or (y1 > height - 10 and y2 > height - 10):
                #    continue

                x_avg = (x1 + x2) // 2

                if abs(slope) > 50:  # classify as perpendicular
                    if slope < 0  and x_avg < mid_x:
                        left_perp.append((x1, y1, x2, y2))
                        all_y_coords.extend([y1, y2])
                    elif slope > 0 and x_avg > mid_x:
                        right_perp.append((x1, y1, x2, y2))
                        all_y_coords.extend([y1, y2])
                else:
                    if slope < 0: #and x_avg < mid_x:
                        left_lines.append((x1, y1, x2, y2))
                        all_y_coords.extend([y1, y2])
                    elif slope > 0: #and x_avg > mid_x:
                        right_lines.append((x1, y1, x2, y2))
                        all_y_coords.extend([y1, y2])

        return left_lines, right_lines, left_perp, right_perp, all_y_coords

    def image_callback(self, msg):
        global clicked_points

        if self.ros_ctrl.Joy_active:
            self.ros_ctrl.pub_cmdVel.publish(Twist())
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width, _ = cv_image.shape
        image_center_x = width / 2

        if not self.got_clicks:
            #clone = cv_image.copy()
            #cv2.imshow("Click 4 Points", clone)
            #cv2.setMouseCallback("Click 4 Points", click_event)
            #while len(clicked_points) < 4:
            #    cv2.waitKey(1)
            #cv2.destroyWindow("Click 4 Points")
            #print("Selected Points: {}".format(clicked_points))
            #self.got_clicks = True
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

        # Choose center_x based on lines
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

        twist = Twist()
        command_text = "PID"

        if left_lines or right_lines:
            total_lines = len(left_lines) + len(right_lines)
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

            if allow_turn:
		twist.linear.x = self.linear
                twist.angular.z = self.angular_pid.update(lateral_error)
                #if self.left_turn_counter >= self.turn_threshold_frames:
                #    command_text = "Turn Left"
                #    #twist.linear.x = -0.05
                #    twist.angular.z = +0.3
                #elif self.right_turn_counter >= self.turn_threshold_frames:
                #    command_text = "Turn Right"
                #    #twist.linear.x = -0.05
                #    twist.angular.z = -0.3
                #else:
                #    twist.linear.x = self.linear
                #    twist.angular.z = self.angular_pid.update(lateral_error)
            else:
                twist.linear.x = self.linear
                twist.angular.z = self.angular_pid.update(lateral_error)
        elif left_perp and right_perp:
            command_text = "Vertical PID"
            twist.linear.x = self.linear
            twist.angular.z = self.angular_pid.update(lateral_error)

        for (x1, y1, x2, y2) in left_lines + right_lines + left_perp + right_perp:
            cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(debug_image, (int(center_x), warped.shape[0]), (int(center_x), int(warped.shape[0]*0.6)), (255, 0, 0), 2)
        cv2.putText(debug_image, "Command: {}".format(command_text), (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
	print(command_text)

        if "DISPLAY" in os.environ:
            cv2.imshow('Lane Detection', debug_image)
            cv2.waitKey(1)

        self.ros_ctrl.pub_cmdVel.publish(twist)
        self.r.sleep()

if __name__ == '__main__':
    LaneFollower()
    rospy.spin()

