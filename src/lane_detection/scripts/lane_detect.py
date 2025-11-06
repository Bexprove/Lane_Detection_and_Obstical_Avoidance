#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from collections import deque, Counter

# Globals
bridge = CvBridge()
turn_history = deque(maxlen=3)

def get_edges_and_roi(frame):
    frame = cv2.resize(frame, (480, 360))
    height = frame.shape[0]
    cropped = frame[int(height * 0.2):, :]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 70, 150)
    return cropped, edges

def detect_lane_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=70, minLineLength=50, maxLineGap=100)
    left_lines, right_lines = [], []
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

def average_lane_line(lines, height):
    if len(lines) == 0:
        return None
    x_coords, y_coords = [], []
    for x1, y1, x2, y2 in lines:
        x_coords += [x1, x2]
        y_coords += [y1, y2]
    poly = np.polyfit(y_coords, x_coords, 1)
    y1 = height
    y2 = int(height * 0.3)
    x1 = int(np.polyval(poly, y1))
    x2 = int(np.polyval(poly, y2))
    return [(x1, y1), (x2, y2)]

def create_drivable_region_mask(cropped, left_line, right_line):
    mask = np.zeros_like(cropped[:, :, 0])
    offset = 20
    if left_line and right_line:
        l0 = (left_line[0][0] - offset, left_line[0][1])
        l1 = (left_line[1][0] - offset, left_line[1][1])
        r1 = (right_line[1][0] + offset, right_line[1][1])
        r0 = (right_line[0][0] + offset, right_line[0][1])
        points = np.array([l0, l1, r1, r0])
        cv2.fillPoly(mask, [points], 255)
    return mask

def get_center_line_points(left_line, right_line, height, num_points=20):
    if not left_line or not right_line:
        return []
    lx_poly = np.polyfit([left_line[0][1], left_line[1][1]], [left_line[0][0], left_line[1][0]], 1)
    rx_poly = np.polyfit([right_line[0][1], right_line[1][1]], [right_line[0][0], right_line[1][0]], 1)

    center_points = []
    for i in range(num_points):
        y = int(height * (1 - 0.3 * i / num_points))
        lx = int(np.polyval(lx_poly, y))
        rx = int(np.polyval(rx_poly, y))
        cx = int((lx + rx) / 2)
        center_points.append((cx, y))
    return center_points

def predict_turn_from_centerline(center_points, left_line, right_line):
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

def smooth_turn(current_turn):
    turn_history.append(current_turn)
    most_common = Counter(turn_history).most_common(1)[0][0]
    return most_common

def image_callback(msg):
    global bridge

    try:
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception as e:
        rospy.logerr("Error converting image: %s", str(e))
        return

    cropped, edges = get_edges_and_roi(frame)
    left_lines, right_lines = detect_lane_lines(edges)
    height = cropped.shape[0]
    left_avg = average_lane_line(left_lines, height)
    right_avg = average_lane_line(right_lines, height)
    drivable_mask = create_drivable_region_mask(cropped, left_avg, right_avg)

    overlay = cropped.copy()
    if left_avg:
        cv2.line(overlay, left_avg[0], left_avg[1], (0, 255, 0), 3)
    if right_avg:
        cv2.line(overlay, right_avg[0], right_avg[1], (0, 255, 0), 3)

    color_mask = cv2.cvtColor(drivable_mask, cv2.COLOR_GRAY2BGR)
    highlight = cv2.addWeighted(overlay, 1.0, color_mask, 0.5, 0)

    center_points = get_center_line_points(left_avg, right_avg, height)
    current_turn = predict_turn_from_centerline(center_points, left_avg, right_avg)
    smooth_turn_prediction = smooth_turn(current_turn)

    cv2.putText(highlight, f"Turn Prediction: {smooth_turn_prediction}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for point in center_points:
        cv2.circle(highlight, point, 3, (255, 100, 100), -1)
    for i in range(len(center_points)-1):
        cv2.line(highlight, center_points[i], center_points[i+1], (255, 0, 255), 2)

    cv2.imshow("Lane Detection", highlight)
    cv2.waitKey(1)

def main():
    rospy.init_node('lane_detection_node', anonymous=True)
    rospy.Subscriber('/camera/rgb/image_raw', Image, image_callback)
    rospy.loginfo("Lane Detection Node Started!")
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

