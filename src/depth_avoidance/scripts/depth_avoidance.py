#!/usr/bin/env python
# coding:utf-8
import rospy
import numpy as np
from common import *
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

class DepthAvoid:
    def __init__(self):
        rospy.on_shutdown(self.cancel)
        self.r = rospy.Rate(20)

        self.Moving = False
        self.switch = False  # manual override switch
        self.ros_ctrl = ROSCtrl()

        # Manually set parameters
        self.linear = 0.2         # Forward speed
        self.angular = 1.0        # Rotation speed
        self.ResponseDist = 0.7  # Distance to react to obstacle (meters)
        self.DepthAngle = 30      # Left/Right sector width (degrees)

        self.sub_depth = rospy.Subscriber('/camera/depth/image_raw', Image, self.registerDepth, queue_size=1)
        rospy.loginfo("Depth avoidance node initialized.")

    def cancel(self):
        self.ros_ctrl.pub_vel.publish(Twist())
        self.ros_ctrl.cancel()
        self.sub_depth.unregister()
        rospy.loginfo("Shutting down this node.")

    def registerDepth(self, data):
        if not isinstance(data, Image):
            return

        try:
            from cv_bridge import CvBridge
            bridge = CvBridge()
            depth_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough").copy()

            if data.encoding == "16UC1":
                depth_image = depth_image.astype(np.float32) / 1000.0
            depth_image[np.isnan(depth_image)] = 10.0

            height, width = depth_image.shape

            left_crop = int(width * (0.5 - self.DepthAngle/180.0))
            right_crop = int(width * (0.5 + self.DepthAngle/180.0))
            cropped = depth_image[:, left_crop:right_crop]

            valid_pixels = cropped[(cropped > 0.1) & (cropped < 10.0)]

            if valid_pixels.size > 0:
                min_depth = np.min(valid_pixels)
            else:
                min_depth = 10.0

            rospy.loginfo("Closest Obstacle Depth: %.2f m", min_depth)

            if self.ros_ctrl.Joy_active or self.switch:
                if self.Moving:
                    self.ros_ctrl.pub_vel.publish(Twist())
                    self.Moving = False
                return

            self.Moving = True
            move_cmd = Twist()

            if min_depth < self.ResponseDist:
                move_cmd.linear.x = 0.0
                move_cmd.angular.z = self.angular
                rospy.loginfo("Obstacle detected! Turning to avoid...")
            else:
                move_cmd.linear.x = self.linear
                move_cmd.angular.z = 0.0

            self.ros_ctrl.pub_vel.publish(move_cmd)
            self.r.sleep()

        except Exception as e:
            rospy.logerr("Depth processing error: %s", str(e))

if __name__ == '__main__':
    rospy.init_node('depth_avoidance', anonymous=False)
    tracker = DepthAvoid()
    rospy.spin()
