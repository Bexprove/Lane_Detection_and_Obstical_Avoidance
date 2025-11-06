#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import numpy as np
import os

SAVE_PATH = os.path.expanduser("~/project_ws/midas_frame.npy")

def save_callback(msg):
    try:
        width = msg.width
        height = msg.height
        dtype = np.float32  # because /midas_depth is 32FC1
        data = np.frombuffer(msg.data, dtype=dtype).reshape((height, width))
        np.save(SAVE_PATH, data)
        print(f"✅ Saved MiDaS inverse depth frame to {SAVE_PATH}")
        rospy.signal_shutdown("Saved one frame")
    except Exception as e:
        print(f"❌ Failed to convert image: {e}")

def main():
    rospy.init_node("save_midas_depth_raw")
    rospy.Subscriber("/midas_topic", Image, save_callback)
    print("📡 Waiting for /midas_topic...")
    rospy.spin()

if __name__ == "__main__":
    main()
