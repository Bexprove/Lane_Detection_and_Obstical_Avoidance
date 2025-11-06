#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist

def move_robot():
    rospy.init_node('auto_motion_node', anonymous=True)
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    move_cmd = Twist()
    move_cmd.linear.x = 0.2  # Move forward
    move_cmd.angular.z = 0.0 # No rotation

    stop_cmd = Twist()  # All zeros

    rospy.loginfo("Moving forward for 5 seconds...")
    start_time = rospy.Time.now()

    while not rospy.is_shutdown():
        elapsed = (rospy.Time.now() - start_time).to_sec()
        if elapsed < 5.0:
            cmd_vel_pub.publish(move_cmd)
        else:
            cmd_vel_pub.publish(stop_cmd)
        rate.sleep()

if __name__ == '__main__':
    try:
        move_robot()
    except rospy.ROSInterruptException:
        pass

