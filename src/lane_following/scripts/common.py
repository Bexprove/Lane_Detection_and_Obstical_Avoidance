#!/usr/bin/env python
# coding:utf-8

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import numpy as np
import time

class ROSCtrl:
    def __init__(self):
        self.Joy_active = False
        self.pub_cmdVel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.sub_JoyState = rospy.Subscriber('/JoyState', Bool, self.JoyStateCallback)

    def JoyStateCallback(self, msg):
        if not isinstance(msg, Bool):
            return
        self.Joy_active = msg.data
        self.pub_cmdVel.publish(Twist())

    def cancel(self):
        self.sub_JoyState.unregister()
        self.pub_cmdVel.unregister()
        rospy.loginfo("ROSCtrl node shutdown.")

class simplePID:
    '''Very simple discrete PID controller'''

    def __init__(self, target, P, I, D):
        self.Kp = np.array(P)
        self.Ki = np.array(I)
        self.Kd = np.array(D)
        self.last_error = 0
        self.integrator = 0
        self.timeOfLastCall = None
        self.setPoint = np.array(target)
        self.integrator_max = float('inf')

    def update(self, current_value):
        current_value = np.array(current_value)
        if self.timeOfLastCall is None:
            self.timeOfLastCall = time.time()
            return 0
        error = self.setPoint - current_value
        currentTime = time.time()
        deltaT = currentTime - self.timeOfLastCall
        self.integrator += error * deltaT
        D = (error - self.last_error) / deltaT
        self.last_error = error
        self.timeOfLastCall = currentTime
        return self.Kp * error + self.Ki * self.integrator + self.Kd * D

