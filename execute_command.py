import rtde_control
import rtde_receive
import numpy as np

# ur_rtde documentation:
# https://sdurobotics.gitlab.io/ur_rtde/

# the two lines below connects to the robot. the robot should be in remote control mode!
rtde_c = rtde_control.RTDEControlInterface("192.168.1.12")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.12")

print(rtde_r.getActualQ())  # this gets the joint angles in radians
print(rtde_r.getActualTCPPose())  # this gets the eef pose in format [x, y, z, rx, ry, rz]

TARGET_POSE = [0.4020901823999471, -0.5617230197876911, 0.022780660249052537, -0.07158596983566785, -3.0871169236746354, 0.1279737481798817, 0.04296715930104256]  # define target pose here in the format of [x, y, z, rx, ry, rz]. rx, ry, rz are in rotation vector format.

# this will move the robot to the target pose
rtde_c.moveL(TARGET_POSE, 0.01, 0.05, False)  # pose, speed, acceleration, asynchronous