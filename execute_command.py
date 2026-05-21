import rtde_control
import rtde_receive

# ur_rtde documentation:
# https://sdurobotics.gitlab.io/ur_rtde/

# the two lines below connects to the robot. the robot should be in remote control mode!
rtde_c = rtde_control.RTDEControlInterface("192.168.1.12")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.12")

print(rtde_r.getActualQ())  # this gets the joint angles in radians
print(rtde_r.getActualTCPPose())  # this gets the eef pose in format [x, y, z, rx, ry, rz]

TARGET_POSE = [0.4228982982168777, -0.5405681651056513, -0.020091319251801132, -0.012761069417938564, -3.1411170207760106, -0.01618970729788005]
# TARGET_POSE = [0.4020901823999471, -0.5617230197876911, 0.022780660249052537, -0.07158596983566785, -3.0871169236746354, 0.1279737481798817]  # define target pose here in the format of [x, y, z, rx, ry, rz]. rx, ry, rz are in rotation vector format.
# TARGET_POSE = [0.36956118287564493, -0.5474824728253321, -0.024981345032776762, -0.2512174384332913, -3.0265238435740014, 0.16983331746829397]

# this will move the robot to the target pose
rtde_c.moveL(TARGET_POSE, 0.1, 0.1)  # pose, speed, acceleration, asynchronous