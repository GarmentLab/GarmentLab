from __future__ import absolute_import
import threading
import rospy
import sys
import tf
import copy
import geometry_msgs.msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sr_robot_commander.sr_robot_commander import SrRobotCommander
from sr_robot_commander.sr_arm_commander import SrArmCommander
from builtins import input
from std_msgs.msg import Float64MultiArray


# The constructors for the SrArmCommander, SrHandCommander and SrRobotCommander
# take a name parameter that should match the group name of the robot to be used.
# How to command the arm separately

switch=True

def callback(data):
    jointdata=list(data.data)
    arm_new_joints_goal = {'ra_shoulder_pan_joint': jointdata[0], 'ra_elbow_joint': jointdata[2],
                        'ra_shoulder_lift_joint': jointdata[1], 'ra_wrist_1_joint': jointdata[3],
                        'ra_wrist_2_joint': jointdata[4], 'ra_wrist_3_joint': jointdata[5]}
    print(jointdata)
    arm_commander.move_to_joint_value_target_unsafe(arm_new_joints_goal,time=0.01,wait=False)


class thread1(threading.Thread):
    def __init__(self, thread_name):
        threading.Thread.__init__(self,name=thread_name)
    def run(self):
        rospy.Subscriber('arm_to_hand',Float64MultiArray,callback)
        print("ready to subscribe\n")
        rospy.spin()




if __name__=="__main__":
    rospy.init_node("arm_receiver", anonymous=True)

    arm_home_joints_goal = {'ra_shoulder_pan_joint': 0.00, 'ra_elbow_joint': 2.00,
                        'ra_shoulder_lift_joint': -1.25, 'ra_wrist_1_joint': -0.733,
                        'ra_wrist_2_joint': 1.5708, 'ra_wrist_3_joint': -3.1416}

    arm_commander = SrArmCommander(name="right_arm")
    arm_commander.set_planner_id("BiTRRT")
    arm_commander.set_pose_reference_frame("ra_base")
    robot_commander = SrRobotCommander(name="right_arm_and_hand")
    arm_commander.move_to_joint_value_target(arm_home_joints_goal)

    send_thread=thread1("send_thread")
    send_thread.start()
    while 1:
        a=input()
        if a=="unlock":
            switch=True
        else:
            switch=False