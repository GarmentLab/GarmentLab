from __future__ import absolute_import
from operator import truediv
import rospy
import sys
from sr_robot_commander.sr_arm_commander import SrArmCommander
from sr_robot_commander.sr_hand_commander import SrHandCommander
from sr_robot_commander.sr_robot_commander import SrRobotCommander
from builtins import input
from std_msgs.msg import Float64MultiArray
from shadow_hand import angle_tensor_to_dict
import numpy as np
import threading

switch=True

def callback(data):
    jointdata=list(data.data)
    jointdata=np.array(jointdata)
    jointdict=angle_tensor_to_dict(jointdata)
    # print(jointdict)
    global switch
    if switch:
        hand_commander.move_to_joint_value_target_unsafe(jointdict,wait=False)
    #print(jointdict)


class thread1(threading.Thread):
    def __init__(self, thread_name):
        threading.Thread.__init__(self,name=thread_name)
    def run(self):
        rospy.Subscriber('hand_joint_value',Float64MultiArray,callback)
        print("ready to subscribe\n")
        rospy.spin()




if __name__=="__main__":
    rospy.init_node("right_hand_receive", anonymous=True)

    hand_commander = SrHandCommander(name="right_hand")
    arm_commander = SrArmCommander(name="right_arm")
    arm_commander.set_planner_id("BiTRRT")
    arm_commander.set_pose_reference_frame("ra_base")
    robot_commander = SrRobotCommander(name="right_arm_and_hand")

    send_thread=thread1("send_thread")
    send_thread.start()
    while 1:
        a=input()
        if a=="unlock":
            switch=True
        else:
            switch=False