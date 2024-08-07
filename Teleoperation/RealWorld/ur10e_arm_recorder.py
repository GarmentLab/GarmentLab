import copy
import rtde_receive
import rtde_control
import time
import rospy
from std_msgs.msg import Float64MultiArray

rtde_r = rtde_receive.RTDEReceiveInterface("192.168.56.1")

if __name__ == "__main__":
    rospy.init_node("arm_rtde_recorder")

    arm_joint = rospy.Publisher("arm_joint", Float64MultiArray, queue_size=100)
    arm_ee = rospy.Publisher("arm_ee", Float64MultiArray, queue_size=100)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        arm_joint_mess = Float64MultiArray(data=rtde_r.getActualQ())
        arm_ee_mess = Float64MultiArray(data=rtde_r.getActualTCPPose())

        arm_joint.publish(arm_joint_mess)
        arm_ee.publish(arm_ee_mess)

        print("arm_joint",arm_joint_mess)
        print("arm_ee",arm_ee_mess)

        rate.sleep()
        
    

