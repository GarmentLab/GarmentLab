import rospy
from Teleoperation.Listener import Listener
from std_msgs.msg import Float64MultiArray

listener = Listener(None, "listener")

listener.launch()

rospy.init_node('leap_motion', anonymous=True)
leap_hand = rospy.Publisher('leap_hand', Float64MultiArray, queue_size=1000)
leap_arm = rospy.Publisher('leap_arm', Float64MultiArray, queue_size=1000)
retarget_hand = rospy.Publisher('retarget_hand', Float64MultiArray, queue_size=1000)

side = "left"

while not listener.unregistered:
    hand_pose_raw, arm_pose_raw, hand_joint_pose, wrist_pos, wrist_ori = env.listener.get_pose(side)

    leap_hand.publish(hand_pose_raw.ravel())
    leap_arm.publish(arm_pose_raw.ravel())
    retarget_hand.publish(hand_joint_pose.ravel())