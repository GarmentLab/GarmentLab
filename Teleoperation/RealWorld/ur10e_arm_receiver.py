import os
import sys
import numpy as np
import rospy
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64MultiArray
import threading

ARM_HOME_POSITION = [-0.5383931325308808, 0.02642760059334514, 0.46995540579416933, -1.2264567031660047, -1.234384858419156, 1.1815867321951872]
ARM_HOME_JOINT_POSITION = [3.500866413116455, -1.3204480570605774, -2.139267921447754, -2.854926725427145, -1.2082536856280726, 3.1336488723754883]

LM_HOME_WRIST_POSITION = np.array([0, 210.0, 40.0])
LM_HOME_PALM_NORMAL = np.array([0.0, -1.0, 0.0])
LM_HOME_DIRECTION = np.array([0.0, 0.0, -1.0])

LM_TO_UR_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
switch = True

def best_fit_transform(A, B):
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

class thread_publisher(threading.Thread):
    def __init__(self, thread_name):
        threading.Thread.__init__(self, name=thread_name)

    def run(self):
        rospy.Subscriber("leap_arm", Float64MultiArray, arm_call_back)
        rospy.spin()


def arm_call_back(msg):
    # end-effector position
        scale = 2.5/600
        # init position
        init_wrist_position = np.copy(LM_HOME_WRIST_POSITION)
        init_palm_normal = np.copy(LM_HOME_PALM_NORMAL)
        init_direction = np.copy(LM_HOME_DIRECTION)
        init_wrist_position *= scale
        init_palm_normal *= 0.2 * scale
        init_direction *= 0.2 * scale

        # fetch data from leap motion
        arm_points = np.array(msg.data).reshape(4,3)
        wrist_position = np.array(arm_points[2])
        palm_normal = np.array(arm_points[1])
        direction = np.array(arm_points[0])
        wrist_position *= scale
        palm_normal *= 0.2 * scale
        direction *= 0.2 * scale

        # convert to ur coordinate
        init_wrist_position = np.dot(LM_TO_UR_ROTATION, init_wrist_position)
        init_palm_normal = np.dot(LM_TO_UR_ROTATION, init_palm_normal)
        init_direction = np.dot(LM_TO_UR_ROTATION, init_direction)

        wrist_position = np.dot(LM_TO_UR_ROTATION, wrist_position)
        palm_normal = np.dot(LM_TO_UR_ROTATION, palm_normal)
        direction = np.dot(LM_TO_UR_ROTATION, direction)

        relative_wrist_position = wrist_position - init_wrist_position

        init_points = np.array([np.zeros(3), init_palm_normal, init_direction])

        point_wrist = relative_wrist_position.copy()
        point_palm_normal = relative_wrist_position + palm_normal
        point_direction = relative_wrist_position + direction
        points = np.array([point_wrist, point_palm_normal, point_direction])

        _, rotation, translation = best_fit_transform(init_points, points)

        # print("rotation: %s" % rotation)
        # print("translation: %s" % translation)

        ur_home_position = np.array(ARM_HOME_POSITION)

        # convert rotation vector to rotation matrix
        init_rotation = R.from_rotvec(ur_home_position[3:]).as_matrix()
        init_translation = ur_home_position[:3]

        # create composed rotation and translation
        composed_rotation = np.dot(init_rotation, rotation)
        composed_translation = np.dot(init_rotation, translation) + init_translation

        # convert rotation matrix to rotation vector
        composed_rotvec = R.from_matrix(composed_rotation).as_rotvec()

        # create new 6d position x, y, z, rx, ry, rz
        current_position = np.zeros(6)
        current_position[:3] = composed_translation
        current_position[3:] = composed_rotvec

        # print("new current_position: %s" % current_position)
        if switch:
            rtde_c.servoL(current_position.tolist(), 0.5, 0.1, 0.2, 0.2, 600)

if __name__ == "__main__":
    rospy.init_node("arm_receriver")

    rtde_c = rtde_control.RTDEControlInterface("192.168.56.1")
    rtde_c.moveJ(ARM_HOME_JOINT_POSITION, 0.5, 0.5)

    arm_control_thread = thread_publisher("arm_control_thread")
    arm_control_thread.start()
    
    while True:
        a = input()
        if a == "unlock":
            switch = True
        elif a == "exit":
            sys.exit(0)
        else:
            switch = False