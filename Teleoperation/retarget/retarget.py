import glob
from ntpath import join
import os
import sys
import time
from typing import List, Optional, Tuple
sys.path.append("/home/isaac/projects/shadow_robot/base/src/sr_interface/shadow-hand-ros-inside/src")
import numpy as np
from align import best_fit_transform
from const import HAND_KEYPOINT_NAMES, HAND_VISULIZATION_LINKS
from scipy import signal
from smooth import VelocityFilter
from solver import MotionMapper
from tqdm import tqdm
from visualization import (
    plot_hand_keypoints,
    plot_hand_motion_keypoints,
    plot_two_hands_motion_keypoints,
)
# from std_msgs.msg import Float64MultiArray
# import rospy
import _thread

joint = np.random.rand(21, 3)
solver_r = MotionMapper("right")
solver_l = MotionMapper("left")

velocity_filter_r = VelocityFilter(5, 5)
velocity_filter_l = VelocityFilter(5, 5)

def optimize(target, side: Optional[str] = "right"):
    global solver_l
    global solver_r
    zero_keypoints = solver_l.get_zero_pose() if side == "left" else solver_r.get_zero_pose() 
    
    scale_factor = calc_scale_factor(target, zero_keypoints)
    target *= scale_factor

    _, R, t = best_fit_transform(
            target[[0, 5, 9, 13]],
            zero_keypoints[[0, 5, 9, 13]],
        )
    
    target = (R @ target.T).T + t
    target += zero_keypoints[[5, 9, 13]].mean(axis=0) - target[
             [5, 9, 13]
        ].mean(axis=0)

    global velocity_filter_r, velocity_filter_l

    if side == "right":
        target = velocity_filter_r(target)
    else:
        target = velocity_filter_l(target)
    target = extend_pinky(target)

    if side == "right":
        return solver_r.step(target)
    else:
        return solver_l.step(target)

def normalized(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def extend_pinky(keypoints: np.ndarray) -> np.ndarray:
    """Extends the pinky finger to the length of mean of the other three fingers."""
    assert keypoints.shape == (21, 3)
    pinky = keypoints[17:21, :].copy()

    index = keypoints[5:9, :]
    middle = keypoints[9:13, :]
    ring = keypoints[13:17, :]

    fingers = np.stack([index, middle, ring], axis=0)

    proximal = np.mean(np.linalg.norm(fingers[:, 1, :] - fingers[:, 0, :], axis=1))
    intermediate = np.mean(np.linalg.norm(fingers[:, 2, :] - fingers[:, 1, :], axis=1))
    distal = np.mean(np.linalg.norm(fingers[:, 3, :] - fingers[:, 2, :], axis=1))

    proximal_direction = normalized(pinky[1, :] - pinky[0, :])
    intermediate_direction = normalized(pinky[2, :] - pinky[1, :])
    distal_direction = normalized(pinky[3, :] - pinky[2, :])

    pinky[1, :] = pinky[0, :] + proximal * proximal_direction
    pinky[2, :] = pinky[1, :] + intermediate * intermediate_direction
    pinky[3, :] = pinky[2, :] + distal * distal_direction

    keypoints = np.concatenate([keypoints[:17, :].copy(), pinky], axis=0)
    return keypoints


# # thread publisher
# def thread_publisher(thredName):
#     pub = rospy.Publisher('hand_joint_value', Float64MultiArray, queue_size=1000)
#     rate = rospy.Rate(10)

#     while not rospy.is_shutdown():
#         latest = optimize(joint)
#         latest = latest.tolist()
#         mess = Float64MultiArray(data=latest)
#         pub.publish(mess)
#         rate.sleep()




def filter_position_sequence(position_seq: np.ndarray, wn=5, fs=25):
    sos = signal.butter(2, wn, "lowpass", fs=fs, output="sos", analog=False)
    seq_shape = position_seq.shape
    if len(seq_shape) < 2:
        raise ValueError(
            f"Joint Sequence must have data with 3-dimension or 2-dimension, but got shape {seq_shape}"
        )
    result_seq = np.empty_like(position_seq)
    if len(seq_shape) == 3:
        for i in range(seq_shape[1]):
            for k in range(seq_shape[2]):
                result_seq[:, i, k] = signal.sosfilt(sos, position_seq[:, i, k])
    elif len(seq_shape) == 2:
        for i in range(seq_shape[1]):
            result_seq[:, i] = signal.sosfilt(sos, position_seq[:, i])

    return result_seq


def calc_link_lengths(keypoints: np.ndarray, links: List[Tuple[int, int]]):
    link_lengths = []
    for start, end in links:
        link_lengths.append(np.linalg.norm(keypoints[start] - keypoints[end]))
    return np.array(link_lengths)


def calc_scale_factor(source: np.ndarray, target: np.ndarray) -> float:
    source_lengths = calc_link_lengths(source, HAND_VISULIZATION_LINKS)
    target_lengths = calc_link_lengths(target, HAND_VISULIZATION_LINKS)
    return np.sum(target_lengths) / np.sum(source_lengths)




def callback(data):
    global joint
    joint=data.data
    joint=list(joint)
    joint=np.array(joint).reshape((21,3))
    print(joint)








# if __name__ == "__main__":
#     # filenames = glob.glob("/home/isaac/projects/shadow_robot/base/src/sr_interface/shadow-hand-project/retarget/hand_pose/*joint*.npy")
#     # filenames = natsorted(filenames)
#     # target = np.stack([np.load(filename) for filename in filenames])

#     # target = np.load("/home/isaac/projects/shadow_robot/base/src/sr_interface/shadow-hand-project/retarget/leap_motion.npy")[::4]

#     # target = target - target[:, 0:1, :]

#     # plot_hand_motion_keypoints(target)

#     # plot_hand_motion_keypoints(target, "target_glove_1.gif")
#     # exit(0)
    
#     rospy.init_node('retarget2robot',anonymous=True)
#     _thread.start_new_thread( thread_publisher, ("Thread-1", ) )

#     rospy.Subscriber('leap_hand',Float64MultiArray,callback)
#     print("ready\n")
#     rospy.spin()