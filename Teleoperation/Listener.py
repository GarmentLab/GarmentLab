import sys
sys.path.append("/home/user/GarmentLab")

sys.path.append("/home/user/GarmentLab/Assets/LeapMotion/leap-sdk-python3")
sys.path.append("/home/user/GarmentLab/Teleoperation/retarget")

import Leap
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from retarget import optimize

def leap_vector_to_numpy(vector) -> np.ndarray:
    return np.array([vector.x, vector.y, vector.z])

def leap_hand_to_keypoints(hand) -> np.ndarray:
    keypoints = np.zeros((21, 3))
    armpoints = np.zeros((4,3))
    keypoints[0, :] = leap_vector_to_numpy(hand.wrist_position)

    for finger in hand.fingers:
        finger_index = finger.type
        for bone_index in range(0, 4):
            bone = finger.bone(bone_index)
            index = 1 + finger_index * 4 + bone_index
            keypoints[index, :] = leap_vector_to_numpy(bone.next_joint)

    armpoints[0, :] = leap_vector_to_numpy(hand.direction)
    armpoints[1, :] = leap_vector_to_numpy(hand.palm_normal)
    armpoints[2, :] = leap_vector_to_numpy(hand.wrist_position)
    armpoints[3, :] = leap_vector_to_numpy(hand.palm_position)
    return keypoints, armpoints

class DeviceListener(Leap.Listener):
    def __init__(self) -> None:
        super().__init__()
        self.cache = []
        self.hand_pose, self.arm_pose = {}, {}

    def on_connect(self, controller):
        print("Device connected")

    def on_disconnect(self, controller):
        print("Device disconnected")

    def on_frame(self, controller):
        frame = controller.frame()

        for hand in frame.hands:
            side = "left" if hand.is_left else "right"
            self.hand_pose[side], self.arm_pose[side] = leap_hand_to_keypoints(hand)

class thread_handler(threading.Thread):
    def __init__(self, listener, thread_name):
        self.listener = listener
        threading.Thread.__init__(self, name=thread_name)

    def run(self):
        self.listener.watch()

TRANSFORM_MAT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
BASE_ROT_MAT_INV = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
ARM_HOME_JOINT_POSITION = [0, -1.25, 2.00, -np.pi / 4, np.pi / 2, 0]
SCALE = np.array([130, 130, 130])
HOME_WRIST_POS = np.array([0.9, 0.0, 0.7])

class Listener:
    def watch(self):
        self.d_listener = DeviceListener()
        controller = Leap.Controller()
        controller.add_listener(self.d_listener)

        if self.app is None:
            while self.app.is_running():
                time.sleep(1)
        else:
            input("Press any key to unregister")

        self.unregistered = True
        controller.remove_listener(self.d_listener)


    def __init__(self, app, thread_name):
        self.app = app
        self.unregistered = False
        self.handler = thread_handler(self, thread_name)
        self.cache = { "left": [], "right": [] }
        self.home_wrist_pos_raw = { }

    def launch(self):
        self.handler.start()

    def get_pose(self, side):
        if side not in self.d_listener.hand_pose:
            return None, None, None, None, None

        hand_pose_raw, arm_pose_raw = self.d_listener.hand_pose[side], self.d_listener.arm_pose[side]

        hand_joint_pose = optimize(hand_pose_raw, side)

        # these joints of left hand are mysteriously reversed
        if side == "left":
            reverse_list = [2, 6, 10, 15, 22]
            for joint in reverse_list:
                hand_joint_pose[joint] = -hand_joint_pose[joint]

        palm_normal = -TRANSFORM_MAT @ np.array(arm_pose_raw[1])
        direction = TRANSFORM_MAT @ np.array(arm_pose_raw[0])

        palm_normal /= np.linalg.norm(palm_normal)
        direction /= np.linalg.norm(direction)
        target_y = np.cross(direction, palm_normal)
        target_y /= np.linalg.norm(target_y)

        # rotation matrix of wrist orientation (x: forward, y: left, z: up)
        r = np.array([direction, target_y, palm_normal]).T @ BASE_ROT_MAT_INV
        wrist_ori = R.from_matrix(r).as_euler('xyz')
        wrist_pos = TRANSFORM_MAT @ np.array(arm_pose_raw[2]) / SCALE
        wrist_pos[0] += 1.6
        wrist_pos[2] -= 1
        wrist_pos[1] += 1.5 if side == "right" else -1.5

        if side not in self.home_wrist_pos_raw:
            self.cache[side].append(wrist_pos)
            if len(self.cache[side]) <= 50:
                return None, None, None, None, None
            self.home_wrist_pos_raw[side] = sum(self.cache[side]) / len(self.cache[side])

        wrist_pos = wrist_pos - (self.home_wrist_pos_raw[side] - HOME_WRIST_POS)



        return hand_pose_raw, arm_pose_raw, hand_joint_pose, wrist_pos, wrist_ori
