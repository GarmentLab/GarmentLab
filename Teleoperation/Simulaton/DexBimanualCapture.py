import sys
import numpy as np

from omni.isaac.kit import SimulationApp

app = SimulationApp({"headless": False})


from Env.env.BaseEnv import BaseEnv
from Teleoperation.Listener import Listener
from Env.Robot.DexLeft import DexLeft
from Env.Robot.DexRight import DexRight
from Env.Config.DexConfig import DexConfig
from Env.Config.GarmentConfig import GarmentConfig
from Env.Garment.Garment import Garment

from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, VisualCuboid

class Dexbimanual(BaseEnv):
    def __init__(self):
        super().__init__()

        configR = DexConfig(env=self, app=app, translation=np.array([0, -0.5, 0.5]))
        configL = DexConfig(env=self, app=app, translation=np.array([0, 0.5, 0.5]))
        garment_config = GarmentConfig(
            usd_path="/home/isaac/GarmentLab/Assets/Garment/Tops/Hooded_Lsleeve_FrontOpen/THLO_Jacket065/THLO_Jacket065_obj.usd",
            pos=np.array([1, -0.6, 0.3]), 
            ori=np.array([0, 0, 0])
        )

        self.garment = Garment(self.world, garment_config)

        self.robotR = DexRight(configR)
        self.robotL = DexLeft(configL)
        self.listener = Listener(app, "handler")

        self.appended_info = []

        self.pedestal1 = VisualCuboid(
            prim_path = "/World/pedestal1", 
            name = "pedestal1", 
            position = np.array([0, -0.5, 0.25]), 
            scale = np.array([0.2, 0.2, 0.5]),
            color = np.array([0.5, 0.5, 0.5])
        )

        self.pedestal2 = VisualCuboid(
            prim_path = "/World/pedestal2", 
            name = "pedestal2", 
            position = np.array([0, 0.5, 0.25]), 
            scale = np.array([0.2, 0.2, 0.5]),
            color = np.array([0.5, 0.5, 0.5])
        )

    def record_callback(self, step_size):
        self.savings.append({ 
            "joint_pos_L": self.robotL.get_joint_positions(), 
            "joint_pos_R": self.robotR.get_joint_positions(),
            "appended_info": self.appended_info 
        })

env = Dexbimanual()
env.reset()
env.listener.launch()

# env.record()

while env.context.is_playing():
    env.step()

    hand_pose_rawR, arm_pose_rawR, hand_joint_poseR, wrist_posR, wrist_oriR = env.listener.get_pose("right")
    hand_pose_rawL, arm_pose_rawL, hand_joint_poseL, wrist_posL, wrist_oriL = env.listener.get_pose("left")

    env.appended_info = [hand_pose_rawL, arm_pose_rawL, hand_pose_rawR, arm_pose_rawR]

    if hand_joint_poseR is not None:
        env.robotR.step(wrist_posR, wrist_oriR, angular_type="euler")
        env.robotR.set_joint_positions(hand_joint_poseR, env.robotR.hand_dof_indices)

    if hand_joint_poseL is not None:
        env.robotL.step(wrist_posL, wrist_oriL, angular_type="euler")
        env.robotL.set_joint_positions(hand_joint_poseL, env.robotL.hand_dof_indices)

# env.stop_record()