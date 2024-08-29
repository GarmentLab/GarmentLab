import sys
import numpy as np

from omni.isaac.kit import SimulationApp


app = SimulationApp({"headless": False})

from Env.env.BaseEnv import BaseEnv
from Teleoperation.Listener import Listener
from Env.Robot.DexLeft import DexLeft
from Env.Config.DexConfig import DexConfig
from Env.Config.GarmentConfig import GarmentConfig
from Env.Garment.Garment import Garment

from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, VisualCuboid

class DexLeftCapture(BaseEnv):
    def __init__(self):
        super().__init__()

        config = DexConfig(env=self, app=app, translation=np.array([0, 0, 0.5]))
        garment_config = GarmentConfig(pos=np.array([1, -0.6, 0.3]), ori=np.array([0, 0, 0]))

        self.garment = Garment(self.world, garment_config)

        self.robot = DexLeft(config)
        self.listener = Listener(app, "handler")
        self.appended_info = []

        self.pedestal = VisualCuboid(
            prim_path = "/World/pedestal", 
            name = "pedestal", 
            position = np.array([0, 0, 0.25]), 
            scale = np.array([0.2, 0.2, 0.5]),
            color = np.array([0.5, 0.5, 0.5])
        )

    def record_callback(self, step_size):
        self.savings.append({ "joint_pos": self.robot.get_joint_positions(), "appended_info": self.appended_info })

env = DexLeftCapture()
env.reset()
env.listener.launch()
env.record()

while env.context.is_playing():
    env.step()
    hand_pose_raw, arm_pose_raw, hand_joint_pose, wrist_pos, wrist_ori = env.listener.get_pose("left")
    env.appended_info = [hand_pose_raw, arm_pose_raw]
    if hand_joint_pose is not None:
        env.robot.step(wrist_pos, wrist_ori, angular_type="euler")
        env.robot.set_joint_positions(hand_joint_pose, env.robot.hand_dof_indices)
    
env.stop_record()