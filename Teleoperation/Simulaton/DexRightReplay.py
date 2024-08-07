import sys
import numpy as np

from omni.isaac.kit import SimulationApp


app = SimulationApp({"headless": False})

from Env.env.BaseEnv import BaseEnv
from Env.Robot.DexRight import DexRight
from Env.Config.DexConfig import DexConfig
from Env.Garment.Garment import Garment
from Env.Config.GarmentConfig import GarmentConfig

from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, VisualCuboid

class DexRightEnv(BaseEnv):
    def __init__(self):
        super().__init__()

        config = DexConfig(env=self, app=app, translation=np.array([0, 0, 0.5]))
        garment_config = GarmentConfig(
            usd_path="/home/user/GarmentLab/Assets/Garment/Tops/Hooded_Lsleeve_FrontOpen/THLO_Jacket065/THLO_Jacket065_obj.usd",
            pos=np.array([1, -0.6, 0.3]), 
            ori=np.array([0, 0, 0])
        )

        self.garment = Garment(self.world, garment_config)
        self.robot = DexRight(config)

        self.pedestal = VisualCuboid(
            prim_path = "/World/pedestal", 
            name = "pedestal", 
            position = np.array([0, 0, 0.25]), 
            scale = np.array([0.2, 0.2, 0.5]),
            color = np.array([0.5, 0.5, 0.5])
        )

        # self.pedestal1 = FixedCuboid(
        #     prim_path = "/World/pedestal1", 
        #     name = "pedestal1", 
        #     position = np.array([1, -0.04, 0.1]), 
        #     scale = np.array([0.1, 0.1, 0.2]),
        #     color = np.array([0.5, 0.5, 0.5])
        # )

    def replay_callback(self, data):
        self.robot.set_joint_positions(data["joint_pos"])

env = DexRightEnv()
env.reset()
env.replay("Assets/Replays/20240806-08:27:54.npy")

while True:
    env.step()
