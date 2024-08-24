import sys
import numpy as np

from omni.isaac.kit import SimulationApp

app = SimulationApp({"headless": False})

from Env.env.BaseEnv import BaseEnv
from Env.Robot.DexRight import DexRight
from Env.Robot.DexLeft import DexLeft
from Env.Config.DexConfig import DexConfig
from Env.Garment.Garment import Garment
from Env.Config.GarmentConfig import GarmentConfig

from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, VisualCuboid

class DexRightEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        configR = DexConfig(env=self, app=app, translation=np.array([0, -0.5, 0.5]))
        configL = DexConfig(env=self, app=app, translation=np.array([0, 0.5, 0.5]))
        garment_config = GarmentConfig(
            usd_path="/home/isaac/GarmentLab/Assets/Garment/Tops/Hooded_Lsleeve_FrontOpen/THLO_Jacket065/THLO_Jacket065_obj.usd",
            pos=np.array([1, -0.6, 0.3]), 
            ori=np.array([0, 0, 0]),
            visual_material_usd=None,
        )

        self.garment = Garment(self.world, garment_config)

        self.robotR = DexRight(configR)
        self.robotL = DexLeft(configL)

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

    def replay_callback(self, data):
        self.robotL.set_joint_positions(data["joint_pos_L"])
        self.robotR.set_joint_positions(data["joint_pos_R"])

env = DexRightEnv()
env.reset()
env.replay("Assets/Replays/20240806-08:47:16.npy")

while True:
    env.step()
