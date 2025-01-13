import sys
import numpy as np

from isaacsim import SimulationApp

# sys.path.append("/home/user/GarmentLab/")

app = SimulationApp({"headless": False})

from BaseEnv import BaseEnv
from Env.Garment.Garment import Garment
from Env.Robot.DexRight import DexRight
from Env.Config.DexConfig import DexConfig
from Env.Config.GarmentConfig import GarmentConfig

from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, VisualCuboid

class DexRightEnv(BaseEnv):
    def __init__(self):
        super().__init__()

        config = DexConfig(env=self, app=app, translation=np.array([0, 0, 1]))
        garment_config = GarmentConfig(pos=np.array([1, -0,5, 0.3]), ori=np.array([0, 0, 0]))

        self.robot = DexRight(config)
        self.garment = Garment(self.world, garment_config)

        self.target = VisualCuboid(prim_path = "/World/target",
            name = "target",
            position = np.array([0.8, 0.7, 0.6]),
            scale = np.array([0.04, 0.04, 0.04]),
            color = np.array([0.5, 0.5, 0.5])
        )


env = DexRightEnv()
env.reset()

while True:
    env.step()
    target_pos, target_ori = env.target.get_world_pose()
    env.robot.move_to(target_pos, target_ori, angular_type="quat")
