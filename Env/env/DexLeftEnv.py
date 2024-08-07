import sys
import numpy as np

from omni.isaac.kit import SimulationApp

sys.path.append("/home/isaac/GarmentLab/")

app = SimulationApp({"headless": False})

from BaseEnv import BaseEnv
from Env.Robot.DexLeft import DexLeft
from Env.Config.DexConfig import DexConfig

from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, VisualCuboid

class DexLeftEnv(BaseEnv):
    def __init__(self):
        super().__init__()

        config = DexConfig(env=self, app=app, translation=np.array([0, 0, 1]))

        self.robot = DexLeft(config)

        self.target = VisualCuboid(
            prim_path = "/World/target",
            name = "target",
            position = np.array([0.8, 0.7, 0.6]),
            scale = np.array([0.04, 0.04, 0.04]),
            color = np.array([0.5, 0.5, 0.5])
        )

env = DexLeftEnv()
env.reset()
env.robot.record()

while env.robot.context.is_playing():
    env.step()
    target_pos, target_ori = env.target.get_world_pose()
    env.robot.move_to(target_pos, target_ori, angular_type="quat")

env.robot.record_stop()