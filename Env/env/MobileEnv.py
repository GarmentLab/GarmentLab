import numpy as np
import sys

from isaacsim import SimulationApp

sys.path.append("/home/user/GarmentLab/")

app = SimulationApp({"headless": False})

from BaseEnv import BaseEnv
from Env.Robot.MobileFranka import MobileFranka
from Env.Config.DexConfig import DexConfig

from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, VisualCuboid

class MobileEnv(BaseEnv):
    def __init__(self):
        super().__init__()

        config = DexConfig(env=self, app=app)

        self.robot = MobileFranka(config)

        self.target = VisualCuboid(prim_path = "/World/target",
            name = "target",
            position = np.array([0.8, 0.7, 0.6]),
            scale = np.array([0.04, 0.04, 0.04]),
            color = np.array([0.5, 0.5, 0.5])
        )


env = MobileEnv()
env.reset()

env.robot.base_move_to(np.array([5, 0, 0]))
env.robot.set_joint_positions([0], [2])
env.robot.base_face_to(np.array([0, 0, 0]))

while True:
    env.step()
    
    # target_pos, target_ori = env.target.get_world_pose()
    # env.robot.move_to(target_pos, target_ori, angular_type="quat")
    

