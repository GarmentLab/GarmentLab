import sys
import numpy as np

from omni.isaac.kit import SimulationApp


app = SimulationApp({"headless": False})

from Env.env.BaseEnv import BaseEnv
from Env.Robot.DexLeft import DexLeft
from Env.Config.DexConfig import DexConfig

class DexLeftEnv(BaseEnv):
    def __init__(self):
        super().__init__()

        config = DexConfig(env=self, app=app, translation=np.array([0, 0, 1]))

        self.robot = DexLeft(config)

    def replay_callback(self, data):
        self.robot.set_joint_positions(data["joint_pos"])

env = DexLeftEnv()
env.reset()
env.replay("Replays/20240805-08:41:57.npy")

while True:
    env.step()
