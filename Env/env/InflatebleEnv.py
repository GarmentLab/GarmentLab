import numpy as np
from isaacsim import SimulationApp
import torch

simulation_app = SimulationApp({"headless": False})
import numpy as np

from Env.Utils.transforms import euler_angles_to_quat
import torch
from Env.Utils.transforms import quat_diff_rad
from Env.env.BaseEnv import BaseEnv
from Env.Garment.Garment import Garment
from Env.Robot.Franka.MyFranka import MyFranka
from Env.env.DefControl import DefControl
from Env.Config.GarmentConfig import GarmentConfig
from Env.Config.FrankaConfig import FrankaConfig
from Env.Deformable.Deformable import Deformable
from Env.Config.DeformableConfig import DeformableConfig
from Env.Config.SceneConfig import SceneConfig
from Env.Config.InflatebleConfig import InflatebleConfig
import open3d as o3d
from Env.Inflatable.Inflatable import Inflatable

class InflatebleEnv(BaseEnv):
    def __init__(self,inflatebleConfig:InflatebleConfig=None,frankaConfig:FrankaConfig=None):
        BaseEnv.__init__(self,deformable=True)
        if inflatebleConfig is None:
            self.inflateble_config=InflatebleConfig()
        else:
            self.inflateble_config=inflatebleConfig
        self.inflateble=Inflatable(self.world,self.inflateble_config)
        # if frankaConfig is None:
        #     self.franka_config=FrankaConfig()
        # else:
        #     self.franka_config=frankaConfig
        # self.robots=self.import_franka(self.franka_config)
        
if __name__=="__main__":
    env=InflatebleEnv()
    env.reset()
    while 1:
        env.step()
        