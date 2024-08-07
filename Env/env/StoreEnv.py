import numpy as np
from isaacsim import SimulationApp
import torch
import sys
sys.path.append("/home/sim/GarmentLab")
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
import open3d as o3d
from Env.Rigid.Rigid import RigidStore


class StoreEnv(BaseEnv):
    def __init__(self,deformbaleConfig:DeformableConfig=None,frankaConfig:FrankaConfig=None):
        BaseEnv.__init__(self,deformable=True)
        if deformbaleConfig is None:
            self.deformable_config=DeformableConfig()
        else:
            self.deformable_config=deformbaleConfig
        self.deformable=Deformable(self.world,self.deformable_config)
        if frankaConfig is None:
            self.franka_config=FrankaConfig()
        else:
            self.franka_config=frankaConfig
        self.rigid = RigidStore()
        self.robots=self.import_franka(self.franka_config)
        self.def_control=DefControl(self.world,self.robots,[self.deformable])

    def get_reward(self):
        pos, _ = self.deformable.deformable.get_world_pose()
        x = pos[0]
        y = pos[1]
        z = pos[2]
        if x >-0.18 and z >0.3:
            print("############### succ ###############")
            return True
        else:
            print("############### fail ###############")
            return False
        
        
        

        
        