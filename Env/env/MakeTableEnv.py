import numpy as np
from isaacsim import SimulationApp
import torch

simulation_app = SimulationApp({"headless": False})
import numpy as np
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka import Franka
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.franka import KinematicsSolver
from omni.isaac.core.utils.types import ArticulationAction
import torch
import sys
sys.path.append("/home/isaac/GarmentLab/")
from Env.Utils.transforms import euler_angles_to_quat
from Env.Utils.transforms import quat_diff_rad
from Env.env.BaseEnv import BaseEnv
from Env.Garment.Garment import Garment
from Env.Robot.Franka.MyFranka import MyFranka
from Env.env.Control import Control
from Env.Config.GarmentConfig import GarmentConfig
from Env.Config.FrankaConfig import MobileFrankaConfig
from Env.Config.DeformableConfig import DeformableConfig
from Env.Garment.TableCloth import TableCloth
from Env.Rigid.Rigid import RigidTable
from Env.Robot.MobileFranka import MobileFranka
import open3d as o3d

class MakeTableEnv(BaseEnv):
    def __init__(self,garment_config:GarmentConfig=None,franka_config:MobileFrankaConfig=None,Deformable_Config:DeformableConfig=None):
        BaseEnv.__init__(self,garment=True)
        if garment_config is None:
            self.garment_config=[GarmentConfig(ori=np.array([0,0,0]))]
        else:
            self.garment_config=garment_config
        self.garment:list[TableCloth]=[]
        for garment_config in self.garment_config:
            self.garment.append(TableCloth(self.world,garment_config))

        self.robot_config=MobileFrankaConfig(pos=[np.array([1.2,1.2,0])])
            
        self.robot=MobileFranka(self.world,self.robot_config)
            
        # self.table=RigidTable(self.world)
        # self.robots=self.import_franka(self.franka_config)
        # self.control=Control(self.world,self.robots,[self.garment[0]])
        

if __name__=="__main__":
    cloth_config=GarmentConfig(usd_path=None,pos=np.array([0,0,1]),scale=np.array([2,2,1]),ori=np.array([0,0,0]))
    cloth_config.particle_contact_offset=0.01
    cloth_config.friction=0.7
    cloth_config.solid_rest_offset=0.008
    env=MakeTableEnv([cloth_config])
    env.reset()
    env.robot.base_move_to(np.array([-1.2,1.2,0]))
    while 1:
        env.step()
        


