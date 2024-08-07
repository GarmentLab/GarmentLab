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
from Env.Utils.transforms import euler_angles_to_quat
import torch
from Env.Utils.transforms import quat_diff_rad
from Env.env.BaseEnv import BaseEnv
from Env.Deformable.Deformable import Deformable
from Env.Robot.Franka.MyFranka import MyFranka
from Env.env.Control import Control
from Env.Config.DeformableConfig import DeformableConfig
from Env.Config.FrankaConfig import FrankaConfig
from Env.Config.DeformableConfig import DeformableConfig
import open3d as o3d

class DeformableEnv(BaseEnv):
    def __init__(self,deformable_config_list:list):
        BaseEnv.__init__(self,garment=True)
     
        for deformable_config in deformable_config_list:
            object=Deformable(self.world,deformable_config)
        

if __name__=="__main__":
    deformable_config_list=[]
    deformable_config1=DeformableConfig(pos=np.array([-1,1,0.5]),youngs_modulus=1e3)
    deformable_config2=DeformableConfig(pos=np.array([0,1,0.5]),youngs_modulus=5e3)
    deformable_config3=DeformableConfig(pos=np.array([1,1,0.5]))
    deformable_config_list.append(deformable_config1)
    deformable_config_list.append(deformable_config2)
    deformable_config_list.append(deformable_config3)
    env=DeformableEnv(deformable_config_list= deformable_config_list)
    env.reset()
    
    while 1:
        env.step()