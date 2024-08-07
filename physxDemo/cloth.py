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
from Env.Garment.Garment import Garment
from Env.Robot.Franka.MyFranka import MyFranka
from Env.env.Control import Control
from Env.Config.GarmentConfig import GarmentConfig
from Env.Config.FrankaConfig import FrankaConfig
from Env.Config.DeformableConfig import DeformableConfig
import open3d as o3d

class ClothDemoEnv(BaseEnv):
    def __init__(self,garment_config_list:list):
        BaseEnv.__init__(self,garment=True)
     
        for garment_config in garment_config_list:
            garment=Garment(self.world,garment_config)
        
    
        


if __name__=="__main__":
    garment_config_list=[]
    garment_config1=GarmentConfig(usd_path="/home/user/GarmentLab/Assets/Garment/Dress/Long_LongSleeve/DLLS_Dress355/DLLS_Dress355_obj.usd",
                                  visual_material_usd="/home/user/GarmentLab/Assets/Material/linen_Beige.usd",
                                  pos=np.array([0,-1,0.5]),particle_contact_offset=0.01)
    garment_config2=GarmentConfig(usd_path="/home/user/GarmentLab/Assets/Garment/Dress/Long_LongSleeve/DLLS_Dress355/DLLS_Dress355_obj.usd",
                                  visual_material_usd="/home/user/GarmentLab/Assets/Material/linen_Beige.usd",
                                  pos=np.array([0,0,0.5]),particle_contact_offset=0.02)
    garment_config3=GarmentConfig(usd_path="/home/user/GarmentLab/Assets/Garment/Dress/Long_LongSleeve/DLLS_Dress355/DLLS_Dress355_obj.usd",
                                  visual_material_usd="/home/user/GarmentLab/Assets/Material/linen_Beige.usd",
                                  pos=np.array([0,1,0.5]),particle_contact_offset=0.04)
    garment_config_list.append(garment_config1)
    garment_config_list.append(garment_config2)
    garment_config_list.append(garment_config3)
    env=ClothDemoEnv(garment_config_list=garment_config_list)
    env.reset()
    
    while 1:
        env.step()