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
from Env.Human.Human import Human
from Env.Rigid.Rigid import Rigid,RigidVisual
from Env.Robot.Franka.MyFranka import MyFranka
from Env.env.Control import Control
from Env.Config.GarmentConfig import GarmentConfig
from Env.Config.FrankaConfig import FrankaConfig
from Env.Config.DeformableConfig import DeformableConfig
class HumanEnv(BaseEnv):
    def __init__(self,garment_config:GarmentConfig=None,franka_config:FrankaConfig=None,Deformable_Config:DeformableConfig=None):
        BaseEnv.__init__(self)
        if garment_config is None:
            self.garmentconfig=GarmentConfig()
        else:
            self.garmentconfig=garment_config
        if franka_config is None:
            self.robotConfig=FrankaConfig(franka_num=2,pos=[np.array([0,0,0]),np.array([0,-1,0])],ori=[np.array([0,0,-np.pi/2]),np.array([0,0,np.pi/2])])
        else:
            self.robotConfig=franka_config
        self.robots=self.import_franka(self.robotConfig)
        self.garment = list()
        
        RigidConfig={
                    "path":"/home/user/GarmentLab/Assets/Rigid/Daybed.usd",
                    "position":np.array([-0.61832, 0.79868, 0.82689]),
                    "orientation":euler_angles_to_quat(np.array([0.,0.,np.pi/2])),
                    "scale":np.array([0.014,0.006,0.004]),
                }
        rigid = Rigid(root_path="/World/Rigid", rigid_config=RigidConfig)
        rigid_visual = RigidVisual()
        human = Human(path = "/home/sim/GarmentLab/Assets/Human/human_2.usd")
        
        particle_system = None
        for config in garment_config:
            garment = Garment(self.world, config, particle_system=particle_system)
            self.garment.append(garment)
            particle_system = garment.get_particle_system()
        self.control=Control(self.world,self.robots,self.garment)
        
        

