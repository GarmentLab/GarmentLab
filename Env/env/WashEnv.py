import numpy as np
from isaacsim import SimulationApp
import torch
import sys
sys.path.append("/home/user/GarmentLab")
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
from Env.Fluid.Fluid import Fluid
from Env.Config.FrankaConfig import FrankaConfig
from Env.Config.GarmentConfig import GarmentConfig
from Env.Config.FluidConfig import FluidConfig
from Env.env.Control import Control
from omni.isaac.core.objects.cuboid import FixedCuboid


class WashEnv(BaseEnv):
    def __init__(self,usd_path):
        BaseEnv.__init__(self)
        self.franka_config=FrankaConfig(pos=[np.array([0,0,0])])
        self.robots=self.import_franka(self.franka_config)
        self.garment_config=GarmentConfig(pos=np.array([0.65,-0.5,0.5]),ori=np.array([0,0,np.pi/2]), usd_path=usd_path,particle_contact_offset=None)
        self.fluid_config=FluidConfig()
        self.fluid=Fluid(self.world,self.fluid_config)
        self.garment=Garment(self.world,self.garment_config,self.fluid.particle_system)
        self.control=Control(self.world,self.robots,[self.garment])




if __name__=="__main__":
    env=WashEnv()
    # env.reset()
    env.reset()
    print(env.garment.get_particle_system_id())
    while 1:
        env.step()
