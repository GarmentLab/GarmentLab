from Env.Config.FrankaConfig import FrankaConfig
import numpy as np
from isaacsim import SimulationApp
import torch

simulation_app = SimulationApp({"headless": False})
import numpy as np
from Env.Deformable.Deformable import Deformable
from Env.Config.DeformableConfig import DeformableConfig
from Env.env.BaseEnv import BaseEnv
from Env.Garment.Garment import Garment
from Env.Config.GarmentConfig import GarmentConfig
import numpy as np

class CustomizedObject(BaseEnv):
    def __init__(self):
        BaseEnv.__init__(self,deformable=True)
        self.deformable_config=DeformableConfig(usd_path="/home/user/GarmentLab/Assets/RealWorld/12_WinnieBear/12_mesh.usd",
                                                scale=np.array([0.001,0.001,0.001]),
                                                pos=np.array([0.3,0.5,0.3]),
                                                visual_material_usd=None,
                                                ori=np.array([np.pi/5,np.pi/2,0]))
        self.deformable=Deformable(self.world,self.deformable_config)
        self.garment_config=GarmentConfig(usd_path="/home/user/GarmentLab/Assets/Garment/Tops/Hooded_Lsleeve_FrontOpen/THLO_Jacket065/THLO_Jacket065_obj.usd",
                                        #   visual_material_usd=None,
                                          pos=np.array([1,-0.3,0.3]),)
        self.garment=Garment(self.world,self.garment_config)
        self.franka_config=FrankaConfig()
        # self.robots=self.import_franka(self.franka_config)

if __name__=="__main__":
    env=CustomizedObject()
    env.reset()
    while 1:
        env.step()