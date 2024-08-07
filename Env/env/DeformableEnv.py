import numpy as np
from isaacsim import SimulationApp
import torch
import sys
sys.path.append("/home/user/GarmentLab")
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
from Env.Rigid.Rigid import RigidHang


class DeformableEnv(BaseEnv):
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
        self.robots=self.import_franka(self.franka_config)
        self.def_control=DefControl(self.world,self.robots,[self.deformable])
        self.rigid = RigidHang()
        
        

        # self.robots=self.import_franka(self.franka_config)
        # self.def_control=DefControl(self.world,self.robots,[self.deformable])
        
        
if __name__=="__main__":
    env=DeformableEnv()
    env.reset()
    # env.robots[0].movel(np.array([0.65,0,0.5]))
    # env.def_control.grasp([np.array([0.65,0,0.08])],[None],[True])
    # env.def_control.move([np.array([0.65,0,0.5])],[None],[True])
    # env.def_control.ungrasp([False])
    # points=env.deformable.get_vertices_positions()
    # points=points.reshape(-1,3)
    # pcd=o3d.geometry.PointCloud()
    # pcd.points=o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd])
    while 1:
        env.step()
        
        