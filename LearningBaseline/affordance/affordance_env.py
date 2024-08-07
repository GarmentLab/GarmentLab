import numpy as np
from isaacsim import SimulationApp
import torch
import os

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
from Env.Rigid.Rigid import RigidAfford
from Env.Robot.Franka.MyFranka import MyFranka
from Env.env.Control import Control
from Env.Config.GarmentConfig import GarmentConfig
from Env.Config.FrankaConfig import FrankaConfig
from Env.Config.DeformableConfig import DeformableConfig

class AffordanceEnv(BaseEnv):
    def __init__(self,garment_config:GarmentConfig=None,franka_config:FrankaConfig=None,Deformable_Config:DeformableConfig=None, task_config = None):
        BaseEnv.__init__(self)
        if garment_config is None:
            self.garmentconfig=GarmentConfig()
        else:
            self.garmentconfig=garment_config
        if franka_config is None:
            self.robotConfig=FrankaConfig(franka_num=1,pos=[np.array([0,0,0]),np.array([0,-1,0])],ori=[np.array([0,0,-np.pi/2]),np.array([0,0,np.pi/2])])
        else:
            self.robotConfig=franka_config
        self.robots=self.import_franka(self.robotConfig)
        self.garment = list()
        particle_system = None
        for config in garment_config:
            garment = Garment(self.world, config, particle_system=particle_system)
            self.garment.append(garment)
            particle_system = garment.get_particle_system()
        self.control=Control(self.world,self.robots,self.garment)
        self.rigid = RigidAfford()
        self.target_point = np.array(task_config["target_point"])
        self.task_name = task_config["task_name"]
        self.garment_name = task_config["garment_name"]


        self.root_path = f"/home/user/isaacgarment/affordance/{self.task_name}_{self.garment_name}"
        if not os.path.exists(self.root_path):
            os.mkdir(self.root_path)

        self.trial_path = os.path.join(self.root_path, "trial_pts.npy")
        self.model_path = os.path.join(self.root_path, "model.npy")

    def get_reward(self, standard_model, garment_id = 0):
        succ_example = np.loadtxt(standard_model)

        pts = self.garment[garment_id].get_vertice_positions()
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        max_scale = np.max(np.abs(pts))
        pts = pts / max_scale
        dist = np.linalg.norm(pts - succ_example, ord=2)
        if dist < 0.1:
            print("################ succ ################")
            return True
        else:
            print("################ fail ################")
            return False

    def get_demo(self, assign_point):
        self.reset()
        self.control.robot_reset()
        for _ in range(20):
            self.world.step()
        point=np.array(assign_point)
        self.control.grasp(pos=[point],ori=[None],flag=[True])
        self.control.move(pos=[self.target_point],ori=[None],flag=[True])
        self.control.move(pos=[self.target_point+np.array([0.1,0,0])],ori=[None],flag=[True])
        for _ in range(100):
            self.world.step()
        final_data=self.garment[0].get_vertice_positions()
        final_path=self.model_path
        np.save(final_path,final_data)
        self.control.ungrasp([False])
        for _ in range(10):
            self.world.step()

    def exploration(self):
        for i in range(800):
            self.reset()
            self.control.robot_reset()
            for _ in range(20):
                self.world.step()
            point=self.allocate_point(i, save_path=self.trial_path)
            self.control.grasp(pos=[point],ori=[None],flag=[True])
            self.control.move(pos=[self.target_point],ori=[None],flag=[True])
            self.control.move(pos=[self.target_point+np.array([0.1,0,0])],ori=[None],flag=[True])
            for _ in range(100):
                self.world.step()
            final_data=self.garment[0].get_vertice_positions()
            final_path=os.path.join(self.root_path,f"final_pts_{i}.npy")
            np.save(final_path,final_data)
            self.control.ungrasp([False])
            for _ in range(10):
                self.world.step()
            self.world.stop()


    def allocate_point(self, index, save_path):
        if index==0:
            self.selected_pool=self.garment[0].get_vertice_positions()*self.garment[0].garment_config.scale
            q = euler_angles_to_quat(self.garment[0].garment_config.ori)
            self.selected_pool=self.Rotation(q,self.selected_pool)
            centroid, _ = self.garment[0].garment_mesh.get_world_pose()
            print(centroid)
            self.selected_pool=self.selected_pool + centroid
            np.savetxt("/home/sim/GarmentLab/select.txt",self.selected_pool)
            indices=torch.randperm(self.selected_pool.shape[0])[:800]
            self.selected_pool=self.selected_pool[indices]
            np.save(save_path, self.selected_pool)
        point=self.selected_pool[index]
        return point

    def Rotation(self,q,vector):
        vector = torch.from_numpy(vector).to(torch.float64)
        q0=q[0].item()
        q1=q[1].item()
        q2=q[2].item()
        q3=q[3].item()
        R=torch.tensor(
            [
                [1-2*q2**2-2*q3**2,2*q1*q2-2*q0*q3,2*q1*q3+2*q0*q2],
                [2*q1*q2+2*q0*q3,1-2*q1**2-2*q3**2,2*q2*q3-2*q0*q1],
                [2*q1*q3-2*q0*q2,2*q2*q3+2*q0*q1,1-2*q1**2-2*q2**2],
            ]
        ).to(torch.float64)
        vector=torch.mm(vector,R.transpose(1,0))
        return vector.numpy()