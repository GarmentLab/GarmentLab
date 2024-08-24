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

class BimanualEnv(BaseEnv):
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
        particle_system = None
        for config in garment_config:
            garment = Garment(self.world, config, particle_system=particle_system)
            self.garment.append(garment)
            particle_system = garment.get_particle_system()
        self.control=Control(self.world,self.robots,self.garment)
        # while simulation_app.is_running():
        #     simulation_app.update()

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

    def get_reward(self,rotation, garment_id = 0):
        for i in range(20):
            self.world.step()
        pts = self.garment[garment_id].get_vertice_positions()
        pts = self.Rotation(rotation,pts)
        path = "/home/isaac/GarmentLab/Assets/succ_example/fling_test_example.txt"
        np.savetxt(path, pts)
        coverage_trial = self.get_current_covered_area(pts)
        self.world.stop()
        self.garment[0].set_pose(pos=np.array([0,0,0.5]),ori=np.array([1,0,0,0]))
        self.world.play()
        for i in range(20):
            self.world.step()
        pts = self.garment[garment_id].get_vertice_positions()
        path = "/home/isaac/GarmentLab/Assets/succ_example/fling_standard_example.txt"
        np.savetxt(path, pts)
        coverage_standard = self.get_current_covered_area(pts)
        if  coverage_trial/coverage_standard > 0.75:
            print("################ succ ################ : coverage ratio", coverage_trial/coverage_standard )
            return True
        else:
            print("################ fail ################ : coverage ratio", coverage_trial/coverage_standard )
            return False


    def get_current_covered_area(self, pos, cloth_particle_radius: float = 0.00625):
        """
        Calculate the covered area by taking max x,y cood and min x,y 
        coord, create a discritized grid between the points
        :param pos: Current positions of the particle states
        """
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 1])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 1])
        init = np.array([min_x, min_y])
        span = np.array([max_x - min_x, max_y - min_y]) / 100.
        pos2d = pos[:, [0, 2]]

        offset = pos2d - init
        slotted_x_low = np.maximum(np.round((offset[:, 0] - cloth_particle_radius) / span[0]).astype(int), 0)
        slotted_x_high = np.minimum(np.round((offset[:, 0] + cloth_particle_radius) / span[0]).astype(int), 100)
        slotted_y_low = np.maximum(np.round((offset[:, 1] - cloth_particle_radius) / span[1]).astype(int), 0)
        slotted_y_high = np.minimum(np.round((offset[:, 1] + cloth_particle_radius) / span[1]).astype(int), 100)
        # Method 1
        grid = np.zeros(10000)  # Discretization
        listx = self.vectorized_range1(slotted_x_low, slotted_x_high)
        listy = self.vectorized_range1(slotted_y_low, slotted_y_high)
        listxx, listyy = self.vectorized_meshgrid1(listx, listy)
        idx = listxx * 100 + listyy
        idx = np.clip(idx.flatten(), 0, 9999)
        grid[idx] = 1
        return np.sum(grid) * span[0] * span[1]

            
    def vectorized_range1(self,start, end):
        """  Return an array of NxD, iterating from the start to the end"""
        N = int(np.max(end - start)) + 1
        idxes = np.floor(np.arange(N) * (end - start)
                        [:, None] / N + start[:, None]).astype('int')
        return idxes

    def vectorized_meshgrid1(self,vec_x, vec_y):
        """vec_x in NxK, vec_y in NxD. Return xx in Nx(KxD) and yy in Nx(DxK)"""
        N, K, D = vec_x.shape[0], vec_x.shape[1], vec_y.shape[1]
        vec_x = np.tile(vec_x[:, None, :], [1, D, 1]).reshape(N, -1)
        vec_y = np.tile(vec_y[:, :, None], [1, 1, K]).reshape(N, -1)
        return vec_x, vec_y

    
    def get_keypoint_groups(self,xzy : np.ndarray):
        x = xzy[:, 0]
        y = xzy[:, 2]

        cloth_height = float(np.max(y) - np.min(y))
        cloth_width = float(np.max(x) - np.min(x))
        
        max_ys, min_ys = [], []
        num_bins = 40
        x_min, x_max = np.min(x),  np.max(x)
        mid = (x_min + x_max)/2
        lin = np.linspace(mid, x_max, num=num_bins)
        for xleft, xright in zip(lin[:-1], lin[1:]):
            max_ys.append(-1 * y[np.where((xleft < x) & (x < xright))].min())
            min_ys.append(-1 * y[np.where((xleft < x) & (x < xright))].max())

        #plot the rate of change of the shirt height wrt x
        diff = np.array(max_ys) - np.array(min_ys)
        roc = diff[1:] - diff[:-1]

        #pad beginning and end
        begin_offset = num_bins//5
        end_offset = num_bins//10
        roc[:begin_offset] = np.max(roc[:begin_offset])
        roc[-end_offset:] = np.max(roc[-end_offset:])
        
        #find where the rate of change in height dips, it corresponds to the x coordinate of the right shoulder
        right_x = (x_max - mid) * (np.argmin(roc)/num_bins) + mid

        #find where the two shoulders are and their respective indices
        xzy_copy = xzy.copy()
        xzy_copy[np.where(np.abs(xzy[:, 0] - right_x) > 0.01), 2] = 10
        right_pickpoint_shoulder = np.argmin(xzy_copy[:, 2])
        right_pickpoint_shoulder_pos = xzy[right_pickpoint_shoulder, :]

        left_shoulder_query = np.array([-right_pickpoint_shoulder_pos[0], right_pickpoint_shoulder_pos[1], right_pickpoint_shoulder_pos[2]])
        left_pickpoint_shoulder = (np.linalg.norm(xzy - left_shoulder_query, axis=1)).argmin()
        left_pickpoint_shoulder_pos = xzy[left_pickpoint_shoulder, :]

        #top left and right points are easy to find
        pickpoint_top_right = np.argmax(x - y)
        pickpoint_top_left = np.argmax(-x - y)

        #to find the bottom right and bottom left points, we need to first make sure that these points are
        #near the bottom of the cloth
        pickpoint_bottom = np.argmax(y)
        diff = xzy[pickpoint_bottom, 2] - xzy[:, 2]
        idx = diff < 0.1
        locations = np.where(diff < 0.1)
        points_near_bottom = xzy[idx, :]
        x_bot = points_near_bottom[:, 0]
        y_bot = points_near_bottom[:, 2]

        #after filtering out far points, we can find the argmax as usual
        pickpoint_bottom_right = locations[0][np.argmax(x_bot + y_bot)]
        pickpoint_bottom_left = locations[0][np.argmax(-x_bot + y_bot)]

        self.bottom_right=pickpoint_bottom_right,
        self.bottom_left=pickpoint_bottom_left,
        self.top_right=pickpoint_top_right,
        self.top_left=pickpoint_top_left,
        self.right_shoulder=right_pickpoint_shoulder,
        self.left_shoulder=left_pickpoint_shoulder,
        

        # get middle point
        middle_point_pos=np.array([0,0.1,0])
        self.middle_point=self.find_nearest_index(middle_point_pos)

        # get left and right points
        middle_band=np.where(np.abs(self.init_position[:,2]-middle_point_pos[2])<0.1)
        self.left_x=np.min(self.init_position[middle_band,0])
        self.right_x=np.max(self.init_position[middle_band,0])
        self.left_point=self.find_nearest_index([self.left_x,0,-0.3])
        self.right_point=self.find_nearest_index([self.right_x,0,-0.3])

        # get top and bottom points
        x_middle_band=np.where(np.abs(self.init_position[:,0]-self.init_position[self.middle_point,0])<0.1)
        self.top_y=np.min(self.init_position[x_middle_band,2])
        self.bottom_y=np.max(self.init_position[x_middle_band,2])
        self.top_point=self.find_nearest_index([0,0,self.top_y])
        self.bottom_point=self.find_nearest_index([0,0,self.bottom_y])
        # self.top_point=np.argmax(self.init_position[x_middle_band,2])
        # self.bottom_point=np.argmin(self.init_position[x_middle_band,2])

        self.keypoint=[self.bottom_left,self.bottom_right,self.top_left,self.top_right,self.left_shoulder,self.right_shoulder,self.middle_point,self.left_point,self.right_point,self.top_point,self.bottom_point]



        

