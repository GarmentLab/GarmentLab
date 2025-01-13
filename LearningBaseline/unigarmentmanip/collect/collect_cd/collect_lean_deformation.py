import os
import sys
sys.path.append(os.getcwd())
sys.path.append("/home/user/DexGarmentLab-master/DexGarmentLab-master/unigarment/unigarmentmlp")     
sys.path.append("/home/user/DexGarmentLab-master/DexGarmentLab-master/unigarment/unigarmentmlp/merger")    
import torch
import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

# import open3d as o3d
import random
import threading
from termcolor import cprint
import omni.replicator.core as rep
import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.core import World
from omni.isaac.core import SimulationContext
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, VisualCuboid
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.prims.rigid_prim import RigidPrim  
from omni.isaac.core.prims.rigid_prim_view import RigidPrimView 
from omni.isaac.core.prims.geometry_prim_view import GeometryPrimView
from omni.isaac.core.materials import OmniGlass
from omni.isaac.sensor import ContactSensor
from omni.physx.scripts import deformableUtils,particleUtils,physicsUtils
from omni.physx import acquire_physx_interface
from pxr import UsdGeom,UsdPhysics,PhysxSchema, Gf
from omni.isaac.core.utils.types import ArticulationAction

from Env.BaseEnv import BaseEnv

from Env_Config.Deformable.Deformable import Deformable

from Env_Config.Garment.Garment import Garment

from Env_Config.Robot.DexLeft_Ur10e import DexLeft_Ur10e
from Env_Config.Robot.DexRight_Ur10e import DexRight_Ur10e
from Env_Config.Robot.BimanualDex_Ur10e import Bimanual_Ur10e

from Env_Config.Camera.Recording_Camera import Recording_Camera

# from Env_Config.Teleoperation.Listener import Listener

# from Env_Config.Utils_Project.AttachmentBlock import AttachmentBlock
from Env_Config.Utils_Project.CollisionGroup import CollisionGroup
from Env_Config.Utils_Project.transforms import Rotation
from Env_Config.Utils_Project.file_operation import get_unique_filename
from unigarment.collect.collect_cd.utils import get_pcd2mesh_correspondence, get_mesh2pcd_correspondence, get_max_sequence_number
from unigarment.collect.pcd_utils import normalize_pcd_points, unnormalize_pcd_points, get_visible_indices
from Env.attachmentblock import AttachmentBlock

import shutil
import open3d as o3d

from unigarment.unigarmentmlp.predict import get_skeleton

class LiftGarment_Collect_Deformation(BaseEnv):
    def __init__(self, garment_path, garment_position, garment_orientation):
        # load BaseEnv
        super().__init__()
        
        # store garment path
        self.garment_path = []
        # load Garment Object
        self.garment = Garment(world=self.world,
                               usd_path=garment_path,
                               pos=garment_position,
                               ori=garment_orientation,
                               friction=10.0,
                               contact_offset=0.018,
                               rest_offset=0.015,
                               particle_contact_offset=0.018,
                               fluid_rest_offset=0.015,
                               solid_rest_offset=0.015,
                               particle_adhesion_scale=0.05,
                               particle_friction_scale=0.05,)
        
        self.garment_path.append(self.garment.garment_prim_path)
              
        # load camera
        self.recording_camera = Recording_Camera(camera_position=np.array([0.0, 0.5, 8]), camera_orientation=np.array([0, 90, -90]))
        
        # initialize world
        self.reset()
        
        # initialize camera, make sure the 'initialization' procedure is behind the 'reset' procedure.
        self.recording_camera.initialize(pc_enable=True, segment_prim_path_list=["/World/Garment/garment"])

        
        cprint("finish creating the world!", color='green')
        
        for i in range(50):
            self.step()
        # cprint(self.bimanual_dex.dexleft.get_joint_positions(), 'cyan')
        cprint("world ready!", color='green')
        
    def record_callback(self, step_size):
        
        joint_pos_L = self.bimanual_dex.dexleft.get_joint_positions()
        
        joint_pos_R = self.bimanual_dex.dexright.get_joint_positions()
        
        action = [*joint_pos_L, *joint_pos_R]
        
        self.saving_data.append({ 
            "action": action,
        })
    
        
    def create_attach_block(self, idx, init_position=np.array([0.0, 0.0, 1.0])):
        '''
        Create attachment block and update the collision group at the same time.
        '''
        # create attach block and finish attach
        print(f"idx: {idx}")
        self.attach = AttachmentBlock(self.world, self.stage, "/World/AttachmentBlock", [self.garment.garment_mesh_prim_path])
        self.attach.create_block(block_name=f"attach_{idx}", block_position=init_position, block_visible=True)
        print("attach finish!")
        # update attach collision group
        # self.collision.update_after_attach()
        for i in range(10):
            simulation_app.update()
        print("Update collision group successfully!")
        
    def set_attach_to_garment(self, attach_position):
        '''
        push attach_block to new grasp point and attach to the garment
        '''
        # set the position of block
        self.attach.set_block_position(attach_position)
        # create attach
        self.attach.attach()
        # render the world
        self.world.step(render=True)
        
  
def collect_data(type, idx, garment_usd_path, rotate_x, rotate_y):
    
    env = LiftGarment_Collect_Deformation(
        garment_path=garment_usd_path,
        garment_position=np.array([0, 0.0, 0.1]),
        garment_orientation=np.array([rotate_x, rotate_y, 0])
    )

    # Render initial frames to stabilize environment
    for _ in range(20):
        env.step()
        
    original_points, _ = env.recording_camera.get_point_cloud_data()
    
    if len(original_points) < 10:
        return
    
    garment_mesh_points = env.garment.get_vertice_positions()
    print(garment_mesh_points.shape)
    
    garment_category = garment_usd_path.split("/")[-1].split(".")[0]
    save_npz_dir = f"data/{type}/unigarment/cd_original/mesh_pcd/{idx}_{garment_category}"
    os.makedirs(save_npz_dir, exist_ok=True)
    save_rgb_dir = f"data/{type}/unigarment/cd_original/cd_rgb_view/{idx}_{garment_category}"
    os.makedirs(save_rgb_dir, exist_ok=True)
    
    # 获取当前目录下已经保存到多少号变形了
    max_sequence_number = get_max_sequence_number(save_npz_dir)
    save_idx = max_sequence_number + 1
    
    cprint(f"Garment_usd path {garment_usd_path} Saving deformation {save_idx}...", "green")
    
    np.savez(os.path.join(save_npz_dir, f"p_{save_idx}.npz"),
             pcd_points=original_points,
             mesh_points=garment_mesh_points)
    
    env.recording_camera.get_rgb_graph(
        save_path=os.path.join(save_rgb_dir, f"rgb_{save_idx}.png")
    )
    
    


if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Collect deformation data for a specific garment.")
    parser.add_argument("type", type=str, help="Type of garment")
    parser.add_argument("idx", type=int, help="Index of the garment")
    parser.add_argument("garment_usd_path", type=str, help="Path to the garment USD file")
    parser.add_argument("rotate_x", type=int, help="Rotation angle around the x-axis")
    parser.add_argument("rotate_y", type=int, help="Rotation angle around the y-axis")
    args = parser.parse_args()  
    
    type = args.type
    idx = args.idx
    garment_usd_path = args.garment_usd_path
    rotate_x = args.rotate_x
    rotate_y = args.rotate_y
    
    # type = "trousers"
    # idx = 0
    # garment_usd_path = "Assets/Garment/Trousers/Long/PL_Pants052/PL_Pants052_obj.usd"
    # rotate_x = 45
    # rotate_y = 0
    collect_data(type, idx, garment_usd_path,
                 rotate_x, rotate_y)
    
    