import os
import sys
import torch
import numpy as np
sys.path.append(os.getcwd())

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

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
from Env_Config.Garment.Deformable_Garment import Deformable_Garment

from Env_Config.Robot.DexLeft_Ur10e import DexLeft_Ur10e
from Env_Config.Robot.DexRight_Ur10e import DexRight_Ur10e
from Env_Config.Robot.BimanualDex_Ur10e import Bimanual_Ur10e

from Env_Config.Camera.Recording_Camera import Recording_Camera

# from Env_Config.Teleoperation.Listener import Listener

from Env_Config.Utils_Project.AttachmentBlock import AttachmentBlock
from Env_Config.Utils_Project.CollisionGroup import CollisionGroup
from Env_Config.Utils_Project.transforms import Rotation
from Env_Config.Utils_Project.file_operation import get_unique_filename

import shutil


class LiftGarment_CollectPointCloud(BaseEnv):
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
                               scale=np.array([0.02, 0.02, 0.02]))
        
        self.camera_pos = self.garment.get_center_point()
        self.camera_pos[2] = 5
        # load camera
        self.recording_camera = Recording_Camera(camera_position=self.camera_pos, camera_orientation=np.array([0, 90, -90]))
        
        # initialize world
        self.reset()
        
        # initialize camera, make sure the 'initialization' procedure is behind the 'reset' procedure.
        self.recording_camera.initialize(pc_enable=True, 
                                         segment_prim_path_list=["/World/Garment/garment"])
        # ["/World/DexLeft", "/World/DexRight", "/World/Garment/garment"]
        # add thread and record gif Asynchronously(use to collect rgb data for generating gif)
        # self.thread_record = threading.Thread(target=self.recording_camera.collect_rgb_graph_for_gif)
        # self.thread_record.daemon = True
        # self.thread_record.start()
        
        cprint("finish creating the world!", color='green')
        
        for i in range(50):
            self.step()
        # cprint(self.bimanual_dex.dexleft.get_joint_positions(), 'cyan')
        cprint("world ready!", color='green')
        

          

def collect_data(type, idx, garment_usd_path):
    garment_category = garment_usd_path.split("/")[-1].split(".")[0]

    # Initialize environment with specific garment path, position, and orientation
    env = LiftGarment_CollectPointCloud(
        garment_path=garment_usd_path,
        garment_position=np.array([0, 0, 0.1]),
        garment_orientation=np.array([90, 0, -90])
    )

    # Render initial frames to stabilize environment
    for _ in range(150):
        env.step()
              
    # Define directories for saving point cloud and RGB data
    save_path = f"data/{type}/skeleton/mesh_pcd"
    os.makedirs(save_path, exist_ok=True)
    save_dir = os.path.join(save_path, f"{idx}_{garment_category}")
    os.makedirs(save_dir, exist_ok=True)
    
    save_rgb_path = f"data/{type}/skeleton/rgb_view"
    os.makedirs(save_rgb_path, exist_ok=True)
    save_rgb_dir = os.path.join(save_rgb_path, f"{idx}_{garment_category}")
    os.makedirs(save_rgb_dir, exist_ok=True)

    # Save point cloud and RGB data
    np.savez(os.path.join(save_dir, "pc_0.npz"),
             pcd_points = env.recording_camera.get_point_cloud_data()[0],
             mesh_points = env.garment.get_vertice_positions())
    
    env.recording_camera.get_rgb_graph(
        save_path=os.path.join(save_rgb_dir, "rgb_0.png")
    )
    
    for _ in range(100000):
        env.step()

    cprint(f"Data collection complete for garment: {idx} - {garment_usd_path}", color='green')


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description="Collect data for a specific garment.")
    # parser.add_argument("type", type=str, help="Type of garment")
    # parser.add_argument("idx", type=int, help="Index of the garment")
    # parser.add_argument("garment_usd_path", type=str, help="Path to the garment USD file")
    # args = parser.parse_args()

    # type = args.type
    # idx = args.idx
    # garment_usd_path = args.garment_usd_path
    
    type = 'glove'
    idx = 2
    garment_usd_path = 'Assets/Garment/Glove/GL_Gloves064/GL_Gloves064_obj.usd'
    collect_data(type, idx, garment_usd_path)