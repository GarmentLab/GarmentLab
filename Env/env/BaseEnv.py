import numpy as np
from isaacsim import SimulationApp
import torch
from time import gmtime, strftime

# simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core import SimulationContext
from omni.isaac.core.objects import DynamicCuboid,FixedCuboid
from Env.Robot.Franka.MyFranka import MyFranka
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.physx.scripts import deformableUtils,particleUtils,physicsUtils
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.prims.rigid_prim import RigidPrim  
import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.core.prims.rigid_prim_view import RigidPrimView
from omni.isaac.core.prims.geometry_prim_view import GeometryPrimView
from pxr import UsdGeom,UsdPhysics,PhysxSchema
from omni.isaac.sensor import ContactSensor
from omni.isaac.core.materials import OmniGlass
from omni.physx import acquire_physx_interface
from Env.Garment.Garment import Garment
from Env.Config.FrankaConfig import FrankaConfig
from Env.Utils.transforms import euler_angles_to_quat
import omni
import open3d as o3d
import omni.replicator.core as rep
from Env.Config.SceneConfig import SceneConfig
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
from omni.isaac.core.prims.xform_prim import XFormPrim



class BaseEnv:
    def __init__(self,scene_config:SceneConfig=None,rigid:bool=False,deformable:bool=False,garment:bool=False):
        self.rigid=rigid
        self.deformable=deformable
        self.garment=garment
        self.recording=False
        self.world=World()
        self.stage=self.world.scene.stage
        self.scene=self.world.scene
        self.context=SimulationContext()
        self.scene.add_default_ground_plane()
        self.savings=[]
        if scene_config is None:
            self.scene_config=SceneConfig()
        else:
            self.scene_config=scene_config
            
        self.room=self.import_room(self.scene_config)
        # self.scene.add_default_ground_plane()
        self.set_physics_scene()
        self.demo_light=rep.create.light(position=[0,0,0],light_type="dome")
        
    def import_franka(self,franka_config:FrankaConfig):
        self.franka_list:list[MyFranka]=[]
        for id in range(franka_config.franka_num):
            self.franka_list.append(MyFranka(self.world,pos=franka_config.pos[id],ori=franka_config.ori[id]))
        return self.franka_list

    def import_room(self,scene_config:SceneConfig):
        self.room_prim_path=find_unique_string_name("/World/Room/room",is_unique_fn=lambda x: not is_prim_path_valid(x))
        self.room_name=find_unique_string_name(initial_name="room",is_unique_fn=lambda x: not self.world.scene.object_exists(x))
        add_reference_to_stage(usd_path=scene_config.room_usd_path,prim_path=self.room_prim_path)
        self.room=XFormPrim(
            prim_path=self.room_prim_path,
            name=self.room_name,
            position=scene_config.pos,
            orientation=euler_angles_to_quat(scene_config.ori),
            scale=scene_config.scale,
            )
    
        
    def set_physics_scene(self):
        # self.physics_interface=acquire_physx_interface()
        # self.physics_interface.overwrite_gpu_setting(1)
        self.physics=self.world.get_physics_context()
        self.physics.enable_ccd(True)
        self.physics.enable_gpu_dynamics(True)
        self.physics.set_broadphase_type("gpu")
        self.physics.enable_stablization(True)
        if self.rigid:
            self.physics.set_solver_type("TGS")
            self.physics.set_gpu_max_rigid_contact_count(10240000)
            self.physics.set_gpu_max_rigid_patch_count(10240000)

    def __replay_callback(self, step_size):
        if self.time_ptr < self.total_ticks:
            self.replay_callback(self.data[self.time_ptr])
            self.time_ptr += 1
    
    def record(self):
        if self.recording == False:
            self.recording = True
            self.replay_file_name = strftime("Assets/Replays/%Y%m%d-%H:%M:%S.npy", gmtime())
            self.context.add_physics_callback("record_callback", self.record_callback)

    def stop_record(self):
        if self.recording == True:
            self.recording = False
            self.context.remove_physics_callback("record_callback")
            np.save(self.replay_file_name, np.array(self.savings))
            self.savings = []
    
    def replay(self, file):
        self.data = np.load(file, allow_pickle=True)
        self.time_ptr = 0
        self.total_ticks = len(self.data)
        self.context.add_physics_callback("replay_callback", self.__replay_callback)
        if self.deformable:
            # self.physics.set_gpu_max_soft_body_contacts(1024000)
            self.physics.set_gpu_collision_stack_size(3000000)
            
        
        
    def reset(self):
        self.world.reset()
    
    def step(self):
        self.world.step(render=True)
        
    def stop(self):
        self.world.stop()
    
    def record_callback(self, step_size):
        pass

    def replay_callback(self, data):
        pass

if __name__=="__main__":
    env=BaseEnv()
    env.reset()
    
    while 1:
        env.step()
    
    
    

    
    