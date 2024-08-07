import numpy as np
from omni.isaac.core import World
from omni.isaac.core.materials.particle_material import ParticleMaterial
from omni.isaac.core.prims.soft.cloth_prim import ClothPrim
from omni.isaac.core.prims.soft.cloth_prim_view import ClothPrimView
from omni.isaac.core.prims.soft.particle_system import ParticleSystem
from omni.isaac.core.utils.nucleus import get_assets_root_path
from pxr import Gf, UsdGeom,Sdf, UsdPhysics, PhysxSchema, UsdLux, UsdShade
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
from omni.isaac.core.utils.semantics import add_update_semantics, get_semantics
from omni.isaac.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
from omni.physx.scripts import physicsUtils, deformableUtils, particleUtils
from omni.isaac.core.objects import DynamicSphere, DynamicCuboid
from Env.Utils.transforms import euler_angles_to_quat
import omni.kit.commands
import omni.physxdemos as demo
from Env.Config.GarmentConfig import GarmentConfig
from omni.isaac.core.materials.physics_material import  PhysicsMaterial

class Human():
    def __init__(self,path):
        self.path=path
        self.prim_path="/World/Avatar"
        add_reference_to_stage(usd_path=path,prim_path=self.prim_path)

        self.rigid_form=XFormPrim(
            prim_path="/World/Avatar",
            name="human",
            position=np.array([-0.28,0,0]),
            orientation=euler_angles_to_quat(np.array([0.,0.,np.pi/2])),
            scale=np.array([1.,1.,0.85])
        )
        self.geom_prim=GeometryPrim(
            prim_path="/World/Avatar",
            collision=True
        )
        self.geom_prim.set_collision_approximation("none")
        
        # self.geom_prim=GeometryPrim(
        #     prim_path=self.prim_path,
        #     collision=True
        # )
        # self.geom_prim.set_collision_approximation("convexHull")

        # self.rigid_form=XFormPrim(
        #     prim_path="/World/Human",#/male_adult_construction_03",
        #     name="human",
        #     position=np.array([0.0,0.0,0.0]),
        #     orientation=euler_angles_to_quat(np.array([0.,0.,np.pi/2]))
        # )
        

        # self.geom_prim=GeometryPrim(
        #     prim_path="/World/Human/male_adult_construction_03/male_adult_construction_03/male_adult_construction_04/trans__organic__whitemaleskin",
        #     collision=True
        # )
        # self.geom_prim.set_collision_approximation("none")
        # self.geom_prim=GeometryPrim(
        #     prim_path="/World/Human/male_adult_construction_03/male_adult_construction_03/male_adult_construction_04/opaque__fabric__jeans",
        #     collision=True
        # )
        # self.geom_prim.set_collision_approximation("none")

