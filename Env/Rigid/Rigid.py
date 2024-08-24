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
from omni.isaac.core.objects import DynamicSphere, DynamicCuboid, FixedCuboid, VisualCuboid
from Env.Utils.transforms import euler_angles_to_quat
import omni.kit.commands
import omni.physxdemos as demo
from Env.Config.GarmentConfig import GarmentConfig
from omni.isaac.core.materials.physics_material import  PhysicsMaterial

class Rigid():
    def __init__(self, root_path, rigid_config):
        self._root = root_path
        self._render_material = False
        self.name="rigid_0"

        add_reference_to_stage(usd_path=rigid_config["path"],prim_path=self._root)

        # define path
        full_path = self._root

        self.rigid_form=XFormPrim(
            prim_path=full_path,
            name=self.name,
            position=rigid_config["position"],
            scale=rigid_config["scale"],
            orientation=rigid_config["orientation"],
        )
        
        self.geom_prim=GeometryPrim(
            prim_path=full_path,
            collision=True
        )
        self.geom_prim.set_collision_approximation("convexHull")

        # self.physics_material=PhysicsMaterial(prim_path=matetial_path, dynamic_friction=0.99,static_friction=0.99)
        # self.geom_prim.apply_physics_material(self.physics_material)

class RigidHang():
    def __init__(self):
        rigid_cube = FixedCuboid(prim_path="/World/cube_0",
                                 name="cube_0",
                                position=np.array([0.5,0.5,0.5]),
                                scale=np.array([0.02,0.02,0.1]),
                                orientation=euler_angles_to_quat(np.array([26.568,-14.472,-26.554])*np.pi/180))
        rigid_cube = FixedCuboid(prim_path="/World/cube_1",
                                 name="cube_1",
                                position=np.array([0.52367,0.51928,0.]),
                                scale=np.array([0.02,0.02,1.1]))


class RigidHangCloth():
    def __init__(self):
        rigid_cube = FixedCuboid(prim_path="/World/cube_0",
                                 name="cube_0",
                                position=np.array([0.00282,-1.85217,0.5]),
                                scale=np.array([0.02,0.02,1.0]),
                                orientation=euler_angles_to_quat(np.array([0,90,0])*np.pi/180))

        visual_cube = VisualCuboid(prim_path="/World/cube_1",
                                 name="cube_1",
                                position=np.array([0.00282,-1.97493,0.5]),
                                scale=np.array([0.02,0.02,0.02]),
                                visible=False)
        

class RigidHangFling():
    def __init__(self):
        rigid_cube = FixedCuboid(prim_path="/World/cube_0",
                                 name="cube_0",
                                position=np.array([0.41957,-0.01923,0.7269]),
                                scale=np.array([0.02,0.02,0.08]),
                                orientation=euler_angles_to_quat(np.array([-180,30,-180])*np.pi/180))

        visual_cube = VisualCuboid(prim_path="/World/cube_1",
                                 name="cube_1",
                                position=np.array([0.00282,0.0,0.5]),
                                scale=np.array([0.02,0.02,0.02]),
                                visible=True)

class RigidAfford():
    def __init__(self):
        rigid_cube = FixedCuboid(prim_path="/World/cube_0",
                                 name="cube_0",
                                position=np.array([0.50402,-0.00277,0.7269]),
                                scale=np.array([0.02,0.02,0.08]),
                                orientation=euler_angles_to_quat(np.array([-180,30,-180])*np.pi/180))

        visual_cube = VisualCuboid(prim_path="/World/cube_1",
                                 name="cube_1",
                                position=np.array([0.00282,0.0,0.5]),
                                scale=np.array([0.02,0.02,0.02]),
                                visible=True)

class RigidVisual():
    def __init__(self):
        visual_cube = VisualCuboid(prim_path="/World/cube_1",
                                 name="cube_1",
                                position=np.array([0.00282,0.0,0.5]),
                                scale=np.array([0.02,0.02,0.02]),
                                visible=True)

class RigidStore():
    def __init__(self):
        add_reference_to_stage(usd_path="/home/isaac/GarmentLab/Assets/Articulated/cabinet.usd",prim_path="/World")
        # self.rigid_form=XFormPrim(
        #     prim_path="/World/cabinet",
        #     name="cabinet",
        #     position=np.array([0,0,-0.5]),
        # )
        visual_cube = VisualCuboid(prim_path="/World/cube_1",
                                 name="cube_1",
                                position=np.array([0.00282,0.0,0.5]),
                                scale=np.array([0.02,0.02,0.02]),
                                visible=True)
        
class RigidTable:
    def __init__(self,world:World):
        add_reference_to_stage("/home/isaac/GarmentLab/Assets/Scene/Willow.usd","/World/table")  
        self.table_rigid_prim:RigidPrim=world.scene.add(RigidPrim(
            prim_path="/World/table",
            name="table",
            position=np.array([0,0,0]),
            scale=np.array([0.01,0.02,0.01]),
        ))
        self.table_geo_prim:GeometryPrim=world.scene.add(GeometryPrim(
            prim_path="/World/table",
            collision=True
        ))
        self.table_geo_prim.set_collision_approximation("convexHull")
        self.table_geo_prim.set_contact_offset(0.01)
        
        