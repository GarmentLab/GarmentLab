import numpy as np
from isaacsim import SimulationApp
import torch

# simulation_app = SimulationApp({"headless": False})
import numpy as np
from omni.isaac.franka import Franka
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
import torch
from Env.env.BaseEnv import BaseEnv
from Env.Robot.Franka.TactileFranka import TactileFranka
from omni.isaac.core.prims.rigid_prim_view import RigidPrimView
import omni
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from pxr import UsdGeom,UsdPhysics,PhysxSchema
import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.prims.rigid_prim import RigidPrim 
from omni.isaac.core.objects import DynamicCuboid,FixedCuboid 


class TactileEnv(BaseEnv):
    def __init__(self):
        super(TactileEnv,self).__init__(rigid=True)
        self.robot=TactileFranka(self.world,Position=torch.from_numpy(np.array([0,0,0])))
        self.left_gripper_prim_path="/World/Franka/panda_leftfinger"
        self.right_gripper_prim_path="/World/Franka/panda_rightfinger"
        self.add_workspace()
        self.add_cube()
        self.add_helper_cube()
        self.init_tactile()
        
    def add_cube(self):
        self.cube_prim_path = find_unique_string_name(
            initial_name="/World/Cube", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        self.cube_name = find_unique_string_name(initial_name="cube", is_unique_fn=lambda x: not self.scene.object_exists(x))
        self.cube_physics_material=PhysicsMaterial(
            prim_path="/World/PhysicsMaterials",
            static_friction=20,
            dynamic_friction=20,
        )
        self._cube = self.scene.add(
            DynamicCuboid(
                name=self.cube_name,
                position=torch.from_numpy(np.array([0.4,0,0.5])),
                prim_path=self.cube_prim_path,
                scale=[0.25, 0.0515, 0.0515],
                size=1.0,
                color=np.array([0, 0, 1]),
                mass=0.1,
                physics_material=self.cube_physics_material
            )
        )
        self.cube_mesh_geo=GeometryPrim(prim_path=self.cube_prim_path,collision=True)
        self.cube_mesh_geo.set_collision_approximation("convexDecomposition")
        self.cube_mesh_geo.apply_physics_material(self.cube_physics_material)
        self.cube_mesh_collision_api=UsdPhysics.MeshCollisionAPI.Apply(prims_utils.get_prim_at_path(self.cube_prim_path))
        self.cube_mesh_collision_api.GetApproximationAttr().Set("sdf")

        self.cube_mesh_sdf_api=PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prims_utils.get_prim_at_path(self.cube_prim_path))
    
    def init_tactile(self):
        self.left_gripper_view=self.scene.add(RigidPrimView(prim_paths_expr=self.left_gripper_prim_path,
                                                            name="left_gripper_view",
                                                    contact_filter_prim_paths_expr=["/World/Cube"],
                                                    max_contact_count=2000))
        self.right_gripper_view=self.scene.add(RigidPrimView(prim_paths_expr=self.right_gripper_prim_path,
                                                             name="right_gripper_view",
                                                    contact_filter_prim_paths_expr=["/World/Cube"],
                                                    max_contact_count=2000))
        
    def get_tactile(self):
        (           friction_forces,
                    friction_points,
                    friction_pair_contacts_count,
                    friction_pair_contacts_start_indices,
                )=self.left_gripper_view.get_friction_data()
        left_friction={"friction_forces":friction_forces,"friction_points":friction_points,"friction_pair_contacts_count":friction_pair_contacts_count,"friction_pair_contacts_start_indices":friction_pair_contacts_start_indices}
        (           friction_forces,
                    friction_points,
                    friction_pair_contacts_count,
                    friction_pair_contacts_start_indices,
                )=self.right_gripper_view.get_friction_data()
        right_friction={"friction_forces":friction_forces,"friction_points":friction_points,"friction_pair_contacts_count":friction_pair_contacts_count,"friction_pair_contacts_start_indices":friction_pair_contacts_start_indices}
        
        (           forces,  # only normal impulses
                    points,
                    normals,
                    distances,
                    pair_contacts_count,
                    pair_contacts_start_indices,
                ) =self.left_gripper_view.get_contact_force_data()
        left_normal={"forces":forces,"points":points,"normals":normals,"distances":distances,"pair_contacts_count":pair_contacts_count,"pair_contacts_start_indices":pair_contacts_start_indices}
        (           forces,  # only normal impulses
                    points,
                    normals,
                    distances,
                    pair_contacts_count,
                    pair_contacts_start_indices,
                ) =self.right_gripper_view.get_contact_force_data()
        right_normal={"forces":forces,"points":points,"normals":normals,"distances":distances,"pair_contacts_count":pair_contacts_count,"pair_contacts_start_indices":pair_contacts_start_indices}
        left_tactile={"friction":left_friction,"normal":left_normal}
        right_tactile={"friction":right_friction,"normal":right_normal}
        return left_tactile,right_tactile      
    
    def add_mesh_cube(self):
        success, tmp_path = omni.kit.commands.execute(
            "CreateMeshPrimWithDefaultXform",
            prim_type="Cube",
            u_patches=10,
            v_patches=10,
            w_patches=10,
            u_verts_scale=1,
            v_verts_scale=1,
            w_verts_scale=1,
            half_scale=10.0,
            select_new_prim=False,)
        self.cube_prim_path = find_unique_string_name(
            initial_name="/World/Cube", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        self.cube_physics_material=material = PhysicsMaterial(
            prim_path="/World/PhysicsMaterials",
            static_friction=30,
            dynamic_friction=30,
        )
        omni.kit.commands.execute("MovePrim", path_from=tmp_path, path_to=self.cube_prim_path)
        self.cube_mesh_geo=GeometryPrim(prim_path=self.cube_prim_path,collision=True)
        self.cube_mesh_geo.set_collision_approximation("convexDecomposition")
        self.cube_mesh_geo.apply_physics_material(self.cube_physics_material)
        self.cube_mesh_collision_api=UsdPhysics.MeshCollisionAPI.Apply(prims_utils.get_prim_at_path(self.cube_prim_path))
        self.cube_mesh_collision_api.GetApproximationAttr().Set("sdf")

        self.cube_mesh_sdf_api=PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prims_utils.get_prim_at_path(self.cube_prim_path))

        self.cube_mesh_rigid=RigidPrim(prim_path=self.cube_prim_path,
                                       mass=0.1,
                                       position=torch.from_numpy(np.array([0.4,0,0.5])),
                                       scale=[1.2,0.2,0.2],  
        )
    def add_workspace(self):
        self.workspace_prim=find_unique_string_name(
            initial_name="/World/WorkSpace", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        self.workspace_name=find_unique_string_name(initial_name="workspace", is_unique_fn=lambda x: not self.scene.object_exists(x))
        self.workspace_material_path=find_unique_string_name(initial_name="/World/PhysicsMaterials1", is_unique_fn=lambda x: not is_prim_path_valid(x))
        self.workspace_physics_material=PhysicsMaterial(prim_path=self.workspace_material_path,static_friction=100,dynamic_friction=100)
        self._workspace=self.scene.add(
            FixedCuboid(prim_path="/World/WorkSpace",
                        name=self.workspace_name,
                        position=np.array([5.05,0,0]),
                        physics_material=self.workspace_physics_material,
                        scale=[10,10,0.1],
            )
        )
    
    def add_helper_cube(self):
        cube_prim_path = find_unique_string_name(
            initial_name="/World/HelperCube", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        cube_name = find_unique_string_name(initial_name="cube", is_unique_fn=lambda x: not self.scene.object_exists(x))
        self.helper_cube_physics_material=material = PhysicsMaterial(
            prim_path="/World/PhysicsMaterials2",
            static_friction=20,
            dynamic_friction=20,
        )
        self._cube = self.scene.add(
            FixedCuboid(
                name=cube_name,
                position=[0.8,0,0.2],
                prim_path=cube_prim_path,
                scale=[0.5,0.5,0.3],
                size=1.0,
                color=np.array([1, 0, 0]),
                physics_material=self.helper_cube_physics_material
            )
        )