import numpy as np
from omni.isaac.core import World
from omni.isaac.core.materials.deformable_material import DeformableMaterial
from omni.isaac.core.prims.soft.deformable_prim import DeformablePrim
from omni.isaac.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
from omni.isaac.core.prims.soft.deformable_prim_view import DeformablePrimView
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import Gf, UsdGeom, UsdLux,UsdPhysics, PhysxSchema
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
from Env.Utils.transforms import euler_angles_to_quat
from Env.Config.DeformableConfig import DeformableConfig
import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.core.materials.visual_material import VisualMaterial
from omni.isaac.core.materials.preview_surface import PreviewSurface
import omni


class Deformable:
    def __init__(self,world:World,deformable_config:DeformableConfig):
        self.deformable_config=deformable_config
        self.usd_path=self.deformable_config.usd_path
        self.stage=world.stage
        self.deformable_view=UsdGeom.Xform.Define(self.stage,"/World/Deformable")
        self.deformable_name=find_unique_string_name(initial_name="deformable",is_unique_fn=lambda x: not world.scene.object_exists(x))  
        self.deformable_prim_path=find_unique_string_name("/World/Deformable/deformable",is_unique_fn=lambda x: not is_prim_path_valid(x))
        
        self.deformable=XFormPrim(
            prim_path=self.deformable_prim_path,
            name=self.deformable_name,
            position=self.deformable_config.pos,
            orientation=euler_angles_to_quat(self.deformable_config.ori),
            scale=self.deformable_config.scale,)
        
        self.deformable_material_path=find_unique_string_name("/World/Deformable/deformable_material",is_unique_fn=lambda x: not is_prim_path_valid(x))
        self.deformable_material = DeformableMaterial(
                prim_path=self.deformable_material_path,
                dynamic_friction=self.deformable_config.dynamic_fricition,
                youngs_modulus=self.deformable_config.youngs_modulus,
                poissons_ratio=self.deformable_config.poissons_ratio,
                damping_scale=self.deformable_config.damping_scale,
                elasticity_damping=self.deformable_config.elasticity_damping,
            )
        add_reference_to_stage(usd_path=self.usd_path,prim_path=self.deformable_prim_path)

        self.deformable_mesh_prim_path=self.deformable_prim_path+"/mesh"
        


        self.deformable_mesh=UsdGeom.Mesh.Get(self.stage, self.deformable_mesh_prim_path)
        # self.deformable_points=self.deformable_mesh.GetPointsAttr().Get()
        # self.deformable_indices=deformableUtils.triangulate_mesh(self.deformable_mesh)
        # self.simulation_resolution=45
        # self.mesh_scale=Gf.Vec3f(0.05,0.05,0.05)
        # self.collision_points,self.collisions_indices=deformableUtils.compute_conforming_tetrahedral_mesh(self.deformable_points,self.deformable_indices)
        # self.simulation_points,self.simulation_indices=deformableUtils.compute_voxel_tetrahedral_mesh(self.collision_points,self.collisions_indices,self.mesh_scale,self.simulation_resolution)



        self.deformable = DeformablePrim(
                name=self.deformable_name,
                prim_path=self.deformable_mesh_prim_path,
                position=self.deformable_config.pos,
                orientation=euler_angles_to_quat(self.deformable_config.ori),
                deformable_material=self.deformable_material,
                vertex_velocity_damping=self.deformable_config.vertex_velocity_damping,
                sleep_damping=self.deformable_config.sleep_damping,
                sleep_threshold=self.deformable_config.sleep_threshold,
                settling_threshold=self.deformable_config.settling_threshold,
                self_collision=self.deformable_config.self_collision,
                solver_position_iteration_count=self.deformable_config.solver_position_iteration_count,
                kinematic_enabled=self.deformable_config.kinematic_enabled,
                simulation_hexahedral_resolution=self.deformable_config.simulation_hexahedral_resolution,
                collision_simplification=self.deformable_config.collision_simplification,
            )
        
        # if self.deformable_config.visual_material_usd is not None:
        #     self.apply_visual_material(self.deformable_config.visual_material_usd)
        
        self.set_contact_offset(0.01)
        self.set_rest_offset(0.008)
    
    def apply_visual_material(self,material_path:str):
        self.visual_material_path=find_unique_string_name(self.deformable_prim_path+"/visual_material",is_unique_fn=lambda x: not is_prim_path_valid(x))
        add_reference_to_stage(usd_path=material_path,prim_path=self.visual_material_path)
        self.visual_material_prim=prims_utils.get_prim_at_path(self.visual_material_path)
        self.material_prim=prims_utils.get_prim_children(self.visual_material_prim)[0]
        self.material_prim_path=self.material_prim.GetPath()
        self.visual_material=PreviewSurface(self.material_prim_path)
        
        self.deformable_mesh_prim=prims_utils.get_prim_at_path(self.deformable_mesh_prim_path)
        self.deformable_submesh=prims_utils.get_prim_children(self.deformable_mesh_prim)
        if len(self.deformable_submesh)==0:
            omni.kit.commands.execute('BindMaterialCommand',
            prim_path=self.deformable_mesh_prim_path, material_path=self.material_prim_path)
        else:
            omni.kit.commands.execute('BindMaterialCommand',
            prim_path=self.deformable_mesh_prim_path, material_path=self.material_prim_path)
            for prim in self.deformable_submesh:
                omni.kit.commands.execute('BindMaterialCommand',
                prim_path=prim.GetPath(), material_path=self.material_prim_path)
                
    def get_vertices_positions(self):
        return self.deformable._get_points_pose()
    
    def set_contact_offset(self,contact_offset:float=0.01):
        self.collsionapi=PhysxSchema.PhysxCollisionAPI.Apply(self.deformable.prim)
        self.collsionapi.GetContactOffsetAttr().Set(contact_offset)
    
    def set_rest_offset(self,rest_offset:float=0.008):
        self.collsionapi=PhysxSchema.PhysxCollisionAPI.Apply(self.deformable.prim)
        self.collsionapi.GetRestOffsetAttr().Set(rest_offset)

        
        
    
        