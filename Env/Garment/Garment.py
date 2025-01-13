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
import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.core.materials.preview_surface import PreviewSurface

class Garment:
    def __init__(self,world:World,garment_config:GarmentConfig,particle_system:ParticleSystem=None):
        self.world=world
        self.garment_config=garment_config
        self.usd_path=self.garment_config.usd_path
        self.stage=world.stage
        self.garment_view=UsdGeom.Xform.Define(self.stage,"/World/Garment")
        self.garment_name=find_unique_string_name(initial_name="garment",is_unique_fn=lambda x: not world.scene.object_exists(x))
        self.garment_prim_path=find_unique_string_name("/World/Garment/garment",is_unique_fn=lambda x: not is_prim_path_valid(x))
        self.particle_material_path=find_unique_string_name("/World/Garment/particleMaterial",is_unique_fn=lambda x: not is_prim_path_valid(x))
        self.stage=world.stage
        self.particle_material=ParticleMaterial(prim_path=self.particle_material_path, friction=self.garment_config.friction)

        if particle_system is None:
            self.particle_system_path=find_unique_string_name("/World/Garment/particleSystem",is_unique_fn=lambda x: not is_prim_path_valid(x))
            self.particle_system = ParticleSystem(
                prim_path=self.particle_system_path,
                simulation_owner=self.world.get_physics_context().prim_path,
                particle_contact_offset=self.garment_config.particle_contact_offset,
                enable_ccd=self.garment_config.enable_ccd,
                global_self_collision_enabled=self.garment_config.global_self_collision_enabled,
                non_particle_collision_enabled=self.garment_config.non_particle_collision_enabled,
                solver_position_iteration_count=self.garment_config.solver_position_iteration_count,
            )
        else:
            self.particle_system_path = particle_system.prim_path
            self.particle_system = particle_system
            self.particle_system=particle_system
            self.particle_system.set_global_self_collision_enabled(self.garment_config.global_self_collision_enabled)
            self.particle_system.set_solver_position_iteration_count(self.garment_config.solver_position_iteration_count)

        add_reference_to_stage(usd_path=self.usd_path,prim_path=self.garment_prim_path)

        self.garment_mesh_prim_path=self.garment_prim_path+"/mesh"
        self.garment=XFormPrim(
            prim_path=self.garment_prim_path,
            name=self.garment_name,
            position=self.garment_config.pos,
            orientation=euler_angles_to_quat(self.garment_config.ori),
            scale=self.garment_config.scale,
            )

        self.garment_mesh=ClothPrim(
            name=self.garment_name+"_mesh",
            prim_path=self.garment_mesh_prim_path,
            particle_system=self.particle_system,
            particle_material=self.particle_material,
            stretch_stiffness=self.garment_config.stretch_stiffness,
            bend_stiffness=self.garment_config.bend_stiffness,
            shear_stiffness=self.garment_config.shear_stiffness,
            spring_damping=self.garment_config.spring_damping,
        )
        # self.world.scene.add(self.garment_mesh)
        self.particle_controller = self.garment_mesh._cloth_prim_view
        if self.garment_config.visual_material_usd is not None:
            self.apply_visual_material(self.garment_config.visual_material_usd)

    def set_mass(self,mass):
        physicsUtils.add_mass(self.world.stage, self.garment_mesh_prim_path, mass)

    def get_particle_system_id(self):
        self.particle_system_api=PhysxSchema.PhysxParticleAPI.Apply(self.particle_system.prim)
        return self.particle_system_api.GetParticleGroupAttr().Get()

    def get_vertices_positions(self):
        return self.garment_mesh._get_points_pose()

    def get_realvertices_positions(self):
        return self.garment_mesh._cloth_prim_view.get_world_positions()

    def apply_visual_material(self,material_path:str):
        self.visual_material_path=find_unique_string_name(self.garment_prim_path+"/visual_material",is_unique_fn=lambda x: not is_prim_path_valid(x))
        add_reference_to_stage(usd_path=material_path,prim_path=self.visual_material_path)
        self.visual_material_prim=prims_utils.get_prim_at_path(self.visual_material_path)
        self.material_prim=prims_utils.get_prim_children(self.visual_material_prim)[0]
        self.material_prim_path=self.material_prim.GetPath()
        self.visual_material=PreviewSurface(self.material_prim_path)

        self.garment_mesh_prim=prims_utils.get_prim_at_path(self.garment_mesh_prim_path)
        self.garment_submesh=prims_utils.get_prim_children(self.garment_mesh_prim)
        if len(self.garment_submesh)==0:
            omni.kit.commands.execute('BindMaterialCommand',
            prim_path=self.garment_mesh_prim_path, material_path=self.material_prim_path)
        else:
            omni.kit.commands.execute('BindMaterialCommand',
            prim_path=self.garment_mesh_prim_path, material_path=self.material_prim_path)
            for prim in self.garment_submesh:
                omni.kit.commands.execute('BindMaterialCommand',
                prim_path=prim.GetPath(), material_path=self.material_prim_path)

    def get_vertice_positions(self):
        return self.garment_mesh._get_points_pose()

    def set_pose(self, pos, ori):
        self.garment.set_world_pose(position=pos, orientation=ori)

    def get_particle_system(self):
        return self.particle_system
