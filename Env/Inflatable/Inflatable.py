from omni.physx.scripts import physicsUtils, particleUtils
from pxr import UsdGeom, Sdf, Gf, PhysxSchema, UsdPhysics, UsdShade
import omni.usd
from Env.Config.InflatebleConfig import InflatebleConfig
from omni.isaac.core.prims.soft.particle_system import ParticleSystem
from omni.isaac.core import World
import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.core.materials.preview_surface import PreviewSurface
from omni.isaac.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.materials.particle_material import ParticleMaterial
from Env.Utils.transforms import euler_angles_to_quat
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
from omni.isaac.core.utils.prims import is_prim_path_valid

class Inflatable:
    def __init__(self,world:World,inflatebleconfig:InflatebleConfig=None,particle_system:ParticleSystem=None):
        self.world=world
        self.inflateble_config=inflatebleconfig
        self.stage=world.stage
        self.inflateble_view=UsdGeom.Xform.Define(self.stage,"/World/Inflateble")
        self.inflateble_name=find_unique_string_name(initial_name="inflateble",is_unique_fn=lambda x: not world.scene.object_exists(x))
        self.inflateble_prim_path=find_unique_string_name("/World/Inflateble/inflateble",is_unique_fn=lambda x: not is_prim_path_valid(x))
        self.particle_system_path=find_unique_string_name("/World/Inflateble/particleSystem",is_unique_fn=lambda x: not is_prim_path_valid(x))
        self.particle_material_path=find_unique_string_name("/World/Inflateble/particleMaterial",is_unique_fn=lambda x: not is_prim_path_valid(x))
        self.stage=world.stage

        
        self.inflateble_mesh_prim_path=self.inflateble_prim_path+"/mesh"

        self.stage=self.world.stage
        self.default_prim_path = Sdf.Path(self.inflateble_prim_path)
        self.particle_system_path = Sdf.Path(self.particle_system_path)
        self.particle_system = PhysxSchema.PhysxParticleSystem.Define(self.stage, self.particle_system_path)
        # create material and assign it to the system:
        particle_material_path = Sdf.Path(self.particle_material_path)
        particleUtils.add_pbd_particle_material(self.stage, particle_material_path)
        physicsUtils.add_physics_material_to_prim(
            self.stage, self.stage.GetPrimAtPath(self.particle_system_path), particle_material_path
        )
        self.set_particle_system()

        self.import_mesh_cube()

        particleUtils.add_physx_particle_cloth(
            stage=self.stage,
            path=self.inflateble_mesh_prim_path,
            dynamic_mesh_path=None,
            particle_system_path=self.particle_system_path,
            spring_stretch_stiffness=self.inflateble_config.stretch_stiffness,
            spring_bend_stiffness=self.inflateble_config.bend_stiffness,
            spring_shear_stiffness=self.inflateble_config.shear_stiffness,
            spring_damping=self.inflateble_config.spring_damping,
            pressure=self.inflateble_config.pressure,
        )

        # configure mass:
        self.set_mass(self.inflateble_config.particle_mass)
        if self.inflateble_config.visual_material_usd is not None:
            self.apply_visual_material(self.inflateble_config.visual_material_usd)

    def set_particle_system(self):
        if self.inflateble_config.particle_contact_offset is not None:
            self.particle_system.GetParticleContactOffsetAttr().Set(self.inflateble_config.particle_contact_offset)
        if self.inflateble_config.rest_offset is not None:
            self.particle_system.GetRestOffsetAttr().Set(self.inflateble_config.rest_offset)
        if self.inflateble_config.solid_rest_offset is not None:
            self.particle_system.GetSolidRestOffsetAttr().Set(self.inflateble_config.solid_rest_offset)
        if self.inflateble_config.fluid_rest_offset is not None:
            self.particle_system.GetFluidRestOffsetAttr().Set(self.inflateble_config.fluid_rest_offset)
        if self.inflateble_config.contact_offset is not None:
            self.particle_system.GetContactOffsetAttr().Set(self.inflateble_config.contact_offset)
    
    def set_mass(self,particle_mass:float):
        num_verts = len(self.inflateble_mesh.GetPointsAttr().Get())
        mass = particle_mass * num_verts
        massApi = UsdPhysics.MassAPI.Apply(self.inflateble_mesh.GetPrim())
        massApi.GetMassAttr().Set(mass)
            
    def import_mesh_cube(self):
        # create a mesh that is turned into an inflatable
        cube_u_resolution = self.inflateble_config.cube_u_resolution
        cube_v_resolution = self.inflateble_config.cube_v_resolution
        cube_w_resolution = self.inflateble_config.cube_w_resolution

        success, tmp_path = omni.kit.commands.execute(
            "CreateMeshPrimWithDefaultXform",
            prim_type="Cube",
            u_patches=cube_u_resolution,
            v_patches=cube_v_resolution,
            w_patches=cube_w_resolution,
            u_verts_scale=self.inflateble_config.u_verts_scale,
            v_verts_scale=self.inflateble_config.v_verts_scale,
            w_verts_scale=self.inflateble_config.w_verts_scale,
            half_scale=self.inflateble_config.half_scale,
            select_new_prim=False,
        )

        omni.kit.commands.execute("MovePrim", path_from=tmp_path, path_to=self.inflateble_mesh_prim_path)
        self.inflateble_mesh = UsdGeom.Mesh.Get(self.stage, self.inflateble_mesh_prim_path)
        physicsUtils.setup_transform_as_scale_orient_translate(self.inflateble_mesh)
        physicsUtils.set_or_add_translate_op(self.inflateble_mesh, Gf.Vec3f(self.inflateble_config.pos.tolist()))
        # physicsUtils.set_or_add_orient_op(cloth_mesh, Gf.Quatf(Gf.Rotation(Gf.Vec3d(self.inflateble_config.ori.tolist()), 10.0).GetQuat()))
        physicsUtils.set_or_add_scale_op(self.inflateble_mesh, Gf.Vec3f(self.inflateble_config.scale.tolist()))
        

    
    def get_particle_system_id(self):
        self.particle_system_api=PhysxSchema.PhysxParticleAPI.Apply(self.particle_system.prim)
        return self.particle_system_api.GetParticleGroupAttr().Get()
    
    def get_vertices_positions(self):
        return self.inflateble_mesh._get_points_pose()
    
    def apply_visual_material(self,material_path:str):
        self.visual_material_path=find_unique_string_name(self.inflateble_prim_path+"/visual_material",is_unique_fn=lambda x: not is_prim_path_valid(x))
        add_reference_to_stage(usd_path=material_path,prim_path=self.visual_material_path)
        self.visual_material_prim=prims_utils.get_prim_at_path(self.visual_material_path)
        self.material_prim=prims_utils.get_prim_children(self.visual_material_prim)[0]
        self.material_prim_path=self.material_prim.GetPath()
        self.visual_material=PreviewSurface(self.material_prim_path)
        
        self.inflateble_mesh_prim=prims_utils.get_prim_at_path(self.inflateble_mesh_prim_path)
        self.inflateble_submesh=prims_utils.get_prim_children(self.inflateble_mesh_prim)
        if len(self.inflateble_submesh)==0:
            omni.kit.commands.execute('BindMaterialCommand',
            prim_path=self.inflateble_mesh_prim_path, material_path=self.material_prim_path)
        else:
            omni.kit.commands.execute('BindMaterialCommand',
            prim_path=self.inflateble_mesh_prim_path, material_path=self.material_prim_path)
            for prim in self.inflateble_submesh:
                omni.kit.commands.execute('BindMaterialCommand',
                prim_path=prim.GetPath(), material_path=self.material_prim_path)