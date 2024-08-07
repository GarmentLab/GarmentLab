from copy import deepcopy
import math
import numpy as np
from omni.physx.scripts import physicsUtils,particleUtils
import omni.kit.commands
import omni.physxdemos as demo
from pxr import UsdGeom, Gf, Sdf, UsdLux, UsdPhysics, UsdShade, PhysxSchema,Vt
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
from omni.isaac.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
from Env.Utils.transforms import euler_angles_to_quat
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.objects.cuboid import VisualCuboid
from omni.isaac.core.prims.soft.particle_system import ParticleSystem
from Env.Config.FluidConfig import FluidConfig

class Fluid:
    def __init__(self,world,fluid_config:FluidConfig,particle_system_prim:ParticleSystem=None):
        self.world=world
        self.stage=world.stage
        self.fluid_config=fluid_config
        self.volume=self.fluid_config.volume
        self.initial_position=self.fluid_config.pos
        if particle_system_prim is None:
            self.particle_system_prim_path = find_unique_string_name(
                initial_name="/World/Fluid/ParticleSystem", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )


            self.particle_system=ParticleSystem(
                prim_path=self.particle_system_prim_path,
                simulation_owner=self.world.get_physics_context().prim_path,
                particle_contact_offset=self.fluid_config.particle_contact_offset,
                contact_offset=self.fluid_config.contact_offset,
                fluid_rest_offset=self.fluid_config.fluid_rest_offset,
                rest_offset=self.fluid_config.rest_offset,
            )

        else:
            self.particle_system=particle_system_prim
            self.particle_system_prim_path=particle_system_prim.prim_path
            self.particle_system.set_fluid_rest_offset(self.fluid_config.fluid_rest_offset)
            self.particle_system.set_rest_offset(self.fluid_config.rest_offset)
            
            
        # create Isosurface
        PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(self.particle_system.prim)
        primVarsAPI = UsdGeom.PrimvarsAPI(self.particle_system.prim)
        primVarsAPI.CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)



        # create volume cube
        self.volume_cube_prim_path=find_unique_string_name(
            initial_name="/World/Fluid/VolumeCube", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )

        _, cube_path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
        self.volume_cube_prim = self.stage.GetPrimAtPath(cube_path)
        self.volume_cube_prim.GetAttribute('xformOp:translate').Set(Gf.Vec3f(self.initial_position[0], self.initial_position[1], self.initial_position[2]+self.volume[2]/2))
        self.volume_cube_prim.GetAttribute('xformOp:scale').Set(Gf.Vec3f(self.volume[0], self.volume[1], self.volume[2]*0.8))


        # create water material
        mtl_created = []
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniSurfacePresets.mdl",
            mtl_name="OmniSurface_ClearWater",
            mtl_created_list=mtl_created,
        )

        self.pbd_particle_material_path = mtl_created[0]

        omni.kit.commands.execute(
            "BindMaterial",
            prim_path=self.particle_system_prim_path,
            material_path=self.pbd_particle_material_path,
        )


        particleUtils.add_pbd_particle_material(
        self.stage,
        self.pbd_particle_material_path,
        cohesion=self.fluid_config.cohesion,
        viscosity=self.fluid_config.viscosity,
        surface_tension=self.fluid_config.surface_tension,
        friction=self.fluid_config.friction,
        damping=self.fluid_config.damping,
        )
        physicsUtils.add_physics_material_to_prim(self.stage, self.particle_system.prim, self.pbd_particle_material_path)
        particle_scale = self.fluid_config.particle_scale   
        particle_mass = self.fluid_config.particle_mass

        # create particle set point instancer
        self.particle_instencer_prim_path = find_unique_string_name(
            initial_name="/World/Fluid/ParticleInstancer", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )

        # bin volume
        omni.kit.commands.execute(
        "AddParticleSamplingCommand",
        prim=self.volume_cube_prim,
        )
        omni.kit.commands.execute(
            "AddPhysicsComponent",
            usd_prim=self.volume_cube_prim,
            component="PhysxParticleSamplingAPI",
        )

        self.particle=particleUtils.add_physx_particleset_pointinstancer(
        stage=self.stage,
        path=Sdf.Path(self.particle_instencer_prim_path),
        positions=Vt.Vec3fArray([]),
        velocities=Vt.Vec3fArray([]),
        particle_system_path=self.particle_system_prim_path,
        self_collision=True,
        fluid=True,
        particle_group=self.fluid_config.particle_group,
        particle_mass=self.fluid_config.particle_mass,
        density=self.fluid_config.density,
        )



        # create box
        self.fluid_box_prim_path = find_unique_string_name(
            initial_name="/World/Fluid/FluidBox", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        self.create_particle_box_collider(
            path=self.fluid_box_prim_path,
            side_length=self.volume[0]*1.2,
            height=self.volume[2]*2/4,
            thickness=self.fluid_config.thickness,
            translate=Gf.Vec3f([self.initial_position[0],self.initial_position[1],self.initial_position[2]+self.volume[2]/2]),
        )
        self.world.reset()
        
    def get_particle_system_id(self):
        self.particle_system_api=PhysxSchema.PhysxParticleAPI.Apply(self.particle_system.prim)
        return self.particle_system_api.GetParticleGroupAttr().Get()

    def in_hull(p, hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        try:
            from scipy.spatial import Delaunay
        except:
            import omni
            omni.kit.pipapi.install("scipy")
            from scipy.spatial import Delaunay
            
        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p)>=0

    def generate_hcp_samples(self,boxMin: Gf.Vec3f, boxMax: Gf.Vec3f, sphereDiameter: float):

        layerDistance = math.sqrt(2.0 / 3.0) * sphereDiameter
        rowShift = math.sqrt(3.0) / 2.0 * sphereDiameter

        result = []
        layer1Offset = (1.0 / 3.0) * (
            Gf.Vec2f(0, 0) + Gf.Vec2f(0.5 * sphereDiameter, rowShift) + Gf.Vec2f(sphereDiameter, 0)
        )

        zIndex = 0
        while True:

            z = boxMin[2] + zIndex * layerDistance
            if z > boxMax[2]:
                break

            yOffset = layer1Offset[1] if zIndex % 2 == 1 else 0

            yIndex = 0
            while True:
                y = boxMin[1] + yIndex * rowShift + yOffset
                if y > boxMax[1]:
                    break

                xOffset = 0
                if zIndex % 2 == 1:
                    xOffset += layer1Offset[0]
                    if yIndex % 2 == 1:
                        xOffset -= 0.5 * sphereDiameter
                elif yIndex % 2 == 1:
                    xOffset += 0.5 * sphereDiameter

                xIndex = 0
                while True:
                    x = boxMin[0] + xIndex * sphereDiameter + xOffset
                    if x > boxMax[0]:
                        break

                    result.append(Gf.Vec3f(x, y, z))
                    xIndex += 1
                yIndex += 1
            zIndex += 1

        return result


    


    def generate_inside_point_cloud(self,sphereDiameter, cloud_points, scale = 1, max_particles = 3000):
        """
        Generate sphere packs inside a point cloud
        """
        offset = 2
        min_x = np.min(cloud_points[:, 0]) + offset
        min_y = np.min(cloud_points[:, 1]) + offset
        min_z = np.min(cloud_points[:, 2]) + offset

        max_x = np.max(cloud_points[:, 0]) 
        max_y = np.max(cloud_points[:, 1]) 
        max_z = np.max(cloud_points[:, 2]) 

        
        min_bound = [min_x, min_y, min_z]
        max_bound = [max_x, max_y, max_z]
        
        min_bound = [item * scale for item in min_bound]
        max_bound = [item * scale for item in max_bound]

        samples = self.generate_hcp_samples(Gf.Vec3f(*min_bound), Gf.Vec3f(*max_bound), sphereDiameter)
        
        finalSamples = []
        contains = self.in_hull(samples, cloud_points)

        for contain, sample in zip(contains, samples):
            if contain and len(finalSamples) < max_particles:
                finalSamples.append(sample)

        
        print("number of particles created: ", len(finalSamples) )
        print("max particles: ", max_particles)
        return finalSamples



    def create_particle_box_collider(
        self,
        path: Sdf.Path,
        side_length: float = 0.01,
        height: float = 0.005,
        translate: Gf.Vec3f = Gf.Vec3f(0, 0, 0),
        thickness: float = 0.001,
        add_cylinder_top=True,
    ):
        """
        Creates an invisible collider box to catch particles. Opening is in y-up

        Args:
            path:           box path (xform with cube collider children that make up box)
            side_length:    inner side length of box
            height:         height of box
            translate:      location of box, w.r.t it's bottom center
            thickness:      thickness of the box walls
        """
        xform = UsdGeom.Xform.Define(self.stage, path)
        # xform.MakeInvisible()
        xform_path = xform.GetPath()
        # physicsUtils.set_or_add_translate_op(xform, translate=translate)
        rotation=euler_angles_to_quat(np.array([-np.pi/2,0,0]))
        physicsUtils.set_or_add_scale_orient_translate(xform,orient=Gf.Quatf(rotation[0],rotation[1],rotation[2],rotation[3]),scale=Gf.Vec3f(1,1,1),translate=translate)
        cube_width = side_length + 2.0 * thickness
        offset = side_length * 0.5 + thickness * 0.5
        # front and back (+/- x)
        cube = UsdGeom.Cube.Define(self.stage, xform_path.AppendChild("front"))
        cube.CreateSizeAttr().Set(1.0)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        physicsUtils.set_or_add_translate_op(cube, Gf.Vec3f(0, height * 0.5, offset))
        physicsUtils.set_or_add_scale_op(cube, Gf.Vec3f(cube_width, height, thickness))

        if add_cylinder_top:
            top_cylinder = UsdGeom.Cylinder.Define(self.stage, xform_path.AppendChild("front_top_cylinder"))
            top_cylinder.CreateRadiusAttr().Set(thickness * 0.5)
            top_cylinder.CreateHeightAttr().Set(cube_width)
            top_cylinder.CreateAxisAttr().Set("X")
            UsdPhysics.CollisionAPI.Apply(top_cylinder.GetPrim())
            physicsUtils.set_or_add_translate_op(top_cylinder, Gf.Vec3f(0, height, offset))

        cube = UsdGeom.Cube.Define(self.stage, xform_path.AppendChild("back"))
        cube.CreateSizeAttr().Set(1.0)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        physicsUtils.set_or_add_translate_op(cube, Gf.Vec3f(0, height * 0.5, -offset))
        physicsUtils.set_or_add_scale_op(cube, Gf.Vec3f(cube_width, height, thickness))

        if add_cylinder_top:
            top_cylinder = UsdGeom.Cylinder.Define(self.stage, xform_path.AppendChild("back_top_cylinder"))
            top_cylinder.CreateRadiusAttr().Set(thickness * 0.5)
            top_cylinder.CreateHeightAttr().Set(cube_width)
            top_cylinder.CreateAxisAttr().Set("X")
            UsdPhysics.CollisionAPI.Apply(top_cylinder.GetPrim())
            physicsUtils.set_or_add_translate_op(top_cylinder, Gf.Vec3f(0, height, -offset))

        # left and right:
        cube = UsdGeom.Cube.Define(self.stage, xform_path.AppendChild("left"))
        cube.CreateSizeAttr().Set(1.0)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        physicsUtils.set_or_add_translate_op(cube, Gf.Vec3f(-offset, height * 0.5, 0))
        physicsUtils.set_or_add_scale_op(cube, Gf.Vec3f(thickness, height, cube_width))

        if add_cylinder_top:
            top_cylinder = UsdGeom.Cylinder.Define(self.stage, xform_path.AppendChild("left_top_cylinder"))
            top_cylinder.CreateRadiusAttr().Set(thickness * 0.5)
            top_cylinder.CreateHeightAttr().Set(cube_width - thickness)
            top_cylinder.CreateAxisAttr().Set("Z")
            UsdPhysics.CollisionAPI.Apply(top_cylinder.GetPrim())
            physicsUtils.set_or_add_translate_op(top_cylinder, Gf.Vec3f(-offset, height, 0))

        cube = UsdGeom.Cube.Define(self.stage, xform_path.AppendChild("right"))
        cube.CreateSizeAttr().Set(1.0)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        physicsUtils.set_or_add_translate_op(cube, Gf.Vec3f(offset, height * 0.5, 0))
        physicsUtils.set_or_add_scale_op(cube, Gf.Vec3f(thickness, height, cube_width))

        if add_cylinder_top:
            top_cylinder = UsdGeom.Cylinder.Define(self.stage, xform_path.AppendChild("right_top_cylinder"))
            top_cylinder.CreateRadiusAttr().Set(thickness * 0.5)
            top_cylinder.CreateHeightAttr().Set(cube_width - thickness)
            top_cylinder.CreateAxisAttr().Set("Z")
            UsdPhysics.CollisionAPI.Apply(top_cylinder.GetPrim())
            physicsUtils.set_or_add_translate_op(top_cylinder, Gf.Vec3f(offset, height, 0))

        xform_path_str = str(xform_path)

        paths = [
            xform_path_str + "/front",
            xform_path_str + "/back",
            xform_path_str + "/left",
            xform_path_str + "/right",
        ]
        for path in paths:
            self.set_glass_material(path)
    def set_glass_material(self, path):
        omni.kit.commands.execute(
            "BindMaterial", prim_path=path, material_path=self.get_glass_material(self.stage)
        )
    
    @staticmethod
    def get_glass_material(stage):
        glassPath = "/World/Looks/OmniGlass"
        if stage.GetPrimAtPath(glassPath):
            return glassPath
        mtl_created = []
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniGlass.mdl",
            mtl_name="OmniGlass",
            mtl_created_list=mtl_created,
        )
        return mtl_created[0]


