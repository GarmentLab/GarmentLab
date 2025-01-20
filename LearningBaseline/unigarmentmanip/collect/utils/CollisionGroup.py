import numpy as np
from omni.isaac.core import World
from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics, PhysxSchema

class CollisionGroup:
    def __init__(self, world:World):
        # get stage
        self.stage = world.stage
        
        # ----------define garment_collision_group---------- #
        # path
        self.garment_group_path = "/World/Collision_Group/garment_group"
        # collision_group
        self.garment_group = UsdPhysics.CollisionGroup.Define(self.stage, self.garment_group_path)
        # filter(define which group can't collide with current group)
        self.filter_garment = self.garment_group.CreateFilteredGroupsRel()
        # includer(push object in the group)
        self.collectionAPI_garment = Usd.CollectionAPI.Apply(self.filter_garment.GetPrim(), "colliders")
        
        # ----------define attach_collision_group---------- #
        self.attach_group_path = "/World/Collision_Group/attach_group"
        self.attach_group = UsdPhysics.CollisionGroup.Define(self.stage, self.attach_group_path)
        self.filter_attach = self.attach_group.CreateFilteredGroupsRel()
        self.collectionAPI_attach = Usd.CollectionAPI.Apply(self.filter_attach.GetPrim(), "colliders")

        # ----------define robot_collision_group---------- #
        self.robot_group_path = "/World/Collision_Group/robot_group"
        self.robot_group = UsdPhysics.CollisionGroup.Define(self.stage, self.robot_group_path)
        self.filter_robot = self.robot_group.CreateFilteredGroupsRel()
        self.collectionAPI_robot = Usd.CollectionAPI.Apply(self.filter_robot.GetPrim(), "colliders")
        
        # push objects to different group
        self.collectionAPI_garment.CreateIncludesRel().AddTarget("/World/Garment")
        self.collectionAPI_attach.CreateIncludesRel().AddTarget("/World/AttachmentBlock")
        self.collectionAPI_robot.CreateIncludesRel().AddTarget("/World/DexLeft")
        self.collectionAPI_robot.CreateIncludesRel().AddTarget("/World/DexRight")
        
        # allocate the filter attribute of different groups
        self.filter_attach.AddTarget(self.garment_group_path)
        self.filter_attach.AddTarget(self.robot_group_path)
        # self.filter_robot.AddTarget(self.garment_group_path)
        
        
        