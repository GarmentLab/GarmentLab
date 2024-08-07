
import numpy as np
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka import Franka
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.franka import KinematicsSolver
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.prims.geometry_prim import GeometryPrim
import omni.isaac.core.utils.prims as prims_utils
from pxr import UsdGeom,UsdPhysics,PhysxSchema
from Env.Utils.transforms import euler_angles_to_quat
import torch
from Env.Utils.transforms import quat_diff_rad
from omni.isaac.core.materials.physics_material import PhysicsMaterial


class TactileFranka:
    def __init__(self,world:World,Position=None,prim_path:str=None,robot_name:str=None,):
        self.world=world
        if prim_path is None:
            self._franka_prim_path = find_unique_string_name(
                initial_name="/World/Franka", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
        else:
            self._franka_prim_path=prim_path

        if robot_name is None:
            self._franka_robot_name = find_unique_string_name(
                initial_name="my_franka", is_unique_fn=lambda x: not self.world.scene.object_exists(x)
            )
        else:
            self._franka_robot_name=robot_name

        self.init_position=Position
        self.world.scene.add(Franka(prim_path=self._franka_prim_path,usd_path="/home/user/Tactile/IsaacTac/Robot/Franka/franka.usd",name=self._franka_robot_name,position=Position))
        self._robot:Franka=self.world.scene.get_object(self._franka_robot_name)
        self.process_franka_gripper()
        self._articulation_controller=self._robot.get_articulation_controller()
        self._controller=RMPFlowController(name="rmpflow_controller",robot_articulation=self._robot)
        self._kinematic_solover=KinematicsSolver(self._robot)
        self._pick_place_controller=PickPlaceController(name="pick_place_controller",robot_articulation=self._robot,gripper=self._robot.gripper)
        self._controller.reset()
        self._pick_place_controller.reset()
        
    def process_franka_gripper(self,):
        self.left_gripper_prim_path="/World/Franka/panda_leftfinger"
        self.left_gripper_prim_collision_path="/World/Franka/panda_leftfinger/collisions"
        self.left_gripper_geo=GeometryPrim(self.left_gripper_prim_collision_path,collision=True)
        self.left_gripper_prim=prims_utils.get_prim_at_path(self.left_gripper_prim_collision_path)
        self.left_gripper_collision_api = UsdPhysics.MeshCollisionAPI.Apply(self.left_gripper_prim)
        self.left_gripper_collision_api.GetApproximationAttr().Set("sdf")
        self.left_gripper_sdf_api=PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(self.left_gripper_prim)
        self.left_gripper_geo.set_contact_offset(0.02)
        self.left_gripper_geo.set_contact_offset(0.01)
        self.gripper_physics_material=PhysicsMaterial(
            prim_path="/World/PhysicsMaterials_gripper",
            static_friction=20,
            dynamic_friction=20,
        )
        # self.left_gripper_sdf_api.GetSdfResolutionAttr().Set(256)
        # self.left_gripper_sdf_api.GetSdfSubgridResolutionAttr().Set(0)
        self.right_gripper_prim_path="/World/Franka/panda_rightfinger"
        self.right_gripper_prim_collision_path="/World/Franka/panda_rightfinger/collisions"
        self.right_gripper_geo=GeometryPrim(self.right_gripper_prim_collision_path,collision=True)
        self.right_gripper_prim=prims_utils.get_prim_at_path(self.right_gripper_prim_collision_path)
        self.right_gripper_collision_api = UsdPhysics.MeshCollisionAPI.Apply(self.right_gripper_prim)
        self.right_gripper_collision_api.GetApproximationAttr().Set("sdf")
        self.right_gripper_sdf_api=PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(self.right_gripper_prim)
        self.right_gripper_geo.set_contact_offset(0.02)
        self.right_gripper_geo.set_contact_offset(0.01)
        self.left_gripper_geo.apply_physics_material(self.gripper_physics_material)
        self.right_gripper_geo.apply_physics_material(self.gripper_physics_material)
    def get_cur_ee_pos(self):
        ee_pos, R = self._controller.get_motion_policy().get_end_effector_as_prim().get_world_pose()
        return ee_pos, R


    def pick_and_place(self,pick,place):
        self._pick_place_controller.reset()
        self._robot.gripper.open()
        while 1:
            self.world.step(render=True)
            actions=self._pick_place_controller.forward(
                picking_position=pick,
                placing_position=place,
                current_joint_positions=self._robot.get_joint_positions(),
                end_effector_offset=np.array([0,0.005,0]),
            )
            if self._pick_place_controller.is_done():
                break
            self._articulation_controller.apply_action(actions)
    
    def open(self):
        for _ in range(100):
            self._robot.gripper.open()
            self.world.step(render=True)
    
    def close(self):
        for _ in range(100):
            self._robot.gripper.close()
            self.world.step(render=True)
        

    @staticmethod
    def interpolate(start_loc, end_loc, speed):
        start_loc = np.array(start_loc)
        end_loc = np.array(end_loc)
        dist = np.linalg.norm(end_loc - start_loc)
        chunks = dist // speed
        if chunks==0:
            chunks=1
        return start_loc + np.outer(np.arange(chunks+1,dtype=float), (end_loc - start_loc) / chunks)
    
    def position_reached(self, target,thres=0.03):
        if target is None:
            return True
        
        ee_pos, R = self._controller.get_motion_policy().get_end_effector_as_prim().get_world_pose()
        pos_diff = np.linalg.norm(ee_pos- target)
        if pos_diff < thres:
            return True
        else:
            return False 
    def rotation_reached(self, target):
        if target is None:
            return True
        
        ee_pos, R = self._controller.get_motion_policy().get_end_effector_as_prim().get_world_pose()
        angle_diff = quat_diff_rad(R, target)[0]
        # print(f'angle diff: {angle_diff}')
        if angle_diff < 0.1:
            return True
        
    def movep(self,end_loc):
        self.world.step(render=True)
        start_loc=self.get_cur_ee_pos()[0]
        cur_step=0
        while 1:
            self.world.step(render=True)
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
            target_joint_positions = self._controller.forward(
                target_end_effector_position=end_loc, target_end_effector_orientation=end_effector_orientation
            )
            self._articulation_controller.apply_action(target_joint_positions)
            cur_step+=1
            if self.position_reached(end_loc):
                break
    
    def movel(self,end_loc,env_ori=None):
        self.world.step(render=True)
        start_loc=self.get_cur_ee_pos()[0]
        cur_step=0
        while 1:
            self.world.step(render=True)
            if env_ori is None:
                env_ori=np.array([0, np.pi, 0])
            end_effector_orientation = euler_angles_to_quat(env_ori)
            target_joint_positions = self._controller.forward(
                target_end_effector_position=end_loc, target_end_effector_orientation=end_effector_orientation
            )
            self._articulation_controller.apply_action(target_joint_positions)
            cur_step+=1
            if self.position_reached(end_loc) and self.rotation_reached(end_effector_orientation):
                break
            if cur_step>1000:
                print("Failed to reach target")
                break
            
            

            



