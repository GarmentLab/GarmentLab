import os
import math
import numpy as np

from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.franka import KinematicsSolver
from omni.isaac.core.prims import XFormPrim
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core import World

from pxr import PhysxSchema

from Env.Utils.set_drive import set_drive
from Env.Utils.transforms import quat_diff_rad
from Env.Config.FrankaConfig import MobileFrankaConfig


import omni.isaac.motion_generation as mg
from omni.isaac.core.articulations import Articulation


class RMPFlowController(mg.MotionPolicyController):
    """[summary]

    Args:
        name (str): [description]
        robot_articulation (Articulation): [description]
        physics_dt (float, optional): [description]. Defaults to 1.0/60.0.
    """

    def __init__(self, name: str, robot_articulation: Articulation, physics_dt: float = 1.0 / 60.0) -> None:
        self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config("Franka", "RMPflow")
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)

        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)

        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        (
            self._default_position,
            self._default_orientation,
        ) = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
        return

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
    
    def mobile_reset(self, position: np.ndarray, orientation: np.ndarray):
        self._motion_policy.set_robot_base_pose(
            robot_position=position, robot_orientation=orientation
        )


class MobileFranka(Robot):
    def __init__(self, world: World, cfg: MobileFrankaConfig):
        self._name = find_unique_string_name("MobileFranka", is_unique_fn=lambda x: not world.scene.object_exists(x))
        self._prim_path = find_unique_string_name("/World/MobileFranka", is_unique_fn=lambda x: not is_prim_path_valid(x))

        self.asset_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../Assets/Robots/mfranka.usd")

        self.translation = cfg.pos
        self.orientation = cfg.ori
        
        if self.orientation.shape[-1]==3:
            self.orientation = euler_angles_to_quat(self.orientation)

        add_reference_to_stage(self.asset_file, self._prim_path)

        super().__init__(
            prim_path=self._prim_path,
            name=self._name,
            translation=self.translation,
            orientation=self.orientation,
            articulation_controller=None,
        )
        self.world=world
        self.world.scene.add(self)

        self.ki_solver = KinematicsSolver(self)

        self.ee = XFormPrim(self.prim_path + '/endeffector', 'endeffector')
        self.franka_base = XFormPrim(self.prim_path + '/panda_link0', 'panda_link')
        self.base = XFormPrim(self.prim_path + '/base_link', 'base_link')
        
        self.gripper_open_position = np.array([0.05, 0.05]) / get_stage_units()
        self.gripper_closed_position = np.array([0.02, 0.02]) / get_stage_units()
        self.deltas = np.array([0.02, 0.02]) / get_stage_units()

        self.gripper = ParallelGripper(
            end_effector_prim_path = self.prim_path + '/endeffector', 
            joint_prim_names = ['panda_finger_joint1', 'panda_finger_joint2'],
            joint_opened_positions = self.gripper_open_position,
            joint_closed_positions = self.gripper_closed_position,
            action_deltas = self.deltas
        )
        self.default_ee_ori=np.array([0,np.pi,0])
        self.controller=RMPFlowController("RMPFlowController",self)
        
    def reach(self,pos,ori=None):
        if ori is None:
            return self.ee_position_reached(pos)
        else:
            return self.ee_reached(pos,ori)

    def open(self):
        for _ in range(10):
            self.gripper.open()
            self.world.step(render=True)
    
    def close(self):
        for _ in range(10):
            self.gripper.close()
            self.world.step(render=True)
            
    def get_cur_ee_pos(self):
        return self.ee.get_world_pose()
    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)

        self.gripper.initialize(
            physics_sim_view=physics_sim_view, 
            articulation_apply_action_func=self.apply_action, 
            get_joint_positions_func=self.get_joint_positions, 
            set_joint_positions_func=self.set_joint_positions, 
            dof_names=self.dof_names
        )

        self.franka_dof_names = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", 
                              "panda_joint6", "panda_joint7", "panda_finger_joint1", "panda_finger_joint2"]
        self.base_dof_names = ["dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint", "dummy_base_revolute_z_joint"]

        self.franka_dof_indicies = [self.get_dof_index(dof_name) for dof_name in self.franka_dof_names]
        self.base_dof_indicies = [self.get_dof_index(dof_name) for dof_name in self.base_dof_names]
        self.joints_config = {
            "panda_link0/panda_joint1": { "drive_type": "angular", "target_type": "position", "default_pos": math.degrees(0.0), "stiffness": 6.98, "damping": 1.40, "max_force": 87, "max_velocity": 124.6 }, 
            "panda_link1/panda_joint2": { "drive_type": "angular", "target_type": "position", "default_pos": math.degrees(-1.0), "stiffness": 6.98, "damping": 1.40, "max_force": 87, "max_velocity": 124.6 },
            "panda_link2/panda_joint3": { "drive_type": "angular", "target_type": "position", "default_pos": math.degrees(0.0), "stiffness": 6.98, "damping": 1.40, "max_force": 87, "max_velocity": 124.6 },
            "panda_link3/panda_joint4": { "drive_type": "angular", "target_type": "position", "default_pos": math.degrees(-2.2), "stiffness": 6.98, "damping": 1.40, "max_force": 87, "max_velocity": 124.6 },
            "panda_link4/panda_joint5": { "drive_type": "angular", "target_type": "position", "default_pos": math.degrees(0.0), "stiffness": 6.98, "damping": 1.40, "max_force": 12, "max_velocity": 149.5 },
            "panda_link5/panda_joint6": { "drive_type": "angular", "target_type": "position", "default_pos": math.degrees(2.4), "stiffness": 6.98, "damping": 1.40, "max_force": 12, "max_velocity": 149.5 },
            "panda_link6/panda_joint7": { "drive_type": "angular", "target_type": "position", "default_pos": math.degrees(0.8), "stiffness": 6.98, "damping": 1.40, "max_force": 12, "max_velocity": 149.5 },
            "panda_hand/panda_finger_joint1": { "drive_type": "linear", "target_type": "position", "default_pos": 0.02, "stiffness": 1e4, "damping": 100, "max_force": 200, "max_velocity": 0.2 },
            "panda_hand/panda_finger_joint2": { "drive_type": "linear", "target_type": "position", "default_pos": 0.02, "stiffness": 1e4, "damping": 100, "max_force": 200, "max_velocity": 0.2 }, 
            "world/dummy_base_prismatic_x_joint": { "drive_type": "linear", "target_type": "velocity", "default_pos": 0.0, "stiffness": 1e4, "damping": 0, "max_force": 4800 },
            "dummy_base_x/dummy_base_prismatic_y_joint": { "drive_type": "linear", "target_type": "velocity", "default_pos": 0.0, "stiffness": 1e4, "damping": 0, "max_force": 4800 },
            "dummy_base_y/dummy_base_revolute_z_joint": { "drive_type": "angular", "target_type": "velocity", "default_pos": 0.0, "stiffness": 1e4, "damping": 0, "max_force": 4800 }
        }

        for joint_name, config in self.joints_config.items():
            absolute_path = f"{self.prim_path}/{joint_name}"
            set_drive(absolute_path, config["drive_type"], config["target_type"], 
                      config["default_pos"], config["stiffness"], config["damping"], config["max_force"])
            if config.get("max_velocity") is not None:
                PhysxSchema.PhysxJointAPI(get_prim_at_path(absolute_path)).CreateMaxJointVelocityAttr().Set(config["max_velocity"])
    def get_prim_path(self):
        return self._prim_path

    def base_position_reached(self, target: np.ndarray, threshold = 0.001):
        current_pos, _ = self.base.get_world_pose()
        delta_pos = target - current_pos
        delta_pos[2] = 0
        distance = np.linalg.norm(delta_pos)
        return distance < threshold
    
    def ee_orientation_reached(self, target: np.ndarray, threshold = 0.01, angular_type = "euler"):
        if angular_type == "euler":
            target = euler_angles_to_quat(target)

        _, R = self.ee.get_world_pose()

        angle_diff = quat_diff_rad(R, target)[0]
        return angle_diff < threshold
    
    def ee_position_reached(self, target: np.ndarray, threshold = 0.01):
        current_pos, _ = self.ee.get_world_pose()
        # print("ee_current_pos",current_pos)
        return np.linalg.norm(current_pos - target) < threshold
    
    def ee_reached(self, target_pos: np.ndarray, target_ori: np.ndarray, pos_threshold = 0.01, 
                   ori_threshold = 0.01, angular_type = "euler"):
        if target_ori is not None:
            return self.ee_position_reached(target_pos, pos_threshold) and self.ee_orientation_reached(target_ori, ori_threshold, angular_type)
        else:
            return self.ee_position_reached(target_pos, pos_threshold)
        
    def move(self, target_pos: np.ndarray, target_ori: np.ndarray=None, angular_type = "euler"):
        if angular_type == "euler":
            target_ori = euler_angles_to_quat(target_ori)
        # action, succ = self.ki_solver.compute_inverse_kinematics(target_pos, target_ori)
        # if succ:
        #     self._articulation_controller.apply_action(action)
        #     self.world.step(render=True) 
        # return succ
        actions=self.controller.forward(
            target_pos,
            target_ori
        )
        self._articulation_controller.apply_action(actions)
        self.world.step(render=True)
        return True
    
    def get_franka_base_pose(self):
        return self.franka_base.get_world_pose()
    
    def gripper_move_to(self, target_pos: np.ndarray, target_ori: np.ndarray=None, angular_type = "euler"):
        self.world.step()
        # if target_ori==None:
        #     target_ori = self.default_ee_ori
        
        franka_base_pos, franka_base_ori = self.get_franka_base_pose()
        self.controller.mobile_reset(franka_base_pos, franka_base_ori)

        while not self.ee_reached(target_pos, target_ori, angular_type=angular_type):
            succ = self.move(target_pos, target_ori, angular_type)
            # self.world.step()

        return True

    def base_move_to(self, target: np.ndarray, velocity = 2.0):
        self.world.step()
        franka_joint=self.get_joint_positions(np.arange(6)+4)
        while not self.base_position_reached(target):
            self.world.step()
            current_pos, _ = self.base.get_world_pose()
            delta_pos = target - current_pos
            delta_pos[2] = 0
            delta_pos = delta_pos / np.linalg.norm(delta_pos)

            dof_vel = delta_pos * velocity
            self.set_joint_velocities(dof_vel, self.base_dof_indicies)
            # self.set_joint_positions(franka_joint,np.arange(6)+4)
            
        self.set_joint_velocities(np.zeros((3, )), self.base_dof_indicies)
        z = self.get_joint_positions([2]).item()
        self.set_joint_positions([z], [2])
        

    def angle(self, target: np.ndarray):
        def reduce_rad(angle):
            angle = angle % (2 * np.pi)
            if angle >= np.pi:
                angle = angle - 2 * np.pi
            return angle

        current_pos, _ = self.base.get_world_pose()
        
        z1 = reduce_rad(self.get_joint_positions([2]).item())
        z2 = reduce_rad(np.arctan2(target[1] - current_pos[1], target[0] - current_pos[0]))     
        return reduce_rad(z2 - z1)

    def base_face_to(self, target: np.ndarray, velocity = 0.3, threshold = 3):
        self.world.step()

        threshold = threshold * np.pi / 180
        delta_t = 0.1
        franka_joint=self.get_joint_positions(np.arange(5)+5)
        while np.abs(self.angle(target)) > threshold:
            self.world.step()
            diff = self.angle(target)
            z = self.get_joint_positions([2]).item()
            self.set_joint_positions([z + delta_t * (velocity if diff > 0 else -velocity)], [2])
            # # self.set_joint_positions(franka_joint,np.arange(5)+5)
            # self.set_joint_positions(np.concatenate(([z + delta_t * (velocity if diff > 0 else -velocity)],franka_joint)),np.concatenate(([2],np.arange(5)+5))) 

    
    
    
        

    