import numpy as np
import os


from omni.isaac.core.robots.robot import Robot
from omni.isaac.universal_robots import KinematicsSolver
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from pxr import PhysxSchema

from Env.Config.DexConfig import DexConfig
from Env.Utils.set_drive import set_drive
from Env.Utils.transforms import quat_diff_rad

class DexLeft(Robot):
    def __init__(self, cfg: DexConfig):
        self.env = cfg.env
        self.app = cfg.app
        self.recording = False

        self._name = "DexLeft" if cfg.name is None else cfg.name
        self._prim_path = "/World/DexLeft" if cfg.prim_path is None else cfg.prim_path

        self.asset_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../Assets/Robots/dexleft.usd")

        self.translation = cfg.translation
        self.orientation = cfg.orientation

        add_reference_to_stage(self.asset_file, self._prim_path)

        super().__init__(
            prim_path=self._prim_path, 
            name=self._name, 
            translation=self.translation, 
            orientation=self.orientation, 
            articulation_controller=None
        )

        self.env.scene.add(self)
        
        self.ki_solver = KinematicsSolver(self, end_effector_frame_name="tool0")
        self.ee = XFormPrim(self._prim_path + "/UR10e/tool0", "end_effector")
    
    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)

        self.hand = self.env.stage.GetPrimAtPath(f"{self._prim_path}/shadowhand")

        # retain acceleration
        for link_prim in self.hand.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(self.env.stage, link_prim.GetPrimPath())
                rb.GetRetainAccelerationsAttr().Set(True)

        self.arm_dof_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.hand_dof_names = ["WRJ2", "WRJ1", "FFJ4", "FFJ3", "FFJ2", "FFJ1", "MFJ4", "MFJ3", "MFJ2", "MFJ1", "RFJ4", 
            "RFJ3", "RFJ2", "RFJ1", "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1", "THJ5", "THJ4", "THJ3", "THJ2", "THJ1"]
        
        self.arm_dof_indices = [self.get_dof_index(dof_name) for dof_name in self.arm_dof_names]
        self.hand_dof_indices = [self.get_dof_index(dof_name) for dof_name in self.hand_dof_names]

        self.joints_config = {
            "forearm/WRJ2": {"stiffness": 1, "damping": 0.1, "max_force": 4.785},
            "wrist/WRJ1": {"stiffness": 1, "damping": 0.1, "max_force": 2.175},
            "palm/FFJ4": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "ffknuckle/FFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "ffproximal/FFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "ffmiddle/FFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
            "palm/MFJ4": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "mfknuckle/MFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "mfproximal/MFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "mfmiddle/MFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
            "palm/RFJ4": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "rfknuckle/RFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "rfproximal/RFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "rfmiddle/RFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
            "palm/LFJ5": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "lfmetacarpal/LFJ4": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "lfknuckle/LFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "lfproximal/LFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "lfmiddle/LFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
            "palm/THJ5": {"stiffness": 0.1, "damping": 0.1, "max_force": 2.3722},
            "thbase/THJ4": {"stiffness": 0.1, "damping": 0.1, "max_force": 1.45},
            "thproximal/THJ3": {"stiffness": 0.1, "damping": 0.1, "max_force": 0.99},
            "thhub/THJ2": {"stiffness": 0.1, "damping": 0.1, "max_force": 1.99},
            "thmiddle/THJ1": {"stiffness": 0.1, "damping": 0.1, "max_force": 1.81},
        }

        for joint_name, config in self.joints_config.items():
            set_drive(f"{self._prim_path}/shadowhand/{joint_name}", "angular", "position", 0, config["stiffness"], config["damping"], config["max_force"])

    def ee_orientation_reached(self, target: np.ndarray, threshold = 0.05, angular_type = "euler"):
        if angular_type == "euler":
            target = euler_angles_to_quat(target)

        _, R = self.ee.get_world_pose()

        angle_diff = quat_diff_rad(R, target)[0]
        return angle_diff < threshold
    
    def ee_position_reached(self, target: np.ndarray, threshold = 0.2):
        current_pos, _ = self.ee.get_world_pose()
        return np.linalg.norm(current_pos - target) < threshold
    
    def ee_reached(self, target_pos: np.ndarray, target_ori: np.ndarray, pos_threshold = 0.2, 
                   ori_threshold = 0.05, angular_type = "euler"):
        return self.ee_position_reached(target_pos, pos_threshold) and self.ee_orientation_reached(target_ori, ori_threshold, angular_type)
        
    def step(self, target_pos: np.ndarray, target_ori: np.ndarray, angular_type = "euler"):
        if angular_type == "euler":
            target_ori = euler_angles_to_quat(target_ori)
        base_pos, _ = self.get_world_pose()
        action, succ = self.ki_solver.compute_inverse_kinematics(target_pos - base_pos, target_ori)
        if succ:
            self._articulation_controller.apply_action(action)
            self.env.world.step(render=True) 
        return succ
    
    def move_to(self, target_pos: np.ndarray, target_ori: np.ndarray, angular_type = "euler"):
        self.env.step()

        while not self.ee_reached(target_pos, target_ori, angular_type=angular_type):
            succ = self.step(target_pos, target_ori, angular_type)
            if not succ:
                return False
            self.env.step()

        return True
    
    
            

            




        
            




        






