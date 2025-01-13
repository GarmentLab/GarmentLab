import numpy as np
import torch

class DeformableConfig:
    def __init__(self,usd_path:str="/home/user/GarmentLab/Assets/Garment/Hat/HA_Hat007/HA_Hat007_obj.usd",pos:np.ndarray=None,ori:np.ndarray=None,scale:np.ndarray=None,visual_material_usd:str="/home/user/GarmentLab/Assets/Material/linen_Pumpkin.usd",youngs_modulus:float=None):
        if pos is None:
            self.pos=np.array([0.6,1.3,0.3])
        else:
            self.pos=pos
        if ori is None:
            self.ori=np.array([np.pi,0,0])
        else:
            self.ori=ori
        if scale is None:
            self.scale=np.array([0.008,0.008,0.008])
        else:
            self.scale=scale
        self.usd_path=usd_path
        self.dynamic_fricition=0.5
        if youngs_modulus is None:
            self.youngs_modulus=None
        else:
            self.youngs_modulus=youngs_modulus
        self.poissons_ratio=None
        self.damping_scale=None
        self.elasticity_damping=None
        self.vertex_velocity_damping=0.0
        self.sleep_damping=10.0
        self.sleep_threshold=0.05
        self.settling_threshold=0.1
        self.self_collision=True
        self.solver_position_iteration_count=16
        self.kinematic_enabled=False
        self.simulation_hexahedral_resolution=10
        self.collision_simplification=True
        self.visual_material_usd=visual_material_usd
