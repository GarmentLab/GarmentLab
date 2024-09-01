import torch
import numpy as np
from typing import Union

class GarmentConfig:
    def __init__(self,usd_path:str="/home/isaac/GarmentLab/Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_074/TCLC_074_obj.usd",pos:Union[torch.Tensor,np.ndarray]=np.array([0.5,-0.5,0.3]),ori:Union[torch.Tensor,np.ndarray]=np.array([0,0,np.pi/2]),scale:np.ndarray=np.array([0.005,0.005,0.005]),visual_material_usd:str="/home/isaac/GarmentLab/Assets/Material/linen_Pumpkin.usd",particle_contact_offset:float=0.02):
        self.usd_path=usd_path
        if isinstance(pos,np.ndarray):
            self.pos=torch.from_numpy(pos)
        else:
            self.pos=pos
        if isinstance(ori,np.ndarray):
            self.ori=torch.from_numpy(ori)
        else:
            self.ori=ori
        if isinstance(scale,np.ndarray):
            self.scale=torch.from_numpy(scale)
        else:
            self.scale=scale
       # self.scale=np.array([0.005, 0.005, 0.005])
        self.stretch_stiffness=1e4
        self.bend_stiffness=100
        self.shear_stiffness=100.0
        self.spring_damping=0.2
        self.particle_contact_offset=particle_contact_offset
        self.enable_ccd=True
        self.global_self_collision_enabled=True
        self.non_particle_collision_enabled=True
        self.solver_position_iteration_count=16
        self.friction=0.3
        self.visual_material_usd=visual_material_usd
        self.contact_offset=None
        self.rest_offset=None
        self.solid_rest_offset=None
        self.fluid_rest_offset=None