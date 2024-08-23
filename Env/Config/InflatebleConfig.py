import numpy as np
import torch

class InflatebleConfig:
    def __init__(self,pos:np.ndarray=None,ori:np.ndarray=None,scale:np.ndarray=None,pressure:float=8.0,visual_material_usd:str="/home/isaac/GarmentLab/Assets/Material/linen_Pumpkin.usd"):
        if pos is None:
            self.pos=np.array([0.8,0,0.5])
        else:
            self.pos=pos
        if ori is None:
            self.ori=np.array([0,0,0])
        else:
            self.ori=ori
        if scale is None:
            self.scale=np.array([0.25,0.25,0.01])
        else:
            self.scale=scale
        self.pressure=pressure
        self.visual_material_usd=visual_material_usd
        self.stretch_stiffness = 20000.0
        self.bend_stiffness = 100.0
        self.shear_stiffness = 100.0
        self.spring_damping = 0.5
        self.particle_mass=0.1
        self.cube_u_resolution=20
        self.cube_v_resolution=20
        self.cube_w_resolution=4
        self.u_verts_scale=1
        self.v_verts_scale=1
        self.w_verts_scale=1
        self.half_scale=60.0
        self.particle_contact_offset=None
        self.contact_offset=None
        self.rest_offset=None
        self.solid_rest_offset=None
        self.fluid_rest_offset=None
        self.solver_position_iteration_count=None