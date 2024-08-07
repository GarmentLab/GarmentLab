import numpy as np
import torch


class FluidConfig:
    def __init__(self,pos:np.ndarray=None,volume:np.ndarray=None,particle_group:int=0):
        if pos is None:
            self.pos=np.array([0.5,0,0])
        else:
            self.pos=pos
        if volume is None:
            self.volume=np.array([0.3,0.3,0.6])
        else:
            self.volume=volume
        self.particle_group=particle_group
        self.particle_contact_offset=0.02
        self.contact_offset=0.01
        self.fluid_rest_offset=0.008
        self.rest_offset=0.01
        self.cohesion=0.01
        self.viscosity=0.0091
        self.surface_tension=0.0074
        self.friction=0.1
        self.damping=0.99
        self.particle_scale=2.5
        self.particle_mass=1e-6*self.particle_scale*self.particle_scale
        self.density=0
        self.side_length=self.volume[0]*1.2
        self.thickness=0.01