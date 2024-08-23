import torch
import numpy as np
from typing import Union

class FrankaConfig:
    def __init__(self,franka_num:int=1,pos:list=[np.array([0,0,0])],ori:list=[np.array([0,0,0])]):
        self.franka_num=franka_num
        assert len(pos)==franka_num,"Length of pos should be equal to franka_num"
        assert len(ori)==franka_num,"Length of ori should be equal to franka_num"
        self.pos=pos
        self.ori=ori
        for i in range(franka_num):
            if isinstance(pos[i],np.ndarray):
                self.pos[i]=torch.from_numpy(pos[i])
            if isinstance(ori[i],np.ndarray):
                self.ori[i]=torch.from_numpy(ori[i])
                

class MobileFrankaConfig:
    def __init__(self,franka_num:int=1,pos=np.array([0,0,0]),ori=np.array([0,0,0])):
        self.franka_num=franka_num
        self.pos=pos
        self.ori=ori
        if isinstance(pos,np.ndarray):
            self.pos=torch.from_numpy(pos)
        if isinstance(ori,np.ndarray):
            self.ori=torch.from_numpy(ori)
                
                

