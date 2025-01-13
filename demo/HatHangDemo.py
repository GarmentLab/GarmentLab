import sys
sys.path.append("/home/user/GarmentLab")
from Env.env.DeformableEnv import DeformableEnv

from Env.Config.GarmentConfig import GarmentConfig
from Env.Config.FrankaConfig import FrankaConfig
import numpy as np


if __name__=="__main__":
    env=DeformableEnv()
    env.reset()
    env.robots[0].movel(np.array([0.65,0,0.5]))
    env.def_control.grasp([np.array([0.65535,-0.03994,0.09283])],[None],[True])
    env.def_control.move([np.array([0.5,-0.0,0.72])],[None],[True])
    env.def_control.move([np.array([0.525,0.525,0.67])],[None],[True])
    env.def_control.ungrasp([False])
    for i in range(50):
        env.world.step()
