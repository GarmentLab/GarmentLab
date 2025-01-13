import sys
sys.path.append("/home/user/GarmentLab/Env")
sys.path.append("/home/user/GarmentLab")
from Env.env.WashEnv import WashEnv
from Env.Config.GarmentConfig import GarmentConfig
from Env.Config.FrankaConfig import FrankaConfig
import numpy as np


if __name__=="__main__":

    env=WashEnv(usd_path="/home/user/GarmentLab/Assets/Garment/Scarf_Tie/ST_Scarf-001/ST_Scarf-001_obj.usd")
    env.reset()
    env.control.robot_reset()
    env.control.grasp([np.array([0,-0.5,0.03467])],[None,np.array([0,np.pi,0])],[True])
    env.control.move([np.array([0.5,0,0.50033])],[None,None],[True,True])
    env.control.move([np.array([0.5,0.1,0.30033])],[None,None],[True,True])
    env.control.move([np.array([0.5,-0.1,0.30033])],[None,None],[True,True])
    env.control.move([np.array([0.5,0.1,0.30033])],[None,None],[True,True])
    env.control.move([np.array([0.5,0.0,0.70033])],[None,None],[True,True])
    while 1:
        env.step()
