import sys
from Env.env.BimanualEnv import BimanualEnv
from Env.Config.GarmentConfig import GarmentConfig
from Env.Config.FrankaConfig import FrankaConfig  
import numpy as np
import torch


if __name__=="__main__":
        config = GarmentConfig(usd_path="/home/user/GarmentLab/Assets/Garment/Dress/Long_ShortSleeve/DLSS_Dress074/DLSS_Dress074_obj.usd")
        config.pos = np.array([0.12837,0.09307,-0.06812])
        config.ori = np.array([0.09024,0.07689,0.57673,0.80829])
        config.scale = np.array([0.006, 0.006, 0.006])
        config.particle_contact_offset = 0.01
        franka = FrankaConfig(franka_num=2, pos=[np.array([-0.3,0.0,0.]),np.array([0.7,0.,0.])], ori=[np.array([0,0,0]),np.array([0,0,np.pi])])


        
        env=BimanualEnv(garment_config=[config], franka_config=franka)

        env.reset()
        
        
        env.control.robot_reset()
        env.control.grasp([np.array([-0.11763,-0.22598,0.03217]),np.array([0.22957,-0.31503,0.05])],[None,np.array([0,np.pi,0])],[True,True])
        env.control.move([np.array([-0.19731,-0.26433,1.0033]),np.array([0.20364,-0.27814,1.0033])],[None,None],[True,True])
        env.control.move([np.array([-0.19731,0.26433,1.0033]),np.array([0.20364,0.27814,1.0033])],[None,None],[True,True])
        env.control.move([np.array([-0.19731,-0.26433,0.2033]),np.array([0.20364,-0.27814,0.2033])],[None,None],[True,True])
        env.control.ungrasp([False,False])


        env.get_reward(rotation=config.ori)
        while 1:
            env.step()

        
