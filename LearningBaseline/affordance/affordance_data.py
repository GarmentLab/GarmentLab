import sys
from affordance_env import AffordanceEnv
from Env.Config.GarmentConfig import GarmentConfig
from Env.Config.FrankaConfig import FrankaConfig  
import numpy as np
import yaml



if __name__=="__main__":

    filename = "/home/isaac/GarmentLab/LearningBaseline/affordance/config/config_5.yaml"
    with open(filename, 'r') as file:
        task_config = yaml.safe_load(file)

    garment_config = GarmentConfig(usd_path=task_config["garment_config"]["garment_path"])
    garment_config.pos = np.array(task_config["garment_config"]["garment_pos"])
    garment_config.ori = np.array(task_config["garment_config"]["garment_ori"])
    garment_config.scale = np.array(task_config["garment_config"]["garment_scale"])
    garment_config.particle_contact_offset = 0.01
    franka_config = FrankaConfig(franka_num=1, pos=[np.array([0,0,0.])], ori=[np.array([0,0,0])])
    
    
    env=AffordanceEnv(garment_config=[garment_config], franka_config=franka_config, task_config=task_config)
    
    env.get_demo(task_config["demo_point"])
    env.exploration()
    

    while 1:
        env.step()