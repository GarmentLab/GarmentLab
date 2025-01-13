import sys
from Env.env.HangEnv import HangEnv
from Env.Config.GarmentConfig import GarmentConfig
from Env.Config.FrankaConfig import FrankaConfig
import numpy as np


if __name__=="__main__":

    garment_config = GarmentConfig(usd_path="/home/user/GarmentLab/Assets/Garment/Dress/Long_ShortSleeve/DLSS_Dress182/DLSS_Dress182_obj.usd")
    garment_config.pos = np.array([-0.65481,-1.27712,0.54132])
    garment_config.ori = np.array([0.47366,-0.32437,-0.46264,-0.67557])
    garment_config.scale = np.array([0.006, 0.006, 0.006])
    garment_config.particle_contact_offset = 0.01
    franka_config = FrankaConfig(franka_num=1, pos=[np.array([-0.4,-1.7,0.])], ori=[np.array([0,0,0])])

    env=HangEnv(garment_config=[garment_config], franka_config=franka_config)

    env.reset()
    env.control.robot_reset()

    # for i in range(10000):
    #     env.world.pause()


    env.control.grasp([np.array([0.19776,-1.52166,0.01756])],[None],[True])
    env.control.move([np.array([0.19933,-1.51923,0.5523])],[None],[True])
    env.control.move([np.array([0.01957,-1.71923,1.04274])],[None],[True])
    env.control.move([np.array([0.01957,-2.11923,0.94274])],[None],[True])
    env.control.ungrasp([False])
    reward_flag = env.get_reward(pick_threshold = 0.5)


    while 1:
        env.step()
