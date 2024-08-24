import sys
sys.path.append("/home/isaac/GarmentLab")
from Env.env.StoreEnv import StoreEnv
from Env.Config.GarmentConfig import GarmentConfig
from Env.Config.FrankaConfig import FrankaConfig  
from Env.Config.DeformableConfig import DeformableConfig
import numpy as np


if __name__=="__main__":
    
    franka_config = FrankaConfig(pos=[np.array([-1.0,0,0])])
    deform_config = DeformableConfig(pos=np.array([-0.62957,1.00223,0.3]))
    env=StoreEnv(frankaConfig=franka_config, deformbaleConfig=deform_config)
    env.reset()

    # open the cabinet
    env.def_control.robot_reset()
    env.def_control.robot[0].open()
    for i in range(200):
        env.def_control.robot_step([np.array([-0.30583,0.23897,0.6649])],[np.array([-np.pi / 2, 0, -np.pi / 2])],[True])
    env.def_control.robot[0].close()
    for i in range(50):
        env.def_control.robot_step([np.array([-0.80583,0.23897,0.6649])],[np.array([-np.pi / 2, 0, -np.pi / 2])],[True])
    env.def_control.robot[0].open()
    for i in range(200):
        env.def_control.robot_step([np.array([-0.80583,0.23897,0.6649])],[np.array([-np.pi / 2, 0, -np.pi / 2])],[True])

    # place the hat in the cabinet
    env.def_control.grasp([np.array([-0.62039,-0.34289,0.09978])],[None],[True])
    env.def_control.move([np.array([-0.12223,0.24242,0.57784])],[None],[True])
    env.def_control.ungrasp([False])

    # close the cabinet
    for i in range(200):
        env.def_control.robot_step([np.array([-0.80583,0.23897,0.6649])],[np.array([-np.pi / 2, 0, -np.pi / 2])],[True])
    for i in range(50):
        env.def_control.robot_step([np.array([-0.82495,0.58052,0.5])],[np.array([-np.pi / 2, 0, -np.pi / 2])],[True])
    for i in range(50):
        env.def_control.robot_step([np.array([-0.60515,0.58052,0.5])],[np.array([-np.pi / 2, 0, -np.pi / 2])],[True])
    for i in range(50):
        env.def_control.robot_step([np.array([-0.60515,0.42882,0.5])],[np.array([-np.pi / 2, 0, -np.pi / 2])],[True])
    for i in range(50):
        env.def_control.robot_step([np.array([-0.40515,0.22882,0.5])],[np.array([-np.pi / 2, 0, -np.pi / 2])],[True])
    for i in range(50):
        env.def_control.robot_step([np.array([-0.10515,0.22882,0.5])],[np.array([-np.pi / 2, 0, -np.pi / 2])],[True])

    reward_flag = env.get_reward()
    
    while 1:
        env.step()