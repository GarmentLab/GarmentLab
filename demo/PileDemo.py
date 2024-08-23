import sys
sys.path.append("/home/isaac/GarmentLab")
from Env.env.PileEnv import PileEnv
from Env.Config.GarmentConfig import GarmentConfig
from Env.Config.FrankaConfig import FrankaConfig  
import numpy as np

if __name__=="__main__":
    
    pile0_path="/home/isaac/GarmentLab/Assets/Garment/Dress/Long_ShortSleeve/DLSS_Dress074/DLSS_Dress074_obj.usd"
    pile1_path="/home/isaac/GarmentLab/Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_082/TCLC_082_obj.usd"
    pile2_path="/home/isaac/GarmentLab/Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top612/TCLC_Top612_obj.usd"
    pile3_path="/home/isaac/GarmentLab/Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_top117/TCLC_top117_obj.usd"
    pile4_path="/home/isaac/GarmentLab/Assets/Garment/Trousers/Short/PS_Short141/PS_Short141_obj.usd"

    
    garment_config_list = list()
    
    config = GarmentConfig(usd_path=pile0_path,visual_material_usd=None)
    config.pos = np.array([0.84139,0.2338,0.14144])
    config.ori = np.array([0.23764,-0.24413,0.30751,0.88846])
    config.scale = np.array([0.006, 0.006, 0.006])
    garment_config_list.append(config)
    
    # config = GarmentConfig(usd_path=pile1_path,visual_material_usd=None)
    # config.pos = np.array([0.84139,-0.07035,0.44144])
    # config.ori = np.array([-0.10213,0.32503,-0.67489,-0.65456])
    # config.scale = np.array([0.0075, 0.0075, 0.0075])
    # garment_config_list.append(config)
    
    config = GarmentConfig(usd_path=pile2_path,visual_material_usd=None)
    config.pos = np.array([0.84139,0.73239,0.34144])
    config.ori = np.array([0.23764,-0.24413,0.30751,0.88846])
    config.scale = np.array([0.0075, 0.0075, 0.0075])
    garment_config_list.append(config)
    
    config = GarmentConfig(usd_path=pile3_path,visual_material_usd=None)
    config.pos = np.array([0.84139,1.51593,0.20144])
    config.ori = np.array([0.23764,-0.24413,0.30751,0.88846])
    config.scale = np.array([0.0075, 0.0075, 0.0075])
    garment_config_list.append(config)
    
    config = GarmentConfig(usd_path=pile4_path,visual_material_usd=None)
    config.pos = np.array([0.84139,0.689,0.1144])
    config.ori = np.array([0.23764,-0.24413,0.30751,0.88846])
    config.scale = np.array([0.0075, 0.0075, 0.0075])
    garment_config_list.append(config)
    
    
    franka = FrankaConfig(franka_num=1, pos=[np.array([-0.3,0.0,0.])], ori=[np.array([0,0,0])])
    
    env=PileEnv(garment_config=garment_config_list, franka_config=franka)
    
    env.reset()

    env.control.robot_reset()
    env.control.grasp([np.array([0.31333,-0.47789,0.02641])],[None],[True],assign_garment=0)
    env.control.move([np.array([0.19721776,0.34098437,0.9674991])],[None],[True])
    env.get_reward(pick_threshold=0.4, garment_id=0)
    
    while 1:
        env.step()
    