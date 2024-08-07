import sys
sys.path.append("/home/user/GarmentLab")
from Env.env.HumanEnv import HumanEnv
from Env.Config.GarmentConfig import GarmentConfig
from Env.Config.FrankaConfig import FrankaConfig  
import numpy as np


if __name__=="__main__":
    
    config0 = GarmentConfig(usd_path="/media/sim/WD_BLACK/ClothesNetData/ClothesNetM_usd/Hat/HA_Hat007/HA_Hat007_obj.usd")
    config0.pos = np.array([-0.62169,-0.5502,-0.1472])
    config0.ori = np.array([0.70711,0.70711,0.0,0.0])
    config0.scale = np.array([0.0075, 0.0075, 0.0075])
    config0.particle_contact_offset = 0.01

    config1 = GarmentConfig(usd_path="/media/sim/WD_BLACK/ClothesNetData/ClothesNetM_usd/Scarf_Tie/ST_Scarf-005/ST_Scarf-005_obj.usd")
    config1.pos = np.array([-0.62169,0.54238,0.4472])
    config1.ori = np.array([0.70711,0.70711,0.0,0.0])
    config1.scale = np.array([0.0075, 0.0075, 0.0075])
    config1.particle_contact_offset = 0.01

    franka = FrankaConfig(franka_num=2, pos=[np.array([-0.7,-0.2,1]),np.array([0.,0.5,1])], ori=[np.array([0.,0.,0]),np.array([0.,0.,np.pi])])
    
    env=HumanEnv(garment_config=[config0,config1], franka_config=franka)
    
    
    env.stop()
    # env.garment.set_pose(pos = config.pos, ori = config.ori)
    env.reset() 
    env.control.robot_reset()
    # 
    env.control.grasp([np.array([-0.57952,0.45522,1.01176]),np.array([-0.46332,0.46664,1.02176])],[None,None],[True,True],assign_garment=1)

    env.control.move([np.array([-0.78477,0.48995,1.79849]),np.array([-0.48333,0.82575,1.79849]),],[None,None],[True,True])
    env.control.move([np.array([-0.80282,0.3686,1.65262]),np.array([0.65775,0.2003,1.66116]),],[None,None],[True,True])
    print("1111111111111111")
    env.control.move([np.array([-0.74086,-0.58022,1.47973]),np.array([0.35775,0.2003,0.87173]),],[None,None],[True,True])
    print("2222222222222222")
    env.control.move([np.array([-0.31494,-0.28949,1.42068]),np.array([0.35775,0.2003,0.87173]),],[None,None],[True,True])
    print("3333333333333333")
    env.control.move([np.array([-0.00227,-0.62359,1.36109]),np.array([0.35775,0.2003,0.87173]),],[None,None],[True,True])
    print("4444444444444444")
    env.control.move([np.array([-0.16306,-0.17359,1.02131]),np.array([0.35775,0.2003,0.87173]),],[None,None],[True,True])

    print("555555555555555555555")
    env.control.ungrasp([False,True], keep=True)
    for i in range(200):
        env.control.robot_step(pos=[np.array([-0.64065,-0.86618,1.6235]),None],ori=[None,None],flag=[True,False])
    print("666666666666666666")
    # env.control.move([np.array([-0.18266,-0.27408,1.42152]),np.array([0.35775,0.2003,0.87173]),],[None,None],[True,True])
    # env.control.move([np.array([-0.47344,-0.16961,1.42152]),np.array([0.35775,0.2003,0.87173]),],[None,None],[True,True])
    # env.control.move([np.array([-0.50344,0.27617,1.42152]),np.array([0.35775,0.2003,0.87173]),],[None,None],[True,True])
    # env.control.move([np.array([-0.63344,0.42617,1.42152]),np.array([0.35775,0.2003,0.87173]),],[None,None],[True,True])
    # env.control.move([np.array([-0.05975,0.00617,1.0145]),np.array([0.35775,0.2003,0.87173]),],[None,None],[True,True])
    # print("##################################")
    
    

    env.control.move([np.array([-0.05975,0.00617,1.0145]),np.array([-0.18266,-0.27408,1.62152]),],[None,None],[False,True])
    env.control.move([np.array([-0.05975,0.00617,1.0145]),np.array([-0.40344,-0.08961,1.50152]),],[None,np.array([np.pi,0,0])],[False,True])
    env.control.move([np.array([-0.05975,0.00617,1.0145]),np.array([-0.50344,0.27617,1.50152]),],[None,np.array([np.pi,0,0])],[False,True])
    env.control.move([np.array([-0.05975,0.00617,1.0145]),np.array([-0.63344,0.42617,1.50152]),],[None,np.array([np.pi,0,0])],[False,True])
    env.control.move([np.array([-0.05975,0.00617,1.0145]),np.array([-0.05975,0.00617,1.0145]),],[None,None],[False,True])
    final_pts = env.garment.get_vertice_positions()
    np.savetxt("/home/isaac/GarmentLab/standard_pts.txt",final_pts)
    
    # env.control.ungrasp([False,False])
    # env.control.grasp([np.array([0,-0.3,0.04]),np.array([0,-0.6,0.02])],[None,None],[True,True])
    # env.control.move([np.array([0,-0.3,0.5]),np.array([0,-0.6,0.5])],[None,None],[True,True])
    # env.control.move([np.array([0,-0.3,0.1]),np.array([0,-0.6,0.5])],[None,None],[True,True])
    # env.control.ungrasp([False,True])
    # env.control.move([None,np.array([0,-0.6,0.1])],[None,None],[False,True])
    while 1:
        env.step()
    