from Env.env.BimanualEnv import BimanualEnv
from Env.Config.GarmentConfig import GarmentConfig
from Env.Config.FrankaConfig import FrankaConfig  
from omni.isaac.core.utils.prims import delete_prim
import numpy as np



# if __name__=="__main__":
#     env=BimanualEnv(garment_config=GarmentConfig(usd_path="/home/user/GarmentLab/Assets/Garment/Trousers/Long/PL_059/PL_059_obj.usd",pos=np.array([0.3317,-0.5,0.3]),scale=np.array([0.006,0.006,0.006])))
#     env.reset()
#     env.control.robot_reset()
#     env.control.grasp([np.array([0.29389,-0.38999,0.015]),np.array([0.29736,-0.62,0.015])],[None,None],[True,True])
#     env.control.move([np.array([0.01558,-0.43,0.2]),np.array([0.01558,-0.58,0.2])],[None,None],[True,True])
#     env.control.move([np.array([-0.29,-0.42,0.07]),np.array([-0.28,-0.58,0.07])],[None,None],[True,True])
#     env.control.ungrasp([True,True])


#     env.control.grasp([np.array([-0.04815,-0.29235,0.02]),np.array([-0.29607,-0.38206,0.02])],[None,None],[True,True])
 
#     delete_prim("/World/Attachment/attach")
#     env.control.move([np.array([-0.04815,-0.45,0.2]),np.array([-0.27879,-0.52,0.2])],[None,None],[True,True])
#     env.control.move([np.array([-0.058,-0.7,0.07]),np.array([-0.27415,-0.7,0.07])],[None,None],[True,True])
#     env.control.ungrasp([True,True])

#     env.control.robot_goto_position([np.array([-0.0,-0.2,0.22]),np.array([0,-0.5,0.22])],[None,None],[True,True])
#     while 1:
#         env.step()

if __name__=="__main__":
    garment_config=GarmentConfig(usd_path="/home/user/GarmentLab/Assets/Garment/Trousers/Long/PL_059/PL_059_obj.usd",pos=np.array([0.3317,-0.5,0.3]),scale=np.array([0.006,0.006,0.006]))
    env=BimanualEnv(garment_config=[garment_config])
    env.reset()
    env.control.robot_reset()
    env.control.grasp([np.array([-0.2,-0.35,0.015]),np.array([0.22,-0.33,0.015])],[None,None],[True,True])
    env.control.move([np.array([-0.2,-0.49426,0.23]),np.array([0.2,-0.49426,0.23])],[None,None],[True,True])
    env.control.move([np.array([-0.25,-0.65,0.07]),np.array([0.25,-0.63604,0.07])],[None,None],[True,True])
    env.control.ungrasp([False,False],keep=[True,False])
    

    env.control.robot_goto_position([None,np.array([0,-0.5,0.22])],[None,None],[True,True])
    
    env.control.grasp([np.array([0.28,-0.62,0.02]),None],[None,None],[True,True])
 
    delete_prim("/World/Attachment/attach_1")
    env.control.move([np.array([-0.0059,-0.63664,0.2]),None],[None,None],[True,False])
    env.control.move([np.array([-0.3,-0.63664,0.07]),None],[None,None],[True,False])
    env.control.ungrasp([False,False],keep=[False,False])

    env.control.robot_goto_position([np.array([-0.0,-0.2,0.22]),None],[None,None],[True,True])
    while 1:
        env.step()

        
