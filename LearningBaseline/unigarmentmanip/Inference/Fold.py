import sys
sys.path.append("/home/user/GarmentLab/Env")
sys.path.append("/home/user/GarmentLab")
from Env.env.FoldEnv import FoldEnv
from Env.Config.GarmentConfig import GarmentConfig
from Env.Config.FrankaConfig import FrankaConfig
import numpy as np
from LearningBaseline.unigarmentmanip.model.UniGarmentManip_Encapsulation import UniGarmentManip_Encapsulation
import open3d as o3d

if __name__=="__main__":
    garment_config=GarmentConfig(usd_path="/home/user/GarmentLab/Assets/Garment/Tops/NoCollar_Lsleeve_FrontClose/TNLC_Dress057/TNLC_Dress057_obj.usd",pos=np.array([0.5,-0.5,0.2]),ori=np.array([0,0,0]),particle_contact_offset=0.01)
    env=FoldEnv(garment_config=[garment_config],
                franka_config=FrankaConfig(ori=[np.array([0,0,0])]))
    env.reset()
    
    model = UniGarmentManip_Encapsulation(catogory="Tops_LongSleeve") 

    for i in range(50):
        env.step()
        
    env.robots[0].movep(np.array([0.05,0,0.5]))
        
    points,color=env.camera.get_point_cloud_data(save_or_not=False,real_time_watch=False)
    
    manipulation_points, _ = model.get_manipulation_points(input_pcd=points, index_list=[488, 883, 683, 1011, 100, 10])
    manipulation_points[0,2]=0.02
    manipulation_points[1,2]=0.05
    manipulation_points[2,2]=0.02
    manipulation_points[3,2]=0.05
    manipulation_points[4,2]=0.02
    manipulation_points[5,2]=0.05

    
    env.control.robot_reset()
    env.control.grasp([manipulation_points[0]],[None],[True])   
    env.control.move([manipulation_points[1]],[None],[True])
    env.control.ungrasp([False],)
    env.control.grasp([manipulation_points[2]],[None],[True])
    manipulation_points[2,2]=0.2
    print("move up")
    env.control.move([manipulation_points[2]],[None],[True])
    print("move up finished")
    env.control.move([manipulation_points[3]],[None],[True])
    env.control.ungrasp([False],)
    env.control.grasp([manipulation_points[4]],[None],[True])
    env.control.move([manipulation_points[5]],[None],[True])
    env.control.ungrasp([False],)
    
    env.robots[0].movep(np.array([0.05,0,0.5]))
    
    
    
    while 1:
        env.step()
