import sys
sys.path.append("/home/user/GarmentLab/Env")
sys.path.append("/home/user/GarmentLab")
from Env.env.FoldEnv import FoldEnv
from Env.Config.GarmentConfig import GarmentConfig
from Env.Config.FrankaConfig import FrankaConfig
import numpy as np
from LearningBaseline.unigarmentmanip.model.UniGarmentManip_Encapsulation import UniGarmentManip_Encapsulation

if __name__=="__main__":
    garment_config=GarmentConfig(usd_path="/home/user/GarmentLab/Assets/Garment/Tops/NoCollar_Lsleeve_FrontClose/TNLC_Dress057/TNLC_Dress057_obj.usd",pos=np.array([-0.1,0,0.2]),ori=np.array([0,0,-np.pi/2]),particle_contact_offset=0.01)
    env=FoldEnv(garment_config=[garment_config],
                franka_config=FrankaConfig(ori=[np.array([0,0,0])]))
    env.reset()

    while 1:
        env.step()
