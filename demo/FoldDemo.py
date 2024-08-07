from Env.env.FoldEnv import FoldEnv
from Env.Config.DeformableConfig import DeformableConfig
from Env.Config.FrankaConfig import FrankaConfig
from Env.Config.GarmentConfig import GarmentConfig
import numpy as np

if __name__=="__main__":
    garment_config=GarmentConfig(usd_path="/home/user/GarmentLab/Assets/Garment/Tops/NoCollar_Lsleeve_FrontClose/TNLC_Dress057/TNLC_Dress057_obj.usd",pos=np.array([0,-0.95,0.3]),ori=np.array([0,0,0]),particle_contact_offset=0.01)
    env=FoldEnv(garment_config=[garment_config],
                franka_config=FrankaConfig(ori=[np.array([0,0,-np.pi/2])]))
    env.reset()
    env.control.robot_reset()
    env.control.grasp([np.array([0.35282,-0.26231,0.02])],[None],[True])

    # env.control.move([np.array([0.28138,-0.26231,0.22])],[None],[True])
    env.control.move([np.array([0.1,-0.26231,0.22])],[None],[True])
    env.control.move([np.array([-0.12,-0.26231,0.1])],[None],[True])
    env.control.ungrasp([False],)

    env.control.grasp([np.array([-0.377,-0.26231,0.02])],[None],[True])

    # env.control.move([np.array([-0.3,-0.26231,0.22])],[None],[True])
    env.control.move([np.array([0.0,-0.26231,0.22])],[None],[True])
    env.control.move([np.array([0.07,-0.26231,0.03])],[None],[True])
    env.control.ungrasp([False],)

    env.control.grasp([np.array([0.0,-0.65,0.015])],[None],[True])

    env.control.move([np.array([-0.0,-0.515,0.22])],[None],[True])
    env.control.move([np.array([0.0,-0.25,0.05])],[None],[True])
 
    env.control.ungrasp([True],)

    env.control.robot_goto_position([np.array([-0.0,-0.566,0.22])],[None],[True])

    while 1:
        env.step()

