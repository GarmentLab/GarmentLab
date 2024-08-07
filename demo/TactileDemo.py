import numpy as np
from isaacsim import SimulationApp
import torch
import open3d as o3d

simulation_app = SimulationApp({"headless": False})

from Env.env.TactileEnv import TactileEnv
import numpy as np





if __name__=="__main__":
    env=TactileEnv()
    env.reset()
    env.robot.open()
    env.robot.movel(np.array([0.35,0,0.5]))
    env.robot.movel(np.array([0.35,0,0.1]))
    env.robot.close()
    # env.robot.movel(np.array([0.3,0,0.09]))
    env.robot.movel(np.array([0.35,0,0.6]))
    left_tactile,right_tactile=env.get_tactile()
    print(left_tactile['normal']['pair_contacts_count'])
    print(right_tactile['normal']['pair_contacts_count'])
    points=left_tactile['normal']['points']
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
    for j in range(50):
        env.step()
    env.robot.movel(np.array([0.45,0,0.6]))
    env.robot.movel(np.array([0.45,0,0.4]))
    env.robot.movel(np.array([0.45,0,0.35]))
    print("done")
    points=left_tactile['normal']['points']
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(points)
    vis=o3d.visualization.Visualizer()
    vis.create_window()
    
    vis.add_geometry(pcd)
    
    render_option=vis.get_render_option()
    render_option.point_size=10
    vis.run()
    vis.destroy_window()
    while 1:
        env.step()
        
    