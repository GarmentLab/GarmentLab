import open3d as o3d
import numpy as np
import os

def visualize_pcd_and_mesh(mesh_points, pcd_points):
    mesh_pcd = o3d.geometry.PointCloud()
    mesh_pcd.points = o3d.utility.Vector3dVector(mesh_points)
    mesh_pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (mesh_points.shape[0], 1)))  # 绿色

    # 可视化 pcd_points (红色)
    pcd_pcd = o3d.geometry.PointCloud()
    pcd_pcd.points = o3d.utility.Vector3dVector(pcd_points)
    pcd_pcd.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (pcd_points.shape[0], 1)))  # 红色

    # 可视化两个点云
    o3d.visualization.draw_geometries([mesh_pcd, pcd_pcd])

if __name__ == "__main__":
    
    type = "ns_tops"
    directory = f"data/trousers/unigarment/cd_rotation/mesh_pcd"
    
    # path = "data/scarf/cd_original/mesh_pcd/0_ST_Scarf-009_obj/p_0.npz"


    # mesh_points = np.load(path)['mesh_points']
    # pcd_points = np.load(path)['pcd_points']
    # visualize_pcd_and_mesh(mesh_points, pcd_points)
    
    # path1 = "data/Hat/unigarment/cd_processed/mesh_pcd/0_HA_Hat044_obj/p_0.npz"
    # path2 = "data/Hat/unigarment/cd_processed/mesh_pcd/0_HA_Hat044_obj/p_0_0.npz"

    # visualize_pcd_and_mesh(np.load(path1)['mesh_points'], np.load(path1)['pcd_points'])
    # for i in range(20):
        
    #     path2 = f"data/Hat/unigarment/cd_processed/mesh_pcd/0_HA_Hat044_obj/p_0_{i}.npz"
    #     points1 = np.load(path1)['pcd_points']
    #     points2 = np.load(path2)['pcd_points']
    #     visualize_pcd_and_mesh(points1, points2)
    
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith("p_11_4.npz"):
                file_path = os.path.join(dirpath, filename)
                
                data = np.load(file_path)
                print(file_path)
                
                mesh_points = data['mesh_points']
                pcd_points = data['pcd_points']
                
                y_values = mesh_points[:, 1]
                max_y = np.max(y_values)
                min_y = np.min(y_values)    
                #print(f"max_y: {max_y}, min_y: {min_y}")
                

                visualize_pcd_and_mesh(mesh_points, pcd_points)