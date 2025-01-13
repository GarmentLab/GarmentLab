import open3d as o3d
import numpy as np
import os
import sys
sys.path.append("/home/user/DexGarmentLab-master/DexGarmentLab-master/unigarment/collect")

from pcd_utils import get_visible_indices

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

def get_rate_skeleton_points(part_points, ratio, sample_num):

    y_values = part_points[:, 1]
    min_y = np.min(y_values)
    max_y = np.max(y_values)
    
    # 手腕
    wrist_y = min_y + ratio * (max_y - min_y)
    distances_to_plane = np.abs(part_points[:, 1] - wrist_y)
    close_points_mask = distances_to_plane <= 0.1
    close_points = part_points[close_points_mask]
    sorted_points = close_points[close_points[:, 0].argsort()]
    x_values = sorted_points[:, 0]
    min_x = np.min(x_values)
    max_x = np.max(x_values)
    indices = np.linspace(0, len(sorted_points) - 1, sample_num, dtype=int)
    wrist_points = sorted_points[indices]
    
    return wrist_points



def get_part_skeleton_points(part_points):
    
    ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
    ans = []
    for ratio in ratios:
        skeleton_points = get_rate_skeleton_points(part_points, ratio, 4)
        ans.extend(skeleton_points)
        
    return ans
    

def get_glove_mesh_skeleton_id(mesh_points):
    
    visible_indices = get_visible_indices(mesh_points)
    invisible_indices = np.setdiff1d(np.arange(len(mesh_points)), visible_indices)
    upper_part = mesh_points[visible_indices]
    lower_part = mesh_points[invisible_indices]    
    
    upper_skeleton_points = get_part_skeleton_points(upper_part)
    lower_skeleton_points = get_part_skeleton_points(lower_part)

    skeleton_points = upper_skeleton_points + lower_skeleton_points

    skeleton_indices = []
    for point in skeleton_points:
        distances = np.linalg.norm(mesh_points - point, axis=1)
        nearest_index = np.argmin(distances)
        skeleton_indices.append(nearest_index)
    
    return skeleton_indices
    
def visualize_skeleton(mesh_points):
    
    skeleton_indices = get_glove_mesh_skeleton_id(mesh_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh_points)
    colors = np.tile([0, 1, 0], (len(mesh_points), 1))  # 绿色 [0, 1, 0]
    colors[skeleton_indices] = [1, 0, 0]  # 红色 [1, 0, 0]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="Skeleton Visualization", width=800, height=600)
    

if __name__ == "__main__":
    
    directory = f"data/glove/skeleton/normalized"
    
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".npz"):
                file_path = os.path.join(dirpath, filename)
                
                data = np.load(file_path)
                print(file_path)
                
                mesh_points = data['mesh_points']
                pcd_points = data['pcd_points']
                
                y_values = mesh_points[:, 1]
                max_y = np.max(y_values)
                min_y = np.min(y_values)    
                print(f"max_y: {max_y}, min_y: {min_y}")
                
                x_values = mesh_points[:, 0]
                max_x = np.max(x_values)
                min_x = np.min(x_values)
                print(f"max_x: {max_x}, min_x: {min_x}")
                
                visualize_skeleton(mesh_points)
                # visualize_pcd_and_mesh(mesh_points, pcd_points)