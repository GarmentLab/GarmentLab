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

def get_rate_skeleton_points(mesh_points, ratio, type):

    z_values = mesh_points[:, 1]
    min_z = np.min(z_values)
    max_z = np.max(z_values)
    
    # 定义一个函数来计算在某个 z 值附近的点数
    def count_nearby_points(z, threshold=0.1):
        # 计算与 z 的差距小于 threshold 的点
        return np.sum(np.abs(z_values - z) < threshold)

    # 遍历 min_z 附近的点
    while count_nearby_points(min_z) <= 10:
        min_z += 0.01  # 向上增加 min_z
        if min_z > max_z:  # 防止越界
            break

    # 遍历 max_z 附近的点
    while count_nearby_points(max_z) <= 10:
        max_z -= 0.01  # 向下减少 max_z
        if max_z < min_z:  # 防止越界
            break

    z_cut = min_z + ratio * (max_z - min_z)
    distances_to_plane = np.abs(mesh_points[:, 2] - z_cut)
    close_points_mask = distances_to_plane <= 0.2
    close_points = mesh_points[close_points_mask]
    
    if len(close_points) == 0:
        print("No points found within the specified distance from the plane.   ", min_z, max_z, z_cut, ratio)

    # 计算每个点与x轴正方向的夹角，使用atan2来获取角度
    angles = np.degrees(np.arctan2(close_points[:, 1], close_points[:, 0]))  # 计算角度，单位为度
    angles = np.mod(angles, 360)  # 确保角度在0到360之间

    sorted_indices = np.argsort(angles)
    sorted_points = close_points[sorted_indices]

    selected_points = []
    if type == 0:
        selected_angles = [0, 90, 180, 270]
    elif type == 1:
        selected_angles = [45, 135, 225, 315]
    
    # print(ratio)
    for angle in selected_angles:

        closest_idx = np.argmin(np.abs(angles - angle))
        selected_points.append(sorted_points[closest_idx])

    return np.array(selected_points)
    


def get_skeleton_points(part_points, type):
    
    if type == 'pos':
        ratios = [0.2, 0.3, 0.4, 0.5, 0.6]
    elif type == 'neg':
        ratios = [0.8, 0.7, 0.6, 0.5, 0.4]
    ans = []
    for (idx, ratio) in enumerate(ratios):
        skeleton_points = get_rate_skeleton_points(part_points, ratio, idx%2)
        ans.extend(skeleton_points)
        
    return ans
    

def get_bowl_hat_mesh_skeleton_id(mesh_points, type):
    
    skeleton_points = get_skeleton_points(mesh_points, type)

    skeleton_indices = []
    for point in skeleton_points:
        distances = np.linalg.norm(mesh_points - point, axis=1)
        nearest_index = np.argmin(distances)
        skeleton_indices.append(nearest_index)
    
    return skeleton_indices
    
def visualize_skeleton(mesh_points, type):
    
    skeleton_indices = get_bowl_hat_mesh_skeleton_id(mesh_points, type)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(mesh_points)
    # colors = np.tile([0, 1, 0], (len(mesh_points), 1))  # 绿色 [0, 1, 0]
    # colors[skeleton_indices] = [1, 0, 0]  # 红色 [1, 0, 0]
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd], window_name="Skeleton Visualization", width=800, height=600)
    

if __name__ == "__main__":
    
    directory = f"data/bowl_hat/unigarment/cd_processed/mesh_pcd"
    
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
                
                z_values = mesh_points[:, 2]
                max_z = np.max(z_values)
                min_z = np.min(z_values)
                print(f"max_z: {max_z}, min_z: {min_z}")
                
                if 'p_1' in filename:
                    visualize_skeleton(mesh_points, 'neg')
                # elif 'p_1' in filename:
                #     visualize_skeleton(mesh_points, 'neg')
                # visualize_pcd_and_mesh(mesh_points, pcd_points)