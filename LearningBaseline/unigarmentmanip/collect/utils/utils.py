import numpy as np              
import open3d as o3d


def get_pcd2mesh_correspondence(garment_mesh_points, pcd_keypoints):
    mesh_keypoints = []
    mesh_keypoints_id = []
    for key_point in pcd_keypoints:
        min_dist = 100000
        min_dist_id = -1
        for i, mesh_point in enumerate(garment_mesh_points):
            dist = np.linalg.norm(mesh_point - key_point)
            if dist < min_dist:
                min_dist = dist
                min_dist_id = i
        mesh_keypoints.append(garment_mesh_points[min_dist_id])
        mesh_keypoints_id.append(min_dist_id)
        
    return mesh_keypoints, mesh_keypoints_id

def get_mesh2pcd_correspondence(mesh_keypoints, pcd_points):
    pcd_keypoints = []
    pcd_keypoints_id = []
    for key_point in mesh_keypoints:
        min_dist = 100000
        min_dist_id = -1
        for i, pcd_point in enumerate(pcd_points):
            dist = np.linalg.norm(pcd_point - key_point)
            if dist < min_dist:
                min_dist = dist
                min_dist_id = i
        pcd_keypoints.append(pcd_points[min_dist_id])
        pcd_keypoints_id.append(min_dist_id)

    return pcd_keypoints, pcd_keypoints_id


def get_visible_mask(pcd_points, key_points):
    x, y, z = pcd_points[:, 0], pcd_points[:, 1], pcd_points[:, 2]
    
    max_z_dict = {}
    
    for i in range(len(pcd_points)):
        xy = (x[i], y[i])
        if xy not in max_z_dict:
            max_z_dict[xy] = z[i]
        else:
            max_z_dict[xy] = max(max_z_dict[xy], z[i])
    
    mask = []
    for key_point in key_points:
        key_x, key_y, key_z = key_point
        max_z_at_xy = max_z_dict.get((key_x, key_y), float('-inf'))
        is_visible = key_z >= max_z_at_xy
        mask.append(is_visible)
    
    return mask
        
import os
import re
def get_max_sequence_number(directory):
    # 获取目录下的所有文件
    files = os.listdir(directory)

    # 筛选出以 "p_" 开头并以 ".npz" 结尾的文件
    pc_files = [f for f in files if f.startswith('p_') and f.endswith('.npz')]

    # 使用正则表达式提取文件名中的序号
    max_sequence = -1
    for file in pc_files:
        match = re.search(r'p_(\d+)\.npz', file)  # 匹配 "pc_i.npz" 格式的文件名
        if match:
            sequence = int(match.group(1))  # 提取序号并转换为整数
            max_sequence = max(max_sequence, sequence)

    return max_sequence


def find_closest_point(initial_mesh_points, pick_point):
    # 计算每个点到 pick_point 的欧几里得距离
    distances = np.linalg.norm(initial_mesh_points - pick_point, axis=1)
    
    # 找到距离最近的点的索引
    closest_index = np.argmin(distances)
    
    # 返回距离最近的点
    closest_point = initial_mesh_points[closest_index]
    
    return closest_point
