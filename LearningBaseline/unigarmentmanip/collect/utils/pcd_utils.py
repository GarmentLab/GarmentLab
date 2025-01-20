import open3d as o3d
import numpy as np
import os

def quaternion_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def rotate_point_cloud(point_cloud, quaternion):
    q_conjugate = quaternion_conjugate(quaternion)
    rotated_points = []

    for point in np.asarray(point_cloud.points):
        # 创建点的四元数表示
        p_q = np.array([0, *point])  # [0, x, y, z]
        
        # 计算逆旋转
        rotated_point = quaternion_multiply(quaternion_multiply(q_conjugate, p_q), quaternion)
        
        # 取出旋转后的 x, y, z
        rotated_points.append(rotated_point[1:])  # 只取 x, y, z

    point_cloud.points = o3d.utility.Vector3dVector(rotated_points)


def normalize_point_cloud(ply_path, quaternion=None, x_range=(-1, 1), y_range=(-1, 1), save_path=None): 
    point_cloud = o3d.io.read_point_cloud(ply_path)

    # 计算质心
    centroid = np.mean(np.asarray(point_cloud.points), axis=0)

    # 将点云平移到原点
    normalized_points = np.asarray(point_cloud.points) - centroid

    # 计算当前范围
    min_coords = np.min(normalized_points[:, :2], axis=0)  
    max_coords = np.max(normalized_points[:, :2], axis=0)  

    # 计算缩放系数
    scale_x = (x_range[1] - x_range[0]) / (max_coords[0] - min_coords[0]) if (max_coords[0] - min_coords[0]) != 0 else 1
    scale_y = (y_range[1] - y_range[0]) / (max_coords[1] - min_coords[1]) if (max_coords[1] - min_coords[1]) != 0 else 1
    scale = min(scale_x, scale_y)  

    # 缩放
    scaled_points = normalized_points * scale
    point_cloud.points = o3d.utility.Vector3dVector(scaled_points)

    # 对点云进行旋转
    if quaternion is not None:
        rotate_point_cloud(point_cloud, quaternion)

    if save_path is not None:
        o3d.io.write_point_cloud(save_path, point_cloud)

    print("Normalized and scaled point cloud saved to:", save_path)
    
    return save_path, point_cloud, quaternion, scale, centroid


def unnormalize_pcd_points(pcd_points, scale, centroid, quaternion=None):

    scaled_points = np.asarray(pcd_points) / scale

    unnormalized_points = scaled_points + centroid

    return unnormalized_points


def normalize_pcd_points(pcd_points, x_range=(-1, 1), y_range=(-1, 1)): 

    # 计算质心
    centroid = np.mean(np.asarray(pcd_points), axis=0)

    # 将点云平移到原点
    normalized_points = np.asarray(pcd_points) - centroid

    # 计算当前范围
    min_coords = np.min(normalized_points[:, :2], axis=0)  
    max_coords = np.max(normalized_points[:, :2], axis=0)  

    # 计算缩放系数
    scale_x = (x_range[1] - x_range[0]) / (max_coords[0] - min_coords[0]) if (max_coords[0] - min_coords[0]) != 0 else 1
    scale_y = (y_range[1] - y_range[0]) / (max_coords[1] - min_coords[1]) if (max_coords[1] - min_coords[1]) != 0 else 1
    scale = min(scale_x, scale_y)  

    # 缩放
    normalized_points = normalized_points * scale
    
    return normalized_points, centroid, scale


def get_visible_indices(mesh_points):
    
    from scipy.spatial import cKDTree
    # 初始化点云
    mesh_pcd = o3d.geometry.PointCloud()
    mesh_pcd.points = o3d.utility.Vector3dVector(mesh_points)

    # 使用 hidden_point_removal 方法计算初始的可见点
    mesh, visible_indices = mesh_pcd.hidden_point_removal(np.asarray([0, 0, 8]), 5000)
    visible_indices = set(visible_indices)  # 转换为集合，便于查找

    # 创建 KD-tree 来计算最近邻
    tree = cKDTree(mesh_points)

    # 遍历所有点
    for i, point in enumerate(mesh_points):
        if i in visible_indices:
            continue  # 如果该点已经可见，跳过

        # 找到该点的最近 4 个邻居
        distances, neighbors = tree.query(point, k=4)  # 包括自己，所以取 k=5
        neighbors = neighbors[1:]  # 排除自己

        # 检查最近 4 个邻居是否全部可见
        if all(neighbor in visible_indices for neighbor in neighbors):
            visible_indices.add(i)  # 如果全部邻居可见，将当前点设为可见

    return sorted(list(visible_indices))  # 返回排序后的可见点索引


def nearest_mesh2pcd(mesh_points, pcd_points, mesh_keypoints_id):
                   
    from scipy.spatial import cKDTree

    mesh_keypoints = mesh_points[mesh_keypoints_id]

    pcd_tree = cKDTree(pcd_points)

    _, nearest_indices = pcd_tree.query(mesh_keypoints)

    return nearest_indices.tolist()


def rotate_point_cloud(points, euler_angles):

    points = np.asarray(points)

    roll, pitch, yaw = np.deg2rad(euler_angles)  # 将角度转换为弧度

    # 绕 x 轴旋转的旋转矩阵
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # 绕 y 轴旋转的旋转矩阵
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # 绕 z 轴旋转的旋转矩阵
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # 计算总的旋转矩阵，旋转顺序为 R_z * R_y * R_x
    rotation_matrix = R_z @ R_y @ R_x

    # 应用旋转矩阵到点云
    rotated_points = points @ rotation_matrix.T  # 转置矩阵进行点乘

    return rotated_points