import open3d as o3d
import numpy as np
import os
import sys
sys.path.append("/home/user/DexGarmentLab-master/DexGarmentLab-master/unigarment/collect")
from scipy.spatial.distance import cdist
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


def get_armpits_points(mesh_points, tolerance=0.05):
    
    y_values = mesh_points[:, 1]
    min_y = np.min(y_values)
    max_y = np.max(y_values)
    
    delta_y = tolerance
    y_cut = min_y
    pre_conti_len = 20
    pre_left_bound_point = None
    pre_right_bound_point = None
    
    while y_cut < max_y:
        mask = np.abs(y_values - y_cut) < tolerance
        points_near_cut = mesh_points[mask]
        if len(points_near_cut) < 20:
            y_cut += delta_y
            continue
        
        points_near_cut_sorted = points_near_cut[np.argsort(points_near_cut[:, 0])]
        
        idx_closest_to_zero = np.argmin(np.abs(points_near_cut_sorted[:, 0]))
        
        distances = cdist(mesh_points, mesh_points)  # 计算所有点之间的距离矩阵
        np.fill_diagonal(distances, np.inf)  # 填充对角线为无穷大，避免自己与自己比较
        min_distances = np.min(distances, axis=1)  # 每个点的最近邻距离
        conti = np.max(min_distances)  # 最大的最小邻距作为连续判断阈值
        
        # 向左侧扩展连续点
        left_length = 0
        left_bound_x = 0
        for i in range(idx_closest_to_zero - 1, -1, -1):
            if np.abs(points_near_cut_sorted[i + 1, 0] - points_near_cut_sorted[i, 0]) <= conti: # 认为是连续
                left_bound_x = points_near_cut_sorted[i, 0]
            else:
                break
        left_length = -left_bound_x
        
        # 向右侧扩展连续点
        right_length = 0
        right_bound_x = 0
        for i in range(idx_closest_to_zero + 1, len(points_near_cut_sorted)):
            if np.abs(points_near_cut_sorted[i, 0] - points_near_cut_sorted[i - 1, 0]) <= conti:
                right_bound_x = points_near_cut_sorted[i, 0]
            else:
                break
            
        right_length = right_bound_x
        total_conti_len = left_length + right_length
        
        cur_left_arpit_point = np.array([- total_conti_len / 2, y_cut])
        distances = np.linalg.norm(mesh_points[:, :2] - cur_left_arpit_point[:2], axis=1)
        closest_idx = np.argmin(distances)
        cur_left_arpit_point = mesh_points[closest_idx]
        
        cur_right_arpit_point = np.array([total_conti_len / 2, y_cut])
        distances = np.linalg.norm(mesh_points[:, :2] - cur_right_arpit_point[:2], axis=1)
        closest_idx = np.argmin(distances)
        cur_right_arpit_point = mesh_points[closest_idx]
        
        # 突增了 认为找到了
        if total_conti_len - pre_conti_len > 0.2:
            return pre_left_bound_point, pre_right_bound_point
        
        # 未找到 继续
        pre_conti_len = total_conti_len
        pre_left_bound_point = cur_left_arpit_point
        pre_right_bound_point = cur_right_arpit_point
        
        y_cut += delta_y  
        
    return pre_left_bound_point, pre_right_bound_point
        
    
    

def get_rate_skeleton_points(part_points, boundary, type, ratio, sample_num):

    if type == 'upper':
        y_values = part_points[part_points[:, 1] >= boundary, 1]
    else:
        y_values = part_points[part_points[:, 1] <= boundary, 1]
    
    if len(y_values) == 0:
        from termcolor import cprint
        cprint("Error: No points found for type", 'green')
        y_values = part_points[:, 1]
        
    min_y = np.min(y_values)
    max_y = np.max(y_values)
    
    # print(type,": ",  max_y-min_y)
    wrist_y = min_y + ratio * (max_y - min_y)
    # print(type, ratio, "wrist_y", wrist_y)
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



def get_half_level_skeleton_points(part_points):
    
    lower_ratios = [0.05, 0.3, 0.6, 0.8, 1]
    upper_ratios = [0.3, 0.6, 0.95]
    ans = []
    boundary = get_armpits_points(part_points)
    for ratio in lower_ratios:
        lower_skeleton_points = get_rate_skeleton_points(part_points, boundary, 'lower', ratio, 4)
        ans.extend(lower_skeleton_points)
    for ratio in upper_ratios:
        upper_skeleton_points = get_rate_skeleton_points(part_points, boundary, 'upper', ratio, 3)
        ans.extend(upper_skeleton_points)
        
    return ans
    

def get_with_sleeve_mesh_skeleton_id(mesh_points):
    
    visible_indices = get_visible_indices(mesh_points)
    invisible_indices = np.setdiff1d(np.arange(len(mesh_points)), visible_indices)
    upper_part = mesh_points[visible_indices]
    lower_part = mesh_points[invisible_indices]    
    
    upper_skeleton_points = get_half_level_skeleton_points(upper_part)
    lower_skeleton_points = get_half_level_skeleton_points(lower_part)

    skeleton_points = upper_skeleton_points + lower_skeleton_points

    skeleton_indices = []
    for point in skeleton_points:
        distances = np.linalg.norm(mesh_points - point, axis=1)
        nearest_index = np.argmin(distances)
        skeleton_indices.append(nearest_index)
    
    return skeleton_indices
    


def visualize_armpits(mesh_points):
    # 获取可见点的索引（这里上半部分的点）
    visible_indices = get_visible_indices(mesh_points)
    upper_part = mesh_points[visible_indices]
    
    # 获取左右腋窝点
    left_armpit_points, right_armpit_points = get_armpits_points(upper_part)
    
    # 创建PointCloud对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(upper_part)  # 只可视化上半部分的点
    
    # 设置颜色：绿色为默认颜色
    colors = np.tile([0, 1, 0], (len(upper_part), 1))  # 所有点初始为绿色 [0, 1, 0]
    
    # 将腋窝点设置为红色
    for point in left_armpit_points:
        # 找到与腋窝点接近的点并设置为红色
        distances = np.linalg.norm(upper_part - point, axis=1)
        closest_idx = np.argmin(distances)
        colors[closest_idx] = [1, 0, 0]  # 红色 [1, 0, 0]
    
    for point in right_armpit_points:
        distances = np.linalg.norm(upper_part - point, axis=1)
        closest_idx = np.argmin(distances)
        colors[closest_idx] = [1, 0, 0]  # 红色 [1, 0, 0]
    
    # 设置点云的颜色
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 可视化
    o3d.visualization.draw_geometries([pcd], window_name="Armpit Visualization", width=800, height=600)




def extract_second_last(path):
    return path.split('/')[-2].split('_')[0]

def sort_paths_by_second_last(paths):
    # 根据倒数第二部分（数字部分）排序
    return sorted(paths, key=lambda path: int(extract_second_last(path))) 

if __name__ == "__main__":
    
    directory = f"data/ls_tops/cd_processed/mesh_pcd"
    path = "data/trousers/unigarment/cd_processed/mesh_pcd/209_PL_pants3_obj/p_0.npz"
    # points = np.load(path)['mesh_points']
    # visualize_skeleton(points)
    
    paths = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith("p_0.npz"):
                file_path = os.path.join(dirpath, filename)
                paths.append(file_path)
                
    paths = sort_paths_by_second_last(paths)
    
    for idx, path in enumerate(paths):

        data = np.load(path)
        mesh_points = data['mesh_points']
        pcd_points = data['pcd_points']
        print(path)
        visualize_armpits(mesh_points)
        # visualize_pcd_and_mesh(mesh_points, pcd_points)
        # visualize_skeleton(mesh_points)