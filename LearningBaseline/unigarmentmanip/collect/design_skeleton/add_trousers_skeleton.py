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

def get_boundary_visual(mesh_points, tolerance=0.02, conti=0.1):
    # 提取x坐标和y坐标
    x_vals = mesh_points[:, 0]
    y_vals = mesh_points[:, 1]
    
    x_cut = 0  
    mask = np.abs(x_vals - x_cut) < tolerance  # 找到x坐标与x_cut近似的点
    y_vals_near_cut = y_vals[mask]  # 取这些点对应的y值
    while len(y_vals_near_cut) == 0:
        tolerance += 0.001
        mask = np.abs(x_vals - (-x_cut)) < tolerance 
        y_vals_near_cut = y_vals[mask]
    
    # 将 y_vals 从大到小排序
    y_vals_sorted = np.sort(y_vals_near_cut)[::-1]  # 从大到小排序
    
    # 扫描y_vals，找到第一个不连续的位置
    for i in range(1, len(y_vals_sorted)):
        if np.abs(y_vals_sorted[i-1] - y_vals_sorted[i]) > conti:
            return y_vals_sorted[i-1], y_vals_sorted[i]  # 返回发现不连续时的最小y值
    
    # 如果没有发现不连续，返回最小的y值
    return y_vals_sorted[-1], 2

def get_boundary_y(mesh_points, tolerance=0.05, conti=0.1):
    # 提取x坐标和y坐标
    x_vals = mesh_points[:, 0]
    y_vals = mesh_points[:, 1]
    
    x_cut = 0  
    mask = np.abs(x_vals - x_cut) < tolerance  # 找到x坐标与x_cut近似的点
    y_vals_near_cut = y_vals[mask]  # 取这些点对应的y值
    while len(y_vals_near_cut) == 0:
        tolerance += 0.001
        mask = np.abs(x_vals - (-x_cut)) < tolerance 
        y_vals_near_cut = y_vals[mask]
    
    # 将 y_vals 从大到小排序
    y_vals_sorted = np.sort(y_vals_near_cut)[::-1]  # 从大到小排序
    
    # 扫描y_vals，找到第一个不连续的位置
    for i in range(1, len(y_vals_sorted)):
        if np.abs(y_vals_sorted[i-1] - y_vals_sorted[i]) > conti:
            return y_vals_sorted[i-1]  # 返回发现不连续时的最小y值
    
    # 如果没有发现不连续，返回最小的y值
    return y_vals_sorted[-1]
    

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
    boundary = get_boundary_y(part_points)
    for ratio in lower_ratios:
        lower_skeleton_points = get_rate_skeleton_points(part_points, boundary, 'lower', ratio, 4)
        ans.extend(lower_skeleton_points)
    for ratio in upper_ratios:
        upper_skeleton_points = get_rate_skeleton_points(part_points, boundary, 'upper', ratio, 3)
        ans.extend(upper_skeleton_points)
        
    return ans
    

def get_trousers_mesh_skeleton_id(mesh_points):
    
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
    

def visualize_skeleton(mesh_points):
    skeleton_indices = get_trousers_mesh_skeleton_id(mesh_points)
    
    # key_point 使用你原始的定义方式，假设 get_boundary_y 是返回 Y 轴坐标的函数
    key_y, nxt_y = get_boundary_visual(mesh_points)
    key_point = np.array([0, key_y, 0])
    nxt_point = np.array([0, nxt_y, 0])
    
    # 创建一个点云对象并将 mesh_points 传递进去
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh_points)
    
    # 为每个点设置颜色，默认绿色
    colors = np.tile([0, 1, 0], (len(mesh_points), 1))  # 绿色 [0, 1, 0]
    
    # 标记骨架点为红色
    colors[skeleton_indices] = [1, 0, 0]  # 红色 [1, 0, 0]
    
    # 设置点云的颜色
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建 key_point 的单独点云，用黄色且较大的点来表示
    key_point_pcd = o3d.geometry.PointCloud()
    key_point_pcd.points = o3d.utility.Vector3dVector([key_point])
    
    # 黄色为 [1, 1, 0]
    key_point_pcd.colors = o3d.utility.Vector3dVector([[1, 1, 0]])  # 黄色
    
    # 创建球体表示的 key_point，增加其可视化效果
    key_point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)  # 设置较大的半径
    key_point_sphere.translate(key_point)  # 移动球体到指定的位置
    key_point_sphere.paint_uniform_color([1, 0, 0])  # 设置颜色为黄色

    # 创建 nxt_point 的球体表示，蓝色球体
    nxt_point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)  # 设置较大的半径
    nxt_point_sphere.translate(nxt_point)  # 移动球体到指定的位置
    nxt_point_sphere.paint_uniform_color([0, 0, 1])  # 设置颜色为蓝色

    # 展示点云、key_point 和 nxt_point 的球体
    o3d.visualization.draw_geometries([pcd, key_point_sphere, nxt_point_sphere], window_name="Skeleton Visualization", width=800, height=600)



def extract_second_last(path):
    return path.split('/')[-2].split('_')[0]

def sort_paths_by_second_last(paths):
    # 根据倒数第二部分（数字部分）排序
    return sorted(paths, key=lambda path: int(extract_second_last(path))) 

if __name__ == "__main__":
    
    directory = f"data/trousers/unigarment/cd_processed/mesh_pcd"
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

        if idx <=251:
            continue
        
        mesh_points = np.load(path)['mesh_points']
        print(path)
        visualize_skeleton(mesh_points)