import os
import sys
sys.path.append(os.getcwd())
sys.path.append("/home/user/DexGarmentLab-master/DexGarmentLab-master/unigarment/unigarmentmlp")
sys.path.append("/home/user/DexGarmentLab-master/DexGarmentLab-master/unigarment/unigarmentmlp/merger")

import numpy as np
import open3d as o3d
from unigarment.collect.pcd_utils import normalize_pcd_points, nearest_mesh2pcd, get_visible_indices
from unigarment.collect.collect_cd.process.process_data import get_bulgy_cloth_keypoints_id


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
    
def visualize(mesh_points):
    mesh_pcd = o3d.geometry.PointCloud()
    mesh_pcd.points = o3d.utility.Vector3dVector(mesh_points)
    o3d.visualization.draw_geometries([mesh_pcd])
    
def visualize_contrast(points1, points2, idx):
    
    points1 = points1 + np.asarray([-2, 0, 0])
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd1.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (points1.shape[0], 1)))  # 黄色

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd2.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (points2.shape[0], 1)))  # 绿色

    # 创建红色点对象
    red_point1 = o3d.geometry.PointCloud()
    red_point1.points = o3d.utility.Vector3dVector(points1[idx:idx+1])
    red_point1.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 红色
    

    red_point2 = o3d.geometry.PointCloud()
    red_point2.points = o3d.utility.Vector3dVector(points2[idx:idx+1])
    red_point2.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 红色

    # 可视化
    print("Visualizing 1: ")
    o3d.visualization.draw_geometries([pcd1, red_point1],
                                      window_name="Contrast Visualization",
                                      point_show_normal=False)
    print("Visualizing 2: ")
    o3d.visualization.draw_geometries([pcd2, red_point2],
                                      window_name="Contrast Visualization",
                                      point_show_normal=False)


def visualize_find_nearest(points1, points2):
  
    idxlist1 = np.random.randint(0, points1.shape[0], 5)
    idxlist2 = nearest_mesh2pcd(points1, points2, idxlist1)
    
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd1.colors = o3d.utility.Vector3dVector(np.tile([1, 1, 0], (points1.shape[0], 1)))  # 黄色

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd2.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (points2.shape[0], 1)))  # 绿色

    # 创建红色点对象
    red_point1 = o3d.geometry.PointCloud()
    red_point1.points = o3d.utility.Vector3dVector(points1[idxlist1])
    red_point1.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 红色

    red_point2 = o3d.geometry.PointCloud()
    red_point2.points = o3d.utility.Vector3dVector(points2[idxlist2])
    red_point2.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 红色

    # 可视化
    o3d.visualization.draw_geometries([pcd1,red_point1],
                                      window_name="Contrast Visualization",
                                      point_show_normal=False)
    
    o3d.visualization.draw_geometries([pcd2, red_point2],
                                      window_name="Contrast Visualization",
                                      point_show_normal=False)


def visualize_visible_points(path):
    
    mesh_points = np.load(path)['mesh_points']
    
    mesh_pcd = o3d.geometry.PointCloud()
    mesh_pcd.points = o3d.utility.Vector3dVector(mesh_points)
    # mesh, visible_indices = mesh_pcd.hidden_point_removal(np.asarray([0, 0, 8]), 5000)
    # visible_indices = np.load(path)['visible_mesh_indices']
    visible_indices = get_visible_indices(mesh_points)

    visible_points = mesh_pcd.select_by_index(visible_indices)
    invisible_points = mesh_pcd.select_by_index(visible_indices, invert=True)

    visible_points.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (len(visible_indices), 1)))  # 红色
    invisible_points.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 0], (len(mesh_points) - len(visible_indices), 1)))  # 黑色
    
    combined_pcd = visible_points + invisible_points

    o3d.visualization.draw_geometries([combined_pcd])


def visualize_keypoints(path):

    data = np.load(path)
    pcd_points = data['pcd_points']  # [N, 3]
    pcd_keypoints_id = data['pcd_keypoints_id']  # [M,] 
    keypoints_visible_mask = data['keypoints_visible_mask']  # [M,] 

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)

    colors = np.ones((pcd_points.shape[0], 3)) * [1, 1, 0]  # Default color is yellow

    for i, keypoint_id in enumerate(pcd_keypoints_id):
        if keypoints_visible_mask[i] == 1:  # Visible keypoint
            colors[keypoint_id] = [1, 0, 0]  # Red color
        else:  # Invisible keypoint
            colors[keypoint_id] = [0, 0, 0]  # Black color

    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])


def visualize_keypoints_mesh(path):

    data = np.load(path)
    mesh_points = data['mesh_points']  # [N, 3]
    mesh_keypoints_id = data['mesh_keypoints_id']  # [M,] 
    keypoints_visible_mask = data['keypoints_visible_mask']  # [M,] 

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh_points)

    colors = np.ones((mesh_points.shape[0], 3)) * [1, 1, 0]  # Default color is yellow

    for i, keypoint_id in enumerate(mesh_keypoints_id):
        if keypoints_visible_mask[i] == 1:  # Visible keypoint
            colors[keypoint_id] = [1, 0, 0]  # Red color
        else:  # Invisible keypoint
            colors[keypoint_id] = [0, 0, 0]  # Black color

    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])


def visualize_keypoints_bulgy_mesh(path):

    points = np.load(path)['normalized_points']
    points, *_ = normalize_pcd_points(points)
    keypoints_id = get_bulgy_cloth_keypoints_id(points)
    visible_indices = get_visible_indices(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = np.ones((points.shape[0], 3)) * [1, 1, 0]  # Yellow color for all points

    for i, keypoint_id in enumerate(keypoints_id):
        if keypoint_id in visible_indices:  # 可见关键点
            colors[keypoint_id] = [1, 0, 0]  # 红色
        else:  # 不可见关键点
            colors[keypoint_id] = [0, 0, 0]  # 黑色

    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])
    

if __name__ == '__main__':
    
    visualize_visible_points("data/scarf/cd_original/mesh_pcd/0_ST_Scarf-009_obj/p_0.npz")
    
    # path1 = "data/glove/skeleton/normalized/0_GL_Gloves079_obj/pc_0.npz"
    # points1 =np.load(path1)['mesh_points']
    # path2 = "Assets/Garment/Glove/GL_Gloves079/GL_Gloves079.obj"
    # points2 = o3d.io.read_triangle_mesh(path2).vertices
    # points2 = np.asarray(points2)
    # num = min(points1.shape[0], points2.shape[0])
    
    # indices = np.random.randint(0, num, 20)
    # for idx in indices:
    #     visualize_contrast(points1, points2, idx)
    # for i in range(8):
    #     path = f"data/Hat/unigarment/cd_original/mesh_pcd/1_HA_Hat045_obj/p_{i}.npz"
    #     data = np.load(path)
    #     mesh_points = data['mesh_points']
    #     pcd_points = data['pcd_points']
    #     visualize_pcd_and_mesh(mesh_points, pcd_points)



# # 加载 npz 文件
# demo_file = 'data/with_sleeves/cd_processed/mesh_pcd/71_TCLC_066_obj/p_0.npz' 
# npz_file = 'data/with_sleeves/cd_processed/mesh_pcd/182_TNLC_Top470_obj/p_0.npz' 

# bulgy_mesh_path = demo_file.replace('mesh_pcd', 'bulgy_mesh')
# # bulgy_mesh_path = os.path.join("data/with_sleeves/cd_processed/bulgy_mesh", npz_file.split("/")[-2], 'p_0.npz')

# bulgy_mesh = np.load(bulgy_mesh_path)['normalized_points']
# bulgy_mesh, *_ = normalize_pcd_points(bulgy_mesh)

# print("鼓起衣服状态验证：")
# # visualize(bulgy_mesh)
# visualize_keypoints_bulgy_mesh(bulgy_mesh_path)
# visualize_visible_points(demo_file)
# visualize_keypoints_mesh(demo_file)
# visualize_keypoints(demo_file)

# print("可见点验证：")
# visualize_visible_points(npz_file)

# # 提取 mesh_points 和 pcd_points
# data = np.load(npz_file)
# mesh_points = data['mesh_points']
# pcd_points = data['pcd_points']
# visible_indices = data['visible_mesh_indices']

# print("找寻最近点验证：")
# visualize_find_nearest(mesh_points, pcd_points)

# contrast_idx = 3323
# print("鼓起衣服&变形状态相同索引验证：")
# visualize_contrast(bulgy_mesh, mesh_points, contrast_idx)

# print("同一状态mesh&pcd验证：")
# visualize_pcd_and_mesh(mesh_points, pcd_points)





