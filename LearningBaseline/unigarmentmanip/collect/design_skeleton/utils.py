import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

path = "data/ls_tops/cd_processed/mesh_pcd/0_DLLS_Dress206_obj/p_0.npz"
points_3d = np.load(path)['mesh_points']

# 投影到XY平面（z坐标设为0）
points_2d = points_3d[:, :2]  # 只取 x 和 y 坐标

# 确保 2D 点云转换为 3D 点云，在 z 轴上设置为 0
points_2d_3d = np.hstack([points_2d, np.zeros((points_2d.shape[0], 1))])  # 将 z 坐标设置为0

# 计算 Delaunay 三角剖分
triangulation = Delaunay(points_2d)

# 计算每个三角形的外接圆半径
def circumradius(p1, p2, p3):
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)
    s = (a + b + c) / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    radius = (a * b * c) / (4 * area)
    return radius

# 设置 alpha 参数，删除外接圆半径大于 alpha 的三角形
alpha = 0.2
valid_edges = []
for simplex in triangulation.simplices:
    p1, p2, p3 = points_2d[simplex]
    radius = circumradius(p1, p2, p3)
    if radius < alpha:
        # 将三角形的边添加到边界列表
        valid_edges.append([simplex[0], simplex[1]])
        valid_edges.append([simplex[1], simplex[2]])
        valid_edges.append([simplex[2], simplex[0]])

# 将有效边界转换为 NumPy 数组
valid_edges = np.array(valid_edges)

# 用 Open3D 可视化
def visualize_boundaries(points, edges):
    # 创建一个 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 创建一个用于表示边界的线集合
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(edges)

    # 可视化边界线和点云
    o3d.visualization.draw_geometries([pcd, line_set], window_name="Point Cloud Boundaries")

# 使用 Open3D 可视化边界
visualize_boundaries(points_2d_3d, valid_edges)