import open3d as o3d
import numpy as np
import os
import random
import torch

def load_and_color_point_cloud(pcd_path, color):
    """
    加载点云文件并设置颜色
    Args:
        pcd_path (str): 点云文件路径
        color (tuple): RGB颜色值，范围 [0, 1]
    Returns:
        open3d.geometry.PointCloud: 带颜色的点云
    """
    point_cloud = o3d.io.read_point_cloud(pcd_path)
    point_cloud.paint_uniform_color(color)  # 设置点云颜色
    return point_cloud

def visualize_paired_pcds(original_dir, normalize_dir, num_samples=5):
    """
    随机选择标准化后的和原始的点云文件对，每对单独可视化
    Args:
        original_dir (str): 原始点云文件夹路径
        normalize_dir (str): 标准化后的点云文件夹路径
        num_samples (int): 随机选择的文件数量
    """
    # 获取 original_pcd 目录下所有的文件路径
    paired_paths = []
    for root, _, files in os.walk(original_dir):
        for file in files:
            if file.endswith(".pcd"):
                # 构建原始 pcd 文件的完整路径
                original_pcd_path = os.path.join(root, file)
                
                # 获取相对于 original_dir 的相对路径
                relative_path = os.path.relpath(original_pcd_path, original_dir)
                
                # 构建标准化 pcd 文件的完整路径
                normalize_pcd_path = os.path.join(normalize_dir, relative_path)
                
                # 检查标准化文件是否存在
                if os.path.exists(normalize_pcd_path):
                    paired_paths.append((original_pcd_path, normalize_pcd_path))

    # 如果配对的文件数量不足，输出警告信息
    if len(paired_paths) < num_samples:
        print(f"警告：可用的配对文件不足 {num_samples} 对，只有 {len(paired_paths)} 对。")
        num_samples = len(paired_paths)

    # 随机选择指定数量的配对文件
    samples = random.sample(paired_paths, num_samples)

    # 可视化每对配对文件
    for original_pcd_path, normalize_pcd_path in samples:
        # 加载原始点云并设置为红色
        original_pcd = load_and_color_point_cloud(original_pcd_path, color=(1, 0, 0))
        
        # 加载标准化点云并设置为绿色
        normalize_pcd = load_and_color_point_cloud(normalize_pcd_path, color=(0, 1, 0))

        # 将原始和标准化点云放在同一个窗口进行可视化
        o3d.visualization.draw_geometries([original_pcd, normalize_pcd], window_name="Original (Red) and Normalized (Green)")



if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 指定原始和标准化的文件夹路径
    original_directory = "dress_data/original_pcd"
    normalize_directory = "dress_data/normalize_pcd"

    # 可视化随机挑选的5对点云
    visualize_paired_pcds(original_directory, normalize_directory, num_samples=5)

