import os
import sys
sys.path.append("unigarment/collect")
import numpy as np
import open3d as o3d
import argparse

from pcd_utils import normalize_pcd_points

def process_all_pcd_files(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".npz"):
                # 原始文件路径
                input_path = os.path.join(root, file)
                
                # 创建对应的输出路径     
                relative_path = os.path.relpath(root, input_dir)
                output_folder = os.path.join(output_dir, relative_path)
                os.makedirs(output_folder, exist_ok=True)

                # 输出文件路径
                output_path = os.path.join(output_folder, file)
                
                # 标准化并保存
                data = np.load(input_path)
                mesh_points = data['mesh_points']
                pcd_points = data['pcd_points']
                mesh_points_normalized, centroid, scale = normalize_pcd_points(mesh_points)
                pcd_points = (pcd_points - centroid) * scale
                np.savez(output_path, 
                         mesh_points=mesh_points_normalized,
                         pcd_points=pcd_points)


if __name__ == "__main__":

    type = "glove"
    input_directory = f"data/{type}/skeleton/mesh_pcd"
    output_dir = f"data/{type}/skeleton/normalized"
    os.makedirs(output_dir, exist_ok=True)
    process_all_pcd_files(input_directory, output_dir)
