
import os
import sys
sys.path.append(os.getcwd())


import wandb
from tqdm import tqdm
import argparse
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import random
from info_nce import InfoNCE
from prefetch_generator import BackgroundGenerator

from LearningBaseline.unigarmentmanip.train.base.config import Config
from LearningBaseline.unigarmentmanip.train.model.pointnet2_Sofa_Model import Sofa_Model
from LearningBaseline.unigarmentmanip.train.val.simple_val import *
import open3d as o3d
from sklearn.decomposition import PCA


import numpy as np
from torch.utils.data import Dataset
import os
import open3d as o3d
import functools
import sys

import tqdm

import tqdm
import torch.nn.functional as F
from info_nce import InfoNCE
import tqdm
import matplotlib.pyplot as plt

config = Config()
config = config.train_config
device = 'cuda'

def convert_path_to_image_filename(file_path):

    folder_path, file_name = os.path.split(file_path)

    folder_name = os.path.basename(folder_path)

    base_name = os.path.splitext(file_name)[0]

    new_filename = f"{folder_name}_{base_name}.png"
    
    return new_filename

def standardize_bbox(pcl:torch.tensor):
        # pcl=pcl
        mins = np.amin(pcl, axis=0)
        maxs = np.amax(pcl, axis=0)
        center = (mins + maxs) / 2.
        scale = np.amax(maxs-mins)
        result = ((pcl - center)/scale).astype(np.float32)  # [-0.5, 0.5]
        return result

def colormap(pointcloud):
        base_point = np.copy(pointcloud[0])
        distance = np.zeros((pointcloud.shape[0],1))
        point1 = np.copy(pointcloud[0])
        point2 = np.copy(pointcloud[0])
        for i in range(pointcloud.shape[0]):#最左下的点
            if pointcloud[i][0]+pointcloud[i][1]<base_point[0]+base_point[1]:
                base_point=pointcloud[i]
        for i in range(pointcloud.shape[0]):#最左上的点(255,0,255)
            if pointcloud[i][0]-pointcloud[i][1]<point1[0]-point1[1]:
                point1 = pointcloud[i]
        for i in range(pointcloud.shape[0]):#最右上的点(170,0,255)
            if pointcloud[i][0]+pointcloud[i][1]>point2[0]+point2[1]:
                point2 = pointcloud[i]
        
        base_point[0]-=0.02
        for i in range(pointcloud.shape[0]):
            distance[i] = np.linalg.norm(pointcloud[i] - base_point)
        max_value = np.max(distance)
        min_value = np.min(distance)
        cmap = plt.cm.get_cmap('jet_r')
        colors = cmap((-distance+max_value)/(max_value-min_value))
        colors = np.reshape(colors,(-1,4))
        color_map = np.zeros((pointcloud.shape[0], 3))
        i=0
        for color in colors:
            color_map[i] = color[:3]
            i=i+1
        color_map2 = np.zeros_like(color_map)
        for i in range(pointcloud.shape[0]):
            distance1 = np.linalg.norm(point1-pointcloud[i])
            distance2 = np.linalg.norm(point2-pointcloud[i])
            dis = np.abs(point1[1]-pointcloud[i][1])
            if dis < 0.4:
                color_map2[i] = np.array([75.0/255.0,0.0,130.0/255.0])*distance2/(distance1+distance2) + np.array([1.0,20.0/255.0,147.0/255.0])*distance1/(distance1+distance2)


        for i in range(pointcloud.shape[0]):
            distance1 = np.linalg.norm(point1-pointcloud[i])
            distance2 = np.linalg.norm(point2-pointcloud[i])
            distance3 = np.linalg.norm(point1-point2)
            dis = np.abs(point1[1]-pointcloud[i][1])
            if dis<0.4:
                color_map[i] = color_map[i]*(dis)/(0.4) + (color_map2[i])*(0.4-dis)/(0.4)
            
        return color_map


def get_query_result(model,demo_flat,deform):
    
    '''
    flat (N, 3)
    deform (N, 3)
    '''
    if type(demo_flat) == np.ndarray:
        demo_flat = torch.from_numpy(demo_flat)
    if type(deform) == np.ndarray:
        deform = torch.from_numpy(deform)
    
    demo_feature=model(demo_flat.unsqueeze(0).float().to(device))[0]
    demo_feature=F.normalize(demo_feature,dim=-1)
    
    deform_feature=model(deform.unsqueeze(0).float().to(device))[0]
    deform_feature=F.normalize(deform_feature,dim=-1)   
    
    deform_query_result=torch.zeros((deform.shape[0])).to(device)


    for j in range(deform.shape[0]):
        deform_query_result[j]=torch.argmax(torch.sum(deform_feature[j]*demo_feature,dim=1))
        
    deform_query_result = deform_query_result.cpu().numpy().astype(int)  # 确保索引是整数类型
    
    return deform_query_result


from copy import deepcopy
def get_color(model, demo_flat,deform):
    
    deform_query_result=get_query_result(model,demo_flat,deform)
    
    if type(deform_query_result) == torch.Tensor:
        deform_query_result=deform_query_result.cpu().numpy()
    
    flat_color = colormap(demo_flat)
    deform_color=flat_color[deform_query_result]

    return deform_color

def visualize_pointcloud_with_colors(points, colors, save_path:str=None, zoom=1.3):

    if type(points) == torch.Tensor:
        points = points.cpu().numpy()
    if type(colors) == torch.Tensor:
        colors = colors.cpu().numpy()
    
    # 创建点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # 初始化可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Point Cloud with Zoom", width=800, height=600)

    # 添加点云到可视化器
    vis.add_geometry(point_cloud)

    # 获取 ViewControl 并设置摄像机参数
    view_control = vis.get_view_control()

    # 设置点云的中心和摄像机视图
    bounding_box = point_cloud.get_axis_aligned_bounding_box()
    center = bounding_box.get_center()

    # 设置视图参数
    view_control.set_lookat(center)  # 设置焦点为点云中心
    view_control.set_up([0, 1, 0])  # 设置上方向
    view_control.set_front([0, 0, -1])  # 设置前方向
    view_control.set_zoom(zoom)  # 设置缩放

    # 更新渲染器
    vis.poll_events()
    vis.update_renderer()
    
    if save_path is not None:
        vis.capture_screen_image(save_path)
        print(f"Saved image to {save_path}")

    # 显示窗口并销毁
    vis.run()
    vis.destroy_window()
  
def demo2deform(demo_npz_path, deform_npz_path):
    # 读取demo数据
    demo_data = np.load(demo_npz_path)
    demo_points = demo_data['points']
    demo_keyid = demo_data['keypoints_id']
    
    # 读取deform数据
    deform_data = np.load(deform_npz_path)
    deform_points = deform_data['points']
    deform_keyid = deform_data['keypoints_id']
    
    # 生成demo点云的颜色映射
    demo_color = colormap(demo_points)

    # 为deform_points指定颜色
    deform_color = np.ones((deform_points.shape[0], 3))  # 初始化为零，即黑色
    for i in range(len(deform_keyid)):
        deform_color[deform_keyid[i]] = demo_color[demo_keyid[i]]
    
    # 创建Open3D点云对象
    deform_pcd = o3d.geometry.PointCloud()
    deform_pcd.points = o3d.utility.Vector3dVector(deform_points)
    deform_pcd.colors = o3d.utility.Vector3dVector(deform_color)

    # 可视化deform点云
    o3d.visualization.draw_geometries([deform_pcd])
   
def keypoints_dot(model, npz1, npz2):
    npz1 = np.load(npz1)
    npz2 = np.load(npz2)
    
    keypoints_visible_mask_1 = npz1['keypoints_visible_mask']
    keypoints_visible_mask_2 = npz2['keypoints_visible_mask']
    # print(f"visible1: {len(keypoints_visible_mask_1[keypoints_visible_mask_1 == 1])}")
    # print(f"visible2: {len(keypoints_visible_mask_2[keypoints_visible_mask_2 == 1])}")

    keypoints_visible_mask = keypoints_visible_mask_1 & keypoints_visible_mask_2
    
    keypoints_id_visible_1 = npz1['pcd_keypoints_id'][keypoints_visible_mask == 1]
    keypoints_id_visible_2 = npz2['pcd_keypoints_id'][keypoints_visible_mask == 1]
    
    points1 = npz1['pcd_points']
    points2 = npz2['pcd_points']
    points1 = np.expand_dims(points1, axis=0)
    points2 = np.expand_dims(points2, axis=0)
    points1 = torch.from_numpy(points1).float().to(device)
    points2 = torch.from_numpy(points2).float().to(device)
    
    with torch.no_grad():
        features1 = model(points1)[0]
        features2 = model(points2)[0]
    
    keypoint_features1 = features1[keypoints_id_visible_1]  # [N, 256]
    keypoint_features2 = features2[keypoints_id_visible_2]   # [N, 256]
    
    keypoint_features1 = F.normalize(keypoint_features1, dim=-1)
    keypoint_features2 = F.normalize(keypoint_features2, dim=-1)
    
    keypoint_features1 = keypoint_features1.cpu().numpy()
    keypoint_features2 = keypoint_features2.cpu().numpy()

    sim = 0
    for i in range(keypoint_features1.shape[0]):
        cos_sim = np.dot(keypoint_features1[i] , keypoint_features2[i])
        sim += cos_sim
    sim /= keypoint_features1.shape[0]
    print('similarity: ', sim)
     
def rotate_point_cloud(points, angle_deg, axis='z'):
    # 将角度转换为弧度
    angle_rad = np.radians(angle_deg)

    # 旋转矩阵构造
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(angle_rad), -np.sin(angle_rad)],
                                    [0, np.sin(angle_rad), np.cos(angle_rad)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                                    [0, 1, 0],
                                    [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                    [np.sin(angle_rad), np.cos(angle_rad), 0],
                                    [0, 0, 1]])

    # 如果 points 是一个 PyTorch 张量，转换为 NumPy 数组
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    # 旋转点云
    rotated_points = np.dot(points, rotation_matrix.T)  # 矩阵乘法

    return rotated_points  
     
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

def get_features(points):
    
    resume_path = "Checkpoint/f2d/checkpoint_7_400.pth"
    model=Sofa_Model(feature_dim=config.feature_dim)
    model.load_state_dict(torch.load(resume_path,map_location=device)['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    points, *_ = normalize_pcd_points(points)
    points = np.expand_dims(points, axis=0)
    
    with torch.no_grad():
        features = model(torch.from_numpy(points).float().to(device))
        print(features.shape)

  
if __name__ == '__main__':
    
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)
    
    points = np.random.rand(2048, 3)
    
    trousers_paths = [
        "data/trousers/unigarment/cd_processed/mesh_pcd/5_PL_LongPants014_obj",
        "data/trousers/unigarment/cd_processed/mesh_pcd/0_PL_Pants052_obj",      
        "data/trousers/unigarment/cd_processed/mesh_pcd/32_PL_M2_048_obj",
        "data/trousers/unigarment/cd_processed/mesh_pcd/57_PL_Pants054_obj",
        "data/trousers/unigarment/cd_processed/mesh_pcd/115_PL_Pants046_obj",
        "data/trousers/unigarment/cd_processed/mesh_pcd/143_PL_086_obj",
        "data/trousers/unigarment/cd_processed/mesh_pcd/174_PL_short010_obj",
        "data/trousers/unigarment/cd_processed/mesh_pcd/205_PL_pants1_obj", # 7
        "data/trousers/unigarment/cd_processed/mesh_pcd/217_PL_022_obj",
        "data/trousers/unigarment/cd_processed/mesh_pcd/248_PS_M1_059_obj",
        "data/trousers/unigarment/cd_processed/mesh_pcd/249_PS_Short062_obj/",
        "data/trousers/unigarment/cd_processed/mesh_pcd/250_PS_033_obj/",  # 11
        "data/trousers/unigarment/cd_processed/mesh_pcd/290_PS_M1_003_obj",
        "data/trousers/unigarment/cd_processed/mesh_pcd/296_PS_Short013_obj",
        "data/trousers/unigarment/cd_processed/mesh_pcd/303_PS_M2_031_obj",
              
    ]
    
    
    resume_path = "Checkpoint/f2d/checkpoint_1_160000.pth"
    model=Sofa_Model(feature_dim=config.feature_dim)
    model.load_state_dict(torch.load(resume_path,map_location=device)['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    npz_paths = trousers_paths
    for (i, path) in enumerate(npz_paths):
        npz_paths[i] = path.replace('cd_processed', 'cd_rotation')
        
    demo_path = os.path.join(npz_paths[0], 'p_0.npz')
    demo_flat = np.load(demo_path)['pcd_points']

    save_dir = "unigarment/affordance_visualization/front_open"
    os.makedirs(save_dir, exist_ok=True)
    demo_flat_colors = colormap(demo_flat)
    visualize_pointcloud_with_colors(demo_flat, demo_flat_colors, None)
    
    save_or_not = False
    
    for i in range (7, len(npz_paths)):
        npz_path = npz_paths[i]
        print(f"testing {npz_path} …………")
        idxs = np.random.randint(1, 10, 5)
        idxs = np.concatenate((np.array([0]), idxs))
        print(f"idxs: {idxs}")
        for idx in idxs:
            path = os.path.join(npz_path, f"p_{15}_{idx}.npz")
            data = np.load(path)
            points = data['pcd_points']  
            
            # points = rotate_point_cloud(points, 45, axis='z')
            
            points = torch.tensor(points).to(config.device)
            deform_color = get_color(model, demo_flat, points) 
            keypoints_dot(model, demo_path, path)   
               
            save_path = convert_path_to_image_filename(path)
            save_path =  os.path.join(save_dir, save_path) if save_or_not else None 
            
            visualize_pointcloud_with_colors(points, deform_color, save_path)
    


