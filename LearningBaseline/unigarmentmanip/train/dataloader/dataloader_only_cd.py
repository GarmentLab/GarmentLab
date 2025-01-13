import os
import sys
sys.path.append("unigarment/train")
sys.path.append("/home/user/DexGarmentLab-master/DexGarmentLab-master/unigarment")
sys.path.append("unigarment/train/dataloader")

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import get_deformation_paths, create_cross_deformation_pairs, create_cross_object_pairs, fps_with_selected, nearest_mesh2pcd, create_cross_only_deformation_pairs
from base.config import Config

configs = Config()
configs = configs.data_config

class Dataset(Dataset):
    
    def __init__(self, mode):
        
        assert mode in ['train', 'val']
        
        self.all_deform_paths = get_deformation_paths(configs.only_deformation_data_dir, configs.garment_data_num, configs.train_ratio, mode)
        self.cross_deformation_pair_path = create_cross_only_deformation_pairs(self.all_deform_paths)
        
        print(self.all_deform_paths[0][0])
        
        print(f"Number of all deformation paths: {len(self.all_deform_paths)}")
        print(f"Number of cross deformation pairs: {len(self.cross_deformation_pair_path)}")
        
    def __len__(self):
        return len(self.cross_deformation_pair_path)
    
    def __getitem__(self, index):
        
        return self.get_cross_deformation_pair(index)
    
    def get_cross_deformation_correspondence(self, index):
        npz1, npz2 = self.cross_deformation_pair_path[index]
        npz1 = np.load(npz1)
        npz2 = np.load(npz2)
            
        # 随机补充（fps）
        random_correspondence_num = configs.correspondence_num
        mesh_points_1 = npz1['mesh_points']
        mesh_points_2 = npz2['mesh_points']
        pcd_points_1 = npz1['pcd_points']
        pcd_points_2 = npz2['pcd_points']
        visible_mesh_id_1 = npz1['visible_mesh_indices']
        visible_mesh_id_2 = npz2['visible_mesh_indices']
        visible_mesh_id = np.intersect1d(visible_mesh_id_1, visible_mesh_id_2)
        random_points_id, random_points = fps_with_selected(mesh_points_1, visible_mesh_id, [], random_correspondence_num)
        pcd_random_points_id_1 = nearest_mesh2pcd(mesh_points_1, pcd_points_1, random_points_id)
        pcd_random_points_id_2 = nearest_mesh2pcd(mesh_points_2, pcd_points_2, random_points_id)

        keypoints_id_visible_1 = np.array(pcd_random_points_id_1)
        keypoints_id_visible_2 = np.array(pcd_random_points_id_2)

        correspondence = np.stack([keypoints_id_visible_1, keypoints_id_visible_2], axis=1)
            
        return correspondence
    
    
    def get_cross_deformation_pair(self, index):
        
        npz1, npz2 = self.cross_deformation_pair_path[index]
        npz1 = np.load(npz1)
        npz2 = np.load(npz2)
        pc1= npz1['pcd_points']
        pc2= npz2['pcd_points']
        correspondence = self.get_cross_deformation_correspondence(index)
        
        return pc1, pc2, correspondence



if __name__ == '__main__':
    
    dataset = Dataset('train')
    print(dataset.__len__())
    pc1, pc2, correspondence= dataset[1]
    print(pc1.shape)
    print(pc2.shape)
    print(correspondence.shape)
    # print(correspondence)
    
    from utils import visualize_point_cloud
    visualize_point_cloud(pc1, correspondence[:, 0], title="Point Cloud 1")
    visualize_point_cloud(pc2, correspondence[:, 1], title="Point Cloud 2")
    
    
    for i in range(configs.correspondence_num):
        visualize_point_cloud(pc1, correspondence[i:i+1, 0], title="Point Cloud 1")
        visualize_point_cloud(pc2, correspondence[i:i+1, 1], title="Point Cloud 2")
    