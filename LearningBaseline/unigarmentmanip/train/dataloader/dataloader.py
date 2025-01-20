import os
import sys


import numpy as np
import torch
from torch.utils.data import Dataset

from utils import get_deformation_paths, create_cross_deformation_pairs, create_cross_object_pairs, fps_with_selected, nearest_mesh2pcd
from base.config import Config

configs = Config()
configs = configs.data_config

class Dataset(Dataset):
    
    def __init__(self, mode):
        
        assert mode in ['train', 'val']
        
        self.all_deform_paths = get_deformation_paths(configs.data_dir, configs.garment_data_num, configs.train_ratio, mode)
        self.cross_deformation_pair_path = create_cross_deformation_pairs(self.all_deform_paths)
        self.cross_object_pair_path = create_cross_object_pairs(self.all_deform_paths)
        # 平衡co cd数量
        self.cross_deformation_pair_index = np.random.randint(0, len(self.cross_deformation_pair_path), len(self.cross_object_pair_path))
        
        print(f"Number of all deformation paths: {len(self.all_deform_paths)}")
        print(f"Number of cross deformation pairs: {len(self.cross_deformation_pair_path)}")
        print(f"Number of cross object pairs: {len(self.cross_object_pair_path)}")
        print(f"Number of cross deformation pairs for cross object pairs: {len(self.cross_deformation_pair_index)}")

    
    def get_cross_deformation_correspondence(self, index):
        npz1, npz2 = self.cross_deformation_pair_path[self.cross_deformation_pair_index[index//2]]
        npz1 = np.load(npz1)
        npz2 = np.load(npz2)
        
        keypoints_visible_mask_1 = npz1['keypoints_visible_mask']
        keypoints_visible_mask_2 = npz2['keypoints_visible_mask']

        keypoints_visible_mask = keypoints_visible_mask_1 & keypoints_visible_mask_2
        
        if np.sum(keypoints_visible_mask) > configs.keypoints_correspondence_num:

            ones_indices = np.where(keypoints_visible_mask == 1)[0]
            
            excess_count = np.sum(keypoints_visible_mask) - configs.keypoints_correspondence_num
            
            indices_to_turn_off = np.random.choice(ones_indices, size=excess_count, replace=False)
            
            keypoints_visible_mask[indices_to_turn_off] = False
        
        keypoints_id_visible_1 = npz1['pcd_keypoints_id'][keypoints_visible_mask == 1]
        keypoints_id_visible_2 = npz2['pcd_keypoints_id'][keypoints_visible_mask == 1]
            
        # 随机补充（fps）
        random_correspondence_num = configs.correspondence_num - np.sum(keypoints_visible_mask)
        mesh_points_1 = npz1['mesh_points']
        mesh_points_2 = npz2['mesh_points']
        pcd_points_1 = npz1['pcd_points']
        pcd_points_2 = npz2['pcd_points']
        visible_mesh_id_1 = npz1['visible_mesh_indices']
        visible_mesh_id_2 = npz2['visible_mesh_indices']
        visible_mesh_id = np.intersect1d(visible_mesh_id_1, visible_mesh_id_2)
        has_selected_points = npz1['pcd_points'][keypoints_id_visible_1]
        random_points_id, random_points = fps_with_selected(mesh_points_1, visible_mesh_id, has_selected_points, random_correspondence_num)
        pcd_random_points_id_1 = nearest_mesh2pcd(mesh_points_1, pcd_points_1, random_points_id)
        pcd_random_points_id_2 = nearest_mesh2pcd(mesh_points_2, pcd_points_2, random_points_id)

        keypoints_id_visible_1 = np.concatenate([keypoints_id_visible_1, pcd_random_points_id_1])
        keypoints_id_visible_2 = np.concatenate([keypoints_id_visible_2, pcd_random_points_id_2])

        correspondence = np.stack([keypoints_id_visible_1, keypoints_id_visible_2], axis=1)
            
        return correspondence
    
    def get_cross_object_correspondence(self, index):
        npz1, npz2 = self.cross_object_pair_path[index//2]
        npz1 = np.load(npz1)
        npz2 = np.load(npz2)
        
        keypoints_visible_mask_1 = npz1['keypoints_visible_mask']
        keypoints_visible_mask_2 = npz2['keypoints_visible_mask']

        keypoints_visible_mask = keypoints_visible_mask_1 & keypoints_visible_mask_2
        
        if np.sum(keypoints_visible_mask) == 0:
            return self.get_cross_deformation_correspondence((index + 2) % self.__len__())
        
        if np.sum(keypoints_visible_mask) > configs.correspondence_num:

            ones_indices = np.where(keypoints_visible_mask == 1)[0]
            
            excess_count = np.sum(keypoints_visible_mask) - configs.correspondence_num
            
            indices_to_turn_off = np.random.choice(ones_indices, size=excess_count, replace=False)
            
            keypoints_visible_mask[indices_to_turn_off] = False
        
        keypoints_id_visible_1 = npz1['pcd_keypoints_id'][keypoints_visible_mask == 1]
        keypoints_id_visible_2 = npz2['pcd_keypoints_id'][keypoints_visible_mask == 1]
        
        correspondence = np.stack([keypoints_id_visible_1, keypoints_id_visible_2], axis=1)
        if correspondence.shape[0] < configs.correspondence_num:

            deficit = configs.correspondence_num - correspondence.shape[0]
            
            indices_to_duplicate = np.random.choice(correspondence.shape[0], size=deficit, replace=True)
            
            additional_rows = correspondence[indices_to_duplicate]
            
            correspondence = np.concatenate([correspondence, additional_rows], axis=0)
            
        return correspondence   
    
    def get_cross_deformation_pair(self, index):
        
        npz1, npz2 = self.cross_deformation_pair_path[self.cross_deformation_pair_index[index//2]]
        npz1 = np.load(npz1)
        npz2 = np.load(npz2)
        pc1= npz1['pcd_points']
        pc2= npz2['pcd_points']
        correspondence = self.get_cross_deformation_correspondence(index)
        
        return pc1, pc2, correspondence
    
    def get_cross_object_pair(self, index):

        npz1, npz2 = self.cross_object_pair_path[index//2]
        npz1 = np.load(npz1)
        npz2 = np.load(npz2)
        pc1= npz1['pcd_points']
        pc2= npz2['pcd_points']
        correspondence = self.get_cross_object_correspondence(index)
        
        return pc1, pc2, correspondence
    
    def __len__(self):
        return 2 * len(self.cross_object_pair_path)
    
    def __getitem__(self, index):
        
        if index % 2 == 0:
            return self.get_cross_deformation_pair(index)
        else:
            return self.get_cross_object_pair(index)


if __name__ == '__main__':
    
    dataset = Dataset('train')
    print(dataset.__len__())
    pc1, pc2, correspondence= dataset[6441]
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
    