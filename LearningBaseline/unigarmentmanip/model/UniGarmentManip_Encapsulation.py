import os
import sys

import numpy as np
import torch
import open3d as o3d
import random
from LearningBaseline.unigarmentmanip.model.pointnet2_UniGarmentManip import UniGarmentManip_Model
from Env.Utils.pointcloud import furthest_point_sampling, normalize_pcd_points

class UniGarmentManip_Encapsulation:

    def __init__(self, catogory:str="Tops_LongSleeve"):
        '''
        load model
        '''
        self.catogory = catogory
        # set resume path
        resume_path = f"./LearningBaseline/unigarmentmanip/checkpoints/{self.catogory}/checkpoint.pth"
        # set seed
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # define model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = UniGarmentManip_Model(normal_channel=False, feature_dim=512).cuda()
        self.model.load_state_dict(torch.load(resume_path))
        self.model = self.model.to(self.device)
        self.model.eval()


    def get_feature(self, input_pcd:np.ndarray, index_list:list=None):
        '''
        get feature of input point cloud
        '''
        normalized_pcd, *_ = normalize_pcd_points(input_pcd)
        normalize_pcd = np.expand_dims(normalized_pcd, axis=0)

        with torch.no_grad():

            pcd_features = self.model(
                torch.from_numpy(normalize_pcd).to(self.device).float(),
            ).squeeze(0)
            # print(pcd_features.shape)

        if index_list is not None:
            target_features_list = []
            for i in index_list:
                target_features_list.append(pcd_features[i])
            return torch.stack(target_features_list)
        else:
            return pcd_features

    def get_manipulation_points(self, input_pcd:np.ndarray, index_list:list=None):
        '''
        get manipulation points of input point cloud
        '''

        #get model output (feature)
        demo_pcd = o3d.io.read_point_cloud(f"./LearningBaseline/unigarmentmanip/checkpoints/{self.catogory}/demo_garment.ply").points
        demo_feature = self.get_feature(demo_pcd, index_list)
        manipulate_feature = self.get_feature(input_pcd)

        # normalize feature
        demo_feature_normalized = torch.nn.functional.normalize(demo_feature, p=2, dim=1)
        manipulate_feature_normalized = torch.nn.functional.normalize(manipulate_feature, p=2, dim=1)
        result = torch.matmul(demo_feature_normalized, manipulate_feature_normalized.T)

        # get max similarity score and indices
        max_values, max_indices = torch.max(result, dim=1)
        print("similarity score: ", max_values)
        print("relevant indices: ", max_indices)

        # get manipulation points
        manipulation_points = input_pcd[max_indices.detach().cpu().numpy()]
        print("manipulation points: \n", manipulation_points)
        return manipulation_points, max_indices.detach().cpu().numpy()
