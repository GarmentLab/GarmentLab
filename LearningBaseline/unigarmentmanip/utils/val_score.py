
import os
import sys

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
from LearningBaseline.unigarmentmanip.train.base.utils import *
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

import torch.nn.functional as F
from info_nce import InfoNCE
import matplotlib.pyplot as plt
from LearningBaseline.unigarmentmanip.train.val.simple_val import *
from LearningBaseline.unigarmentmanip.train.dataloader.utils import get_deformation_paths

config = Config()

def get_val_paths():
    return get_deformation_paths(config.data_config.data_dir, None, 0.8, 'train')



if __name__ == '__main__':
    
    resume_path = "Checkpoint/f2d/checkpoint_11_2000.pth"
    demo_path = "dress_data/cross_deformation/0_DLLS_Dress132_obj/p_0.npz"
    
    model=Sofa_Model(feature_dim=config.train_config.feature_dim)
    model.load_state_dict(torch.load(resume_path,map_location=config.train_config.device)['model_state_dict'])
    model = model.to(config.train_config.device)
    
    val_paths = get_val_paths()

    val_points = []
    val_outputs = []
    val_correspondences = []
    for garment_path in val_paths:
        for val_path in garment_path:
            val_point = np.load(val_path)['points']
            val_point = np.expand_dims(val_point, axis=0)
            val_point = torch.tensor(val_point).to(config.train_config.device)
            val_points.append(val_point)
            with torch.no_grad():
                val_output = model(torch.tensor(val_point).to(config.train_config.device))
            val_outputs.append(val_output)
            
            val_correspondence = np.load(val_path)['keypoints_id']
            val_correspondence = np.expand_dims(val_correspondence, axis=0)
            val_correspondence = torch.tensor(val_correspondence).to(config.train_config.device)
            val_correspondences.append(val_correspondence)

    
    val_points = torch.cat(val_points, dim=0)
    print(f"val_points shape: {val_points.shape}")
    val_outputs = torch.cat(val_outputs, dim=0) 
    print(f"val_outputs shape: {val_outputs.shape}")
    val_correspondences = torch.cat(val_correspondences, dim=0)
    print(f"val_correspondences shape: {val_correspondences.shape}")
    
    val_len = val_outputs.shape[0]
    
    demo_point = np.load(demo_path)['points']
    demo_point = np.expand_dims(demo_point, axis=0)
    
    demo_points = np.repeat(demo_point, val_len, axis=0)
    demo_points = torch.tensor(demo_points).to(config.train_config.device)
    print(f"demo_points shape: {demo_points.shape}")
    
    with torch.no_grad():
        demo_output = model(torch.tensor(demo_point).to(config.train_config.device))
    demo_outputs = demo_output.repeat(val_len,1,1)
    print(f"demo_output shape: {demo_output.shape}")
    
    demo_correspondences = np.load(demo_path)['keypoints_id']
    demo_correspondences = torch.tensor(demo_correspondences).to(config.train_config.device)
    demo_correspondences = demo_correspondences.repeat(val_len,1)
    print(f"demo_correspondences shape: {demo_correspondences.shape}")
    
    correspondence = torch.stack((demo_correspondences, val_correspondences), dim=2)
    print(f"correspondence shape: {correspondence.shape}")

    inference = cal_inference_pair(demo_outputs,val_outputs,correspondence).long()
    distance,accuracy=cal_distance_accuracy(demo_points,val_points,inference,correspondence)
    distance=distance.item()
    accuracy=accuracy.item()
    
    print(f"distance: {distance}")
    print(f"accuracy: {accuracy}")