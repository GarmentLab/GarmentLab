import os
import sys

import numpy as np
curpath=os.getcwd()
sys.path.append(curpath)
sys.path.append(os.path.join(curpath,'train'))

from base.config import Config
import argparse
from base.utils import *
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import random
from info_nce import InfoNCE
import open3d as o3d

config = Config()
config = config.train_config

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def cal_inference_pair(feature1,feature2,correspondence):
    # feature1 batchsize*num_points*feature_dim
    # feature2 batchsize*num_points*feature_dim
    # correspondence batchsize*num_correspondence*2
    batchsize=feature1.shape[0]
    num_correspondence=correspondence.shape[1]
    feature_dim=feature1.shape[2]

    batch_index=torch.arange(batchsize).to(config.device)

    #query batchsize*num_correspondence*feature_dim
    # query=torch.stack([feature1[batch_index,correspondence[batch_index,i,0]] for i in range(num_correspondence)],dim=1)
    query=feature1.gather(1,correspondence[:,:,0].unsqueeze(-1).expand(-1,-1,feature_dim))
    query=F.normalize(query,dim=-1)
    feature2=F.normalize(feature2,dim=-1)
    #inferece batchsize*num_correspondence
    inference=torch.zeros(batchsize,num_correspondence).to(config.device)
    # for i in range(batchsize):
    #     for j in range(num_correspondence):
    #         inference[i,j]=torch.argmax(torch.sum(query[i,j]*feature2[i],dim=1,keepdim=True))
    # inference = torch.argmax(torch.sum(query[:,:,None,:] * feature2[:,None,:,:],dim=3),dim=2)# b x n x n x d
    for i in range(batchsize):
        inference[i]=torch.argmax(torch.sum(query[i,:,None,:]*feature2[i,None,:,:],dim=2),dim=1)
    return inference


def cal_distance_accuracy(pc1,pc2,inference,correspondence, distance_threshold):
    #pc1 batchsize*num_points*3
    #pc2 batchsize*num_points*3

    batchsize=pc1.shape[0]
    num_points=pc1.shape[1]
    num_correspondence=correspondence.shape[1]

    batch_index=torch.arange(batchsize).to(config.device)
    #pc1_pos
    # pc1_pos=torch.stack([pc1[batch_index,correspondence[batch_index,i,0]] for i in range(num_correspondence)],dim=1)
    pc1_pos=pc1.gather(1,correspondence[:,:,0].unsqueeze(-1).expand(-1,-1,3))

    #gt_pos batchsize*num_correspondence*3
    # gt_pos=torch.stack([pc2[batch_index,correspondence[batch_index,i,1],:3] for i in range(num_correspondence)],dim=1)
    gt_pos=pc2.gather(1,correspondence[:,:,1].unsqueeze(-1).expand(-1,-1,3))

    #inference_pos batchsize*num_correspondence*3
    # print(inference)
    # inference_pos=torch.stack([pc2[batch_index,inference[batch_index,i],:3] for i in range(num_correspondence)],dim=1)
    inference_pos=pc2.gather(1,inference.unsqueeze(-1).expand(-1,-1,3))
    
    #cal distance
    # distance=torch.zeros(batchsize,num_correspondence).to(config.train_config.device)
    # for i in range(batchsize):
    #     for j in range(num_correspondence):
    #         # print(inference_pos[i,j,:3])
    #         # print(gt_pos[i,j,:3])
    #distance batchsize*num_correspondence
    distance=torch.norm(inference_pos.reshape(-1,3)-gt_pos.reshape(-1,3),dim=1).reshape(batchsize,num_correspondence).to(config.device)
    
    #cal accuracy
    correct=distance<distance_threshold
    #accuracy batchsize
    accuracy=torch.sum(correct,dim=1)/num_correspondence

    return distance.mean().mean(),accuracy.mean()

def visualize(pc1,pc2,inference,correspondence):
    #pc1 batchsize*num_points*3
    #pc2 batchsize*num_points*3
    #inference batchsize*num_correspondence
    #correspondence batchsize*num_correspondence*2
    batchsize=pc1.shape[0]
    num_correspondence=correspondence.shape[1]
    for i in range(batchsize):
        pcd1=o3d.geometry.PointCloud()
        points1=pc1[i][:,:3].cpu().numpy().reshape(-1,3)
        colors1=pc1[i][:,3:].cpu().numpy().reshape(-1,3)
        points1[:,0]-=0.5
        pcd1.points=o3d.utility.Vector3dVector(points1)
        pcd1.colors=o3d.utility.Vector3dVector(colors1)
        pcd2=o3d.geometry.PointCloud()
        points2=pc2[i][:,:3].cpu().numpy().reshape(-1,3)
        colors2=pc2[i][:,3:].cpu().numpy().reshape(-1,3)
        points2[:,0]+=0.5
        pcd2.points=o3d.utility.Vector3dVector(points2)
        pcd2.colors=o3d.utility.Vector3dVector(colors2)
        gt_correspondence=[]
        for j in range(num_correspondence):
            gt_correspondence.append([correspondence[i,j,0],correspondence[i,j,1]])
        inference_correspondence=[]
        for j in range(num_correspondence):
            inference_correspondence.append([correspondence[i,j,0],inference[i,j]])
        
        gt_corr=o3d.geometry.LineSet().create_from_point_cloud_correspondences(pcd1,pcd2,gt_correspondence)
        gt_corr.colors=o3d.utility.Vector3dVector(np.tile(np.array([0,1,0]),(len(gt_correspondence),1)))
        inference_corr=o3d.geometry.LineSet().create_from_point_cloud_correspondences(pcd1,pcd2,inference_correspondence)
        inference_corr.colors=o3d.utility.Vector3dVector(np.tile(np.array([1,0,0]),(len(inference_correspondence),1)))
        o3d.visualization.draw_geometries([pcd1,pcd2,gt_corr,inference_corr])




            










