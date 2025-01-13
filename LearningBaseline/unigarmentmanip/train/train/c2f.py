'''coarse to fine single can visualize'''
import argparse
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import sys
sys.path.append(os.getcwd())
sys.path.append("unigarment/train")

import torch.nn.functional as F
from info_nce import InfoNCE
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.pointnet2_Sofa_Model import Sofa_Model
from unigarment.train.dataloader.utils import *
from unigarment.train.base.config import c2f_Config

configs = c2f_Config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

class C2fDataset(Dataset):
    
    def __init__(self, model: Sofa_Model):
        
        self.model = model 
        self.device = device
        
        self.all_deform_paths = get_deformation_paths(configs.data_dir, configs.garment_data_num, 1)
        self.cross_deformation_pair_path = create_flat2deform_pairs(self.all_deform_paths)
        
        self.criterion=InfoNCE(negative_mode='paired',temperature=0.1)
        self.optimizer=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-5)
   
        print(f"Number of all deformation paths: {len(self.all_deform_paths)}")
        print(f"Number of cross deformation pairs: {len(self.cross_deformation_pair_path)}")
     
    def __len__(self):
        return len(self.cross_deformation_pair_path)
    
    def get_cross_deformation_correspondence(self, index):
        
        npz1, npz2 = self.cross_deformation_pair_path[index]
        npz1 = np.load(npz1)
        npz2 = np.load(npz2)
        
        mesh_points_1 = npz1['mesh_points']
        mesh_points_2 = npz2['mesh_points']
        pcd_points_1 = npz1['pcd_points']
        pcd_points_2 = npz2['pcd_points']
        visible_mesh_id_1 = npz1['visible_mesh_indices']
        visible_mesh_id_2 = npz2['visible_mesh_indices']
        visible_mesh_id = np.intersect1d(visible_mesh_id_1, visible_mesh_id_2)
        visible_flat_pcd_id = nearest_mesh2pcd(mesh_points_1, pcd_points_1, visible_mesh_id)
        visible_deform_pcd_id = nearest_mesh2pcd(mesh_points_2, pcd_points_2, visible_mesh_id)
        
        mesh_id = nearest_pcd2mesh(pcd_points_2, mesh_points_2, np.arange(0, pcd_points_2.shape[0]))
        
        correspondence = nearest_mesh2pcd(mesh_points_1, pcd_points_1, mesh_id)
        
        correspondence = np.array(correspondence)
            
        return correspondence, visible_flat_pcd_id, visible_deform_pcd_id
        
        
       
    def get_cross_deformation_pair(self, index):
        
        npz1, npz2 = self.cross_deformation_pair_path[index]
        npz1 = np.load(npz1)
        npz2 = np.load(npz2)
        pc1= npz1['pcd_points']
        pc2= npz2['pcd_points']
        correspondence, visible_flat_pcd_id, visible_deform_pcd_id = self.get_cross_deformation_correspondence(index)
        
        return pc1, pc2, correspondence, visible_flat_pcd_id, visible_deform_pcd_id
    

    def process(self,index):
        flat, deform, correspondence, visible_flat_pcd_id, visible_deform_pcd_id = self.get_cross_deformation_pair(index)
        # vis=visualize(flat,deform,gt_correspondence)
        # vis.show_gt()
        flat = torch.from_numpy(flat).float().to(device)
        deform = torch.from_numpy(deform).float().to(device)
        correspondence = torch.from_numpy(correspondence).long().to(device)
        # print(f"correspondence_shape: {correspondence.shape}")
        
        for repetition in range(5):
            
            flat_feature = self.model(flat.unsqueeze(0).float().to(device))
            flat_feature = F.normalize(flat_feature,dim=-1)
            deform_feature = self.model(deform.unsqueeze(0).float().to(device))
            deform_feature = F.normalize(deform_feature,dim=-1)
            
            flat_feature = flat_feature.squeeze(0)  #  N * 512
            deform_feature = deform_feature.squeeze(0)  #  N * 512
            
            # 将 visible_deform_pcd_id 和 visible_flat_pcd_id 转换为集合，以加速 in 操作
            visible_deform_pcd_id_set = set(visible_deform_pcd_id)
            visible_flat_pcd_id_set = set(visible_flat_pcd_id)

            # 批量计算
            deform_query_result = torch.full((flat.shape[0],), -1, dtype=torch.long, device=device)  # 初始化为 -1

            # 计算所有点的相似度
            similarity_matrix = torch.sum(deform_feature.unsqueeze(1) * flat_feature.unsqueeze(0), dim=2)  # (N, N)

            # 获取每个变形衣物点在平展衣物中的最相似点
            _, deform_query_result = similarity_matrix.max(dim=1)

            # 筛选有效的匹配点，使用集合加速查找
            for j in range(flat.shape[0]):
                if j in visible_deform_pcd_id_set:  # 使用集合检查
                    if deform_query_result[j] in visible_flat_pcd_id_set:  # 使用集合检查
                        deform_query_result[j] = deform_query_result[j]  # 保留原值
                    else:
                        deform_query_result[j] = -1  # 不匹配的点设置为 -1
                else:
                    deform_query_result[j] = -1  # 如果变形点不可见，设置为 -1
                
            # 变形衣服中每个点在flat中特征最匹配的点
            deform_query_result = deform_query_result.long()    #  N
            
            valid_indices = deform_query_result != -1
            valid_indices = torch.nonzero(valid_indices).reshape(-1)
            gt_corr_pos = flat[correspondence[valid_indices], :3]   #  N * 3 物理意义上的对应点
            query_corr_pos = flat[deform_query_result[valid_indices], :3]  #  N * 3 结构意义上的对应点（feature最相近的点）
            distance = torch.sum((gt_corr_pos-query_corr_pos)**2,dim=1)
            
            mistake = distance > 0.08
            
            mistake_index = torch.nonzero(mistake).reshape((-1,))   # K 认为是匹配错误的点的deform中的索引
            
            
            if len(mistake_index) <= 100:
                break
            # vis.show_mistake(mistake)
            deform_query_result=torch.zeros((mistake_index.shape[0]),150).to(self.device)   # K * 150
            for j in range(mistake_index.shape[0]):
                # 找相似度最大的150个点 作为负点（因为这是认为的匹配错误的点）
                deform_query_result[j]=torch.argsort(torch.sum(deform_feature[mistake_index[j]]*flat_feature,dim=1),descending=True)[:150]
            
            deform_query_result=deform_query_result.long()  # K * 150
            assert deform_query_result.shape[1] == 150, "deform_query_result_shape error: {deform_query_result.shape}"
            # print(f"deform_query_result_shape: {deform_query_result.shape}")


            randindex=torch.randperm(flat.shape[0])[:200]   # flat中随机选200个点的索引

            if mistake_index.shape[0] > 500:
                mistake_index=mistake_index[torch.randperm(mistake_index.shape[0])[:500]]  # 控制匹配错误的点的数量不超过500 K<=500
                deform_query_result=deform_query_result[torch.randperm(deform_query_result.shape[0])[:500]]  # 控制匹配错误的点的数量不超过500 K<=500
            
            query=deform_feature[mistake_index]  # K * 512
            query=torch.cat((query,deform_feature[randindex]),dim=0) # (K + 200) * 512
            positive=flat_feature[correspondence[mistake_index]]   # K * 512
            positive=torch.cat((positive,flat_feature[correspondence[randindex]]),dim=0)   # (K + 200) * 512
            # print(f"deform_query_result_shape: {deform_query_result.shape}")
            negative=flat_feature[deform_query_result.reshape((-1,))].reshape((mistake_index.shape[0],150,-1))
            rand_negative_index=torch.randint(0,flat.shape[0],(200,150)).to(self.device)  # 200 * 150
            negative=torch.cat((negative,flat_feature[rand_negative_index.reshape(-1)].reshape(200,150,-1)),dim=0)
            
            self.optimizer.zero_grad()
            loss=self.criterion(query,positive,negative)
            loss.backward()
            if loss.item()<4:
                break
            self.optimizer.step()
 

        
        
        
    
if __name__=="__main__":

    model_path = "Checkpoint/f2d/checkpoint_4_36000.pth"
    model = Sofa_Model(feature_dim=512)
    model.load_state_dict(torch.load(model_path,map_location=device)['model_state_dict'])
    model=model.to(device)

    dataset=C2fDataset(model)
    checkpoint_dir = "Checkpoint/c2f"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    with tqdm(total=len(dataset)) as pbar:
        for i in range(len(dataset)):
            dataset.process(i)

            if i % 100 == 0:
                torch.save({"model_state_dict":dataset.model.state_dict(),
                            "optimizer_state_dict":dataset.optimizer.state_dict(),
                            "optimizer":dataset.optimizer}, os.path.join(checkpoint_dir, "model_checkpoint_"+str(i)+".pth"))
            # print("process {}th mesh".format(i))
            pbar.update(1)


    

    