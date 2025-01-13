import torch
class Config:
    def __init__(self):
        self.data_config=data_Config()
        self.train_config=train_Config()
        self.record_config=record_Config()


class data_Config:
    def __init__(self):
        self.data_dir="data/front_open/unigarment/cd_rotation/mesh_pcd"
        self.only_deformation_data_dir = "data/scarf/cd_processed/mesh_pcd"
        self.garment_data_num=None
        self.train_ratio=1
        self.flag='deform'#'deform' or 'object'
        self.deform_level=30#how messy the cloth is
        self.pair_num=50#how many pairs of correspondence to be sampled
        self.factor=0.9 #the factor of train val split
        self.distance_control=0.25
        self.distance_var=0.5
        self.distance_threshold=0.5
        self.correspondence_num=30  #########
        self.keypoints_correspondence_num=8

class train_Config:
    def __init__(self):
        self.lr=0.001
        self.batch_size=64
        self.epoch=1000
        self.num_workers=4
        self.weight_decay=1e-5
        self.device=torch.device('cuda')
        self.temperature=0.1
        self.num_negative=150
        self.correspondence_num=30   #######
        self.feature_dim=512
        self.distance_threshold=0.05   # ##
        self.batch_num=40000
        self.fine_num=150
        self.fine_negative=150
        self.smooth_num=64
        self.info_nce_weight=3
        self.num_sparse=10
        self.sparse_threshold=0.2
        self.upper_limit=300
        self.ratio=0.8
        
        self.save_pth_interval=4000

class record_Config:
    def __init__(self):
        self.record_interval=200 #record checkpoint between batch

class c2f_Config:
    def __init__(self):
        self.data_dir="/home/user/DexGarmentLab-master/DexGarmentLab-master/data/with_sleeves/cd_processed/mesh_pcd"
        self.garment_data_num=None
        self.train_ratio=1
        self.pair_num=50#how many pairs of correspondence to be sampled
        self.distance_control=0.25
        self.distance_var=0.5
        self.distance_threshold=0.5
        self.correspondence_num=30  #########
        self.keypoints_correspondence_num=8
        
        self.lr=0.001
        self.batch_size=64
        self.epoch=1000
        self.num_workers=4
        self.weight_decay=1e-5
        self.device=torch.device('cuda')
        self.temperature=0.1
        self.num_negative=150
        self.feature_dim=512
        self.distance_threshold=0.05   # ##
        self.batch_num=40000
        self.fine_num=150
        self.fine_negative=150
        self.smooth_num=64
        self.info_nce_weight=3
        self.num_sparse=10
        self.sparse_threshold=0.2
        self.upper_limit=300
        self.ratio=0.8
        
        self.save_pth_interval=2000