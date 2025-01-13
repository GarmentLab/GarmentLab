
import os
import sys
sys.path.append(os.getcwd())
sys.path.append("unigarment/train")

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

from base.config import Config
from base.utils import *
from model.pointnet2_Sofa_Model import Sofa_Model
from val.simple_val import *
from dataloader.dataloader_only_cd import Dataset


config = Config()
config = config.train_config


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def train(checkpoint_dir:str, resume_path:str=None):
    # produce dataset train val dataloader
    train_dataset=Dataset("train")
    print("load dataset successfully")
        
    train_dataloader=DataLoader(train_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 num_workers=0)
      
    
    model=Sofa_Model(feature_dim=config.feature_dim)
    
    if resume_path:
        print("resume from {}".format(resume_path))
        model.load_state_dict(torch.load(resume_path,map_location=config.device)['model_state_dict'])
        optimizer=torch.load(resume_path,map_location=config.device)['optimizer']
    else:
        # produce optimizer
        optimizer=torch.optim.Adam(model.parameters(),lr=config.lr,weight_decay=config.weight_decay)

    # produce model
    model.to(config.device)

    # produce loss function
    criterion=InfoNCE(negative_mode='paired',temperature=config.temperature)

    
    wandb.init(project="unigarment-scarf", 
               name="train_session", 
               config={
                    "learning_rate": config.lr,
                    "batch_size": config.batch_size,
                    "epoch": config.epoch
    })

    # train
    train_step=0
    val_step=0 
    print("start training")
    for epoch in range(config.epoch):
        model.train()
        train_length=len(train_dataloader)
        total_loss=0
        with tqdm(enumerate(train_dataloader), total=train_length, desc=f"Epoch {epoch + 1}/{config.epoch}", disable=False) as t:
            for i, (pc1, pc2, correspondence) in t:
                # pc1 batchsize*num_points*3
                # pc2 batchsize*num_points*3
                # correspondence batchsize*num_correspondence*2
                batchsize=pc1.shape[0]
                num_points=pc1.shape[1]
                num_correspondence=correspondence.shape[1]

                pc1=pc1.to(config.device).float()
                pc2=pc2.to(config.device).float()
                correspondence=correspondence.to(config.device)


                #pc1_output batchsize*num_points*feature_dim
                #pc2_output batchsize*num_points*feature_dim
                pc1_output=model(pc1)
                pc2_output=model(pc2)
                feature_dim=pc1_output.shape[2]



                batch_index=torch.arange(0,pc1.shape[0])

                # query batchsize*num_correspondence*feature_dim
                # query=torch.stack([pc1_output[batch_index,correspondence[:,i,0]]for i in range(correspondence.shape[1])],dim=1)
                # print("query shape",query.shape)
                query=pc1_output.gather(1,correspondence[:,:,0].unsqueeze(2).expand(-1,-1,feature_dim))


                # positive batchsize*num_correspondence*feature_dim
                # positive=torch.stack([pc2_output[batch_index,correspondence[:,i,1]]for i in range(correspondence.shape[1])],dim=1)
                # print("positive shape",positive.shape)
                positive=pc2_output.gather(1,correspondence[:,:,1].unsqueeze(2).expand(-1,-1,feature_dim))



                # negative batchsize*num_correspondence*num_negative*feature_dim
                # negative here is random select not in correspondence
                # negative_index batchsize*num_correspondence*num_negative
                # print(f"num_points: {num_points}, batchsize: {batchsize}, num_correspondence: {num_correspondence}")
                # assert num_points > 0, "num_points must be positive"
                # assert num_points > config.num_negative, "num_points must be greater than config.num_negative"

                # print(f"num_points: {num_points}, config.num_negative: {config.num_negative}")
                negative_index=torch.randint(0,num_points-1,(batchsize,num_correspondence,config.num_negative)).to(config.device)
                negative_index=negative_index.reshape(batchsize,num_correspondence*config.num_negative)
                negative=pc2_output.gather(1,negative_index.unsqueeze(2).expand(-1,-1,feature_dim))
                negative=negative.reshape(batchsize,num_correspondence,config.num_negative,feature_dim)
                # print("negative shape",negative.shape)


                num_negative=negative.shape[2]
                query=query.reshape(batchsize*num_correspondence,feature_dim)
                positive=positive.reshape(batchsize*num_correspondence,feature_dim)
                negative=negative.reshape(batchsize*num_correspondence,num_negative,feature_dim)


                loss=criterion(query,positive,negative)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss+=loss.item()

                inference = cal_inference_pair(pc1_output,pc2_output,correspondence).long()
                distance1,accuracy1=cal_distance_accuracy(pc1,pc2,inference,correspondence, 0.05)
                distance1=distance1.item()
                accuracy1=accuracy1.item()
                distance2,accuracy2=cal_distance_accuracy(pc2,pc1,inference,correspondence, 0.02)
                distance2=distance2.item()
                accuracy2=accuracy2.item()

                
                # print("train: epoch: {}, batch: {},loss: {}, average_loss: {}, distance: {}, accuracy: {}".format(
                #     epoch, i, loss.item(), total_loss / (i + 1), distance, accuracy
                # ))

                wandb.log({
                    'epoch': epoch,
                    'train_loss': loss.item(),
                    'train_average_loss': total_loss / (i + 1),
                    'train_distance': distance1,
                    'train_accuracy_0.05': accuracy1,
                    'train_accuracy_0.02': accuracy2,
                }, step=train_step)
                train_step += 1
                
                
                if i > config.batch_num:
                    break
                
                if (i+1) % 2900 == 0:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save({'epoch':epoch,
                                'model_state_dict':model.state_dict(),
                                'optimizer':optimizer},
                                os.path.join(checkpoint_dir,f'checkpoint_{epoch+1}_{i+1}.pth'))


        # model.eval()
        # with torch.no_grad():
        #     val_total_loss=0
        #     for i,(pc1,pc2,correspondence) in enumerate(val_dataloader):
        #         # pc1 batchsize*num_points*6
        #         # pc2 batchsize*num_points*6
        #         # correspondence batchsize*num_correspondence*2
        #         batchsize=pc1.shape[0]
        #         num_points=pc1.shape[1]
        #         num_correspondence=correspondence.shape[1]



        #         pc1=pc1.to(config.train_config.device)
        #         pc2=pc2.to(config.train_config.device)
        #         correspondence=correspondence.to(config.train_config.device)




        #         #pc1_output batchsize*num_points*feature_dim
        #         #pc2_output batchsize*num_points*feature_dim
        #         pc1_output=model(pc1)
        #         pc2_output=model(pc2)
        #         feature_dim=pc1_output.shape[2]


        #         batch_index=torch.arange(0,pc1.shape[0])

        #         # query batchsize*num_correspondence*feature_dim
        #         query=pc1_output.gather(1,correspondence[:,:,0].unsqueeze(2).expand(-1,-1,feature_dim))


        #         # positive batchsize*num_correspondence*feature_dim
        #         positive=pc2_output.gather(1,correspondence[:,:,1].unsqueeze(2).expand(-1,-1,feature_dim))



        #         # negative batchsize*num_correspondence*num_negative*feature_dim
        #         # negative here is random select not in correspondence
        #         # negative_index batchsize*num_correspondence*num_negative
        #         negative_index=torch.randint(0,num_points,(batchsize,num_correspondence,config.train_config.num_negative)).to(config.train_config.device)
        #         negative_index=negative_index.reshape(batchsize,num_correspondence*config.train_config.num_negative)
        #         negative=pc2_output.gather(1,negative_index.unsqueeze(2).expand(-1,-1,feature_dim))
        #         negative=negative.reshape(batchsize,num_correspondence,config.train_config.num_negative,feature_dim)



                
        #         num_negative=negative.shape[2]
        #         query=query.reshape(batchsize*num_correspondence,feature_dim)
        #         positive=positive.reshape(batchsize*num_correspondence,feature_dim)
        #         negative=negative.reshape(batchsize*num_correspondence,num_negative,feature_dim)
        #         loss=criterion(query,positive,negative)
        #         val_total_loss+=loss.item()

        #         inference = cal_inference_pair(pc1_output,pc2_output,correspondence,config).long()
        #         distance,accuracy=cal_distance_accuracy(pc1,pc2,inference,correspondence,config)
        #         distance=distance.item()
        #         accuracy=accuracy.item()



        #         print("val: epoch:{},batch:{},loss:{},average_loss:{},distance:{},accuracy:{}".format(epoch,i,loss.item(),val_total_loss/(i+1),distance,accuracy))
        #         with open(log_val_path,'a') as f:
        #             f.write("val: epoch:{},batch:{},loss:{},average_loss:{},distance:{},accuracy:{}\n".format(epoch,i,loss.item(),val_total_loss/(i+1),distance,accuracy))
        #         writer.add_scalar('val_loss',loss.item(),val_step)
        #         writer.add_scalar('val_average_loss',val_total_loss/(i+1),val_step)
        #         writer.add_scalar('val_distance',distance,val_step)
        #         writer.add_scalar('val_accuracy',accuracy,val_step)
        #         if i > config.train_config.batch_num//100:
        #             break
        #     torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer':optimizer,"model":model},os.path.join(checkpoint_dir,'checkpoint_{}.pth'.format(epoch)))



    

if __name__ == '__main__':

    checkpoint_dir = "Checkpoint/f2d"

    train(checkpoint_dir)
    
