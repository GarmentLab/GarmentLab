import numpy as np
import torch
import argparse
from affordance_utils import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()  
    
    parser.add_argument('--epoch_num', type=int, required=True)  
    parser.add_argument('--train_path', type=str, default="/home/isaac/isaacgarment/affordance/")  
    parser.add_argument('--eval_path', type=str, default="/home/isaac/isaacgarment/affordance/")  
    parser.add_argument('--task_name', type=str, default="hang")  
    parser.add_argument('--device', type=str, default="cuda:0")  
    
    args = parser.parse_args()  

    pts, label = preprocess(task_name = args.task_name, data_root_path = args.train_path, device=args.device)

    affordance=get_model(num_classes=1).to(args.device)
    opt = torch.optim.Adam(affordance.parameters(),lr=0.001)
    func = torch.nn.L1Loss()
    for idx in range(360):
        score_hat=affordance(pts)
        loss=func(score_hat,label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"training loss of epoch {idx} : {loss.item():.4f}")
