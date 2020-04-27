import torch
import torch.nn as nn

from dataset import Base_Op,Wraped_Data
import baseline
from train import train_for_deep
import utils
import config
import os
import pickle as pkl

if __name__=='__main__':
    opt=config.parse_opt()
    torch.cuda.set_device(opt.CUDA_DEVICE)
    torch.manual_seed(opt.SEED)
    
    dictionary=Base_Op()
    dictionary.init_dict()
    
    train_set=Wraped_Data(opt,dictionary,'train')
    test_set=Wraped_Data(opt,dictionary,'test')
    constructor='build_baseline'
    model=getattr(baseline,constructor)(train_set,opt).cuda()
    model.w_emb.init_embedding()
    print ('Length of train:',len(train_set))
    print ('Length of test:',len(test_set))
    train_for_deep(model,test_set,opt,train_set)
    exit(0)