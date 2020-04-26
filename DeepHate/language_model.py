import torch
import torch.nn as nn
import numpy as np
import config
import os
import torch.nn.functional as F
from torch.autograd import Variable
import math

opt=config.parse_opt()

class Para_Embedding(nn.Module):
    def __init__(self,ntoken,emb_dim,dropout):
        super(Para_Embedding,self).__init__()
        self.emb=nn.Embedding(ntoken+1,emb_dim,padding_idx=ntoken)
        self.dropout=nn.Dropout(dropout)
        self.ntoken=ntoken
        self.emb_dim=emb_dim

    def init_embedding(self):
        print ('Initializing glove Embedding...')
        if opt.DATASET=='dt':       
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.OFFENSIVE_DATA,'para_embedding.npy')))
        elif opt.DATASET=='founta':       
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.FOUNTA_DATA,'para_embedding.npy'))) 
        elif opt.DATASET=='dt_full':       
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.OFFENSIVE_FULL_DATA,'para_embedding.npy')))
        elif opt.DATASET=='wz':
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.WZ_DATA,'para_embedding.npy')))
        elif opt.DATASET=='total':
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.TOTAL_DATA,'para_embedding.npy')))
        self.emb.weight.data[:self.ntoken]=glove_weight
  
    def forward(self,x):
        emb=self.emb(x)
        emb=self.dropout(emb)
        return emb

    
class Senti_Embedding(nn.Module):
    def __init__(self,ntoken,emb_dim,dropout):
        super(Senti_Embedding,self).__init__()
        self.emb=nn.Embedding(ntoken+1,emb_dim,padding_idx=ntoken)
        self.dropout=nn.Dropout(dropout)
        self.ntoken=ntoken
        self.emb_dim=emb_dim

    def init_embedding(self):
        print ('Initializing glove Embedding...')
        if opt.DATASET=='dt':       
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.OFFENSIVE_DATA,'senti_embedding.npy')))
        elif opt.DATASET=='founta':       
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.FOUNTA_DATA,'senti_embedding.npy'))) 
        elif opt.DATASET=='dt_full':       
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.OFFENSIVE_FULL_DATA,'senti_embedding.npy')))
        elif opt.DATASET=='wz':
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.WZ_DATA,'senti_embedding.npy')))
        elif opt.DATASET=='total':
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.TOTAL_DATA,'senti_embedding.npy')))
        self.emb.weight.data[:self.ntoken]=glove_weight
  
    def forward(self,x):
        emb=self.emb(x)
        emb=self.dropout(emb)
        return emb

class Word_Embedding(nn.Module):
    def __init__(self,ntoken,emb_dim,dropout):
        super(Word_Embedding,self).__init__()
        self.emb=nn.Embedding(ntoken+1,emb_dim,padding_idx=ntoken)
        self.dropout=nn.Dropout(dropout)
        self.ntoken=ntoken
        self.emb_dim=emb_dim

    def init_embedding(self):
        print ('Initializing glove Embedding...')
        if opt.DATASET=='dt':       
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.OFFENSIVE_DATA,'glove_embedding.npy')))
        elif opt.DATASET=='founta':       
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.FOUNTA_DATA,'glove_embedding.npy'))) 
        elif opt.DATASET=='dt_full':       
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.OFFENSIVE_FULL_DATA,'glove_embedding.npy')))
        elif opt.DATASET=='wz':
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.WZ_DATA,'glove_embedding.npy')))
        elif opt.DATASET=='total':
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.TOTAL_DATA,'glove_embedding.npy')))
        self.emb.weight.data[:self.ntoken]=glove_weight
  
    def forward(self,x):
        emb=self.emb(x)
        emb=self.dropout(emb)
        return emb

class Fast_Embedding(nn.Module):
    def __init__(self,ntoken,emb_dim,dropout):
        super(Fast_Embedding,self).__init__()
        self.emb=nn.Embedding(ntoken+1,emb_dim,padding_idx=ntoken)
        self.dropout=nn.Dropout(dropout)
        self.ntoken=ntoken
        self.emb_dim=emb_dim

    def init_embedding(self):
        print ('Initializing glove Embedding...')
        if opt.DATASET=='dt':       
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.OFFENSIVE_DATA,'fast_embedding.npy')))
        elif opt.DATASET=='founta':       
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.FOUNTA_DATA,'fast_embedding.npy'))) 
        elif opt.DATASET=='dt_full':       
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.OFFENSIVE_FULL_DATA,'fast_embedding.npy')))
        elif opt.DATASET=='wz':
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.WZ_DATA,'fast_embedding.npy')))
        elif opt.DATASET=='total':
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.TOTAL_DATA,'fast_embedding.npy')))
        self.emb.weight.data[:self.ntoken]=glove_weight
  
    def forward(self,x):
        emb=self.emb(x)
        emb=self.dropout(emb) 
        return emb
        