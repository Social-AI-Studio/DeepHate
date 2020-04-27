import torch
import torch.nn as nn
import config
import numpy as np
import torch.nn.functional as F

from full_rnn import Full_RNN
from language_model import Word_Embedding
from classifier import SimpleClassifier

class Deep_Basic(nn.Module):
    def __init__(self,w_emb,fc,opt,rnn):
        super(Deep_Basic,self).__init__()
        self.opt=opt
        self.w_emb=w_emb
        self.fc=fc
        self.rnn=rnn
        
        
    def forward(self,text):
        batch_size=text.shape[0]
        length=text.shape[1]
        #word level cnn
        word_emb=self.w_emb(text)
        word_rnn=self.rnn(word_emb)
        word_pool=F.max_pool1d(word_rnn.transpose(1,2).contiguous(),kernel_size=length).squeeze(2)
        logits=self.fc(word_pool)
        
        return logits 
        
        
def build_baseline(dataset,opt): 
    opt=config.parse_opt()
    w_emb=Word_Embedding(dataset.dictionary.ntoken(),opt.EMB_DIM,opt.EMB_DROPOUT,opt)  
    final_dim=3
    fc=SimpleClassifier(opt.NUM_HIDDEN*2,opt.MID_DIM,final_dim,opt.FC_DROPOUT)
    rnn=Full_RNN(opt.EMB_DIM,opt.NUM_HIDDEN,opt.NUM_LAYER,opt.BIDIRECT,opt.L_RNN_DROPOUT)
    return Deep_Basic(w_emb,fc,opt,rnn)
    