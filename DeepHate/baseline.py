import torch
import torch.nn as nn
import config
import numpy as np
import torch.nn.functional as F

from full_rnn import Full_RNN,CNN_Model,Gate_combine_three
from language_model import Word_Embedding,Fast_Embedding,Para_Embedding,Senti_Embedding
from classifier import SimpleClassifier,SingleClassifier
from attention import Attention, Gate_Attention

class Deep_Basic(nn.Module):
    def __init__(self,w_emb,para_emb,fast_emb,opt,rnn1,rnn2,rnn3,cnn1,cnn2,cnn3,fc,senti_emb,rnn4,cnn4,proj_t,gate,att1,att2,att3,att4):
        super(Deep_Basic,self).__init__()
        self.opt=opt
        self.w_emb=w_emb
        self.para_emb=para_emb
        self.fast_emb=fast_emb
        self.senti_emb=senti_emb
        
        self.rnn1=rnn1
        self.rnn2=rnn2
        self.rnn3=rnn3
        self.rnn4=rnn4
        self.cnn1=cnn1
        self.cnn2=cnn2
        self.cnn3=cnn3
        self.cnn4=cnn4
        self.att1=att1
        self.att2=att2
        self.att3=att3
        self.att4=att4
        
        
        self.gate=gate
        self.proj_t=proj_t
        self.fc=fc
        
        #attention weights for output after convolution operation with different filter size
        fsz=len(self.opt.FILTER_SIZE.split(','))
        self.a=nn.Parameter(torch.Tensor(fsz))
        self.a.data.uniform_(0,1)
        self.b=nn.Parameter(torch.Tensor(fsz))
        self.b.data.uniform_(0,1)
        self.c=nn.Parameter(torch.Tensor(fsz))
        self.c.data.uniform_(0,1)
        self.d=nn.Parameter(torch.Tensor(fsz))
        self.d.data.uniform_(0,1)
        
        #attention weights for three different kinds of embeddings
        self.total=nn.Parameter(torch.Tensor(3))
        self.total.data.uniform_(0,1)
        
    def forward(self,text,topic):
        glove_emb=self.w_emb(text)
        fast_emb=self.fast_emb(text)
        para_emb=self.para_emb(text)
        senti_emb=self.senti_emb(text)
        
        glove=[]
        senti=[]
        fast=[]
        para=[]
        glove_cnn=self.cnn1(glove_emb)
        fast_cnn=self.cnn2(fast_emb)
        para_cnn=self.cnn3(para_emb)
        senti_cnn=self.cnn4(senti_emb)
        for i,fsz in enumerate(self.opt.FILTER_SIZE.split(',')):
            g=glove_cnn[i]
            f=fast_cnn[i]
            p=para_cnn[i]
            s=senti_cnn[i]
            glove_rnn,hidden_g=self.rnn1(g)
            fast_rnn,hidden_f=self.rnn2(f)
            para_rnn,hidden_p=self.rnn3(p)
            senti_rnn,hidden_s=self.rnn4(s)
            glove_rnn=self.att1(glove_rnn,hidden_g)
            fast_rnn=self.att2(fast_rnn,hidden_f)
            para_rnn=self.att3(para_rnn,hidden_p)
            senti_rnn=self.att4(senti_rnn,hidden_s)
            glove.append(glove_rnn.unsqueeze(2))
            fast.append(fast_rnn.unsqueeze(2))
            para.append(para_rnn.unsqueeze(2))
            senti.append(senti_rnn.unsqueeze(2))
        glove=torch.cat(glove,dim=2)
        para=torch.cat(para,dim=2)
        fast=torch.cat(fast,dim=2)
        senti=torch.cat(senti,dim=2)
        glove=torch.sum(self.a*glove,dim=2)
        para=torch.sum(self.b*para,dim=2)
        fast=torch.sum(self.c*fast,dim=2)
        senti=torch.sum(self.d*senti,dim=2)
        
        total=[]
        total.append(glove.unsqueeze(2))
        total.append(para.unsqueeze(2))
        total.append(fast.unsqueeze(2))
        total=torch.cat(total,dim=2)
        semantic=torch.sum(self.total*total,dim=2)
        
        proj_topic=self.proj_t(topic)
        joint=self.gate(senti,proj_topic,semantic)
        logits=self.fc(joint)
        
        return logits 
        
        
def build_baseline(dataset,opt): 
    opt=config.parse_opt()
    w_emb=Word_Embedding(dataset.dictionary.ntoken(),opt.EMB_DIM,opt.EMB_DROPOUT) 
    para_emb=Para_Embedding(dataset.dictionary.ntoken(),opt.EMB_DIM,opt.EMB_DROPOUT) 
    fast_emb=Fast_Embedding(dataset.dictionary.ntoken(),opt.EMB_DIM,opt.EMB_DROPOUT) 
    senti_emb=Senti_Embedding(dataset.dictionary.ntoken(),opt.EMB_DIM,opt.EMB_DROPOUT) 
    
    if opt.DATASET=='founta' or opt.DATASET=='wz':
        final_dim=4
    elif opt.DATASET=='dt_full' :
        final_dim=3
    else:
        final_dim=2
        
    fc=SimpleClassifier(opt.NUM_HIDDEN,opt.MID_DIM,final_dim,opt.FC_DROPOUT)
    
    rnn1=Full_RNN(opt.NUM_FILTER,opt.NUM_HIDDEN,opt.NUM_LAYER,opt.BIDIRECT,opt.L_RNN_DROPOUT)
    rnn2=Full_RNN(opt.NUM_FILTER,opt.NUM_HIDDEN,opt.NUM_LAYER,opt.BIDIRECT,opt.L_RNN_DROPOUT)
    rnn3=Full_RNN(opt.NUM_FILTER,opt.NUM_HIDDEN,opt.NUM_LAYER,opt.BIDIRECT,opt.L_RNN_DROPOUT)
    rnn4=Full_RNN(opt.NUM_FILTER,opt.NUM_HIDDEN,opt.NUM_LAYER,opt.BIDIRECT,opt.L_RNN_DROPOUT)
    cnn1=CNN_Model(opt.EMB_DIM,opt.NUM_HIDDEN,opt)
    cnn2=CNN_Model(opt.EMB_DIM,opt.NUM_HIDDEN,opt)
    cnn3=CNN_Model(opt.EMB_DIM,opt.NUM_HIDDEN,opt)
    cnn4=CNN_Model(opt.EMB_DIM,opt.NUM_HIDDEN,opt)
    att1=Attention(opt)
    att2=Attention(opt)
    att3=Attention(opt)
    att4=Attention(opt)
    
    gate=Gate_combine_three(opt.NUM_HIDDEN,opt.MID_DIM,opt.ATT_DROPOUT)
    proj_t=SingleClassifier(opt.NUM_TOPICS,opt.NUM_HIDDEN,opt.FC_DROPOUT)
    
    
    return Deep_Basic(w_emb,para_emb,fast_emb,opt,rnn1,rnn2,rnn3,cnn1,cnn2,cnn3,fc,senti_emb,rnn4,cnn4,proj_t,gate,att1,att2,att3,att4)
    
    