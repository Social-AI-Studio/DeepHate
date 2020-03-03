import torch
import torch.nn as nn
import config
import numpy as np
import torch.nn.functional as F

from full_rnn import Full_RNN,Part_RNN,CNN_Model,Capsule
from language_model import Word_Embedding,Fast_Embedding,Para_Embedding,Senti_Embedding
from classifier import SimpleClassifier,SingleClassifier
from attention import Intra,Attention, Gate_Attention, MFB

class Deep_Basic(nn.Module):
    def __init__(self,w_emb,para_emb,fast_emb,opt,rnn1,rnn2,rnn3,cnn1,cnn2,cnn3,fc,senti_emb,rnn4,cnn4,proj_t,proj_s,gate,att,att1,att2,att3,proj_w):
        super(Deep_Basic,self).__init__()
        self.opt=opt
        self.w_emb=w_emb
        self.fc=fc
        self.para_emb=para_emb
        self.fast_emb=fast_emb
        self.rnn1=rnn1
        self.rnn2=rnn2
        self.rnn3=rnn3
        self.rnn4=rnn4
        self.cnn1=cnn1
        self.cnn2=cnn2
        self.cnn3=cnn3
        self.cnn4=cnn4
        self.senti_emb=senti_emb
        
        self.gate=gate
        self.att=att
        self.att1=att1
        self.att2=att2
        self.att3=att3
        self.proj_t=proj_t
        self.proj_s=proj_s
        self.proj_w=proj_w
        self.fc=fc
        
        
    def forward(self,text,topic):
        batch_size=text.shape[0]
        
        #char level cnn
        glove_emb=self.w_emb(text)
        fast_emb=self.fast_emb(text)
        para_emb=self.para_emb(text)
        senti_emb=self.senti_emb(text)
        
        glove_cnn=self.cnn1(glove_emb)
        fast_cnn=self.cnn2(fast_emb)
        para_cnn=self.cnn3(para_emb)
        senti_cnn=self.cnn4(senti_emb)
        
        
        glove_rnn,hidden_g=self.rnn1(glove_cnn)
        fast_rnn,hidden_f=self.rnn2(fast_cnn)
        para_rnn,hidden_p=self.rnn3(para_cnn)
        
        glove_rnn=self.att(glove_rnn,hidden_g)
        fast_rnn=self.att1(fast_rnn,hidden_f)
        para_rnn=self.att2(para_rnn,hidden_p)
        
        total_rnn=(glove_rnn + fast_rnn + para_rnn) / 3
        senti_rnn,hidden_s=self.rnn4(senti_cnn)
        senti_rnn=self.att3(senti_rnn,hidden_s)
        
        #capsule1=torch.cat((senti_rnn,total_rnn,topic),dim=1)
        #capsule2=self.proj_t(torch.cat((total_rnn,topic),dim=1))
        #capsule=self.gate(capsule1,capsule2)
        capsule1=self.proj_s(senti_rnn)
        capsule2=self.proj_t(topic)
        capsule3=self.proj_w(total_rnn)
        capsule=capsule1+capsule2+capsule3
        #capsule=capsule1+capsule2
        logits=self.fc(capsule)
        
        return logits 
        
        
def build_baseline(dataset,opt): 
    opt=config.parse_opt()
    w_emb=Word_Embedding(dataset.dictionary.ntoken(),opt.EMB_DIM,opt.EMB_DROPOUT) 
    para_emb=Para_Embedding(dataset.dictionary.ntoken(),opt.EMB_DIM,opt.EMB_DROPOUT) 
    fast_emb=Fast_Embedding(dataset.dictionary.ntoken(),opt.EMB_DIM,opt.EMB_DROPOUT) 
    senti_emb=Senti_Embedding(dataset.dictionary.ntoken(),opt.EMB_DIM,opt.EMB_DROPOUT) 
    
    if opt.DATASET=='founta' or opt.DATASET=='wz':
        final_dim=4
    elif opt.DATASET=='dt_full':
        final_dim=3
    else:
        final_dim=2
    fc=SimpleClassifier(opt.NUM_HIDDEN,opt.MID_DIM,final_dim,opt.FC_DROPOUT)
    rnn1=Full_RNN(len(opt.FILTER_SIZE.split(','))*opt.NUM_FILTER,opt.NUM_HIDDEN,opt.NUM_LAYER,opt.BIDIRECT,opt.L_RNN_DROPOUT)
    rnn2=Full_RNN(len(opt.FILTER_SIZE.split(','))*opt.NUM_FILTER,opt.NUM_HIDDEN,opt.NUM_LAYER,opt.BIDIRECT,opt.L_RNN_DROPOUT)
    rnn3=Full_RNN(len(opt.FILTER_SIZE.split(','))*opt.NUM_FILTER,opt.NUM_HIDDEN,opt.NUM_LAYER,opt.BIDIRECT,opt.L_RNN_DROPOUT)
    rnn4=Full_RNN(len(opt.FILTER_SIZE.split(','))*opt.NUM_FILTER,opt.NUM_HIDDEN,opt.NUM_LAYER,opt.BIDIRECT,opt.L_RNN_DROPOUT)
    '''rnn1=Part_RNN(opt.EMB_DIM,opt.NUM_HIDDEN,opt.NUM_LAYER,opt.BIDIRECT,opt.L_RNN_DROPOUT)
    rnn2=Part_RNN(opt.EMB_DIM,opt.NUM_HIDDEN,opt.NUM_LAYER,opt.BIDIRECT,opt.L_RNN_DROPOUT)
    rnn3=Part_RNN(opt.EMB_DIM,opt.NUM_HIDDEN,opt.NUM_LAYER,opt.BIDIRECT,opt.L_RNN_DROPOUT)
    rnn4=Part_RNN(opt.EMB_DIM,opt.NUM_HIDDEN,opt.NUM_LAYER,opt.BIDIRECT,opt.L_RNN_DROPOUT)'''
    cnn1=CNN_Model(opt.EMB_DIM,opt.NUM_HIDDEN,opt)
    cnn2=CNN_Model(opt.EMB_DIM,opt.NUM_HIDDEN,opt)
    cnn3=CNN_Model(opt.EMB_DIM,opt.NUM_HIDDEN,opt)
    cnn4=CNN_Model(opt.EMB_DIM,opt.NUM_HIDDEN,opt)
    gate=Gate_Attention(opt.NUM_HIDDEN,opt.NUM_HIDDEN,opt.NUM_HIDDEN)
    proj_t=SingleClassifier(opt.NUM_TOPICS,opt.NUM_HIDDEN,opt.FC_DROPOUT)
    proj_s=SingleClassifier(opt.NUM_HIDDEN,opt.NUM_HIDDEN,opt.FC_DROPOUT)
    proj_w=SingleClassifier(opt.NUM_HIDDEN,opt.NUM_HIDDEN,opt.FC_DROPOUT)
    
    attention=Attention(opt)
    att1=Attention(opt)
    att2=Attention(opt)
    att3=Attention(opt)
    return Deep_Basic(w_emb,para_emb,fast_emb,opt,rnn1,rnn2,rnn3,cnn1,cnn2,cnn3,fc,senti_emb,rnn4,cnn4,proj_t,proj_s,gate,attention,att1,att2,att3,proj_w)
    
    