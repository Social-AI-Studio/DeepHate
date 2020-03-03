import torch
import torch.nn as nn
import numpy as np
import config
import torch.nn.functional as F
from torch.autograd import Variable
import math

opt=config.parse_opt()
class Word_Embedding(nn.Module):
    def __init__(self,ntoken,emb_dim,dropout):
        super(Word_Embedding,self).__init__()
        self.emb=nn.Embedding(ntoken+1,emb_dim,padding_idx=ntoken)
        self.dropout=nn.Dropout(dropout)
        self.ntoken=ntoken
        self.emb_dim=emb_dim

    def init_embedding(self):
        print 'Initializing glove Embedding...'
        if opt.DATASET=='tgif-qa':       
            glove_weight=torch.from_numpy(np.load(opt.EMB_DIR))
        elif opt.DATASET=='tvqa':     
            glove_weight=torch.from_numpy(np.load(opt.TVQA_EMB_DIR))   
        elif opt.DATASET=='anet-qa': 
            glove_weight=torch.from_numpy(np.load(opt.ANET_EMB_DIR))  
        self.emb.weight.data[:self.ntoken]=glove_weight
  
    def forward(self,x):
        emb=self.emb(x)
        emb=self.dropout(emb)
        return emb
    
class Attention(nn.Module):
    def __init__(self,opt):
        super(Attention,self).__init__()
        self.proj1=nn.Linear(opt.NUM_HIDDEN ,opt.PROJ_MID_DIM)
        self.proj2=nn.Linear(opt.NUM_HIDDEN * 3,opt.PROJ_MID_DIM)
        self.dropout=nn.Dropout(opt.ATT_DROPOUT)
        self.proj3=nn.Linear(opt.PROJ_MID_DIM,1)
    
    def forward(self,a,b):
        middle=F.tanh(self.proj1(a)+self.proj2(b))
        final=self.proj3(middle)
        return final

class RNN_Cell(nn.Module):
    def __init__(self,opt):
        super(RNN_Cell,self).__init__()
        in_size=opt.NUM_HIDDEN * 3
        hidden= opt.FINAL_HIDDEN
        self.fact_proj1=nn.Linear(in_size,hidden)
        self.fact_proj2=nn.Linear(in_size,hidden)
        self.h_proj1=nn.Linear(hidden,hidden)
        self.h_proj2=nn.Linear(hidden,hidden)
    
    def forward(self,fact,g_att,h_past):
        r=F.sigmoid(self.fact_proj1(fact) + self.h_proj1(h_past))
        h_comp=F.tanh(self.fact_proj2(fact) + self.h_proj2(h_past))
        h_update=g_att * h_comp + (1-g_att) * h_past
        return h_update
    

class Ans_Self_Att(nn.Module):
    def __init__(self,opt):
        super(Ans_Self_Att,self).__init__()
        self.opt=opt
        self.in_size=opt.NUM_HIDDEN * 3
        self.attention=Attention(opt)
        self.gru_cell=RNN_Cell(opt)
        
    def forward(self,ans_mean,ans_att):
        ans_mean=ans_mean / self.opt.ANS_LEN
        h_0 = Variable(torch.zeros(ans_att.size()[0],self.opt.FINAL_HIDDEN)).cuda()
        for idx in range(ans_att.size()[1]):
            ans_cur=ans_att[:,idx,:]
            weight=self.attention(ans_mean,ans_cur)
            if idx==0:
                h_update = h_0
            h_update=self.gru_cell(ans_cur,weight,h_update)
        return h_update

    
class AnswerEmbedding(nn.Module):
    def __init__(self,in_dim,num_hidden,num_layer,bidirect,dropout,rnn_type='LSTM'):
        super(AnswerEmbedding,self).__init__()
        rnn_cls=nn.LSTM if rnn_type=='LSTM' else nn.GRU
        self.rnn=rnn_cls(in_dim,num_hidden,num_layer,bidirectional=bidirect,dropout=dropout,batch_first=True)
        self.in_dim=in_dim
        self.num_hidden=num_hidden
        self.num_layer=num_layer
        self.rnn_type=rnn_type
        self.num_bidirect=1+int(bidirect)

    def init_hidden(self,batch,video_out):
        video_out=torch.unsqueeze(video_out,0)
        weight=next(self.parameters()).data
        hid_shape=(self.num_layer * self.num_bidirect,batch,self.num_hidden)
        if self.rnn_type =='LSTM':
            return (Variable(video_out).cuda(),Variable(torch.zeros([1,batch,self.num_hidden])).cuda())
        else:
            return Variable(video_out).cuda()

    def forward(self,x,video_out):
        batch=x.size(0)
        hidden=self.init_hidden(batch,video_out)
        self.rnn.flatten_parameters()
        output,hidden=self.rnn(x,hidden)
        if self.num_bidirect==1:
            return output[:,-1,:]
        forward_=output[:,-1,self.num_hidden]
        backward_=output[:,0,self.num_hidden]
        return torch.cat((forward_,backward_),dim=1)
    
    def forward_all(self,x):
        batch=x.size(0)
        hidden=self.init_hidden(batch)
        self.rnn_flatten_parameters()
        output,hidden=self.rnn(x,hidden)
        return output


    