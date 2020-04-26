
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from fc import FCNet
from classifier import SimpleClassifier

class Attention(nn.Module):
    def __init__(self,opt):
        super(Attention,self).__init__()
        self.opt=opt
        self.v_proj=FCNet(self.opt.NUM_HIDDEN,self.opt.PROJ_DIM,self.opt.FC_DROPOUT)
        self.q_proj=FCNet(self.opt.NUM_HIDDEN,self.opt.PROJ_DIM,self.opt.FC_DROPOUT)
        self.att=FCNet(self.opt.PROJ_DIM,1,self.opt.FC_DROPOUT)
        self.softmax=nn.Softmax()
        
    def forward(self,v,q):
        v_proj=self.v_proj(v)
        q_proj=torch.unsqueeze(self.q_proj(q),1)
        vq_proj=F.relu(v_proj +q_proj)
        proj=torch.squeeze(self.att(vq_proj))
        w_att=torch.unsqueeze(self.softmax(proj),2)
        vatt=v * w_att
        att=torch.sum(vatt,1)
        return att

class Gate_Attention(nn.Module):
    def __init__(self,num_hidden_a,num_hidden_b,num_hidden):
        super(Gate_Attention,self).__init__()
        self.hidden=num_hidden
        self.w1=nn.Parameter(torch.Tensor(num_hidden_a,num_hidden))
        self.w2=nn.Parameter(torch.Tensor(num_hidden_b,num_hidden))
        self.bias=nn.Parameter(torch.Tensor(num_hidden))
        self.reset_parameter()
        
    def reset_parameter(self):
        stdv1=1. / math.sqrt(self.hidden)
        stdv2=1. / math.sqrt(self.hidden)
        stdv= (stdv1 + stdv2) / 2.
        self.w1.data.uniform_(-stdv1,stdv1)
        self.w2.data.uniform_(-stdv2,stdv2)
        self.bias.data.uniform_(-stdv,stdv)
        
    def forward(self,a,b):
        wa=torch.matmul(a,self.w1)
        wb=torch.matmul(b,self.w2)
        gated=wa+wb+self.bias
        gate=torch.sigmoid(gated)
        output=gate * a + (1-gate) * b
        return output