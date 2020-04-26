import os
import time 
import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
import config
import numpy as np
import h5py
import pickle as pkl
import json
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,classification_report,precision_recall_fscore_support
    
def log_hyperpara(logger,opt):
    dic = vars(opt)
    for k,v in dic.items():
        logger.write(k + ' : ' + str(v))

def bce_for_loss(logits,labels):
    loss=nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss*=labels.size(1)
    return loss

def compute_score(logits,labels):
    logits=torch.max(logits,1)[1]
    labels=torch.max(labels,1)[1]
    score=logits.eq(labels)
    score=score.sum().float()
    return score

def compute_other(logits,labels):
    acc=compute_score(logits,labels)
    logits=np.argmax(logits.cpu().numpy(),axis=1)
    label=np.argmax(labels.cpu().numpy(),axis=1)
    length=logits.shape[0]
    '''f1=f1_score(label,logits,average='micro',labels=np.unique(label))
    recall=recall_score(label,logits,average='micro',labels=np.unique(label))
    precision=precision_score(label,logits,average='micro',labels=np.unique(label))'''
    
    f1=f1_score(label,logits,average='weighted',labels=np.unique(label))
    recall=recall_score(label,logits,average='weighted',labels=np.unique(label))
    precision=precision_score(label,logits,average='weighted',labels=np.unique(label))
  
    m_f1=f1_score(label,logits,average='macro',labels=np.unique(label))
    m_recall=recall_score(label,logits,average='macro',labels=np.unique(label))
    m_precision=precision_score(label,logits,average='macro',labels=np.unique(label))
    return f1,recall,precision,acc,m_f1,m_recall,m_precision

def train_for_deep(model,test_set,opt,train_set):
    #freeze the parameters in the sentiment embedding layer during training
    for name, value in model.named_parameters():
        if name=='senti_emb.emb.weight':
            value.requires_grad=False
    params = filter(lambda p: p.requires_grad, model.parameters())
    optim=torch.optim.Adamax(params)
    if opt.DATASET=='offensive':
        logger=utils.Logger(os.path.join(opt.OFFENSIVE_RESULT,'log'+str(opt.SAVE_NUM)+'.txt'))
    elif opt.DATASET=='dt_full':
        logger=utils.Logger(os.path.join(opt.OFFENSIVE_FULL_RESULT,'log'+str(opt.SAVE_NUM)+'.txt'))
    elif opt.DATASET=='wz':
        logger=utils.Logger(os.path.join(opt.WZ_RESULT,'log'+str(opt.SAVE_NUM)+'.txt'))
    elif opt.DATASET=='founta':
        logger=utils.Logger(os.path.join(opt.FOUNTA_RESULT,'log'+str(opt.SAVE_NUM)+'.txt'))
    elif opt.DATASET=='total':
        logger=utils.Logger(os.path.join(opt.TOTAL_RESULT,'log'+str(opt.SAVE_NUM)+'.txt'))
    
    log_hyperpara(logger,opt)
    train_size=len(train_set)
    test_size=len(test_set)
    train_loader=DataLoader(train_set,opt.BATCH_SIZE,shuffle=True,num_workers=1)
    test_loader=DataLoader(test_set,opt.BATCH_SIZE,shuffle=True,num_workers=1)
    for epoch in range(opt.EPOCHS):
        total_loss=0
        train_score=0.0
        eval_loss=0
        eval_score=0.0
        t=time.time()
        for i,(tokens,labels,prob) in enumerate(train_loader):
            tokens=tokens.cuda()
            labels=labels.float().cuda()
            prob=prob.float().cuda()
            pred=model(tokens,prob)
            loss=bce_for_loss(pred,labels)
            total_loss+=loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            batch_score=compute_score(pred,labels)
            train_score+=batch_score
            if i==0:
                t_labels=labels
                t_pred=pred
            else:
                t_labels=torch.cat((t_labels,labels),0)
                t_pred=torch.cat((t_pred,pred),0)
        print ('Epoch', epoch,'for training loss:',total_loss)
        f1,recall,precision,acc,m_f1,m_recall,m_precision=compute_other(t_pred.detach(),t_labels)
        model.train(False)
        evaluate_score,test_loss,e_f1,e_recall,e_precision,m_f1,m_recall,m_precision=evaluate_for_offensive(model,test_loader,opt,epoch)
        eval_score=100 * evaluate_score /test_size
        total_loss = total_loss /train_size
        train_score=100 * train_score / train_size
        e_f1=100.0 * e_f1 
        e_recall=100.0 * e_recall 
        e_precision=100.0 * e_precision 
        m_f1=100.0 * m_f1 
        m_recall=100.0 * m_recall 
        m_precision=100.0 * m_precision 
        print ('Epoch:',epoch,'evaluation score:',eval_score,' loss:',eval_loss)
        print ('Epoch:',epoch,'evaluation f1:',e_f1,' recall:',e_recall)
        logger.write('epoch %d, time: %.2f' %(epoch, time.time() -t))
        logger.write('\ttrain_loss: %.2f, accuracy: %.2f' % (total_loss, train_score))
        logger.write('\teval accuracy: %.2f ' % ( eval_score))
        logger.write('\teval f1: %.2f ' % ( e_f1))
        logger.write('\teval precision: %.2f ' % ( e_precision))
        logger.write('\teval recall: %.2f ' % (e_recall))
        logger.write('\teval macro f1: %.2f ' % ( m_f1))
        logger.write('\teval macro precision: %.2f ' % ( m_precision))
        logger.write('\teval macro recall: %.2f ' % (m_recall))
        model.train(True)
    return eval_score,e_f1,e_precision,e_recall,m_f1,m_recall,m_precision
    
def evaluate_for_offensive(model,test_loader,opt,epoch):
    score=0.0
    total_loss=0
    f1=0.0
    precision=0.0
    recall=0.0
    acc=0.0
    total_num=len(test_loader.dataset)
    print ('The length of the loader is:',len(test_loader.dataset))
    for i,(tokens,labels,prob) in enumerate(test_loader):
        with torch.no_grad():
            tokens=tokens.cuda()
            labels=labels.float().cuda()
            prob=prob.float().cuda()
            pred=model(tokens,prob)
        batch_score=compute_score(pred,labels)
        batch_loss=bce_for_loss(pred,labels)
        total_loss+=batch_loss
        score+=batch_score
        if i==0:
            t_labels=labels
            t_pred=pred
        else:
            t_labels=torch.cat((t_labels,labels),0)
            t_pred=torch.cat((t_pred,pred),0)
    f1,recall,precision,acc,m_f1,m_recall,m_precision=compute_other(t_pred,t_labels)
    avg_loss=total_loss   
    return score,avg_loss,f1,recall,precision,m_f1,m_recall,m_precision    
            
            
            