import torch
import torch.nn as nn
from torch.utils.data import Subset,ConcatDataset

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
    
    #result saving
    if opt.DATASET=='dt':
        logger=utils.Logger(os.path.join(opt.OFFENSIVE_RESULT,'final_'+str(opt.SAVE_NUM)+'.txt'))
    elif opt.DATASET=='dt_full':
        logger=utils.Logger(os.path.join(opt.OFFENSIVE_FULL_RESULT,'final_'+str(opt.SAVE_NUM)+'.txt'))
    else:
        logger=utils.Logger(os.path.join(opt.FOUNTA_RESULT,'final_'+str(opt.SAVE_NUM)+'.txt'))
    
    dictionary=Base_Op()
    dictionary.init_dict()
    split_dataset=pkl.load(open(os.path.join(opt.SPLIT_DATASET,opt.DATASET+'_new.pkl'),'rb'))
    
    constructor='build_baseline'
    #definitions for criteria
    score=0.0
    f1=0.0
    recall=0.0
    precision=0.0
    m_f1=0.0
    m_recall=0.0
    m_precision=0.0
    for i in range(opt.CROSS_VAL):
        train_set=Wraped_Data(opt,dictionary,split_dataset,i)
        test_set=Wraped_Data(opt,dictionary,split_dataset,i,'test')
        model=getattr(baseline,constructor)(train_set,opt).cuda()
        model.w_emb.init_embedding()
        model.fast_emb.init_embedding()
        model.senti_emb.init_embedding()
        model.para_emb.init_embedding()
        s,f,p,r,m_f,m_r,m_p=train_for_deep(model,test_set,opt,train_set)
        score+=s
        f1+=f
        precision+=p
        recall+=r
        m_f1+=m_f
        m_precision+=m_p
        m_recall+=m_r
        logger.write('validation folder %d' %(i+1))
        logger.write('\teval score: %.2f ' % (s))
        logger.write('\teval precision: %.2f ' % (p))
        logger.write('\teval recall: %.2f ' % (r))
        logger.write('\teval f1: %.2f ' % (f))
        logger.write('\teval macro precision: %.2f ' % (m_p))
        logger.write('\teval macro recall: %.2f ' % (m_r))
        logger.write('\teval macro f1: %.2f ' % (m_f))
    score/=opt.CROSS_VAL
    f1/=opt.CROSS_VAL
    precision/=opt.CROSS_VAL
    recall/=opt.CROSS_VAL
    m_f1/=opt.CROSS_VAL
    m_precision/=opt.CROSS_VAL
    m_recall/=opt.CROSS_VAL
    logger.write('\n final result')
    logger.write('\teval score: %.2f ' % (score))
    logger.write('\teval precision: %.2f ' % (precision))
    logger.write('\teval recall: %.2f ' % (recall))
    logger.write('\teval f1: %.2f ' % (f1))
    logger.write('\teval macro precision: %.2f ' % (m_precision))
    logger.write('\teval macro recall: %.2f ' % (m_recall))
    logger.write('\teval macro f1: %.2f ' % (m_f1))
    exit(0)
    