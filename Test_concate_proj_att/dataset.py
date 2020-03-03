import os
import pandas as pd
import re
import json
import pickle as pkl
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import utils
from tqdm import tqdm
import config
import itertools
import random
import string
from sparse_lda import SparseLDA

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data
    
def read_hdf5(path):
    data=h5py.File(path,'rb')
    return data

def read_csv(path):
    data=pd.read_csv(path)
    return data

def read_csv_sep(path):
    data=pd.read_csv(path,sep='\t')
    return data
    
def dump_pkl(path,info):
    pkl.dump(info,open(path,'wb'))  
    
def read_json(path):
    utils.assert_exits(path)
    data=json.load(open(path,'rb'))
    '''in anet-qa returns a list'''
    return data

def pd_pkl(path):
    data=pd.read_pickle(path)
    return data

def read_jsonl(path):
    total_info=[]
    with open(path,'rb')as f:
        d=f.readlines()
    for i,info in enumerate(d):
        data=json.loads(info)
        total_info.append(data)
    return total_info

class Base_Op(object):
    def __init__(self):
        self.opt=config.parse_opt()
        
    def expand_match(self,contraction):
        contraction_mapping = {
            "isn't": "is not",
            "aren't": "are not",
            "con't": "cannot",
            "can't've": "cannot have",
            "you'll've": "your will have",
            "you're": "you are",
            "you've": "you have"
        }
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
        if contraction_mapping.get(match)\
        else contraction_mapping.get(match.lower())                      
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
    
    def tokenize(self,text):
        url_pattern=re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))'r'[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})')
        emojis_pattern=re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
        hash_pattern=re.compile(r'#\w*')
        single_letter_pattern=re.compile(r'(?<![\w\-])\w(?![\w\-])')
        blank_spaces_pattern=re.compile(r'\s{2,}|\t')
        reserved_pattern=re.compile(r'(RT|rt|FAV|fav|VIA|via)')
        mention_pattern=re.compile(r'@\w*')
        CONTRACTION_MAP = {
            "isn't": "is not",
            "aren't": "are not",
            "con't": "cannot",
            "can't've": "cannot have",
            "you'll've": "your will have",
            "you're": "you are",
            "you've": "you have"
        }
        constraction_pattern=re.compile('({})'.format('|'.join(CONTRACTION_MAP.keys())),flags=re.IGNORECASE|re.DOTALL)
        Whitespace = re.compile(u"[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)
        urls=re.sub(pattern=url_pattern, repl='', string=text)
        mentions=re.sub(pattern=mention_pattern, repl='', string=urls)
        hashtag=re.sub(pattern=hash_pattern, repl='', string=mentions)
        reserved=re.sub(pattern=reserved_pattern, repl='', string=hashtag)
        reserved=Whitespace.sub(" ", reserved)
        reserved=constraction_pattern.sub(self.expand_match,reserved)
        punct="[{}]+".format(string.punctuation)
        punctuation=re.sub(punct,'',reserved)
        single=re.sub(pattern=single_letter_pattern, repl='', string=punctuation)
        blank=re.sub(pattern=blank_spaces_pattern, repl=' ', string=single)
        blank=blank.lower()
        #print blank
        return blank.split()
    
    def get_tokens(self,sent):
        tokens=self.tokenize(sent)
        #print tokens
        token_num=[]
        for t in tokens:
            if t in self.word2idx:
                token_num.append(self.word2idx[t])
            else:
                token_num.append(self.word2idx['UNK'])
        return token_num
    
    def token_sent(self):
        cur=0
        data=pkl.load(open(os.path.join(self.opt.SPLIT_DATASET,self.opt.DATASET+'.pkl'),'rb'))             
        for j,line in enumerate(data.keys()):
            cur_info=data[line]#it's a list
            for info in cur_info:
                tweet=info['sent']
                tokens=self.tokenize(tweet)
                for t in tokens:
                    if t not in self.word_count:
                        self.word_count[t]=1
                    else:
                        self.word_count[t]+=1
        for word in self.word_count.keys():
            if self.word_count[word]>=self.opt.MIN_OCC:
                self.word2idx[word]=cur
                self.idx2word.append(word)
                cur+=1
        self.idx2word.append('UNK')
        self.word2idx['UNK']=len(self.idx2word)-1   
        if self.opt.DATASET=='dt':
            dump_pkl(os.path.join(self.opt.OFFENSIVE_DATA,'dictionary.pkl'),[self.word2idx,self.idx2word])
        elif self.opt.DATASET=='founta':
            dump_pkl(os.path.join(self.opt.FOUNTA_DATA,'dictionary.pkl'),[self.word2idx,self.idx2word])
        elif self.opt.DATASET=='dt_full':
            dump_pkl(os.path.join(self.opt.OFFENSIVE_FULL_DATA,'dictionary.pkl'),[self.word2idx,self.idx2word])
        elif self.opt.DATASET=='wz':
            dump_pkl(os.path.join(self.opt.WZ_DATA,'dictionary.pkl'),[self.word2idx,self.idx2word])
    
    def create_dict(self):
        self.word_count={}
        self.word2idx={}
        self.idx2word=[]
        self.token_sent()
    
    def create_embedding(self):
        print (self.opt.DATASET)
        word2emb={}
        with open(self.opt.GLOVE_PATH,'r') as f:
            entries=f.readlines()
        emb_dim=len(entries[0].split(' '))-1
        weights=np.zeros((len(self.idx2word),emb_dim),dtype=np.float32)
        for entry in entries:
            word=entry.split(' ')[0]
            word2emb[word]=np.array(list(map(float,entry.split(' ')[1:])))
        for idx,word in enumerate(self.idx2word):
            if word not in word2emb:
                continue
            weights[idx]=word2emb[word]
        
            
        if self.opt.DATASET=='dt':
            np.save(os.path.join(self.opt.OFFENSIVE_DATA,'glove_embedding.npy'),weights)
        elif self.opt.DATASET=='founta':
            np.save(os.path.join(self.opt.FOUNTA_DATA,'glove_embedding.npy'),weights)
        elif self.opt.DATASET=='dt_full':
            np.save(os.path.join(self.opt.OFFENSIVE_FULL_DATA,'glove_embedding.npy'),weights)
        elif self.opt.DATASET=='wz':
            np.save(os.path.join(self.opt.WZ_DATA,'glove_embedding.npy'),weights)
        return weights
    
    def create_fasttext(self):
        word2emb={}
        with open(self.opt.FAST_TEXT,'r') as f:
            entries=f.readlines()
        emb_dim=len(entries[1].split(' '))-1
        weights=np.zeros((len(self.idx2word),emb_dim),dtype=np.float32)
        for i,entry in enumerate(entries):
            if i==0:
                continue
            word=entry.split(' ')[0]
            word2emb[word]=np.array(list(map(float,entry.split(' ')[1:])))
        for idx,word in enumerate(self.idx2word):
            if word not in word2emb:
                continue
            weights[idx]=word2emb[word]
            
        if self.opt.DATASET=='dt':
            np.save(os.path.join(self.opt.OFFENSIVE_DATA,'fast_embedding.npy'),weights)
        elif self.opt.DATASET=='founta':
            np.save(os.path.join(self.opt.FOUNTA_DATA,'fast_embedding.npy'),weights)
        elif self.opt.DATASET=='dt_full':
            np.save(os.path.join(self.opt.OFFENSIVE_FULL_DATA,'fast_embedding.npy'),weights)
        elif self.opt.DATASET=='wz':
            np.save(os.path.join(self.opt.WZ_DATA,'fast_embedding.npy'),weights)
        return weights
    
    def create_para(self):
        word2emb={}
        with open(self.opt.PARA_TEXT,'rb') as f:
            entries=f.readlines()
        emb_dim=self.opt.EMB_DIM
        weights=np.zeros((len(self.idx2word),emb_dim),dtype=np.float32)
        for i,entry in enumerate(entries):
            entry=str(entry)[:-2]
            word_list=entry.split(' ')
            word=word_list[0]
            num=bytes(word_list[1],encoding='utf-8')
            word2emb[word]=np.array(list(map(float,num)))
        for idx,word in enumerate(self.idx2word):
            if word not in word2emb:
                continue
            weights[idx]=word2emb[word]
            
        if self.opt.DATASET=='dt':
            np.save(os.path.join(self.opt.OFFENSIVE_DATA,'para_embedding.npy'),weights)
        elif self.opt.DATASET=='founta':
            np.save(os.path.join(self.opt.FOUNTA_DATA,'para_embedding.npy'),weights)
        elif self.opt.DATASET=='dt_full':
            np.save(os.path.join(self.opt.OFFENSIVE_FULL_DATA,'para_embedding.npy'),weights)
        elif self.opt.DATASET=='wz':
            np.save(os.path.join(self.opt.WZ_DATA,'para_embedding.npy'),weights)
        return weights
    
    def create_senti(self):
        word2emb={}
        senti_file=pkl.load(open(os.path.join(self.opt.SENT_EMB,self.opt.DATASET+'_sentiment_embdding.pkl'),'rb'),encoding='iso-8859-1')
        emb_dim=300
        weights=np.zeros((len(self.idx2word),emb_dim),dtype=np.float32)
        for i,word in enumerate(senti_file.keys()):
            word2emb[word]=np.array(senti_file[word],dtype=np.float32)
        for idx,word in enumerate(self.idx2word):
            if word not in word2emb:
                continue
            weights[idx]=word2emb[word]
            
        if self.opt.DATASET=='dt':
            np.save(os.path.join(self.opt.OFFENSIVE_DATA,'senti_embedding.npy'),weights)
        elif self.opt.DATASET=='founta':
            np.save(os.path.join(self.opt.FOUNTA_DATA,'senti_embedding.npy'),weights)
        elif self.opt.DATASET=='dt_full':
            np.save(os.path.join(self.opt.OFFENSIVE_FULL_DATA,'senti_embedding.npy'),weights)
        elif self.opt.DATASET=='wz':
            np.save(os.path.join(self.opt.WZ_DATA,'senti_embedding.npy'),weights)
        return weights
    
    def init_dict(self):
        if self.opt.CREATE_DICT:
            print ('Creating Dictionary...')
            self.create_dict()
        else:
            print ('Loading Dictionary...')
            if self.opt.DATASET=='dt':
                created_dict=load_pkl(os.path.join(self.opt.OFFENSIVE_DATA,'dictionary.pkl'))
            elif self.opt.DATASET=='founta':
                created_dict=pkl.load(open(os.path.join(self.opt.FOUNTA_DATA,'dictionary.pkl'),'rb'),encoding='iso-8859-1') 
            elif self.opt.DATASET=='dt_full':
                created_dict=load_pkl(os.path.join(self.opt.OFFENSIVE_FULL_DATA,'dictionary.pkl'))
            elif self.opt.DATASET=='wz':
                created_dict=load_pkl(os.path.join(self.opt.WZ_DATA,'dictionary.pkl'))
                
            
            self.word2idx=created_dict[0]
            self.idx2word=created_dict[1]
            
        if self.opt.CREATE_EMB:
            print ('Creating Embedding...;')
            self.senti=self.create_senti()
            self.glove_weights=self.create_embedding()
            self.fast_weights=self.create_fasttext()
            self.para_weight=self.create_para()
        else:
            print ('Loading Embedding...')
            if self.opt.DATASET=='dt':
                self.glove_weights=np.load(os.path.join(self.opt.OFFENSIVE_DATA,'glove_embedding.npy'))
                self.fast_weights=np.load(os.path.join(self.opt.OFFENSIVE_DATA,'fast_embedding.npy'))
                self.para_weights=np.load(os.path.join(self.opt.OFFENSIVE_DATA,'para_embedding.npy'))
                self.senti_weights=np.load(os.path.join(self.opt.OFFENSIVE_DATA,'senti_embedding.npy'))
            elif self.opt.DATASET=='founta':
                self.glove_weights=np.load(os.path.join(self.opt.FOUNTA_DATA,'glove_embedding.npy'))
                self.fast_weights=np.load(os.path.join(self.opt.FOUNTA_DATA,'fast_embedding.npy'))
                self.para_weights=np.load(os.path.join(self.opt.FOUNTA_DATA,'para_embedding.npy'))
                self.senti_weights=np.load(os.path.join(self.opt.FOUNTA_DATA,'senti_embedding.npy'))
            elif self.opt.DATASET=='dt_full':
                self.glove_weights=np.load(os.path.join(self.opt.OFFENSIVE_FULL_DATA,'glove_embedding.npy'))
                self.fast_weights=np.load(os.path.join(self.opt.OFFENSIVE_FULL_DATA,'fast_embedding.npy'))
                self.para_weights=np.load(os.path.join(self.opt.OFFENSIVE_FULL_DATA,'para_embedding.npy'))
                self.senti_weights=np.load(os.path.join(self.opt.OFFENSIVE_FULL_DATA,'senti_embedding.npy'))
            elif self.opt.DATASET=='wz':
                self.glove_weights=np.load(os.path.join(self.opt.WZ_DATA,'glove_embedding.npy'))
                self.fast_weights=np.load(os.path.join(self.opt.WZ_DATA,'fast_embedding.npy'))
                self.para_weights=np.load(os.path.join(self.opt.WZ_DATA,'para_embedding.npy'))
                self.senti_weights=np.load(os.path.join(self.opt.WZ_DATA,'senti_embedding.npy'))
        self.ntoken()
        
    def ntoken(self):
        self.ntokens=len(self.word2idx)
        print ('Number of Tokens:',self.ntokens)
        return self.ntokens
    
    
    def __len__(self):
        return len(self.word2idx)
        
class Wraped_Data(Base_Op):
    def __init__(self,opt,dictionary,split_data,test_num,mode='training'):
        super(Wraped_Data,self).__init__()
        self.opt=config.parse_opt()
        self.dictionary=dictionary
        self.split_data=split_data
        self.test_num=test_num
        self.mode=mode
        self.entries=self.load_tr_val_entries()
        if self.opt.DATASET=='dt':
            self.classes=2
        elif self.opt.DATASET=='dt_full':
            self.classes=3
        else:
            self.classes=4
        self.tokenize()
        self.tensorize()
   
    def load_tr_val_entries(self):
        all_data=[]
        #loading dataset for training and testing
        if self.mode=='training':
            for i in range(self.opt.CROSS_VAL):
                all_data.extend(self.split_data[str(i)])
        else:
            all_data.extend(self.split_data[str(self.test_num)])
        #classify types of tweet
        entries=[]
        for info in all_data:
            sent=info['sent']
            label=info['label']
            prob=info['topic']
            index=info['key']
            entry={
                'tweet':sent,
                'answer':label,
                'prob':prob,
            }
            entries.append(entry)
        return entries
    
    def padding_sent(self,tokens,length):
        if len(tokens)<length:
            padding=[self.dictionary.ntokens]*(length-len(tokens))
            tokens=padding+tokens
        else:
            tokens=tokens[:length]
        return tokens
    
    def padding_chars(self,tokens,length):
        if len(tokens)<length:
            padding=[26]*(length-len(tokens))
            tokens=padding+tokens
        else:
            tokens=tokens[:length]
        return tokens
    
    def tokenize(self):
        print('Tokenize Tweets...')
        length=self.opt.LENGTH
        for entry in tqdm(self.entries):
            tokens=self.dictionary.get_tokens(entry['tweet'])
            pad_tokens=self.padding_sent(tokens,length)
            entry['tokens']=np.array((pad_tokens),dtype=np.int64)
            
    def tensorize(self):
        print ('Tesnsorize all Information...')
        count=0
        for entry in tqdm(self.entries):
            entry['text_tokens']=torch.from_numpy(entry['tokens'])
            target=torch.from_numpy(np.zeros((self.classes),dtype=np.float32))
            target[entry['answer']]=1.0
            entry['label']=target
            prob=np.array(entry['prob'],dtype=np.float64)
            entry['prob']=torch.from_numpy(prob)
                        
                               
    
    def __getitem__(self,index):
        entry=self.entries[index]
        tweet=entry['text_tokens']
        label=entry['label']
        prob=entry['prob']
        return tweet,label,prob
        
        
    def __len__(self):
        return len(self.entries)
    
