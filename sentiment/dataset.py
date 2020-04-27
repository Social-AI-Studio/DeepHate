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
    pkl.dump(info,open(path,'wb'),protocol = 2)  
    
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
        return blank.split()
    
    def get_tokens(self,sent):
        tokens=self.tokenize(sent)
        token_num=[]
        for t in tokens:
            if t in self.word2idx:
                token_num.append(self.word2idx[t])
            else:
                token_num.append(self.word2idx['UNK'])
        return token_num
    
    def token_sent(self):
        cur=0
        tweets=[]
        if self.opt.DATASET == 'dt' or self.opt.DATASET == 'dt_full':
            data=pd.read_csv(os.path.join(self.opt.DATA_PATH,(self.opt.DATASET+'.csv'))) 
            tweets.extend(data['tweet'])
        elif self.opt.DATASET=='wz':
            data=load_pkl('./wz_train.pkl')
            for line in data:
                tweets.append(line['tweet'])
        else:
            founta=pkl.load(open('/home/caorui/py35env/hate/split_data/founta_new.pkl','rb'))
            for i in range(5):
                info=founta[str(i)]
                for row in info:
                    tweets.append(row['sent'])
        print ('Creating for train...')
        for j,tweet in enumerate(tweets):
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
        dump_pkl('./'+self.opt.DATASET+'_dictionary.pkl',[self.word2idx,self.idx2word])
    
    def create_dict(self):
        self.word_count={}
        self.word2idx={}
        self.idx2word=[]
        self.token_sent()
    
    def create_embedding(self):
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
            
        np.save('./'+self.opt.DATASET+'_glove_embedding.npy',weights)
    
    def init_dict(self):
        if self.opt.CREATE_DICT:
            print ('Creating Dictionary...')
            self.create_dict()
        else:
            print ('Loading Dictionary...')
            created_dict=load_pkl('./'+self.opt.DATASET+'_dictionary.pkl')
            self.word2idx=created_dict[0]
            self.idx2word=created_dict[1]
            
        if self.opt.CREATE_EMB:
            print ('Creating Embedding...;')
            self.glove_weights=self.create_embedding()
        else:
            print ('Loading Embedding...')
            self.glove_weights=np.load('./'+self.opt.DATASET+'_glove_embedding.npy')
        self.ntoken()
        
    def ntoken(self):
        self.ntokens=len(self.word2idx)
        print ('Number of Tokens:',self.ntokens)
        return self.ntokens
    
    def __len__(self):
        return len(self.word2idx)
        
class Wraped_Data(Base_Op):
    def __init__(self,opt,dictionary,mode):
        super(Wraped_Data,self).__init__()
        self.opt=config.parse_opt()
        self.dictionary=dictionary
        self.mode=mode
        if opt.DATASET=='wz':
            self.entries=self.load_wz_val_entries()
        elif opt.DATASET is not 'wz':
            self.entries=self.load_tr_val_entries()
            length=len(self.entries)
            part=int(length*self.opt.TRAIN_SIZE)
            if mode=='train':
                self.entries=self.entries[:part]
            else:
                self.entries=self.entries[part:]
        self.tokenize()
        self.tensorize()
    
    def load_tr_val_entries(self):
        entries=[]
        tweets=[]
        labels=json.load(open(os.path.join(self.opt.DATA_PATH,self.opt.DATASET+'.json'),'r'))
        if self.opt.DATASET == 'dt' or self.opt.DATASET == 'dt_full':
            data=pd.read_csv(os.path.join(self.opt.DATA_PATH,(self.opt.DATASET+'.csv'))) 
            tweets.extend(data['tweet'])
        else:
            founta=pkl.load(open('/home/caorui/py35env/hate/split_data/founta_new.pkl','rb'))
            for i in range(5):
                info=founta[str(i)]
                for row in info:
                    tweets.append(row['sent'])
        for i,tweet in enumerate(tweets):
            entry={
                'tweet':tweet,
                'answer':labels[i]
            }
            entries.append(entry)
        return entries
    
    def load_wz_val_entries(self):
        entries=[]
        tweets=load_pkl('./wz_'+self.mode+'.pkl')
        for i,tweet in enumerate(tweets):
            entry={
                'tweet':tweet['tweet'],
                'answer':tweet['label']
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
    
    def tokenize(self):
        print('Tokenize Tweets...')
        length=self.opt.LENGTH
        for entry in tqdm(self.entries):
            tokens=self.dictionary.get_tokens(entry['tweet'])
            pad_tokens=self.padding_sent(tokens,length)
            entry['tokens']=np.array((pad_tokens),dtype=np.int64)
            
    def tensorize(self):
        print ('Tesnsorize all Information...')
        for entry in tqdm(self.entries):
            entry['text_tokens']=torch.from_numpy(entry['tokens'])
            target=torch.from_numpy(np.zeros((3),dtype=np.float32))
            target[entry['answer']]=1.0
            entry['label']=target                           
    
    def __getitem__(self,index):
        entry=self.entries[index]
        tweet=entry['text_tokens']
        label=entry['label']
        return tweet,label
        
    def __len__(self):
        return len(self.entries)
    
