import pandas as pd
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle as pkl

if __name__ == '__main__':
    '''dt=pd.read_csv('/home/caorui/disk_data/videoQA/hate_data/dt.csv')
    dt_full=pd.read_csv('/home/caorui/disk_data/videoQA/hate_data/dt_full.csv')'''
    #founta=pd.read_csv('/home/caorui/disk_data/videoQA/hate_data/founta.csv',sep='\t')
    founta=pkl.load(open('/home/caorui/py35env/hate/split_data/founta_new.pkl','rb'))
    analyser=SentimentIntensityAnalyzer()
    
    #for dt or dt_full
    '''total_data=dt_full
    tweets=total_data['tweet']'''
    tweets=[]
    for i in range(5):
        info=founta[str(i)]
        for row in info:
            tweets.append(row['sent'])
    whole_info=[]
    for i,tweet in enumerate(tweets):
        if i%100==0:
            print ('Number',i,'tweet')
        senti_info=analyser.polarity_scores(tweet)
        label=0
        score=senti_info['neg']
        if score<senti_info['neu']:
            label=1
            score=senti_info['neu']
        if score<senti_info['pos']:
            label=2
        whole_info.append(label)
        #print (label)
    print ('length of dataset is',len(whole_info))
    json.dump(whole_info,open('./founta.json','w'))