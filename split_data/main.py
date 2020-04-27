import pickle as pkl

def get_info(list_info):
    count_dict={'0':0,'1':0,'2':0,'3':0}
    for entity in list_info:
        label=entity['label']
        count_dict[str(label)]+=1
    print (count_dict)

if __name__ == '__main__':
    dt=pkl.load(open('./dt_full_new.pkl','rb'))
    wz=pkl.load(open('./wz_new.pkl','rb'))
    founta=pkl.load(open('./founta_new1.pkl','rb'))
    
    #get data distribution
    print ('Getting data distribution for wz...')
    get_info(wz['0'])
    print ('Getting data distribution for dt...')
    get_info(dt['0'])
    print ('Getting data distribution for founta...')
    get_info(founta['0'])
    
    """
    data distribution:
    WZ: 0-racisim, 1-sexism, 2-neither, 3-both
    DT: 0-hate, 1-neither, 2-offensive
    FOUNTA: 0-hate, 1-abusive, 2-spam, 3-normal
    TOTAL: 0-hate, 1-abusive, 2-normal
    """
    
    total_list={}
    for i in range(5):
        entries=[]
        one_wz=wz[str(i)]
        one_dt=dt[str(i)]
        one_founta=founta[str(i)]
        count_dict={'0':0,'1':0,'2':0}
        for tweet_info in one_wz:
            label=tweet_info['label']
            sent=tweet_info['sent']
            topic=tweet_info['topic']
            if label==0 or label==1 or label==3 :
                entry={
                    'answer':0,
                    'sent':sent,
                    'prob':topic,
                    'dataset':'wz'
                }
                count_dict['0']+=1
                entries.append(entry)
            else:
                entry={
                    'answer':2,
                    'sent':sent,
                    'prob':topic,
                    'dataset':'wz'
                }
                count_dict['2']+=1
                entries.append(entry)
        print ('Information for folder of wz is:',count_dict)
        count_dict={'0':0,'1':0,'2':0}
        for tweet_info in one_dt:
            label=tweet_info['label']
            sent=tweet_info['sent']
            topic=tweet_info['topic']
            if label==0:
                entry={
                    'answer':0,
                    'sent':sent,
                    'prob':topic,
                    'dataset':'dt'
                }
                count_dict['0']+=1
                entries.append(entry)
            elif label==1:
                entry={
                    'answer':2,
                    'sent':sent,
                    'prob':topic,
                    'dataset':'dt'
                }
                count_dict['2']+=1
                entries.append(entry)
            elif label==2:
                entry={
                    'answer':1,
                    'sent':sent,
                    'prob':topic,
                    'dataset':'dt'
                }
                count_dict['1']+=1
                entries.append(entry)
        print ('Information for folder of dt is:',count_dict)
        count_dict={'0':0,'1':0,'2':0}
        for tweet_info in one_founta:
            label=tweet_info['label']
            sent=tweet_info['sent']
            topic=tweet_info['topic']
            if label==0:
                entry={
                    'answer':0,
                    'sent':sent,
                    'prob':topic,
                    'dataset':'founta'
                }
                count_dict['0']+=1
                entries.append(entry)
            elif label==3:
                entry={
                    'answer':2,
                    'sent':sent,
                    'prob':topic,
                    'dataset':'founta'
                }
                count_dict['2']+=1
                entries.append(entry)
            elif label==1:
                entry={
                    'answer':1,
                    'sent':sent,
                    'prob':topic,
                    'dataset':'founta'
                }
                count_dict['1']+=1
                entries.append(entry)
        print ('Information for folder of founta is:',count_dict)
        print ('The length of one folder is:',len(entries))
        total_list[str(i)]=entries
    pkl.dump(total_list,open('merge_data.pkl','wb'))