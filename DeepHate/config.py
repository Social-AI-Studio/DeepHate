import argparse 

def parse_opt():
    parser=argparse.ArgumentParser()
    '''path configuration'''
    parser.add_argument('--GLOVE_PATH',type=str,default='/home/caorui/glove.6B.300d.txt')
    #path for pre-precessing and result saving
    parser.add_argument('--OFFENSIVE_RESULT',type=str,default='./offensive/result')
    parser.add_argument('--OFFENSIVE_DATA',type=str,default='./offensive/dictionary')
    parser.add_argument('--OFFENSIVE_FULL_RESULT',type=str,default='./offensive_full/result')
    parser.add_argument('--OFFENSIVE_FULL_DATA',type=str,default='./offensive_full/dictionary')
    parser.add_argument('--WZ_RESULT',type=str,default='./wz/result')
    parser.add_argument('--WZ_DATA',type=str,default='./wz/dictionary')
    parser.add_argument('--FOUNTA_RESULT',type=str,default='./founta/result')
    parser.add_argument('--FOUNTA_DATA',type=str,default='./founta/dictionary')
    parser.add_argument('--TOTAL_RESULT',type=str,default='./total/result')
    parser.add_argument('--TOTAL_DATA',type=str,default='./total/dictionary')
    #path for the split dataset
    parser.add_argument('--SPLIT_DATASET',type=str,default='/home/caorui/Reinforceenv/hate/split_data')
    parser.add_argument('--FAST_TEXT',type=str,default='/home/caorui/Reinforceenv/hate/wiki-news-300d-1M.vec')
    parser.add_argument('--PARA_TEXT',type=str,default='/home/caorui/Reinforceenv/hate/paragram_300_sl999.txt')
    parser.add_argument('--SENT_EMB',type=str,default='/home/caorui/Reinforceenv/hate/Sentiment-Analysis-with-Word-Embeddings')
    
    
    '''hyper parameters configuration'''
    parser.add_argument('--EMB_DROPOUT',type=float,default=0.5)
    parser.add_argument('--FC_DROPOUT',type=float,default=0.2) 
    parser.add_argument('--CNN_DROPOUT',type=float,default=0.0) 
    parser.add_argument('--MIN_OCC',type=int,default=3)
    parser.add_argument('--TEST_NUM',type=int,default=0)
    parser.add_argument('--BATCH_SIZE',type=int,default=128)
    parser.add_argument('--EMB_DIM',type=int,default=300)
    parser.add_argument('--MID_DIM',type=int,default=128)
    parser.add_argument('--PROJ_DIM',type=int,default=32)
    parser.add_argument('--NUM_HIDDEN',type=int,default=200)
    parser.add_argument('--NUM_LAYER',type=int,default=1)
    parser.add_argument('--NUM_FILTER',type=int,default=150)
    parser.add_argument('--FILTER_SIZE',type=str,default="1,2,3")
    parser.add_argument('--BIDIRECT',type=bool,default=False)
    parser.add_argument('--L_RNN_DROPOUT',type=float,default=0.3)
    parser.add_argument('--ATT_DROPOUT',type=float,default=0.4)  
    
    '''
    wz for WZ
    fouta for FOUNTA
    dt_full for DT
    total for TOTAL
    '''
    parser.add_argument('--DATASET',type=str,default='dt_full')
    parser.add_argument('--LENGTH',type=int,default=30)
    
    parser.add_argument('--CREATE_DICT',type=bool,default=False)
    parser.add_argument('--CREATE_EMB',type=bool,default=False)
    parser.add_argument('--SAVE_NUM',type=int,default=0)
    parser.add_argument('--EPOCHS',type=int,default=3)
    parser.add_argument('--CROSS_VAL',type=int,default=5)
    
    parser.add_argument('--SEED', type=int, default=1111, help='random seed')
    parser.add_argument('--CUDA_DEVICE', type=int, default=0)
    
    #for topic modeling
    parser.add_argument('--NUM_TOPICS', type=int, default=15)
    args=parser.parse_args()
    return args
