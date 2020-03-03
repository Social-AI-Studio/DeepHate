import argparse 

def parse_opt():
    parser=argparse.ArgumentParser()
    
    parser.add_argument('--MODEL',type=str,default='basic')
    
    '''path configuration'''
    parser.add_argument('--GLOVE_PATH',type=str,default='/home/caorui/disk_data/videoQA/glove.6B.300d.txt')
    #path for pre-precessing and result saving
    parser.add_argument('--OFFENSIVE_RESULT',type=str,default='./offensive/result')
    parser.add_argument('--OFFENSIVE_DATA',type=str,default='./offensive/dictionary')
    parser.add_argument('--OFFENSIVE_FULL_RESULT',type=str,default='./offensive_full/result')
    parser.add_argument('--OFFENSIVE_FULL_DATA',type=str,default='./offensive_full/dictionary')
    parser.add_argument('--WZ_RESULT',type=str,default='./wz/result')
    parser.add_argument('--WZ_DATA',type=str,default='./wz/dictionary')
    parser.add_argument('--FOUNTA_RESULT',type=str,default='./founta/result')
    parser.add_argument('--FOUNTA_DATA',type=str,default='./founta/dictionary')
    #path for the split dataset
    parser.add_argument('--SPLIT_DATASET',type=str,default='/home/caorui/py35env/hate/split_data')
    parser.add_argument('--FAST_TEXT',type=str,default='/home/caorui/py35env/hate/wiki-news-300d-1M.vec')
    parser.add_argument('--PARA_TEXT',type=str,default='/home/caorui/py35env/hate/paragram_300_sl999.txt')
    parser.add_argument('--SENT_EMB',type=str,default='/home/caorui/py35env/hate/Sentiment-Analysis-with-Word-Embeddings')
    
    
    '''hyper parameters configuration'''
    parser.add_argument('--EMB_DROPOUT',type=float,default=0.5)
    parser.add_argument('--FC_DROPOUT',type=float,default=0.0) 
    parser.add_argument('--CNN_DROPOUT',type=float,default=0.0) 
    parser.add_argument('--MIN_OCC',type=int,default=2)
    parser.add_argument('--TEST_NUM',type=int,default=0)
    parser.add_argument('--BATCH_SIZE',type=int,default=128)
    parser.add_argument('--EMB_DIM',type=int,default=300)
    parser.add_argument('--NUM_CHAR',type=int,default=30)
    parser.add_argument('--MID_DIM',type=int,default=128)
    parser.add_argument('--PROJ_DIM',type=int,default=128)
    parser.add_argument('--NUM_HIDDEN',type=int,default=256)
    parser.add_argument('--NUM_LAYER',type=int,default=1)
    parser.add_argument('--NUM_FILTER',type=int,default=150)
    parser.add_argument('--FILTER_SIZE',type=str,default="3")
    parser.add_argument('--BIDIRECT',type=bool,default=False)
    parser.add_argument('--L_RNN_DROPOUT',type=float,default=0.3)
    parser.add_argument('--ATT_DROPOUT',type=float,default=0.3)  
    
    '''dt is the DT dataset
    fouta for founta
    dt_full for DT-Full'''
    parser.add_argument('--DATASET',type=str,default='wz')
    parser.add_argument('--CHAR_VOCB',type=int,default=26)
    parser.add_argument('--LENGTH',type=int,default=14)
    parser.add_argument('--NUM_CHARACTER',type=int,default=50)
    
    parser.add_argument('--CREATE_DICT',type=bool,default=False)
    parser.add_argument('--CREATE_EMB',type=bool,default=False)
    parser.add_argument('--SAVE_NUM',type=int,default=0)
    parser.add_argument('--EPOCHS',type=int,default=15)
    parser.add_argument('--CROSS_VAL',type=int,default=5)
    
    parser.add_argument('--SEED', type=int, default=1111, help='random seed')
    parser.add_argument('--CUDA_DEVICE', type=int, default=0)
    
    parser.add_argument('--FOUNTA_TIMES',type=int,default=8)
    parser.add_argument('--OFFENSIVE_TIMES',type=int,default=9)
    parser.add_argument('--FULL_TIMES',type=int,default=2)
    
    parser.add_argument('--NUM_ITER', type=int, default=3)
    parser.add_argument('--NUM_ROUTINE', type=int, default=3)
    
    parser.add_argument('--SAMPLE_RATE',type=float,default=0.8) 
    
    #for topic modeling
    parser.add_argument('--NUM_TOPICS', type=int, default=15)
    parser.add_argument('--MIN_NUM_WORDS', type=int, default=3)
    args=parser.parse_args()
    return args
