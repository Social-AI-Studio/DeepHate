import argparse 

def parse_opt():
    parser=argparse.ArgumentParser()
    parser.add_argument('--GLOVE_PATH',type=str,default='/home/caorui/disk_data/videoQA/glove.6B.300d.txt')
    parser.add_argument('--DATASET_ENCODING',type=str,default="ISO-8859-1")
    parser.add_argument('--TRAIN_SIZE',type=float,default=0.8)
    parser.add_argument('--DATA_PATH',type=str,default='/home/caorui/py35env/hate/Sentiment-Analysis-with-Word-Embeddings/hate_data')
    parser.add_argument('--SENT_LEN',type=int,default=50)
    parser.add_argument('--EMB_DIM',type=int,default=300)
    parser.add_argument('--EMB_DROPOUT',type=float,default=0.5)
    parser.add_argument('--FC_DROPOUT',type=float,default=0.0)  
    parser.add_argument('--MIN_OCC',type=int,default=2)
    parser.add_argument('--BATCH_SIZE',type=int,default=128)
    parser.add_argument('--MID_DIM',type=int,default=256)
    parser.add_argument('--PROJ_DIM',type=int,default=256)
    parser.add_argument('--NUM_HIDDEN',type=int,default=100)
    parser.add_argument('--NUM_LAYER',type=int,default=1)
    parser.add_argument('--NUM_FILTER',type=int,default=5)
    parser.add_argument('--BIDIRECT',type=bool,default=True)
    parser.add_argument('--L_RNN_DROPOUT',type=float,default=0.3)
    parser.add_argument('--ATT_DROPOUT',type=float,default=0.3)  
    parser.add_argument('--EPOCHS',type=int,default=8)
    
    parser.add_argument('--CUDA_DEVICE', type=int, default=0)
    parser.add_argument('--SEED', type=int, default=1111, help='random seed')
    parser.add_argument('--SAVE_NUM',type=int,default=0)
    
    parser.add_argument('--CREATE_DICT',type=bool,default=False)
    parser.add_argument('--CREATE_EMB',type=bool,default=False)
    parser.add_argument('--DATASET',type=str,default='wz')
    
    parser.add_argument('--LENGTH',type=int,default=30)
    
    args=parser.parse_args()
    return args