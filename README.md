The code implementation for the paper "DeepHate: Hate Speech Detection via Multi-Faceted Text Representations (WebSci'20)".

To cite:
```
@inproceedings{deephate20,
    title={DeepHate: Hate Speech Detection via Multi-Faceted Text Representations,
    author={Rui, Cao, Roy Ka-Wei, Lee and Tuan-Anh, Hoang},
    booktitle={ACM Web Science Conference},
    month={July}
    year={2020}
    publisher={ACM}
}
```

**DT dataset**
Parameters for below results:  
Num_hidden_state = 200  
emb_dropout = 0.5  
fc_dropout = 0.2  
num_filter = 50 for each kind of filter size  
filter_size = 1,2,3   
length of tweets = 30  
epoch = 7  

Without Attention  

| Model                          | Prec | Rec | F1 |
|:-------------------------------|:----:|:---:|:--:|
|LSTM (Glove)                    | 88.93      | 89.86     | 88.83    |
|LSTM (Fastext)                  | 89.51      | 90.42     | 89.53    |
|LSTM (Wiki)                     | 89.34      | 90.27     | 89.35    |
|CNN (Glove)                     | 89.13      | 89.81     | 89.22    |
|CNN (Fastext)                   | 89.25      | 90.30     | 89.42    |
|CNN (Wiki)                      | 89.66      | 90.62     | 89.62    |

Fusion Comparison

| Model                          | Prec | Rec | F1 |
|:-------------------------------|:----:|:---:|:--:|
|LSTM (3 word_emb) - gate-two    | 90.10      | 90.60     | 90.10    |
|LSTM (3 word_emb) - gate-three  | 89.75      | 90.61     | 89.88    |
|LSTM (3 word_emb) - concate     | 89.42      | 90.42     | 89.45    |
|LSTM (glove,fastext) - gate     | 89.79      | 90.58     | 89.84    |
|LSTM (glove,fastext) - concate  | 90.02      | 90.67     | 89.94    |
|LSTM (glove,wiki) - gate        | 89.42      | 90.13     | 89.42    |
|LSTM (glove,wiki) - concate     | 89.04      | 90.03     | 89.12    |
|LSTM (fastext,wiki) - gate      | 89.83      | 90.55     | 89.85    |
|LSTM (fastext,wiki) - concate   | 90.06      | 90.66     | 90.05    |
|LSTM (3 word_emb,senti) - gate  | 89.68      | 90.46     | 89.66    |
|LSTM (3 word_emb,senti) - concate| 89.66     | 90.46     | 89.75    |
|LSTM (3 word_emb,topic) - gate  | 89.37      | 90.25     | 89.49    |
|LSTM (3 word_emb,topic) - concate| 89.37     | 90.25     | 89.49    |
|LSTM (3 word_emb,senti,topic) - gate  |89.57      |90.50           |89.74          |
|LSTM (3 word_emb,senti,topic) - concate|89.68     |90.55           |89.83          |
|LSTM (glove,senti) - gate       | 89.35      | 90.05     | 89.07    |
|LSTM (glove,senti) - concate    | 88.93      | 89.96     | 89.01    |
|LSTM (glove,topic) - gate       | 89.69      | 90.50     | 89.72    |
|LSTM (glove,topic) - concate    | 88.65      | 89.66     | 88.81    |
|LSTM (glove,senti,topic) - gate  |89.01           |89.98           |89.14          |
|LSTM (glove,senti,topic) - concate|89.09          |90.15           |89.16          |
|CNN (3 word_emb) - gate-two    |            |           |          |
|CNN (3 word_emb) - gate-three  |89.59            |90.46           |89.64          |
|CNN (3 word_emb) - concate     |89.59            |90.40           |89.68          |
|CNN (glove,fastext) - gate     |89.39            |90.41           |89.52          |
|CNN (glove,fastext) - concate  |89.32            |90.07           |89.31          |
|CNN (glove,wiki) - gate        |89.47            |90.40           |89.51          |
|CNN (glove,wiki) - concate     |89.55            |90.30           |89.45          |
|CNN (fastext,wiki) - gate      |            |           |          |
|CNN (fastext,wiki) - concate   |            |           |          |
|CNN (3 word_emb,senti) - gate  |            |           |          |
|CNN (3 word_emb,senti) - concate|            |           |          |
|CNN (3 word_emb,topic) - gate  |            |           |          |
|CNN (3 word_emb,topic) - concate|            |           |          |
|CNN (3 word_emb,senti,topic) - gate  |89.60      |90.50           |89.77          |
|CNN (3 word_emb,senti,topic) - concate|89.35     |90.22           |89.47          |
|CNN (glove,senti) - gate       |            |           |          |
|CNN (glove,senti) - concate    |            |           |          |
|CNN (glove,topic) - gate       |            |           |          |
|CNN (glove,topic) - concate    |            |           |          |
|CNN (glove,senti,topic) - gate  |           |           |          |
|CNN (glove,senti,topic) - concate|88.84           |89.80           |88.97          |
|CNN-LSTM (3 word_emb) - gate-two    |            |           |          |
|CNN-LSTM (3 word_emb) - gate-three  |            |           |          |
|CNN-LSTM (3 word_emb) - concate     |            |           |          |
|CNN-LSTM (glove,fastext) - gate     |            |           |          |
|CNN-LSTM (glove,fastext) - concate  |            |           |          |
|CNN-LSTM (glove,wiki) - gate        |            |           |          |
|CNN-LSTM (glove,wiki) - concate     |            |           |          |
|CNN-LSTM (fastext,wiki) - gate      |            |           |          |
|CNN-LSTM (fastext,wiki) - concate   |            |           |          |
|CNN-LSTM (3 word_emb,senti) - gate  |            |           |          |
|CNN-LSTM (3 word_emb,senti) - concate|            |           |          |
|CNN-LSTM (3 word_emb,topic) - gate  |            |           |          |
|CNN-LSTM (3 word_emb,topic) - concate|            |           |          |
|CNN-LSTM (3 word_emb,senti,topic) - gate  |      |           |          |
|CNN-LSTM (3 word_emb,senti,topic) - concate|     |           |          |
|CNN-LSTM (glove,senti) - gate       |            |           |          |
|CNN-LSTM (glove,senti) - concate    |            |           |          |
|CNN-LSTM (glove,topic) - gate       |            |           |          |
|CNN-LSTM (glove,topic) - concate    |            |           |          |
|CNN-LSTM (glove,senti,topic) - gate  |           |           |          |
|CNN-LSTM (glove,senti,topic) - concate|          |           |          |

Other baselines

| Model                          | Prec | Rec | F1 |
|:-------------------------------|:----:|:---:|:--:|
|CNN-W                           | 89.13      | 89.81     | 89.22    |
|CNN-C                           | 60.53      | 77.43     | 67.60    |
|CNN-B                           | 78.02      | 80.33     | 77.01    |
|LSTM-W                          | 88.93      | 89.86     | 88.83    |
|LSTM-C                          | 77.21      | 79.88     | 76.47    |
|LSTM-B                          | 59.97      | 77.44     | 67.60    |
|HybridCNN                       | 89.02      | 89.68     | 88.88    |
|CNN-GRU                         | 88.98      | 89.91     | 89.09    |


**Founta dataset**
Parameters for below results:  
Num_hidden_state = 200  
emb_dropout = 0.5  
fc_dropout = 0.2  
num_filter = 50 for each kind of filter size  
filter_size = 1,2,3   
length of tweets = 30  
epoch = 10    

Without Attention    


| Model                          | Prec | Rec | F1 |
|:-------------------------------|:----:|:---:|:--:|
|LSTM (Glove)                    | 80.12      | 81.20     | 80.29    |
|LSTM (Fastext)                  | 80.10      | 81.40     | 80.36    |
|LSTM (Wiki)                     | 79.29      | 80.56     | 79.57    |
|CNN (Glove)                     | 80.06      | 80.59     | 80.07    |
|CNN (Fastext)                   | 80.24      | 81.06     | 80.34    |
|CNN (Wiki)                      | 79.64      | 80.23     | 79.66    |

Fusion Comparison

| Model                          | Prec | Rec | F1 |
|:-------------------------------|:----:|:---:|:--:|
|LSTM (3 word_emb) - gate-two    | 79.32      | 80.33     | 79.61    |
|LSTM (3 word_emb) - gate-three  | 79.46      | 80.44     | 79.68    |
|LSTM (3 word_emb) - concate     | 79.13      | 80.13     | 79.50    |
|LSTM (glove,fastext) - gate     | 80.05      | 80.98     | 80.30    |
|LSTM (glove,fastext) - concate  | 80.10      | 81.26     | 80.39    |
|LSTM (glove,wiki) - gate        | 79.28      | 80.44     | 79.57    |
|LSTM (glove,wiki) - concate     | 79.34      | 80.45     | 79.66    |
|LSTM (fastext,wiki) - gate      | 79.28      | 80.45     | 79.57    |
|LSTM (fastext,wiki) - concate   | 79.25      | 80.67     | 79.59    |
|LSTM (3 word_emb,senti) - gate  | 79.28      | 80.55     | 79.60    |
|LSTM (3 word_emb,senti) - concate| 79.85     | 81.12     | 79.88    |
|LSTM (3 word_emb,topic) - gate  | 79.37      | 80.39     | 79.68    |
|LSTM (3 word_emb,topic) - concate| 79.37     | 80.39     | 79.68    |
|LSTM (3 word_emb,senti,topic) - gate  |79.60      |80.55           |79.78          |
|LSTM (3 word_emb,senti,topic) - concate|79.22     |80.39           |79.51          |
|LSTM (glove,senti) - gate       | 80.14      | 81.20     | 80.34    |
|LSTM (glove,senti) - concate    | 80.01      | 80.97     | 80.16    |
|LSTM (glove,topic) - gate       | 79.36      | 80.54     | 79.65    |
|LSTM (glove,topic) - concate    | 80.15      | 81.21     | 80.32    |
|LSTM (glove,senti,topic) - gate  |80.15           |81.16           |80.22          |
|LSTM (glove,senti,topic) - concate|79.98          |80.75           |80.05          |
|CNN (3 word_emb) - gate-two    |            |           |          |
|CNN (3 word_emb) - gate-three  |79.36            |79.88           |79.37          |
|CNN (3 word_emb) - concate     |79.32            |79.93           |79.37          |
|CNN (glove,fastext) - gate     |79.82            |80.39           |79.79          |
|CNN (glove,fastext) - concate  |80.03            |80.63           |80.11          |
|CNN (glove,wiki) - gate        |79.44            |80.06           |79.49          |
|CNN (glove,wiki) - concate     |79.62            |80.17           |79.62          |
|CNN (fastext,wiki) - gate      |            |           |          |
|CNN (fastext,wiki) - concate   |            |           |          |
|CNN (3 word_emb,senti) - gate  |            |           |          |
|CNN (3 word_emb,senti) - concate|            |           |          |
|CNN (3 word_emb,topic) - gate  |            |           |          |
|CNN (3 word_emb,topic) - concate|            |           |          |
|CNN (3 word_emb,senti,topic) - gate  |79.30      |79.80           |79.16          |
|CNN (3 word_emb,senti,topic) - concate|79.23     |80.07           |79.44          |
|CNN (glove,senti) - gate       |            |           |          |
|CNN (glove,senti) - concate    |            |           |          |
|CNN (glove,topic) - gate       |            |           |          |
|CNN (glove,topic) - concate    |            |           |          |
|CNN (glove,senti,topic) - gate  |           |           |          |
|CNN (glove,senti,topic) - concate|79.95           |80.55           |79.91          |
|CNN-LSTM (3 word_emb) - gate-two    |            |           |          |
|CNN-LSTM (3 word_emb) - gate-three  |            |           |          |
|CNN-LSTM (3 word_emb) - concate     |            |           |          |
|CNN-LSTM (glove,fastext) - gate     |            |           |          |
|CNN-LSTM (glove,fastext) - concate  |            |           |          |
|CNN-LSTM (glove,wiki) - gate        |            |           |          |
|CNN-LSTM (glove,wiki) - concate     |            |           |          |
|CNN-LSTM (fastext,wiki) - gate      |            |           |          |
|CNN-LSTM (fastext,wiki) - concate   |            |           |          |
|CNN-LSTM (3 word_emb,senti) - gate  |            |           |          |
|CNN-LSTM (3 word_emb,senti) - concate|            |           |          |
|CNN-LSTM (3 word_emb,topic) - gate  |            |           |          |
|CNN-LSTM (3 word_emb,topic) - concate|            |           |          |
|CNN-LSTM (3 word_emb,senti,topic) - gate  |      |           |          |
|CNN-LSTM (3 word_emb,senti,topic) - concate|     |           |          |
|CNN-LSTM (glove,senti) - gate       |            |           |          |
|CNN-LSTM (glove,senti) - concate    |            |           |          |
|CNN-LSTM (glove,topic) - gate       |            |           |          |
|CNN-LSTM (glove,topic) - concate    |            |           |          |
|CNN-LSTM (glove,senti,topic) - gate  |           |           |          |
|CNN-LSTM (glove,senti,topic) - concate|          |           |          |

Other baselines

| Model                          | Prec | Rec | F1 |
|:-------------------------------|:----:|:---:|:--:|
|CNN-W                           | 80.06      | 80.59     | 80.07    |
|CNN-C                           | 55.77      | 60.64     | 54.15    |
|CNN-B                           | 69.85      | 70.50     | 66.08    |
|LSTM-W                          | 80.12      | 81.20     | 80.29    |
|LSTM-C                          | 70.80      | 71.57     | 67.95    |
|LSTM-B                          | 54.33      | 61.02     | 54.02    |
|HybridCNN                       | 79.97      | 80.77     | 79.98    |
|CNN-GRU                         | 79.91      | 80.51     | 79.96    |

**WZ dataset**
Parameters for below results:  
Num_hidden_state = 200  
emb_dropout = 0.5  
fc_dropout = 0.2  
num_filter = 50 for each kind of filter size  
filter_size = 1,2,3   
length of tweets = 30  
epoch = 7  

Without Attention    

| Model                          | Prec | Rec | F1 |
|:-------------------------------|:----:|:---:|:--:|
|LSTM (Glove)                    | 77.77      | 80.10     | 77.62    |
|LSTM (Fastext)                  | 75.85      | 77.81     | 76.14    |
|LSTM (Wiki)                     | 78.49      | 79.87     | 78.49    |
|CNN (Glove)                     | 78.08      | 79.67     | 77.50    |
|CNN (Fastext)                   | 76.66      | 78.32     | 76.68    |
|CNN (Wiki)                      | 79.59      | 79.99     | 78.73    |

Fusion Comparison

| Model                          | Prec | Rec | F1 |
|:-------------------------------|:----:|:---:|:--:|
|LSTM (3 word_emb) - gate-two    | 78.05      | 79.19     | 77.77    |
|LSTM (3 word_emb) - gate-three  | 77.85      | 78.90     | 77.61    |
|LSTM (3 word_emb) - concate     | 77.83      | 78.86     | 77.89    |
|LSTM (glove,fastext) - gate     | 77.08      | 79.10     | 76.55    |
|LSTM (glove,fastext) - concate  | 77.05      | 78.52     | 76.64    |
|LSTM (glove,wiki) - gate        | 78.53      | 79.05     | 77.45    |
|LSTM (glove,wiki) - concate     | 78.70      | 79.67     | 77.61    |
|LSTM (fastext,wiki) - gate      | 76.97      | 77.76     | 75.00    |
|LSTM (fastext,wiki) - concate   | 77.82      | 78.93     | 77.34    |
|LSTM (3 word_emb,senti) - gate  | 77.30      | 79.23     | 77.41    |
|LSTM (3 word_emb,senti) - concate| 77.93     | 79.37     | 77.77    |
|LSTM (3 word_emb,topic) - gate  | 77.28      | 79.23     | 76.99    |
|LSTM (3 word_emb,topic) - concate| 77.28     | 79.23     | 76.99    |
|LSTM (3 word_emb,senti,topic) - gate  |77.97      |78.83           |77.49          |
|LSTM (3 word_emb,senti,topic) - concate|77.98          |79.38           |77.91          |
|LSTM (glove,senti) - gate       | 77.50      | 79.85     | 76.93    |
|LSTM (glove,senti) - concate    | 77.57      | 79.63     | 76.96    |
|LSTM (glove,topic) - gate       | 77.28      | 79.28     | 77.33    |
|LSTM (glove,topic) - concate    | 77.63      | 79.67     | 76.55    |
|LSTM (glove,senti,topic) - gate  |77.48           |79.08           |76.95          |
|LSTM (glove,senti,topic) - concate|77.57          |80.30           |76.87          |
|CNN (3 word_emb) - gate-two    |            |           |          |
|CNN (3 word_emb) - gate-three  |78.51            |79.61           |78.35          |
|CNN (3 word_emb) - concate     |78.40            |80.04           |78.24          |
|CNN (glove,fastext) - gate     |76.82            |78.83           |76.18          |
|CNN (glove,fastext) - concate  |77.21            |79.27           |77.11          |
|CNN (glove,wiki) - gate        |79.49            |79.89           |78.48          |
|CNN (glove,wiki) - concate     |79.48            |80.25           |78.95          |
|CNN (fastext,wiki) - gate      |            |           |          |
|CNN (fastext,wiki) - concate   |            |           |          |
|CNN (3 word_emb,senti) - gate  |            |           |          |
|CNN (3 word_emb,senti) - concate|            |           |          |
|CNN (3 word_emb,topic) - gate  |            |           |          |
|CNN (3 word_emb,topic) - concate|            |           |          |
|CNN (3 word_emb,senti,topic) - gate  |78.10      |79.76           |78.22          |
|CNN (3 word_emb,senti,topic) - concate|78.44     |79.87           |77.59          |
|CNN (glove,senti) - gate       |            |           |          |
|CNN (glove,senti) - concate    |            |           |          |
|CNN (glove,topic) - gate       |            |           |          |
|CNN (glove,topic) - concate    |            |           |          |
|CNN (glove,senti,topic) - gate  |           |           |          |
|CNN (glove,senti,topic) - concate|77.66           |79.34           |76.79          |
|CNN-LSTM (3 word_emb) - gate-two    |            |           |          |
|CNN-LSTM (3 word_emb) - gate-three  |            |           |          |
|CNN-LSTM (3 word_emb) - concate     |            |           |          |
|CNN-LSTM (glove,fastext) - gate     |            |           |          |
|CNN-LSTM (glove,fastext) - concate  |            |           |          |
|CNN-LSTM (glove,wiki) - gate        |            |           |          |
|CNN-LSTM (glove,wiki) - concate     |            |           |          |
|CNN-LSTM (fastext,wiki) - gate      |            |           |          |
|CNN-LSTM (fastext,wiki) - concate   |            |           |          |
|CNN-LSTM (3 word_emb,senti) - gate  |            |           |          |
|CNN-LSTM (3 word_emb,senti) - concate|            |           |          |
|CNN-LSTM (3 word_emb,topic) - gate  |            |           |          |
|CNN-LSTM (3 word_emb,topic) - concate|            |           |          |
|CNN-LSTM (3 word_emb,senti,topic) - gate  |      |           |          |
|CNN-LSTM (3 word_emb,senti,topic) - concate|     |           |          |
|CNN-LSTM (glove,senti) - gate       |            |           |          |
|CNN-LSTM (glove,senti) - concate    |            |           |          |
|CNN-LSTM (glove,topic) - gate       |            |           |          |
|CNN-LSTM (glove,topic) - concate    |            |           |          |
|CNN-LSTM (glove,senti,topic) - gate  |           |           |          |
|CNN-LSTM (glove,senti,topic) - concate|          |           |          |

Other baselines

| Model                          | Prec | Rec | F1 |
|:-------------------------------|:----:|:---:|:--:|
|CNN-W                           | 78.08      | 79.67     | 77.50    |
|CNN-C                           | 54.77      | 74.01     | 62.95    |
|CNN-B                           | 76.30      | 79.08     | 74.78    |
|LSTM-W                          | 77.77      | 80.10     | 77.62    |
|LSTM-C                          | 74.82      | 78.13     | 71.95    |
|LSTM-B                          | 54.77      | 74.01     | 62.95    |
|HybridCNN                       | 77.97      | 79.73     | 77.53    |
|CNN-GRU                         | 75.58      | 79.70     | 74.98    |

