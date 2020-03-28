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
emb_dropout = 
fc_dropout = 
num_filter =
filter_size =
epoch = 

| Model                          | Prec | Rec | F1 |
|:-------------------------------|:----:|:---:|:--:|
|LSTM (Glove)                    |      |     |    |
|LSTM (Fastext)                  |      |     |    |
|LSTM (Wiki)                     |      |     |    |
|CNN (Glove)                     |      |     |    |
|CNN (Fastext)                   |      |     |    |
|CNN (Wiki)                      |      |     |    |

Fusion Comparison
| Model                          | Prec | Rec | F1 |
|:-------------------------------|:----:|:---:|:--:|
|LSTM (3 word_emb) - gate-two    |      |     |    |
|LSTM (3 word_emb) - gate-three  |      |     |    |
|LSTM (glove,fastext) - gate     |      |     |    |
|LSTM (glove,wiki) - gate        |      |     |    |
|LSTM (fastext,wiki) - gate      |      |     |    |
|LSTM (3 word_emb,senti) - gate  |      |     |    |
|LSTM (3 word_emb,topic) - gate  |      |     |    |
|LSTM (glove,senti) - gate       |      |     |    |
|LSTM (glove,topic) - gate       |      |     |    |

Other baselines
| Model                          | Prec | Rec | F1 |
|:-------------------------------|:----:|:---:|:--:|
|CNN-W                           |      |     |    |
|CNN-C                           |      |     |    |
|CNN-B                           |      |     |    |
|LSTM-W                          |      |     |    |
|LSTM-C                          |      |     |    |
|LSTM-B                          |      |     |    |
|HybridCNN                       |      |     |    |
|CNN-GRU                         |      |     |    |


**Founta dataset**
Parameters for below results:
Num_hidden_state = 200
emb_dropout = 
fc_dropout = 
num_filter =
filter_size =
epoch = 

| Model                          | Prec | Rec | F1 |
|:-------------------------------|:----:|:---:|:--:|
|LSTM (Glove)                    |      |     |    |
|LSTM (Fastext)                  |      |     |    |
|LSTM (Wiki)                     |      |     |    |
|CNN (Glove)                     |      |     |    |
|CNN (Fastext)                   |      |     |    |
|CNN (Wiki)                      |      |     |    |

Fusion Comparison
| Model                          | Prec | Rec | F1 |
|:-------------------------------|:----:|:---:|:--:|
|LSTM (3 word_emb) - gate-two    |      |     |    |
|LSTM (3 word_emb) - gate-three  |      |     |    |
|LSTM (glove,fastext) - gate     |      |     |    |
|LSTM (glove,wiki) - gate        |      |     |    |
|LSTM (fastext,wiki) - gate      |      |     |    |
|LSTM (3 word_emb,senti) - gate  |      |     |    |
|LSTM (3 word_emb,topic) - gate  |      |     |    |
|LSTM (glove,senti) - gate       |      |     |    |
|LSTM (glove,topic) - gate       |      |     |    |

Other baselines
| Model                          | Prec | Rec | F1 |
|:-------------------------------|:----:|:---:|:--:|
|CNN-W                           |      |     |    |
|CNN-C                           |      |     |    |
|CNN-B                           |      |     |    |
|LSTM-W                          |      |     |    |
|LSTM-C                          |      |     |    |
|LSTM-B                          |      |     |    |
|HybridCNN                       |      |     |    |
|CNN-GRU                         |      |     |    |

**WZ dataset**
Parameters for below results:
Num_hidden_state = 200
emb_dropout = 
fc_dropout = 
num_filter =
filter_size =
epoch = 

| Model                          | Prec | Rec | F1 |
|:-------------------------------|:----:|:---:|:--:|
|LSTM (Glove)                    |      |     |    |
|LSTM (Fastext)                  |      |     |    |
|LSTM (Wiki)                     |      |     |    |
|CNN (Glove)                     |      |     |    |
|CNN (Fastext)                   |      |     |    |
|CNN (Wiki)                      |      |     |    |

Fusion Comparison
| Model                          | Prec | Rec | F1 |
|:-------------------------------|:----:|:---:|:--:|
|LSTM (3 word_emb) - gate-two    |      |     |    |
|LSTM (3 word_emb) - gate-three  |      |     |    |
|LSTM (glove,fastext) - gate     |      |     |    |
|LSTM (glove,wiki) - gate        |      |     |    |
|LSTM (fastext,wiki) - gate      |      |     |    |
|LSTM (3 word_emb,senti) - gate  |      |     |    |
|LSTM (3 word_emb,topic) - gate  |      |     |    |
|LSTM (glove,senti) - gate       |      |     |    |
|LSTM (glove,topic) - gate       |      |     |    |

Other baselines
| Model                          | Prec | Rec | F1 |
|:-------------------------------|:----:|:---:|:--:|
|CNN-W                           |      |     |    |
|CNN-C                           |      |     |    |
|CNN-B                           |      |     |    |
|LSTM-W                          |      |     |    |
|LSTM-C                          |      |     |    |
|LSTM-B                          |      |     |    |
|HybridCNN                       |      |     |    |
|CNN-GRU                         |      |     |    |

