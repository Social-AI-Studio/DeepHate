# DeepHate: Hate Speech Detection via Multi-Faceted Text Representations  
The code implementation for the paper "DeepHate: Hate Speech Detection via Multi-Faceted Text Representations (WebSci'20)".  

## Dependencies:  

- Python 3.5
- Pytorch 1.0.0

## Prerequisites  
In order to use our model for hate speech detection:

- You need to use sentiment analysis tool **VADER** to label the sentiment plority of each tweet  
- Use the network provided in the file named **sentiment** to train a network for sentiment classification  
- Extract the embedding layer of the sentiment classification network as the sentiment embedding  
- Use the codes in the file of **topic_modeling** to obtain topic vectors

## Data  

- For each dataset, we split the whole dataset into five folders in advance making sure each share almost equal number of tweets from each label
- You can find data for datasets in the file of **split_data**

## Getting Started

- The training and testing process have been included in the **main** file, to start:  

``` python main.py ```
    
- The configuration are set as default in the **config.py**

## Acknowledgements  
- [VADER](https://github.com/cjhutto/vaderSentiment): the lexicon-based tool for sentiment analysis
To cite:
```
@inproceedings{deephate20,
    title={DeepHate: Hate Speech Detection via Multi-Faceted Text Representations,
    author={Cao, Rui and Lee, Roy Ka-Wei and Hoang, Tuan-Anh},
    booktitle={Proceedings of the 11th ACM Conference on Web Science},
    month={July}
    year={2020}
    publisher={ACM}
}
```