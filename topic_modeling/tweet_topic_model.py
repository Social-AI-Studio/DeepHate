import pandas as pd
import sys
from nltk.corpus import stopwords
import tweet_tokenize
from sparse_lda import *


def read_data(dataset_name, dataset_path):
    tweets = []
    if dataset_name == 'founta':
        data_file = open(dataset_path, 'r')
        for line in data_file:
            tweet = dict()
            fields = line.strip().split('\t')
            tweet['raw_text'] = fields[0]
            tweet['tokens'] = [w.lower() for w in tweet_tokenize.tokenizeRawTweetText(fields[0])]
            tweet['label'] = fields[1]
            tweets.append(tweet)
        data_file.close()
    elif dataset_name == 'dt':
        df = pd.read_csv(dataset_path)
        for _, row in df.iterrows():
            tweet = dict()
            tweet['raw_text'] = row['tweet']
            tweet['tokens'] = [w.lower() for w in tweet_tokenize.tokenizeRawTweetText(row['tweet'])]
            tweet['label'] = row['class']
            tweets.append(tweet)
    else:
        print('not defined!')
        sys.exit(1)
    print('read in %d tweets' % (len(tweets)))
    return tweets


def contains_letter(token):
    for c in token:
        if c.isalpha():
            return True
    return False


def filter_tweets_words(tweets, min_num_words, min_tweet_frequency):
    removed_words = set(stopwords.words('english'))
    removed_words.add('UNK')
    removed_words.add('unk')
    tweet_frequency = dict()
    for tweet in tweets:
        for w in tweet['tokens']:
            if w in removed_words:
                continue
            elif w in tweet_frequency:
                tweet_frequency[w] += 1
            elif w.startswith('http:') or w.startswith('https:'):
                w.startswith('http:')
            elif contains_letter(w):
                tweet_frequency[w] = 1
            else:
                removed_words.add(w)

    removed_tweets = set()
    while True:
        flag = True
        for t in range(len(tweets)):
            if t in removed_tweets:
                continue
            remains = [w for w in tweets[t]['tokens'] if (w not in removed_words) and (
                    w in tweet_frequency and tweet_frequency[w] >= min_tweet_frequency)]
            if len(remains) < min_num_words:
                removed_tweets.add(t)
                flag = False
                for w in remains:
                    tweet_frequency[w] -= 1
        if flag:
            break

    for t in range(len(tweets)):
        remains = [w for w in tweets[t]['tokens'] if (w not in removed_words) and (
                w in tweet_frequency and tweet_frequency[w] >= min_tweet_frequency)]
        tweets[t]['words'] = remains
    vocabs = [w for w in tweet_frequency if tweet_frequency[w] >= min_tweet_frequency]
    return tweets, vocabs


def topic_model(tweets, vocabs, min_num_words, num_topics):
    word2index = {}
    for i in range(len(vocabs)):
        word2index[vocabs[i]] = i
    tweet2document = {}
    documents = []
    for t in range(len(tweets)):
        bow = [word2index[w] for w in tweets[t]['words']]
        if len(bow) >= min_num_words:
            tweet2document[t] = len(documents)
            documents.append({'bow': bow})

    lda = SparseLDA(num_topics, burning_period=10, max_iterations=50, sampling_gap=5)
    # lda = SparseLDA(num_topics, burning_period=0, max_iterations=2, sampling_gap=1)
    lda.fit(documents, len(vocabs))
    for t in range(len(tweets)):
        if t in tweet2document:
            j = tweet2document[t]
            tweets[t]['theta'] = lda.thetas[j]
        else:
            tweets[t]['theta'] = np.array([1 / num_topics] * num_topics)
    loglikelihood = lda.get_likelihood(documents)
    return tweets, lda.topics, loglikelihood, word2index


def get_tweet_topics(dataset_name, dataset_path, num_topics, min_num_words=3, min_tweet_frequency=5):
    print('reading tweets')
    tweets = read_data(dataset_name, dataset_path)
    print('filtering stop/less-frequent words and too short tweets')
    tweets, vocabs = filter_tweets_words(tweets, min_num_words, min_tweet_frequency)
    print('mining topics using sparse-LDA model with Gibbs sampling')
    tweets, topics, likelihood, word2index = topic_model(tweets, vocabs, min_num_words, num_topics)
    vocabs = [None] * len(word2index)
    for word in word2index:
        index = word2index[word]
        vocabs[index] = word
    print('done!')
    return tweets, vocabs, topics, likelihood


def topic_modeling(num_topics, dataset_name, data, dictionary, min_num_words=3, min_tweet_frequency=5):
    """

    :param num_topics:
    :param dataset_name: name of the dataset, i.e., value of "opt.DATASET"
    :param data: a list of raw tweet
    :param dictionary:
    :param min_num_words:
    :param min_tweet_frequency:
    :return:
    """
    entries = []
    if dataset_name == 'dt':
        for j in range(len(data)):
            info = data[j]
            entries.extend(info['hate'])
            entries.extend(info['no_hate'])
    elif dataset_name == 'dt_full':
        for j in range(len(data)):
            info = data[j]
            entries.extend(info['hate'])
            entries.extend(info['no_hate'])
            entries.extend(info['offensive'])
    else:
        for j, info in enumerate(data):
            entries.extend(info['hate'])
            entries.extend(info['abusive'])
            entries.extend(info['spam'])
            entries.extend(info['normal'])
    print ('\t\t\t\tlen(entries) = ', len(entries), '\tentries[10] = ', entries[10])
    tweets = []
    for i in range(len(entries)):
        tweet = dict()
        tweet['tokens'] = dictionary.get_tokens(entries[i])
        tweets.append(tweet)

    tweets, vocabs = filter_tweets_words(tweets, min_num_words, min_tweet_frequency)
    tweets, topics, loglikelihood = topic_model(tweets, vocabs, min_num_words, num_topics)
    tweet2prob = [tweets[t]['theta'] for t in range(len(tweets))]
    return tweet2prob, loglikelihood
