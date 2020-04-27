import tweet_topic_model
import numpy as np

dataset_name = 'dt'
dataset_path = '../data/davidson_icwsm17/labeled_data.csv'

# dataset_name = 'founta'
# dataset_path = '../data/founta_icwsm18/hatespeech_text_label_vote_RESTRICTED.csv'

num_topics = 5

tweets, vocabs, topics, likelihood = tweet_topic_model.get_tweet_topics(dataset_name, dataset_path, num_topics,
                                                                        min_num_words=3,
                                                                        min_tweet_frequency=5)
print(likelihood)
# print(vocabs)
print('Topwords:')
for z in range(num_topics):
    ids = list(np.argsort(topics[z])[-10:])
    ids.reverse()
    top_words = [vocabs[i] for i in ids]
    print('topic-%d: %s' % (z, ' '.join(top_words)))

