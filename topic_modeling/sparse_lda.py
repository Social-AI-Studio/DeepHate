import numpy as np
import random


def compute_entropy(weights):
    probs = weights / np.sum(weights) + 1e-12
    entropy = -np.sum(probs * np.log(probs))
    return entropy


def update_entropy(current_weight, delta, sum_current_weights, curr_e):
    current_weight += 1e-12
    new_entropy = (curr_e * sum_current_weights + (current_weight * np.log(current_weight)
                                                   - (current_weight + delta) * np.log(current_weight + delta))
                   + np.log(delta + sum_current_weights) * (delta + sum_current_weights)
                   - sum_current_weights * np.log(sum_current_weights)) / (sum_current_weights + delta)
    return new_entropy


class SparseLDA:
    def __init__(self, num_topics, entropy_reg=0.5, burning_period=100, max_iterations=500, sampling_gap=20):
        self.num_topics = num_topics
        self.alpha = 50 / num_topics
        self.beta = 0.01
        self.entropy_reg = entropy_reg

        self.documents = None

        self.n_dz = None
        self.n_zw = None
        self.sum_nzw = None

        self.cumm_n_dz = None
        self.cumm_n_zw = None
        self.cumm_sum_nzw = None

        self.states = None

        self.burning_period = burning_period
        self.max_iterations = max_iterations
        self.sampling_gap = sampling_gap

        self.thetas = None
        self.topics = None

    def init_states_counts(self, documents, vocab_size, bow):
        num_docs = len(documents)
        self.n_dz = np.zeros((num_docs, self.num_topics))
        self.sum_ndz = np.zeros(num_docs)
        self.n_zw = np.zeros((self.num_topics, vocab_size))
        self.sum_nzw = np.zeros(self.num_topics)

        self.cumm_n_dz = np.zeros((num_docs, self.num_topics))
        self.cumm_n_zw = np.zeros((self.num_topics, vocab_size))
        self.cumm_sum_nzw = np.zeros(self.num_topics)

        self.states = []
        for d in range(num_docs):
            topic_assignments = []
            for i in range(len(documents[d][bow])):
                w = documents[d][bow][i]
                z = random.randint(0, self.num_topics - 1)
                topic_assignments.append(z)
                self.n_dz[d, z] += 1
                self.n_zw[z, w] += 1
                self.sum_nzw[z] += 1
            self.sum_ndz[d] = len(documents[d][bow])
            self.states.append(topic_assignments)

    def update_cumm_counts(self):
        num_docs = self.n_dz.shape[0]
        vocab_size = self.n_zw.shape[1]
        for d in range(num_docs):
            for z in range(self.num_topics):
                self.cumm_n_dz[d, z] += self.n_dz[d, z]
        for z in range(self.num_topics):
            for w in range(vocab_size):
                self.cumm_n_zw[z, w] += self.n_zw[z, w]
                self.cumm_sum_nzw[z] += self.sum_nzw[z]

    def sample(self, d, i, w, vocab_size):
        curr_z = self.states[d][i]
        self.n_zw[curr_z][w] -= 1
        self.n_dz[d][curr_z] -= 1

        weights = np.array([self.n_dz[d, z] * (self.n_zw[z, w] + self.beta) / (self.sum_nzw[z] + self.beta * vocab_size)
                            for z in range(self.num_topics)])

        # print('weights.shape = ', weights.shape)
        curr_entropy = compute_entropy(self.n_dz[d])
        # print('curr_entropy.shape = ', curr_entropy.shape)
        entropies = np.array([update_entropy(self.n_dz[d, z], 1, self.sum_ndz[d], curr_entropy)
                              for z in range(self.num_topics)])
        # print('entropies.shape = ', entropies.shape)
        regs = np.exp(-entropies * entropies / (2 * self.entropy_reg))
        # print('regs.shape = ', regs.shape)
        weights = weights * regs
        # print('weights.shape = ', weights.shape)
        probs = weights / np.sum(weights)
        z = np.random.multinomial(1, probs, size=None).argmax()
        self.states[d][i] = z
        self.n_zw[z][w] += 1
        self.n_dz[d][z] += 1

    def fit(self, documents, vocab_size, bow='bow'):

        self.init_states_counts(documents, vocab_size, bow)
        num_docs = len(documents)
        for iter in range(self.max_iterations):
            print('iter = ', iter)
            for d in range(num_docs):
                for i in range(len(documents[d]['bow'])):
                    w = documents[d]['bow'][i]
                    self.sample(d, i, w, vocab_size)
            if iter < self.burning_period:
                continue
            if iter % self.sampling_gap == 0:
                self.update_cumm_counts()
        self.thetas = (self.cumm_n_dz + self.alpha) / (
                np.sum(self.cumm_n_dz, axis=1).reshape(-1, 1) + self.num_topics * self.alpha)
        self.topics = (self.cumm_n_zw + self.beta) / (
                np.sum(self.cumm_n_zw, axis=1).reshape(-1, 1) + self.beta * vocab_size)

    def get_likelihood(self, documents):
        if self.thetas is None:
            print('model is not yet learned')
            return None
        else:
            loglik = 0.0
            num_docs = len(documents)
            for d in range(num_docs):
                theta = self.thetas[d]
                for i in range(len(documents[d]['bow'])):
                    w = documents[d]['bow'][i]
                    pw = 0.0
                    for z in range(self.num_topics):
                        pw += theta[z] * self.topics[z][w]
                    loglik += np.log(pw)
            return loglik
