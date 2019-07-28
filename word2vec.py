# -- Dependencies ---

import sys
import tensorflow as tf
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize 
from collections import defaultdict
from argparse import ArgumentParser

class W2VConfig:
    
    def __init__(self, args):
        self.embed_size = 0
        self.corpus_path=args.corpus_path
        self.methods = args.model_types


class W2VArgParser:
    ''' config class argument parser used for word2vec methods. '''

    def __init__(self):
        self.p = ArgumentParser(description='The parameters for word2vec methods.')
        self.p.add_argument('-i', dest='corpus_path', default='./alice.txt', type=str, help='Path to the corpus. Default: ./alice.txt')        
        self.p.add_argument('-opt', dest='model_types',  metavar='N', type=int, nargs='+', choices={1, 2}, default={1},
                                    help='Types of model: 1 = skip-gram model, 2 = bag of words.')

    def get_args(self, args):
        return self.p.parse_args(args)


class W2V: 

    def __init__(self, config):
        self.config = config
        
        self.settings = {}
        self.settings['n'] = 5                 # dimension of word embeddings
        self.settings['window_size'] = 2       # context window +/- center word
        self.settings['min_count'] = 0         # minimum word count
        self.settings['epochs'] = 5            # number of training epochs
        self.settings['neg_samp'] = 10         # number of negative words to use during training
        self.settings['learning_rate'] = 0.01  # learning rate

        self.tot_words = 0
        self.word2idx  = {}
        self.idx2word  = {}
        self.words_list= []


    def preprocess(self, corpus, methods='sg'):

        word_counts = defaultdict(int) # all new dictionary item will have 0 count. 
        for sentence in sent_tokenize(corpus):
            for word in word_tokenize(sentence): 
                word_counts[word.lower()] += 1

        structured_corpus = []
        for sentence in sent_tokenize(corpus):
            structured_corpus.append([word.lower() for word in word_tokenize(sentence)])

        self.tot_words = len(list(word_counts.keys()))
        self.words_list = sorted(list(word_counts.keys()),reverse=False)
        self.word2idx = dict((word, i) for i, word in enumerate(self.words_list))
        self.idx2word = dict((i, word) for i, word in enumerate(self.words_list))
        
        # step 1: get idx for target and context pairs. 
        data_idx = []
        for sentence in structured_corpus:
            tot_words_sentence = len(sentence)

            for idx, word in enumerate(sentence):
                target_word_idx = self.word2idx[word]

                context_word_idx = []
                for win_idx in range(idx - self.settings['window_size'], idx + self.settings['window_size'] + 1):
                    if win_idx != idx and win_idx < tot_words_sentence and win_idx >= 0:
                        context_word = sentence[win_idx]
                        context_word_idx.append(self.word2idx[context_word])

                data_idx.append((target_word_idx, context_word_idx))

        # step 2: transform those idxs to corresponding one-hot vectors.
        data_vec = []
        for target_idx, context_word_idxs in data_idx:
            target_vec = self.get_onehot_vec(target_idx)
            context_vec = [self.get_onehot_vec(context_word_idx) for context_word_idx in context_word_idxs]
            data_vec.append([target_vec, context_vec])            

        return data_vec
    
    def train(self, data_vec):
        x_train = [] # input word
        y_train = [] # output word
        for target_vec, context_vecs in data_vec:
            for context_vec in context_vecs:
                x_train.append(target_vec)
                y_train.append(context_vec)

        # convert them to numpy arrays
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        # making placeholders for x_train and y_train
        x = tf.placeholder(tf.float32, shape=(None, self.tot_words))
        y_label = tf.placeholder(tf.float32, shape=(None, self.tot_words))

        # encoder 
        W1 = tf.Variable(tf.random_normal([self.tot_words, self.settings['n']]))
        b1 = tf.Variable(tf.random_normal([self.settings['n']])) #bias
        hidden_representation = tf.add(tf.matmul(x,W1), b1)

        # decoder
        W2 = tf.Variable(tf.random_normal([self.settings['n'], self.tot_words]))
        b2 = tf.Variable(tf.random_normal([self.tot_words]))
        prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))

        # define the loss function:
        cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))

        # define the training step:
        train_step = tf.train.GradientDescentOptimizer(self.settings['learning_rate']).minimize(cross_entropy_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer()) #make sure you do this!

        # train for n_iter iterations
        for idx in range(self.settings['epochs']):
            self.sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
            print('epoch', idx, ' loss is : ', self.sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))

        self.word_embeddings = self.sess.run(W1+b1)

    def get_embeddings(self):
        return self.word_embeddings

    
    def visualize(self, vectors):
        from sklearn.manifold import TSNE

        model = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        vectors = model.fit_transform(vectors) 

        from sklearn import preprocessing

        normalizer = preprocessing.Normalizer()
        vectors =  normalizer.fit_transform(vectors, 'l2')

        print(vectors)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        print(words)
        for word in words:
            print(word, vectors[word2int[word]][1])
            ax.annotate(word, (vectors[word2int[word]][0],vectors[word2int[word]][1] ))
        plt.show()

    def euclidean_dist(self, vec1, vec2):
        return np.sqrt(np.sum((vec1-vec2)**2))

    def find_closest(self, word_index, vectors):
        min_dist = 10000 # to act like positive infinity
        min_index = -1
        query_vector = vectors[word_index]
        for index, vector in enumerate(vectors):
            if self.euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
                min_dist = self.euclidean_dist(vector, query_vector)
                min_index = index
        return min_index


    def get_onehot_vec(self, idx):
        onehot_vec = np.zeros(self.tot_words)
        onehot_vec[idx] = 1
        return onehot_vec

def main(args):

    config = W2VConfig(W2VArgParser().get_args(args))
    w2v = W2V(config)
    corpus_f = open('./alice.txt', 'r').read()
    corpus_f = corpus_f.replace("\n", " ")
    data = w2v.preprocess(corpus_f)
    w2v.train(data)

if __name__ == "__main__": 
    main(sys.argv[1:])