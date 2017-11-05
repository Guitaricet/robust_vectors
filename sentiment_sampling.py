import codecs

from nltk import RegexpTokenizer, NaiveBayesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn import svm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
import os
import argparse
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
import scipy
import numpy as np
from tqdm import tqdm
from random import random, choice
from time import time
import pandas as pd


from six.moves import cPickle

from sample import sample_multi
from utils import noise_generator
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_svmlight_files

import math
from nltk.tokenize import word_tokenize



from keras.layers import LSTM, Convolution1D, Flatten, Dropout, Dense
from keras.layers.embeddings import Embedding
from keras.models import Sequential


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="random, word2vec, robust", type=str, default="random")
    parser.add_argument("-s", "--save-dir", help="directory with stored robust model", type=str, default="save")
    parser.add_argument("-w", "--word2vec-model", help="path to word2vec binary model", type=str, default="")
    parser.add_argument("-n", "--noise-level", help="probability of typo appearance", type=float, default=0)
    parser.add_argument("-p", "--percent_of_test", help="percent of data used to train", type=float, default=1)
    parser.add_argument("-t", "--model_type", help="type of model used to train", type=str, default="lstm")
    parser.add_argument("-d", "--data_path", type=str, default="./data/tweets")
    args = parser.parse_args()
    return args


def linear_svm(training_data, testing_data, training_target, testing_target):
    start = time()
    clf_linear = BernoulliNB()
    print("Training ...")
    clf_linear.fit(training_data, training_target)
    print(testing_data.shape)
    predict_test = clf_linear.predict(testing_data)
    print(len(predict_test))
    print(len(training_target))
    print(predict_test[:30])
    print(testing_target[:30])
    result = roc_auc_score(testing_target, predict_test)
    #result = f1_score(testing_target, predict_test,labels=[0,1,2], average='micro')
    end = time()
    print("Training time: {}".format(end - start))
    print("mean accuracy:{}".format(clf_linear.score(testing_data, testing_target)))
    return result


def keras_test(train, test, train_label, test_label):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=300))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train, train_label, epochs=9, batch_size=8, verbose=2)
    score = model.evaluate(test, test_label, batch_size=1, verbose=2)
    print(score[1])
    with open(args.model_type + "_results_sentiment.txt", "at") as f_out:
        f_out.write("robust,%.2f,%.3f\n" % (args.noise_level, score[1]))



def samping_sentiment_data(args, data, labels):
    idx_for_split = int(0.2 * len(data))
    print(data[0], labels[0])
    print(data[1], labels[1])
    results = np.squeeze(np.vsplit(sample_multi(args.save_dir, data, args.model_type), len(data)))
    print(results[0].shape)
    train = results[idx_for_split:]
    test = results[:idx_for_split]
    train_label = labels[idx_for_split:]
    test_label = labels[:idx_for_split]
    print(len(train_label))
    print(len(test_label))
    keras_test(train, test, train_label, test_label)
    roc_auc_score = linear_svm(train, test, train_label, test_label)
    with open(args.model_type + "_results_sentiment.txt", "at") as f_out:
        f_out.write("robust,%.2f,%.3f\n" % (args.noise_level, roc_auc_score))


if __name__ == '__main__':
    args = get_args()
    full_data = pd.read_csv(args.data_path)[:800]
    df = full_data[['SentimentText', 'Sentiment']]
    dt = df.loc[:, 'SentimentText']
    y_target = df['Sentiment'].values

    global chars

    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, _ = cPickle.load(f)

    samping_sentiment_data(args, dt, y_target)

