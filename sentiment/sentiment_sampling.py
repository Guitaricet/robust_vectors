import codecs

from nltk import RegexpTokenizer
from sklearn.metrics import roc_auc_score, f1_score
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


from nltk.tokenize import TweetTokenizer

from six.moves import cPickle

from sample import sample_multi
from utils import noise_generator
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_svmlight_files

import math
from nltk.tokenize import word_tokenize




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


def linear_svm(training_data, testing_data, training_target, testing_target, infer=False):
    start = time()
    clf_linear = SVC(probability=True, kernel="linear", class_weight="balanced")
    print("Training ...")
    clf_linear.fit(training_data, training_target)
    print("Training Done")
    print("Testing ...")
    predict_test = clf_linear.predict(testing_data)
    print(len(predict_test))
    print(len(training_target))
    result = roc_auc_score(testing_target, predict_test)
    end = time()
    print("Training time: {}".format(end - start))
    return result


def samping_sentiment_data(args, data, labels):
    idx_for_split = int(0.2 * len(data))
    results = np.squeeze(np.vsplit(sample_multi(args.save_dir, data, args.model_type), len(data)))
    train = results[idx_for_split:]
    test = results[:idx_for_split]
    train_label = labels[idx_for_split:]
    test_label = labels[:idx_for_split]
    roc_auc_score = linear_svm(train, test, train_label, test_label)
    with open("results_sentiment.txt", "at") as f_out:
        # f_out.write("robust,%.2f,%.3f\n" % (args.noise_level, mean_squared_error(true, pred)))
        f_out.write("robust,%.2f,%.3f\n" % (args.noise_level, roc_auc_score))

if __name__ == '__main__':
    args = get_args()
    full_data = pd.read_table(args.data_path)[:1000]
    df = full_data[['SentimentText', 'Sentiment']]
    dt = df.loc[:, 'SentimentText']
    y_target = df['Sentiment'].values
    samping_sentiment_data(args, dt, y_target)

