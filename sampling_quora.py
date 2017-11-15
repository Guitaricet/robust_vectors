import codecs

from nltk import word_tokenize
from sklearn.metrics import roc_auc_score
import os
import argparse
import numpy as np
import pandas as pd
import math
from gensim.models import Word2Vec

from sentiment_sampling import linear_svm
from utils import noise_generator
from tqdm import tqdm
from random import random, choice
from six.moves import cPickle

from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
stop = set(stopwords.words('english'))
snowball_stemmer = SnowballStemmer("english")



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


def preproc_sentence(sentence):
   return " ".join([ snowball_stemmer.stem(i) for i in sentence.lower().split() if i not in stop])


def load_data(args):
    full_data = pd.read_csv(args.data_path, sep='\t')[:500]
    df = full_data[['sentence1','sentence2', 'duplicate']]
    dt = df[['sentence1', 'sentence2']]
    y_target = df['duplicate'].values
    return dt, y_target


def svm_robust_score(args,data, labels):
    idx_for_split = int(0.2 * len(data))
    phrases = []
    pred = []
    for index, row in data.iterrows():
        phrases.append(noise_generator(row["sentence1"], args.noise_level, chars))
        phrases.append(noise_generator(row["sentence2"], args.noise_level, chars))
    from sample import sample_multi
    results = np.squeeze(np.vsplit(sample_multi(args.save_dir, phrases, args.model_type), len(phrases)))
    #pairs_vectors = zip(results[0::2], results[1::2])
    # df = pd.DataFrame(columns=['cosine', 'canberra', 'cityblock', 'euclidean', 'minkowski', 'braycurtis',"skew_q1","skew_q2"\
    #                            "kur_q1","kur_q2", "skew_diff", "kur_diff"])
    # df['cosine'] = [cosine(x, y) for (x, y) in pairs_vectors]
    # print(len(df))
    # df['canberra'] = [canberra(x, y) for (x, y) in zip(results[0::2], results[1::2])]
    # df['cityblock'] = [cityblock(x, y) for (x, y) in zip(results[0::2], results[1::2])]
    # df['euclidean'] = [euclidean(x, y) for (x, y) in zip(results[0::2], results[1::2])]
    # df['minkowski'] = [minkowski(x, y, 3) for (x, y) in zip(results[0::2], results[1::2])]
    # df['braycurtis'] = [braycurtis(x, y) for (x, y) in zip(results[0::2], results[1::2])]
    # question1_vec = results[0::2]
    # question2_vec = results[1::2]
    # data['skew_q1'] = [skew(x) for x in question1_vec]
    # data['skew_q2'] = [skew(x) for x in question2_vec]
    # data['kur_q1'] = [kurtosis(x) for x in question1_vec]
    # data['kur_q2'] = [kurtosis(x) for x in question2_vec]
    #
    # data['skew_diff'] = np.abs(data['skew_q1'] - data['skew_q2'])
    # data['kur_diff'] = np.abs(data['kur_q1'] - data['kur_q2'])

    for i in range(0, len(results), 2):
        v1 = results[i]
        v2 = results[i + 1]
        if (v1 == np.zeros_like(v1)).all() or (v2 == np.zeros_like(v2)).all():
            print(i)
        pred.append(1 - cosine(v1, v2))
        if math.isnan(pred[-1]):
            pred[-1] = 0.5
    # pr = pd.DataFrame(pred)
    # train = df.iloc[idx_for_split:]
    # test = df.iloc[:idx_for_split]
    # train_label = labels[idx_for_split:]
    # test_label = labels[:idx_for_split]
    roc_auc= roc_auc_score(labels, pred)
    # clf = catboost.CatBoostClassifier(depth=6, iterations=5000, learning_rate=0.1, thread_count=16)
    # clf.fit(train, train_label)
    # y_proba = clf.predict_proba(test)[:, 1]
    # roc_auc = roc_auc_score(test_label, y_proba)
    with open("results_quora" + args.model_type + ".txt", "at") as f_out:
        # f_out.write("robust,%.2f,%.3f\n" % (args.noise_level, mean_squared_error(true, pred)))
        f_out.write(args.mode + ",%.2f,%.3f\n" % (args.noise_level, roc_auc))

def get_w2v_results(args, pairs, true):
    pred = []
    w2v = Word2Vec.load(args.word2vec_model)

    def get_mean_vec(phrase):
        tokens = word_tokenize(phrase)
        vectors = [np.zeros((w2v.vector_size,))]
        for token in tokens:
            if token in w2v:
                vector = w2v[token]
                vectors.append(vector)
        return np.mean(vectors, axis=0)

    for index, pair in pairs.iterrows():
        v1 = get_mean_vec(noise_generator(pair["sentence1"], args.noise_level, chars))
        v2 = get_mean_vec(noise_generator(pair["sentence2"], args.noise_level, chars))
        pred.append(1 - cosine(v1, v2))
    with open("results" + args.mode + ".txt", "at") as f_out:
        # f_out.write("word2vec,%.2f,%.3f\n" % (args.noise_level, mean_squared_error(true, pred)))
        f_out.write("word2vec,%.2f,%.3f\n" % (args.noise_level, roc_auc_score(true, pred)))

if __name__ == '__main__':
    args = get_args()


    global chars

    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, _ = cPickle.load(f)

    dt, y_target = load_data(args)

    if(args.mode == "robust"):
        svm_robust_score(args, dt, y_target)

    if(args.mode =="w2v"):
        get_w2v_results(args, dt, y_target)
