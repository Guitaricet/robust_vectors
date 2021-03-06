import codecs
from sklearn.metrics import roc_auc_score
import os
import argparse
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
import numpy as np
from tqdm import tqdm
from random import random, choice
from six.moves import cPickle
import math
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer



parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="random, word2vec, robust", type=str, default="word2vec")
parser.add_argument("-s", "--save-dir", help="directory with stored robust model", type=str, default="save")
parser.add_argument("-w", "--word2vec-model", help="path to word2vec binary model", type=str, default="")
parser.add_argument("-n", "--noise-level", help="probability of typo appearance", type=float, default=0)

args = parser.parse_args()

with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, _ = cPickle.load(f)


def noise_generator(string):
    noised = ""
    for c in string:
        if random() > args.noise_level:
            noised += c
        if random() < args.noise_level:
            noised += choice(chars)
    return noised

pairs = []
for filename in ["msr_paraphrase_test.txt", "msr_paraphrase_train.txt"]:
    with codecs.open(os.path.join("data", "MRPC", filename), encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            pair = {"text_1": parts[3], "text_2": parts[4], "decision": float(parts[0])}
            pairs.append(pair)

# pos = filter(lambda x: x["class"] == "1", pairs)
# neg = filter(lambda x: x["class"] == "0", pairs)
# min_len = min(len(pos), len(neg))
# pairs = pos[:min_len] + neg[:min_len]
true = [x["decision"] for x in pairs]

if "random" in args.mode:
    pred = [random() for _ in true]
    with open("results3.txt", "at") as f_out:
        # f_out.write("random,%.2f,%.3f\n" % (args.noise_level, mean_squared_error(true, pred)))
        f_out.write("random,%.2f,%.3f\n" % (args.noise_level, roc_auc_score(true, pred)))

if "word2vec" in args.mode:
    pred = []
    w2v = Word2Vec.load(os.path.join(args.save_dir, "w2v_reuters"))
    stemmer = SnowballStemmer("english")

    def get_mean_vec(phrase):
        tokens = [stemmer.stem(t) for t in word_tokenize(phrase)]
        vectors = [np.zeros((w2v.vector_size,))]
        for token in tokens:
            if token in w2v:
                vector = w2v[token]
                vectors.append(vector)
        return np.mean(vectors, axis=0)
    for pair in tqdm(pairs):
        v1 = get_mean_vec(noise_generator(pair["text_1"]))
        v2 = get_mean_vec(noise_generator(pair["text_2"]))
        pred.append(1 - cosine(v1, v2))
    with open("results4.txt", "at") as f_out:
        # f_out.write("word2vec,%.2f,%.3f\n" % (args.noise_level, mean_squared_error(true, pred)))
        f_out.write("word2vec,%.2f,%.3f\n" % (args.noise_level, roc_auc_score(true, pred)))
    # print "ROC\t\t=\t%.2f" % roc_auc_score(true, pred)

if "robust" in args.mode:
    pred = []
    phrases = []
    for pair in pairs:
        phrases.append(noise_generator(pair["text_1"]))
        phrases.append(noise_generator(pair["text_2"]))
    from sample import sample_multi
    results = np.vsplit(sample_multi(args.save_dir, phrases), len(phrases))
    for i in range(0, len(results), 2):
        v1 = results[i]
        v2 = results[i + 1]
        pred.append(1 - cosine(v1, v2))
        if math.isnan(pred[-1]):
            pred[-1] = 0.5
    with open("results3.txt", "at") as f_out:
        # f_out.write("robust,%.2f,%.3f\n" % (args.noise_level, mean_squared_error(true, pred)))
        f_out.write("robust,%.2f,%.3f\n" % (args.noise_level, roc_auc_score(true, pred)))
    # print "ROC\t\t=\t%.2f" % roc_auc_score(true, pred)

# print "Class ratio\t=\t%.2f" % (float(len(filter(None, true)))/len(true))
# print "F1\t=\t%.2f" % f1_score(true, pred)
