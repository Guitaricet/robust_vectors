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
import sbl2py.utils


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="random, word2vec, robust", type=str, default="random")
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
for filename in ["TuPC_test_set.txt", "TuPC_train_set.txt"]:
    with codecs.open(os.path.join("data", "TuPC-2016", filename), encoding="iso-8859-9") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            pair = {"text_1": parts[0], "text_2": parts[1], "decision": float(parts[2])}
            pairs.append(pair)

# pos = filter(lambda x: x["class"] == "1", pairs)
# neg = filter(lambda x: x["class"] == "0", pairs)
# min_len = min(len(pos), len(neg))
# pairs = pos[:min_len] + neg[:min_len]
true = [x["decision"] for x in pairs]

if "random" in args.mode:
    pred = [random() for _ in true]
    with open("results6.txt", "at") as f_out:
        # f_out.write("random,%.2f,%.3f\n" % (args.noise_level, mean_squared_error(true, pred)))
        f_out.write("random,%.2f,%.3f\n" % (args.noise_level, roc_auc_score(true, pred)))

if "word2vec" in args.mode:
    pred = []
    w2v = Word2Vec.load(args.word2vec_model)

    with open("data/turkish.sbl") as sbl:
        sbl_code = sbl.read()

    py_code = sbl2py.translate_string(sbl_code)
    turkish = sbl2py.utils.module_from_code('demo_module', py_code)

    def get_mean_vec(phrase):
        tokens = word_tokenize(phrase.lower())
        vectors = [np.zeros((w2v.vector_size,)) + 1e-10]
        for token in tokens:
            if token in w2v:
                stemmed = turkish.stem(token)
                if stemmed in w2v:
                    vector = w2v[stemmed]
                    vectors.append(vector)
        return np.mean(vectors, axis=0)
    for pair in tqdm(pairs):
        v1 = get_mean_vec(noise_generator(pair["text_1"]))
        v2 = get_mean_vec(noise_generator(pair["text_2"]))
        pred.append(1 - cosine(v1, v2))
    with open("results6.txt", "at") as f_out:
        # f_out.write("word2vec,%.2f,%.3f\n" % (args.noise_level, mean_squared_error(true, pred)))
        f_out.write("word2vec,%.2f,%.3f\n" % (args.noise_level, roc_auc_score(true, pred)))
    # print "ROC\t\t=\t%.2f" % roc_auc_score(true, pred)

if "fasttext" in args.mode:
    ft = {}
    with codecs.open("data/TuPC-2016/word_vectors.txt", encoding="iso-8859-9") as f:
        parts = f.readline().strip().split()
        ft[parts[0]] = np.array(map(float, parts[1:]))
    pred = []

    def get_mean_vec(phrase):
        tokens = word_tokenize(phrase)
        vectors = []
        for token in tokens:
            vectors.append(ft[token])
        return np.mean(vectors, axis=0)

    for pair in tqdm(pairs):
        v1 = get_mean_vec(noise_generator(pair["text_1"]))
        v2 = get_mean_vec(noise_generator(pair["text_2"]))
        pred.append(1 - cosine(v1, v2))

    with open("results6.txt", "at") as f_out:
        # f_out.write("robust,%.2f,%.3f\n" % (args.noise_level, mean_squared_error(true, pred)))
        f_out.write("fasttext,%.2f,%.3f\n" % (args.noise_level, roc_auc_score(true, pred)))
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
    with open("results6.txt", "at") as f_out:
        # f_out.write("robust,%.2f,%.3f\n" % (args.noise_level, mean_squared_error(true, pred)))
        f_out.write("robust,%.2f,%.3f\n" % (args.noise_level, roc_auc_score(true, pred)))
    # print "ROC\t\t=\t%.2f" % roc_auc_score(true, pred)

# print "Class ratio\t=\t%.2f" % (float(len(filter(None, true)))/len(true))
# print "F1\t=\t%.2f" % f1_score(true, pred)
