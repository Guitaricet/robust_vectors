import codecs
from lxml import etree
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import os
import argparse
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
import numpy as np
from tqdm import tqdm
from pymystem3 import Mystem
from random import random, choice
from six.moves import cPickle


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="random, word2vec, robust", type=str, default="random")
parser.add_argument("-s", "--save-dir", help="directory with stored robust model", type=str, default="save")
parser.add_argument("-i", "--input-file", help="data to check against", type=str)
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

with codecs.open(args.input_file, encoding="utf-8") as f:
    doc = etree.parse(f)
corpus = doc.find("corpus")
pairs = []
for child in corpus:
    paraph = {}
    for value in child:
        paraph[value.get("name")] = value.text
    if paraph["class"] != "0":
        if paraph["class"] == "-1":
            paraph["class"] = "0"
        # elif paraph["class"] == "0":
        #     paraph["class"] = "1"
        paraph.pop('id_1', None)
        paraph.pop('id_2', None)
        paraph.pop('jaccard', None)
        pairs.append(paraph)

# pos = filter(lambda x: x["class"] == "1", pairs)
# neg = filter(lambda x: x["class"] == "0", pairs)
# min_len = min(len(pos), len(neg))
# pairs = pos[:min_len] + neg[:min_len]
true = [int(x["class"]) for x in pairs]

if "random" in args.mode:
    pred = [random() for _ in true]
    # pred = [0 if random() < 0.5 else 1 for _ in true]
    print "ROC\t\t=\t%.2f" % roc_auc_score(true, pred)

if "word2vec" in args.mode:
    pred = []
    w2v = Word2Vec.load_word2vec_format(args.word2vec_model, binary=True)
    m = Mystem()

    def get_mean_vec(phrase):
        tokens = m.analyze(phrase)
        vectors = []
        for token in tokens:
            if "analysis" not in token:
                continue
            if token["analysis"]:
                tag = token["analysis"][0]["gr"].split(',')[0]
                if tag[-1] == "=":
                    tag = tag[:-1]
                lemma = token["analysis"][0]["lex"] + "_" + tag
                vector = np.zeros((w2v.vector_size,))
                if lemma in w2v:
                    vector = w2v[lemma]
                vectors.append(vector)
        return np.mean(vectors, axis=0)
    for pair in tqdm(pairs):
        v1 = get_mean_vec(noise_generator(pair["text_1"]))
        v2 = get_mean_vec(noise_generator(pair["text_2"]))
        pred.append(1 - cosine(v1, v2))
    print "ROC\t\t=\t%.2f" % roc_auc_score(true, pred)

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
    print "ROC\t\t=\t%.2f" % roc_auc_score(true, pred)

print "Class ratio\t=\t%.2f" % (float(len(filter(None, true)))/len(true))
# print "F1\t=\t%.2f" % f1_score(true, pred)
