import codecs
from lxml import etree
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from random import random
import argparse
from gensim.models import Word2Vec
import pymorphy2
from pymorphy2.tokenizers import simple_word_tokenize
from scipy.spatial.distance import cosine
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="random, word2vec, robust", type=str, default="random")
parser.add_argument("-s", "--save-dir", help="directory with stored robust model", type=str, default="save")
parser.add_argument("-i", "--input-file", help="data to check against", type=str)
parser.add_argument("-w", "--word2vec-model", help="path to word2vec binary model", type=str, default="")
parser.add_argument("-n", "--noise-level", help="probability of typo appearance", type=float, default=0)

args = parser.parse_args()

with codecs.open(args.input_file, encoding="utf-8") as f:
    doc = etree.parse(f)
corpus = doc.find("corpus")
pairs = []
true = []
for child in corpus:
    paraph = {}
    for value in child:
        paraph[value.get("name")] = value.text
    if paraph["class"] != "0":
        if paraph["class"] == "-1":
            paraph["class"] = "0"
        paraph.pop('id_1', None)
        paraph.pop('id_2', None)
        paraph.pop('jaccard', None)
        pairs.append(paraph)
        true.append(int(paraph["class"]))

pred = []

if args.mode == "random":
    pred = [random() for _ in true]
    # pred = [0 if random() < 0.5 else 1 for _ in true]

elif args.mode == "word2vec":
    w2v = Word2Vec.load_word2vec_format(args.word2vec_model, binary=True)
    morph = pymorphy2.MorphAnalyzer()
    for pair in tqdm(pairs):
        def get_mean_vec(phrase):
            tokens = simple_word_tokenize(phrase)
            vectors = []
            for token in tokens:
                vector = np.zeros((w2v.vector_size,))
                if token in w2v:
                    vector = w2v[token]
                elif token.lower() in w2v:
                    vector = w2v[token.lower()]
                vectors.append(vector)
            return np.mean(vectors)
        v1 = get_mean_vec(pair["text_1"])
        v2 = get_mean_vec(pair["text_2"])
        pred.append(1 - cosine(v1, v2))

elif args.mode == "robust":
    phrases = []
    for pair in pairs:
        phrases.append(pair["text_1"])
        phrases.append(pair["text_2"])
    from sample import sample_multi
    results = sample_multi(args.save_dir, phrases)
    for i in range(0, len(phrases), 2):
        v1 = np.mean(phrases[i])
        v2 = np.mean(phrases[i + 1])
        pred.append(1 - cosine(v1, v2))

else:
    raise AttributeError("Unknown working mode!")

print "ROC\t=\t%.2f" % roc_auc_score(true, pred)
# print "F1\t=\t%.2f" % f1_score(true, pred)
