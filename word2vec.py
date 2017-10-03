# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
import os
import codecs
from nltk.corpus import reuters
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
from glob import glob
import argparse
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", help="path to directory with data", type=str, default="reuters")

args = parser.parse_args()


def create_clean_model():
    corpus = []
    print("reading data")
    for f in tqdm(glob(os.path.join(args.data_dir, "*"))):
        if not f.endswith(".txt"):
            continue
        with open(f) as f_in:
            corpus.append(sent_tokenize(f_in.read()))

    clean_corpus = []
    logging.info(corpus)
    stemmer = SnowballStemmer("english")

    for c in tqdm(corpus):
        for s in c:
            clean_corpus.append(list(filter(lambda x: x.isalpha(), word_tokenize(stemmer.stem(s).lower()))))
    print("start training")
    print(clean_corpus[:10])
    model = Word2Vec(clean_corpus, size=300, window=5, min_count=2, workers=4, iter=200)

    with codecs.open("save/word2vec_MRPC", "wb") as f_out:
        model.save(f_out)
    return clean_corpus

create_clean_model()
print("reading data")
# sents = []
# stemmer = SnowballStemmer("english")
# for s in tqdm(reuters.sents()):
#     phrase = filter(lambda x: x, [filter(str.isalnum, stemmer.stem(t).lower()) for t in s])
#     sents.append(phrase)
#
# print("start training")
# print(sents[:10])
# model = Word2Vec(sents, size=256, window=5, min_count=2, workers=4, iter=200)
# with codecs.open(os.path.join("save", "w2v_reuters"), "wb") as f_out:
#     model.save(f_out)
