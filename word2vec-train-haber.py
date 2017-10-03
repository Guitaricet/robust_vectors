import os
import pickle

from gensim.models import Word2Vec
from glob import glob
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def create_clean_model():
    corpus = []
    print("reading data")
    for f in tqdm(glob(os.path.join("data/42bin_haber", "*"))):
        if not f.endswith(".txt"):
            continue
        with open(f) as f_in:
            corpus.append(sent_tokenize(f_in.read().encode().decode("iso-8859-9")))

    clean_corpus = []
    logging.info(corpus)
    print("tokenizing")
    for c in tqdm(corpus):
        for s in c:
            clean_corpus.append(list(filter(lambda x: x.isalpha(), word_tokenize(s.lower()))))

    print("training")
    print(clean_corpus)
    model = Word2Vec(clean_corpus)
    print(len(model.raw_vocab))
    with open("save/word2vec_haber", "wb") as f_out:
        model.save(f_out)
    return clean_corpus


def create_stemmed_model():
    print("stemming")
    stemmed_file = "data/42bin_haber.pkl"
    if os.path.exists(stemmed_file):
        with open(stemmed_file, "rb") as f_in:
            stemmed_corpus = pickle.load(f_in)

    print("training on stemmed")
    model = Word2Vec(stemmed_corpus, size=256, window=5, workers=4, iter=200)
    with open("save/word2vec_haber_stemmed", "wb") as f_out:
        model.save(f_out)


if __name__ == '__main__':
    create_stemmed_model()