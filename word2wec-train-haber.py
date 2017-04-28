import os
import pickle

import sbl2py
import sbl2py.utils
from gensim.models import Word2Vec
from glob import glob
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

corpus = []
print "reading data"
for f in tqdm(glob("data/42bin_haber/*/*/*")):
    if not f.endswith(".txt"):
        continue
    with open(f) as f_in:
        corpus.append(sent_tokenize(f_in.read().decode("iso-8859-9")))

clean_corpus = []
print "tokenizing"
for c in tqdm(corpus):
    for s in c:
        clean_corpus.append(filter(lambda x: x.isalpha(), word_tokenize(s.lower())))

print "training"
model = Word2Vec(clean_corpus)
print len(model.raw_vocab)
with open("save/word2vec_haber", "wb") as f_out:
    model.save(f_out)


with open("data/turkish.sbl") as sbl:
    sbl_code = sbl.read()

py_code = sbl2py.translate_string(sbl_code)
turkish = sbl2py.utils.module_from_code('demo_module', py_code)

print "stemming"
stemmed_file = "data/42bin_haber.pkl"
if not os.path.exists(stemmed_file):
    stemmed_corpus = []
    for sent in tqdm(clean_corpus):
        stemmed_sent = []
        for word in sent:
            stemmed_sent.append(turkish.stem(word))
        stemmed_corpus.append(stemmed_sent)
    with open(stemmed_file, "wb") as f_out:
        pickle.dump(stemmed_corpus, f_out)
else:
    with open(stemmed_file, "rb") as f_in:
        stemmed_corpus = pickle.load(f_in)

print "training on stemmed"
model = Word2Vec(stemmed_corpus, size=256, window=5, workers=4, iter=200)
with open("save/word2vec_haber_stemmed", "wb") as f_out:
    model.save(f_out)
