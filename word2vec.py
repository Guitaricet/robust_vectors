# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
import os
import codecs
from nltk.corpus import reuters
from nltk.stem import SnowballStemmer
from tqdm import tqdm
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print "reading data"
sents = []
stemmer = SnowballStemmer("english")
for s in tqdm(reuters.sents()):
    phrase = filter(lambda x: x, [filter(unicode.isalnum, stemmer.stem(t).lower()) for t in s])
    sents.append(phrase)

print "start training"
model = Word2Vec(sents, size=256, window=5, min_count=3, workers=4, iter=100, sample=1e-5)
with codecs.open(os.path.join("save", "w2v_reuters"), "wb") as f_out:
    model.save(f_out)
