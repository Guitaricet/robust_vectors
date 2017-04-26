from gensim.models import Word2Vec
from glob import glob
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer
import sbl2py

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

import sbl2py.utils
with open("data/turkish.sbl") as sbl:
    sbl_code = sbl.read()

py_code = sbl2py.translate_string(sbl_code)
turkish = sbl2py.utils.module_from_code('demo_module', py_code)

print "stemming"
stemmed_corpus = []
for sent in tqdm(clean_corpus):
    stemmed_sent = []
    for word in sent:
        stemmed_sent.append(turkish.stem(word))
    stemmed_corpus.append(stemmed_sent)

print "training on stemmed"
model = Word2Vec(stemmed_corpus)
with open("save/word2vec_haber_stemmed", "wb") as f_out:
    model.save(f_out)

