import os
import collections

import pymorphy2
from six.moves import cPickle
import numpy as np
import codecs
from gensim.models import Word2Vec
from pymorphy2.tokenizers import simple_word_tokenize
from tqdm import tqdm


def letters2vec(word, vocab, dtype=np.uint8):
    base = np.zeros(len(vocab), dtype=dtype)

    middle = np.copy(base)
    for char in word:
        middle[vocab[char]] += 1

    first = np.copy(base)
    first[vocab[word[0]]] += 1
    second = np.copy(base)
    if len(word) > 1:
        second[vocab[word[1]]] += 1
    third = np.copy(base)
    if len(word) > 2:
        third[vocab[word[2]]] += 1

    end_first = np.copy(base)
    end_first[vocab[word[-1]]] += 1
    end_second = np.copy(base)
    if len(word) > 1:
        end_second[vocab[word[-2]]] += 1
    end_third = np.copy(base)
    if len(word) > 2:
        end_third[vocab[word[-3]]] += 1

    return np.concatenate([first, second, third, middle, end_third, end_second, end_first])


class TextLoader:
    def __init__(self, args, chars=None, vocab=None):
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length

        self.chars = chars
        self.vocab = vocab

        input_file = os.path.join(args.data_dir, "input.txt")
        vocab_file = os.path.join(args.data_dir, "vocab.pkl")
        tensor_file1 = os.path.join(args.data_dir, "xdata.npy")
        tensor_file2 = os.path.join(args.data_dir, "ydata.npy")
        w2v_file = args.w2v_model

        self.xdata = None
        self.ydata = None

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file1) and os.path.exists(tensor_file2)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file1, tensor_file2, w2v_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file1, tensor_file2)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, x_file, y_file, w2v_file):
        self.create_vocab(vocab_file, input_file)

        if self.vocab_size < 256:
            dtype = np.uint8
        else:
            dtype = np.uint16

        w2v = Word2Vec.load_word2vec_format(w2v_file, binary=True)
        self.w2v_size = w2v.vector_size
        morph = pymorphy2.MorphAnalyzer()
        with codecs.open(input_file, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                tokens = simple_word_tokenize(line)
                true_vectors = []
                letter_vectors = []
                for token in tokens:
                    lemma = morph.parse(token)[0].normal_form
                    true_vector = np.zeros(w2v.vector_size)
                    if lemma in w2v:
                        true_vector = w2v[lemma]
                    letter_vector = letters2vec(token, self.vocab, dtype)
                    true_vectors.append(true_vector)
                    letter_vectors.append(letter_vector)
                if self.xdata is None:
                    self.xdata = np.vstack(letter_vectors)
                elif letter_vectors:
                    self.xdata = np.vstack([self.xdata, np.vstack(letter_vectors)])
                if self.ydata is None:
                    self.ydata = np.vstack(true_vectors)
                elif true_vectors:
                    self.ydata = np.vstack([self.ydata, np.vstack(true_vectors)])

        np.save(x_file, self.xdata)
        np.save(y_file, self.ydata)

    def load_preprocessed(self, vocab_file, x_file, y_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.xdata = np.load(x_file)
        self.ydata = np.load(y_file)
        self.w2v_size = self.ydata.shape[1]

    def create_batches(self):
        self.num_batches = int(self.xdata.shape[0] / (self.batch_size * self.seq_length))
        self.xdata = self.xdata[:self.num_batches * self.batch_size * self.seq_length]
        self.xdata = self.ydata[:self.num_batches * self.batch_size * self.seq_length]

        self.x_batches = np.split(self.xdata.reshape(self.batch_size, -1, self.xdata.shape[1]),
                                  self.num_batches, 1)
        self.y_batches = np.split(self.ydata.reshape(self.batch_size, -1, self.xdata.shape[1]),
                                  self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

    def create_vocab(self, vocab_file, input_file):
        # preparation of vocab
        with codecs.open(input_file, "r", encoding="utf-8") as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        temp_chars, _ = zip(*count_pairs)
        if self.chars is None:
            self.chars = temp_chars
        elif not set(self.chars).issuperset(set(temp_chars)):
            os.write(2, "Incompatible charsets. Using substitute.")
        self.vocab_size = len(self.chars)

        if self.vocab is None:
            self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
