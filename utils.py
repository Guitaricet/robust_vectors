import os

from six.moves import cPickle
import numpy as np
import codecs
from pymorphy2.tokenizers import simple_word_tokenize
from tqdm import tqdm
from collections import Counter


def letters2vec(word, vocab, dtype=np.uint8):
    base = np.zeros(len(vocab), dtype=dtype)

    def update_vector(vector, char):
        if char in vocab:
            vector[vocab[char]] += 1

    middle = np.copy(base)
    for char in word:
        update_vector(middle, char)

    first = np.copy(base)
    update_vector(first, word[0])
    second = np.copy(base)
    if len(word) > 1:
        update_vector(second, word[1])
    third = np.copy(base)
    if len(word) > 2:
        update_vector(third, word[2])

    end_first = np.copy(base)
    update_vector(end_first, word[-1])
    end_second = np.copy(base)
    if len(word) > 1:
        update_vector(end_second, word[-2])
    end_third = np.copy(base)
    if len(word) > 2:
        update_vector(end_third, word[-3])

    return np.concatenate([first, second, third, middle, end_third, end_second, end_first])


class TextLoader:
    def __init__(self, args, chars=None, vocab=None):
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length

        self.chars = chars
        self.vocab = vocab
        self.noise_level = args.noise_level

        input_file = os.path.join(args.data_dir, "input.txt")
        vocab_file = os.path.join(args.data_dir, "vocab.pkl")
        tensor_file = os.path.join(args.data_dir, "data.npy")
        letter_vocab_file = os.path.join(args.data_dir, "letter_vocab.npy")

        self.xdata = None
        self.ydata = None

        if not (os.path.exists(vocab_file)
                and os.path.exists(tensor_file)
                and os.path.exists(letter_vocab_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file, letter_vocab_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file, letter_vocab_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file, letter_vocab_file,):
        print "creating char vocab"
        self.create_vocab(vocab_file, input_file)

        if self.vocab_size < 256:
            dtype = np.uint8
        else:
            dtype = np.uint16

        with codecs.open(input_file, "rt", encoding="utf-8") as f:
            all_tokens = simple_word_tokenize(f.read())
        uniq_tokens = Counter(all_tokens)
        count_pairs = sorted(uniq_tokens.items(), key=lambda x: -x[1])
        tokens, _ = zip(*count_pairs)
        tokens_vocab = dict(zip(tokens, xrange(len(tokens))))
        letter_vectors = []
        print "creating vocabs for w2v & letters"
        for token in tqdm(tokens):
            letter_vector = letters2vec(token, self.vocab, dtype)
            letter_vectors.append(letter_vector)

        self.letter_vocab = np.vstack(letter_vectors)
        self.tensor = np.array(list(map(tokens_vocab.get, all_tokens)))
        self.word_vocab_size = len(uniq_tokens)
        self.letter_size = self.letter_vocab.shape[1]

        np.save(tensor_file, self.tensor)
        np.save(letter_vocab_file, self.letter_vocab)

    def load_preprocessed(self, vocab_file, tensor_file, letter_vocab_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.letter_vocab = np.load(letter_vocab_file)
        self.letter_size = self.letter_vocab.shape[1]
        self.word_vocab_size = self.letter_vocab.shape[0]

    def create_batches(self):
        temp_tensor = np.zeros((len(self.tensor) - self.seq_length) * self.seq_length)
        for index in range(len(self.tensor) - self.seq_length):
            temp_tensor[index * self.seq_length:index * self.seq_length + self.seq_length] \
                = self.tensor[index:index + self.seq_length]
        self.tensor = temp_tensor
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]

        self.batches = np.split(self.tensor.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        batch = self.batches[self.pointer]

        def lookup(x):
            d = np.array(self.letter_vocab[x, :]).astype(np.float32)
            n1 = np.random.choice([0, 1], size=(self.letter_size,), p=[0.95, 0.05])
            n2 = np.random.choice([0, 1], size=(self.letter_size,), p=[1 - self.noise_level, self.noise_level])
            d -= n1.astype(np.float32)
            d += n2.astype(np.float32)
            d[d < 0] = 0
            return d
        v_lookup = np.vectorize(lookup, otypes=[np.ndarray])
        out = v_lookup(batch.flat)
        self.pointer += 1
        return np.array(out.tolist()).reshape([self.batch_size, self.seq_length, -1])

    def reset_batch_pointer(self):
        self.pointer = 0

    def create_vocab(self, vocab_file, input_file):
        # preparation of vocab
        with codecs.open(input_file, "rt", encoding="utf-8") as f:
            data = f.read()
        counter = Counter(data)
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
