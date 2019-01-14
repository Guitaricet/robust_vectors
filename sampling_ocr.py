import argparse
from glob import glob
from os.path import join

import numpy as np
from six.moves import cPickle

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="random, word2vec, robust", type=str, default="robust")
parser.add_argument("-s", "--save-dir", help="directory with stored robust model", type=str, default="save")
parser.add_argument("-t", "--model_type", help="type of model used to train", type=str, default="biSRU")
parser.add_argument("-i", "--input-dir", help="dir to go through")


args = parser.parse_args()

with open(join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, _ = cPickle.load(f)

if "robust" in args.mode:
    filenames = []
    phrases = []

    for filename in glob(join(args.input_dir, "*.txt")):
        filenames.append(filename)
        with open(filename, "rt") as f:
            lines = [line.strip() for line in f.readlines()]
            phrases.append(" ".join(lines))

    from sample import sample_multi
    results = np.vsplit(sample_multi(args.save_dir, phrases, args.model_type), len(phrases))

    for i in range(len(results)):
        np.savetxt(filenames[i] + ".rove", results[i])
