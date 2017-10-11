import tensorflow as tf

import argparse
import os
from six.moves import cPickle

from model import Model
from biLstm_model import BiLSTM, StackedBiLstm
from conv_model import Conv3LayerModel
from tqdm import tqdm
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('--prime', type=str, default=' ',
                        help='prime text or file with prime text')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='0 to use max at each timestep, '
                             '1 to sample at each timestep, '
                             '2 to sample on spaces')
    args = parser.parse_args()
    sample(args)


def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        _, vocab = cPickle.load(f)
    model = Model(saved_args, True)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

            print(model.sample(sess, vocab, args.prime))


def sample_multi(save_dir, data):
    with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'rb') as f:
        _, vocab = cPickle.load(f)
    #ATTENTION # TODO understand what model we want to choose.
    model = Conv3LayerModel(saved_args, True)
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25))
    with tf.Session(config=config) as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            vector = np.mean(model.sample(sess, vocab, data[0]), axis=0)
            vectors = np.zeros((len(data), vector.shape[0]))
            vectors[0, :] = vector
            for i in tqdm(range(1, len(data))):
                vectors[i, :] = np.mean(model.sample(sess, vocab, data[i]), axis=0)
                if(vectors[i, :] == np.zeros_like(vector)).all():
                    print(model.sample(sess, vocab, data[i]))
                    print(data[i])
                    print(len(data[i]))
                    exit()
    return vectors

if __name__ == '__main__':
    main()
