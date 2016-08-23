import tensorflow as tf

import argparse
import os
from six.moves import cPickle
from model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=500,
                        help='number of characters to sample')
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

            print model.sample(sess, vocab, args.prime)

if __name__ == '__main__':
    main()
