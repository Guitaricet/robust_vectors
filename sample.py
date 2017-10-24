import tensorflow as tf

import argparse
import os
from six.moves import cPickle

from cnn_lstm_model import ConvLSTMModel
from model import Model
from biLstm_model import BiLSTM, StackedBiLstm
from conv_model import Conv3LayerModel, Conv6LayerModel, Conv1d3Layer
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
    #sample(args)


def sample(save_dir, phrase):
    with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'rb') as f:
        _, vocab = cPickle.load(f)
    # TODO: adaptive model
    model = StackedBiLstm(saved_args, True)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

            return model.sample(sess, vocab, phrase)


def sample_multi(save_dir, data, model_type):
    with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'rb') as f:
        _, vocab = cPickle.load(f)
    #ATTENTION # TODO understand what model we want to choose.
    if model_type == 'biLSTM':
        model = BiLSTM(saved_args, True)
    elif model_type ==  'stackBiLstm':
        model = StackedBiLstm(saved_args, True)
    elif model_type == 'cnn3layers':
        model = Conv3LayerModel(saved_args, True)
    elif model_type == 'conv1d':
        model = Conv1d3Layer(saved_args, True)
    elif model_type == 'cnn6layers':
        model = Conv6LayerModel(saved_args, True)
    elif model_type == 'cnn_lstm':
        model = ConvLSTMModel(saved_args, True)
    else:
        model = Model(saved_args, True)

    #model = Conv3LayerModel(saved_args, True)
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
    return vectors

if __name__ == '__main__':
    main()
