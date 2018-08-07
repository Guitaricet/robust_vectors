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
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

            return model.sample(sess, vocab, phrase)


def sample_multi(save_dir, data, model_type):
    with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'rb') as f:
        _, vocab = cPickle.load(f)

    if model_type == 'biLSTM':
        model = BiLSTM(saved_args, True)
    elif model_type == 'biSRU':
        model = BiLSTM(saved_args, True)
    elif model_type == 'stackBiLstm':
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

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25))
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            vector = np.mean(model.valid_run(sess, vocab, data[0]), axis=0)
            vectors = np.zeros((len(data), vector.shape[0]))
            vectors[0, :] = vector
            for i in tqdm(range(1, len(data))):
                vectors[i, :] = np.mean(model.valid_run(sess, vocab, data[i]), axis=0)
    return vectors


class RoVeSampler:
    def __init__(self, model_dir, model_type, sess, batch_size=64):
        with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
        with open(os.path.join(model_dir, 'chars_vocab.pkl'), 'rb') as f:
            _, vocab = cPickle.load(f)

        saved_args.batch_size = batch_size
        saved_args.seq_length = 1

        if model_type == 'biLSTM':
            model = BiLSTM(saved_args, True)
        elif model_type == 'biSRU':
            model = BiLSTM(saved_args, True)
        elif model_type == 'stackBiLstm':
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
            model = Model(saved_args, False)

        self.model = model
        self.vocab = vocab
        self.sess = sess
        self.saver = tf.train.Saver()
        self.ckpt = tf.train.get_checkpoint_state(model_dir)
        if sess is not None:
            self.saver.restore(sess, self.ckpt.model_checkpoint_path)

        assert self.ckpt and self.ckpt.model_checkpoint_path

    def restore(self, sess):
        self.sess = sess
        print('Model checkpoint path: ', self.ckpt.model_checkpoint_path)
        self.saver.restore(sess, self.ckpt.model_checkpoint_path)

    def sample_multi(self, texts_batch, pad=None):
        """
        :param texts_batch: list of strings
        :param pad: if specified, pads vectors and returns numpy ndarray of size
                     (batch_size=len(texts_batch), seq_len=pad, vector_size)
        :return: list of lists of numpy ndarrays if not pad
                  or numpy ndarray if pad
        """
        if self.sess is None:
            raise RuntimeError('get a not None session using .restore()')
        vectors = []
        for text in texts_batch:
            vector = self.model.valid_run(self.sess, self.vocab, text)
            vectors.append(vector)

        if pad is not None:
            vectors = self._pad(vectors, pad)

        return vectors

    def sample(self, texts_batch, pad=None):
        if self.sess is None:
            raise RuntimeError('get a not None session using .restore()')

        return self.model.sample(self.sess, self.vocab, texts_batch, batch_size=len(texts_batch), pad=pad)

    @staticmethod
    def _pad(vectors, seq_len):
        """
        Zero pad the vectors
        :param vectors: list of lists of word vectors
        :param seq_len: sequence length
        :return:
        """
        batch_size = len(vectors)
        vector_size = vectors[0][0].shape[0]

        padded = np.zeros([batch_size, seq_len, vector_size])
        for i, text in enumerate(vectors):
            for j, vector in enumerate(text):
                padded[i, j, :] = vector

        return padded


if __name__ == '__main__':
    main()

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25))
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()

        sampler = RoVeSampler('save/brown', 'sru', sess)
        not_vectors = sampler.sample_multi(['this is a text', 'this is anotuer'], pad=5)
        not_vectors2 = sampler.sample(['this is a text', 'this is anotuer'])
        pass
