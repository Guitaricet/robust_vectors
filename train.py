from __future__ import print_function

from tqdm import tqdm
import tensorflow as tf

import argparse
import time
import os
import numpy as np
import codecs
import logging
from six.moves import cPickle

from scipy.spatial.distance import cosine
from sklearn.metrics import roc_auc_score

import math
from utils import TextLoader, noise_generator
from model import Model
from biLstm_model import BiLSTM, StackedBiLstm
from conv_model import  Conv3LayerModel, Conv6LayerModel

logging.basicConfig(filename='validation.log', filemode='w', level=logging.INFO)
logging.getLogger("tensorflow").setLevel(logging.WARNING) # suppress tf use info logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/42bin_haber',
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=256,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='cnn, rnn, stackBiLstm, biLSTM, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=8,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=25,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=2000,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                        help='decay rate for rmsprop')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.8,
                        help='decay rate for rmsprop')
    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    parser.add_argument('--w2v_size', type=int, default=300,
                        help='number of dimensions in word embedding')
    parser.add_argument('--noise_level', type=float, default=0.05,
                        help='probability og typo')
    args = parser.parse_args()
    print(args)
    train(args)


def get_validate_phrases(args):
    pairs = []
    phrases = []
    for filename in ["msr_paraphrase_valid.txt"]:
        with codecs.open(os.path.join("data", "MRPC", filename), encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.strip().split("\t")
                pair = {"text_1": parts[3], "text_2": parts[4], "decision": float(parts[0])}
                pairs.append(pair)
    pairs = pairs
    true = [x["decision"] for x in pairs]
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, _ = cPickle.load(f)

    for pair in pairs:
        phrases.append(noise_generator(pair["text_1"], args.noise_level, chars))
        phrases.append(noise_generator(pair["text_2"], args.noise_level, chars))

    return phrases, true


def train(args):
    # check compatibility if training is continued from previously saved model
    if args.init_from is None:
        print(args.init_from)
        data_loader = TextLoader(args)
        ckpt = ''
    else:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from), " %s must be a a path" % args.init_from
        assert os.path.isfile(
            os.path.join(args.init_from, "config.pkl")), "config.pkl file does not exist in path %s" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,
                                           "chars_vocab.pkl")), "chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        # open old config and check if models are compatible
        with codecs.open(os.path.join(args.init_from, 'config.pkl'),'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme] == vars(args)[
                checkme], "Command line argument and saved model disagree on '%s' " % checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with codecs.open(os.path.join(args.init_from, 'chars_vocab.pkl'), 'rb') as f:
            saved_chars, saved_vocab = cPickle.load(f)

        data_loader = TextLoader(args, chars=saved_chars, vocab=saved_vocab)

        assert saved_chars == data_loader.chars, "Data and loaded model disagree on character set!"
        assert saved_vocab == data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    args.vocab_size = data_loader.vocab_size
    args.letter_size = data_loader.letter_size
    args.word_vocab_size = data_loader.word_vocab_size

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    if args.model == 'biLSTM':
        model = BiLSTM(args)
        train_bidirectional_model(model, data_loader, args, ckpt)
    elif args.model == 'stackBiLstm':
        model = StackedBiLstm(args)
        train_bidirectional_model(model, data_loader, args, ckpt)
    elif args.model == 'cnn3layers':
        model = Conv3LayerModel(args)
        train_cnn_model(model, data_loader, args, ckpt)
    elif args.model == 'cnn6layers':
        model = Conv6LayerModel(args)
        train_cnn_model(model, data_loader, args, ckpt)
    else:
        model = Model(args)
        train_one_forward_model(model, data_loader, args, ckpt)


def train_one_forward_model(model, data_loader, args, ckpt):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3))
    with codecs.open(os.path.join(args.save_dir, 'chars_vocab.pkl'),'rb') as f:
        saved_chars, saved_vocab = cPickle.load(f)

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for step in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** step)))
            data_loader.reset_batch_pointer()
            state = model.initial_state.eval()
            for b in tqdm(range(data_loader.num_batches)):
                start = time.time()
                batch, change = data_loader.next_batch()
                feed = {model.input_data: batch, model.change: change, model.initial_state: state}
                if b % 113 != 0:
                    train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                else:
                    train_loss = sess.run([model.cost], feed)
                    end = time.time()
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                          .format(step * data_loader.num_batches + b,
                                  args.num_epochs * data_loader.num_batches,
                                  step, train_loss[0], end - start))
                if (step * data_loader.num_batches + b) % args.save_every == 0:
                    print("Validation")
                    valid_data, true_labels = get_validate_phrases(args)
                    vector = np.mean(model.valid_run(sess, saved_vocab, valid_data[0]), axis=0)
                    vectors = np.zeros((len(valid_data), vector.shape[0]))
                    vectors[0, :] = vector
                    for i in tqdm(range(1, len(valid_data))):
                        vectors[i, :] = np.mean(model.valid_run(sess, saved_vocab, valid_data[i]), axis=0)
                    valid_results = np.vsplit(vectors,len(valid_data))
                    pred = []
                    for i in range(0, len(valid_results), 2):
                        v1 = valid_results[i]
                        v2 = valid_results[i + 1]
                        pred.append(1 - cosine(v1, v2))
                        if math.isnan(pred[-1]):
                            pred[-1] = 0.5
                    print("="*30)
                    print("RocAuc at step %d: %f" % (step, roc_auc_score(true_labels, pred)))
                    print("="*30)

                    # Save model and checkpoints
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))
        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=args.num_epochs * data_loader.num_batches)
        print("final model saved to {}".format(checkpoint_path))


def train_bidirectional_model(model, data_loader, args, ckpt):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3))
    with codecs.open(os.path.join(args.save_dir, 'chars_vocab.pkl'),'rb') as f:
        saved_chars, saved_vocab = cPickle.load(f)

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            model.initial_state_fw = tf.convert_to_tensor(model.initial_state_fw)
            model.initial_state_bw = tf.convert_to_tensor(model.initial_state_bw)
            state_fw = model.initial_state_fw.eval()
            state_bw = model.initial_state_bw.eval()
            for step in tqdm(range(data_loader.num_batches)):
                start = time.time()
                batch, change = data_loader.next_batch()
                feed = {model.input_data: batch,
                        model.change: change,
                        model.initial_state_fw: state_fw,
                        model.initial_state_bw: state_bw
                        }
                if step % 113 != 0:
                    train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                    state_fw = state[0]
                    state_bw = state[1]
                else:
                    train_loss = sess.run([model.cost], feed)
                    end = time.time()
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                          .format(e * data_loader.num_batches + step,
                                  args.num_epochs * data_loader.num_batches,
                                  e, train_loss[0], end - start))
                if (e * data_loader.num_batches + step) % args.save_every == 0:
                    #Validation
                    print("Validation")
                    valid_data, true_labels = get_validate_phrases(args)
                    vector = np.mean(model.valid_run(sess, saved_vocab, valid_data[0]), axis=0)
                    vectors = np.zeros((len(valid_data), vector.shape[0]))
                    vectors[0, :] = vector
                    for i in tqdm(range(1, len(valid_data))):
                        vectors[i, :] = np.mean(model.valid_run(sess, saved_vocab, valid_data[i]), axis=0)
                    valid_results = np.vsplit(vectors,len(valid_data))
                    pred = []
                    for i in range(0, len(valid_results), 2):
                        v1 = valid_results[i]
                        v2 = valid_results[i + 1]
                        pred.append(1 - cosine(v1, v2))
                        if math.isnan(pred[-1]):
                            pred[-1] = 0.5
                    roc_auc_validation_score = roc_auc_score(true_labels, pred)
                    print("="*30)
                    print("RocAuc at step %d: %f" % (step, roc_auc_validation_score))
                    print("="*30)
                    logging.info("RocAuc at step %d and epoch %d : %f"%( step, e, roc_auc_validation_score))
                    # Save model to save directory
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + step)
                    print("model saved to {}".format(checkpoint_path))
        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=args.num_epochs * data_loader.num_batches)
        print("final model saved to {}".format(checkpoint_path))


def train_cnn_model(model, data_loader, args, ckpt):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3))
    with codecs.open(os.path.join(args.save_dir, 'chars_vocab.pkl'),'rb') as f:
        saved_chars, saved_vocab = cPickle.load(f)

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for epoch in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** epoch)))
            data_loader.reset_batch_pointer()
            for _step in tqdm(range(data_loader.num_batches)):
                start = time.time()
                batch, change = data_loader.next_batch()
                feed = {model.input_data: batch}
                if _step % 113 != 0:
                    train_loss, _ = sess.run([model.cost, model.train_op], feed)
                else:
                    train_loss = sess.run([model.cost], feed)
                    end = time.time()
                    print("{}/{} (epoch{}), train_loss = {:.3f}, time/batch = {:.3f}" \
                          .format(epoch * data_loader.num_batches + _step,
                                  args.num_epochs * data_loader.num_batches,
                                  epoch, train_loss[0], end - start))
                if (epoch * data_loader.num_batches + _step) % args.save_every == 0:
                    if _step == 0:
                        continue
                    print("Validation")
                    valid_data, true_labels = get_validate_phrases(args)
                    vector = np.mean(model.valid_run(sess, saved_vocab, valid_data[0]), axis=0)
                    vectors = np.zeros((len(valid_data), vector.shape[0]))
                    vectors[0, :] = vector
                    for i in tqdm(range(1, len(valid_data))):
                        vectors[i, :] = np.mean(model.valid_run(sess, saved_vocab, valid_data[i]), axis=0)
                    valid_results = np.vsplit(vectors,len(valid_data))
                    pred = []
                    for i in range(0, len(valid_results), 2):
                        v1 = valid_results[i]
                        v2 = valid_results[i + 1]
                        pred.append(1 - cosine(v1, v2))
                        if math.isnan(pred[-1]):
                            logging.info("prediction contains nan")
                            pred[-1] = 0.5
                    roc_auc_validation_score = roc_auc_score(true_labels, pred)
                    print("="*30)
                    print("RocAuc at epoch %d: %f" % (epoch, roc_auc_validation_score))
                    print("="*30)
                    logging.info("RocAuc at epoch %d and step %d : %f"%(epoch, _step, roc_auc_validation_score))
                    # Save model and checkpoints
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=epoch * data_loader.num_batches + _step)
                    print("model saved to {}".format(checkpoint_path))


        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=args.num_epochs * data_loader.num_batches)
        print("final model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
    main()
