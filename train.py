from __future__ import print_function

import argparse
import codecs
import logging
import math
import os
import time

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine
from six.moves import cPickle
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

from biLstm_model import BiLSTM, StackedBiLstm
from cnn_lstm_model import ConvLSTMModel
from conv_model import Conv3LayerModel, Conv6LayerModel, Conv1d3Layer
from rnn_model import RNNModel
from sentiment_sampling import linear_svm
from utils import TextLoader, noise_generator

logging.basicConfig(filename='validation.log', filemode='w', level=logging.INFO)
logging.getLogger("tensorflow").setLevel(logging.WARNING)  # suppress tf use info logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data',
                        help='data directory containing input.txt (plaintext)')
    parser.add_argument('--save-dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--rnn-size', type=int, default=256,
                        help='size of RNN hidden state')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='cnn, rnn, stackBiLstm, biLSTM, gru, or lstm')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64,
                        help='minibatch size')
    parser.add_argument('--seq-length', dest='seq_length', type=int, default=32,
                        help='RNN sequence length')
    parser.add_argument('--num-epochs', type=int, default=1,
                        help='number of epochs')
    parser.add_argument('--save-every', type=int, default=2000,
                        help='save frequency')
    parser.add_argument('--grad-clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning-rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--decay-rate', type=float, default=0.97,
                        help='decay rate for rmsprop')
    parser.add_argument('--dropout-keep-prob', type=float, default=0.8,
                        help='dropout keep probability')
    parser.add_argument('--type', type=str, default=None,
                        help="""type of task
                                1)para - paraphrase
                                2)sentiment
                                3)entail - entailment detection
                                4)quora
                                """)

    parser.add_argument('--init-from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    parser.add_argument('--w2v-size', type=int, default=300,
                        help='number of dimensions in word embedding')
    parser.add_argument('--noise-level', type=float, default=0.05,
                        help='probability og typo')
    args = parser.parse_args()
    print(args)
    train(args)


def get_validate_phrases(args):
    pairs = []
    phrases = []
    for filename in ["valid.txt"]:
        with codecs.open(os.path.join(args.data_dir, filename), encoding="utf-8") as f:
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


def get_validate_entailment(args):
    pairs = []
    phrases = []
    import pandas as pd
    valid_path = os.path.join(args.data_dir, "valid.txt")
    if valid_path.__contains__("quora"):
        full_df = pd.read_csv(valid_path, sep='\t')[:300]
        decision = "duplicate"
    else:
        decision = "gold_label"
        full_df = pd.read_csv(valid_path)
    for index, row in full_df.iterrows():
        pair = {"text_1": row['sentence1'], "text_2": row["sentence2"],
                "decision": int(row[decision])} #ist(filter(lambda x: x.isdigit(), row["gold_label"]))[0]
        pairs.append(pair)

    pairs = pairs
    true = [x["decision"] for x in pairs]
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, _ = cPickle.load(f)

    for pair in pairs:
        phrases.append(noise_generator(pair["text_1"], args.noise_level, chars))
        phrases.append(noise_generator(pair["text_2"], args.noise_level, chars))

    return phrases, true


def get_data_for_sentiment(args):
    import pandas as pd
    valid_path = os.path.join(args.data_dir, "valid.txt")
    full_data = pd.read_csv(valid_path)[:300]
    dt = full_data.loc[:, 'SentimentText']
    y_target = full_data['Sentiment']
    print(dt[:2], y_target[:2])
    return dt, y_target.values


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
        with codecs.open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
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

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    if args.model == 'biLSTM':
        model = BiLSTM(args)
        train_bidirectional_model(model, data_loader, args, ckpt)
    elif args.model == 'biSRU':
        model = BiLSTM(args)
        train_bidirectional_model(model, data_loader, args, ckpt)
    elif args.model == 'stackBiLstm':
        model = StackedBiLstm(args)
        train_bidirectional_model(model, data_loader, args, ckpt)
    elif args.model == 'cnn3layers':
        model = Conv3LayerModel(args)
        train_cnn_model(model, data_loader, args, ckpt)
    elif args.model == 'conv1d':
        model = Conv1d3Layer(args)
        train_cnn_model(model, data_loader, args, ckpt)
    elif args.model == 'cnn6layers':
        model = Conv6LayerModel(args)
        train_cnn_model(model, data_loader, args, ckpt)
    elif args.model == 'cnn_lstm':
        model = ConvLSTMModel(args)
        train_one_forward_model(model, data_loader, args, ckpt)
    else:
        model = RNNModel(args)
        train_one_forward_model(model, data_loader, args, ckpt)


def train_one_forward_model(model, data_loader, args, ckpt):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3))
    with codecs.open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        saved_chars, saved_vocab = cPickle.load(f)

    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
    save_model_path = os.path.join(args.save_dir, 'model.tf')
    # input_ = {
    #     'x': model.input_data,
    #     'change': model.change,
    #     'init_state': model.initial_state
    # }
    # output_ = {
    #     ''
    # }

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
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"\
                          .format(step * data_loader.num_batches + b,
                                  args.num_epochs * data_loader.num_batches,
                                  step, train_loss[0], end - start))
                if (step * data_loader.num_batches + b) % args.save_every == 0:
                    if args.type is not None:
                        print("Validation")
                        if args.type != 'sentiment':
                            if args.type == 'para':
                                valid_data, true_labels = get_validate_phrases(args)
                            elif args.type == 'entail':
                                valid_data, true_labels = get_validate_entailment(args)
                            else:
                                raise ValueError(args.type)
                            vector = np.mean(model.valid_run(sess, saved_vocab, valid_data[0]), axis=0)
                            vectors = np.zeros((len(valid_data), vector.shape[0]))
                            vectors[0, :] = vector
                            for i in tqdm(range(1, len(valid_data))):
                                vectors[i, :] = np.mean(model.valid_run(sess, saved_vocab, valid_data[i]), axis=0)
                            valid_results = np.vsplit(vectors, len(valid_data))
                            pred = []
                            for i in range(0, len(valid_results), 2):
                                v1 = valid_results[i]
                                v2 = valid_results[i + 1]
                                pred.append(1 - cosine(v1, v2))
                                if math.isnan(pred[-1]):
                                    pred[-1] = 0.5
                            roc_auc_validation_score = roc_auc_score(true_labels, pred)
                            f1_validation_score = f1_score(true_labels, pred)
                            acc_validation_score = accuracy_score(true_labels, pred)

                        elif args.type == 'sentiment':
                            valid_data, true_labels = get_data_for_sentiment(args)
                            vector = np.mean(model.valid_run(sess, saved_vocab, valid_data[0]), axis=0)
                            vectors = np.zeros((len(valid_data), vector.shape[0]))
                            vectors[0, :] = vector
                            for i in tqdm(range(1, len(valid_data))):
                                vectors[i, :] = np.mean(model.valid_run(sess, saved_vocab, valid_data[i]), axis=0)
                            valid_results = np.squeeze(np.vsplit(vectors, len(valid_data)))
                            idx_tosplit = int(0.2 * len(valid_results))
                            valid_train = valid_results[idx_tosplit:]
                            valid_test = valid_results[:idx_tosplit]
                            train_label = true_labels[idx_tosplit:]
                            test_label = true_labels[:idx_tosplit]
                            roc_auc_validation_score = linear_svm(valid_train, valid_test, train_label, test_label)
                        else:
                            raise ValueError(args.type)

                        print("="*30)
                        print("RocAuc at step %d: %f" % (step, roc_auc_validation_score))
                        print("="*30)
                        logging.info("RocAuc at step %d and epoch %d : %f" % (step, b, roc_auc_validation_score))

                    # Save model and checkpoints
                    saver.save(sess, checkpoint_path, global_step=step * data_loader.num_batches + b)
                    # tf.saved_model.simple_save(sess, save_model_path)
                    print("model saved to {}".format(checkpoint_path))

        saver.save(sess, checkpoint_path, global_step=args.num_epochs * data_loader.num_batches)
        # tf.saved_model.simple_save(sess, save_model_path)
        print("final model saved to {}".format(checkpoint_path))


def train_bidirectional_model(model, data_loader, args, ckpt):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3))
    with codecs.open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
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
                    # Save model to save directory
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + step)
                    print("model saved to {}".format(checkpoint_path))
                    # if step == 0:
                    #     continue
                    # Validation
                    print("Validation")
                    if args.type is not None:
                        if args.type != 'sentiment':
                            if args.type == 'para':
                                valid_data, true_labels = get_validate_phrases(args)
                            if args.type == 'entail':
                                valid_data, true_labels = get_validate_entailment(args)
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
                        if args.type == 'sentiment':
                            valid_data, true_labels = get_data_for_sentiment(args)
                            vector = np.mean(model.valid_run(sess, saved_vocab, valid_data[0]), axis=0)
                            vectors = np.zeros((len(valid_data), vector.shape[0]))
                            vectors[0, :] = vector
                            for i in tqdm(range(1, len(valid_data))):
                                vectors[i, :] = np.mean(model.valid_run(sess, saved_vocab, valid_data[i]), axis=0)
                            valid_results = np.squeeze(np.vsplit(vectors, len(valid_data)))
                            idx_tosplit = int(0.2 * len(valid_results))
                            valid_train = valid_results[idx_tosplit:]
                            valid_test = valid_results[:idx_tosplit]
                            train_label = true_labels[idx_tosplit:]
                            test_label = true_labels[:idx_tosplit]
                            roc_auc_validation_score = linear_svm(valid_train, valid_test, train_label, test_label)
                        print("="*30)
                        print("RocAuc at epoch %d: %f" % (e, roc_auc_validation_score))
                        print("="*30)
                        logging.info("RocAuc at step %d and epoch %d : %f" % (step, e, roc_auc_validation_score))
                    else:
                        warning_message = 'No validation is performed due to non-specified --type parameter'
                        print(warning_message)
                        logging.warning(warning_message)

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
            for step in tqdm(range(data_loader.num_batches)):
                start = time.time()
                batch, change = data_loader.next_batch()
                feed = {model.input_data: batch}
                if step % 113 != 0:
                    train_loss, _ = sess.run([model.cost, model.train_op], feed)
                else:
                    train_loss = sess.run([model.cost], feed)
                    end = time.time()
                    print("{}/{} (epoch{}), train_loss = {:.3f}, time/batch = {:.3f}" \
                          .format(epoch * data_loader.num_batches + step,
                                  args.num_epochs * data_loader.num_batches,
                                  epoch, train_loss[0], end - start))
                if (epoch * data_loader.num_batches + step) % args.save_every == 0:
                    # Save model and checkpoints
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=epoch * data_loader.num_batches + step)
                    print("model saved to {}".format(checkpoint_path))

                    print("Validation")
                    if args.type != 'sentiment':
                        if args.type == 'para':
                            valid_data, true_labels = get_validate_phrases(args)
                        if args.type == 'entail':
                            valid_data, true_labels = get_validate_entailment(args)
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
                    if (args.type == 'sentiment'):
                        valid_data, true_labels = get_data_for_sentiment(args)
                        vector = np.mean(model.valid_run(sess, saved_vocab, valid_data[0]), axis=0)
                        vectors = np.zeros((len(valid_data), vector.shape[0]))
                        vectors[0, :] = vector
                        for i in tqdm(range(1, len(valid_data))):
                            vectors[i, :] = np.mean(model.valid_run(sess, saved_vocab, valid_data[i]), axis=0)
                        valid_results = np.squeeze(np.vsplit(vectors, len(valid_data)))
                        idx_tosplit = int(0.2 * len(valid_results))
                        valid_train = valid_results[idx_tosplit:]
                        valid_test = valid_results[:idx_tosplit]
                        train_label = true_labels[idx_tosplit:]
                        test_label = true_labels[:idx_tosplit]
                        roc_auc_validation_score = linear_svm(valid_train, valid_test, train_label, test_label)
                    print("="*30)
                    print("RocAuc at epoch %d: %f" % (epoch, roc_auc_validation_score))
                    print("="*30)
                    logging.info("RocAuc at epoch %d and step %d : %f"%(epoch, step, roc_auc_validation_score))

        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=args.num_epochs * data_loader.num_batches)
        print("final model saved to {}".format(checkpoint_path))


if __name__ == '__main__':
    main()
