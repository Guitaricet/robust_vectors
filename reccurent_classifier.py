import os
import re
import logging
from time import time
from random import random, choice
from datetime import datetime
from six.moves import cPickle

import colored_traceback.auto
from comet_ml import Experiment
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from tensorboardX import SummaryWriter
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score, accuracy_score

from sru import SRUCell
from utils import letters2vec


with open('comet.apikey') as f:
    apikey = f.read()
experiment = Experiment(api_key=apikey, project_name='rove_classifier', auto_metric_logging=False, disabled=True)

logger = logging.getLogger()

formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M'
)

fileHandler = logging.FileHandler('reccurent_classifier.log')
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)

logger.setLevel(logging.INFO)


ALPHABET = ['<UNK>', '\n'] + [s for s in """ abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'’’/\|_@#$%ˆ&* ̃‘+-=<>()[]{}"""]
ALPHABET = [t for t in ALPHABET if t not in ('(', ')')]


class IMDBDataset:
    ...


class MokoronDataset:
    """
    Dataset class for .csv-based datasets

    Alphabet should be the same as at RoVe train
    Can be used only in tf.session
    """

    def __init__(self,
                 filepath,
                 text_field,
                 label_field,
                 vocab,
                 alphabet=None,
                 max_text_length=128,
                 noise_level=0.0):
        self.noise_level = noise_level
        self.alphabet = alphabet or ALPHABET
        self.text_field = text_field
        self.label_field = label_field
        self.data = pd.read_csv(filepath)
        assert self.label_field in self.data.columns
        self.label2int = {l: i for i, l in enumerate(sorted(self.data[self.label_field].unique()))}
        self.label_placeholder = np.zeros(len(self.label2int))
        self.max_text_length = max_text_length
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data.iloc[idx]
        text = line[self.text_field].lower()
        # for mokoron dataset we should remove smiles
        if ')' not in self.alphabet:
            text = re.sub('[()]', '', text)

        label = self.label2int[line[self.label_field]]

        if self.noise_level > 0:
            text = self._noise_generator(text)

        text = self._preprocess(text)

        return text, label

    def _noise_generator(self, string):
        noised = ""
        for c in string:
            if random() > self.noise_level:
                noised += c
            if random() < self.noise_level:
                noised += choice(self.alphabet)
        return noised

    def _preprocess(self, text):
        tokens_vecs = np.zeros((self.max_text_length, 7*len(self.vocab)))
        tokens = word_tokenize(text)
        if len(tokens) > self.max_text_length:
            tokens = tokens[:self.max_text_length]
        for i, t in enumerate(tokens):
            x = letters2vec(t, self.vocab).reshape((1, 1, -1))
            tokens_vecs[i] = x

        return tokens_vecs


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._batch_pointer = 0
        self._indices = list(range(len(dataset)))
        if shuffle:
            self._shuffle()

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __next__(self):
        if self._batch_pointer * self.batch_size + self.batch_size >= len(self.dataset):
            raise StopIteration

        batch = []
        labels = []
        for bi in range(self.batch_size):
            idx = self._indices[self._batch_pointer * self.batch_size + bi]
            x, y = self.dataset[idx]
            batch.append(x)
            labels.append(y)

        self._batch_pointer += 1

        batch = np.array(batch)
        assert batch.shape[0] == self.batch_size
        return batch, labels

    def __iter__(self):
        self._reset_batch_pointer()
        return self

    def _reset_batch_pointer(self):
        self._batch_pointer = 0
        if self.shuffle:
            self._shuffle()

    def _shuffle(self):
        np.random.shuffle(self._indices)


# based on github.com/roomylee/rnn-text-classification-tf
class RNN:
    def __init__(self, sequence_length, num_classes,
                 cell_type, embeddings_size, hidden_size, input_tensor=None):

        with tf.variable_scope('placeholder'):
            self.input_vectors = tf.placeholder(tf.float32, shape=[None, sequence_length, embeddings_size], name='input_vectors')
            self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')
            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

        if input_tensor is not None:
            self.input_vectors = input_tensor

        labels_one_hot = tf.one_hot(self.labels, num_classes)
        text_length = self._length(self.input_vectors)

        # Recurrent Neural Network
        with tf.variable_scope('rnn'):
            rnn_input_layer = tf.layers.Dense(hidden_size, activation=tf.nn.relu, name='linear')
            rnn_input = rnn_input_layer.apply(self.input_vectors)

            tf.summary.histogram('rnn_input_linear', rnn_input_layer.weights[0])
            tf.summary.histogram('rnn_input_linear_bias', rnn_input_layer.weights[1])

            cell = self._get_cell(hidden_size, cell_type)
            self.h_outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                                                  inputs=rnn_input,
                                                  sequence_length=text_length,
                                                  dtype=tf.float32)

            self.last_output = self.h_outputs[:, -1, :]
            self.last_output = tf.layers.dropout(self.last_output, self.dropout_prob)

        with tf.variable_scope('outputs'):
            logits_layer = tf.layers.Dense(num_classes,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           name='logits')
            self.logits = logits_layer.apply(self.last_output)
            self.predictions = tf.argmax(self.logits, 1, name='predictions')

            tf.summary.histogram('logits', logits_layer.weights[0])
            tf.summary.histogram('logits_bias', logits_layer.weights[1])

            # Calculate mean cross-entropy loss
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=labels_one_hot)
            self.loss = tf.reduce_mean(losses, name='loss')

        with tf.variable_scope('metrics'):
            self.labels_int = tf.argmax(labels_one_hot, axis=1)
            accuracy, self.accuracy_op = tf.metrics.accuracy(self.labels_int, self.predictions, name='accuracy')
            precision, self.precision_op = tf.metrics.precision(self.labels_int, self.predictions, name='precision')
            recall, self.recall_op = tf.metrics.recall(self.labels_int, self.predictions, name='recall')
            # see https://stackoverflow.com/a/50251763 for multilabel f1 score in tf
            self.metrics_ops = tf.group(self.accuracy_op, self.precision_op, self.recall_op)

        tf.summary.scalar('loss', self.loss)

        streaming_metrics_key = 'streaming_metrics'
        acc = tf.summary.scalar('accuracy_streaming', accuracy, collections=[streaming_metrics_key])
        pre = tf.summary.scalar('precision_streaming', precision, collections=[streaming_metrics_key])
        rec = tf.summary.scalar('recall_streaming', recall, collections=[streaming_metrics_key])

        self.summary = tf.summary.merge_all()
        self.metrics_summary = tf.summary.merge([acc, pre, rec])

    @staticmethod
    def _get_cell(hidden_size, cell_type):
        cell_type = cell_type.lower()

        if cell_type == "vanilla":
            return tf.nn.rnn_cell.BasicRNNCell(hidden_size)
        elif cell_type == "lstm":
            return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        elif cell_type == "gru":
            return tf.nn.rnn_cell.GRUCell(hidden_size)
        elif cell_type == "sru":
            return SRUCell(hidden_size, state_is_tuple=False)
        else:
            raise ValueError(cell_type)

    # Lengths of the sequence data
    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.reduce_sum(tf.abs(seq), axis=2))
        length = tf.reduce_sum(relevant, axis=1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def clip_grads(loss, clip_norm, variables=None):
        if variables is None:
            variables = tf.trainable_variables()
        grads = tf.gradients(loss, variables)
        clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm)
        return zip(clipped_grads, variables)

    def evaluate(self, dataloader, sess, rove_input, frac=1.0):
        """
        This function is needed because tf.metrics does not support multiclass weighted f1
        """
        predictions = []
        labels = []

        max_iters = int(len(dataloader) * frac)
        for i, (batch, label) in enumerate(dataloader):
            if i >= max_iters:
                break
            feed_dict = {
                rove_input: batch,
                self.labels: label,
                self.dropout_prob: 0
            }
            preds, labs = sess.run([self.predictions, self.labels_int], feed_dict=feed_dict)
            predictions += list(preds)
            labels += list(labs)

        print(f'true : {labels[:20]}')
        print(f'pred : {predictions[:20]}')
        res = {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted')
        }
        return res


def train(epochs=10,
          lr=1e-3,
          batch_size=64,
          rnn_size=256,
          dropout=0.5,
          seq_len=32,
          use_annealing=False,
          use_gradclip=False,
          gradclip_norm=5,
          noise_level=0.05,
          max_iters=2500,
          save_model_path='save/classifier/mokoron'):

    hyperparams = {'epochs': epochs,
                   'dropout': dropout,
                   'learning_rate': lr,
                   'use_annealing': use_annealing,
                   'use_gradclip': use_gradclip,
                   'rnn_size': rnn_size,
                   'batch_size': batch_size,
                   'noise_level': noise_level,
                   'seq_len': seq_len,
                   'max_iters': max_iters}

    if use_gradclip:
        hyperparams['gradclip_norm'] = gradclip_norm

    experiment.log_multiple_params(hyperparams)

    rove_path = 'save/ruscorpora_bisru'
    with open(os.path.join(rove_path, 'chars_vocab.pkl'), 'rb') as f:
        _, vocab = cPickle.load(f)

    save_results_path = 'results/%s_%s.csv' % ('rove', 'mokoron')

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    save_model_path = save_model_path + current_time
    writer_name = f'runs/{current_time}/mokoron'
    writer = SummaryWriter(writer_name + '/X', comment='_test')

    logger.info('Preparing datasets')
    train_dataset = MokoronDataset('../text_classification/data/mokoron/train.csv',
                                   text_field='text_spellchecked',
                                   label_field='sentiment',
                                   vocab=vocab,
                                   noise_level=noise_level,
                                   max_text_length=seq_len)
    val_dataset = MokoronDataset('../text_classification/data/mokoron/validation.csv',
                                 text_field='text_spellchecked',
                                 label_field='sentiment',
                                 vocab=vocab,
                                 noise_level=noise_level,
                                 max_text_length=seq_len)
    val_original_dataset = MokoronDataset('../text_classification/data/mokoron/validation.csv',
                                          text_field='text_original',
                                          label_field='sentiment',
                                          vocab=vocab,
                                          noise_level=0,
                                          max_text_length=seq_len)
    test_dataset = MokoronDataset('../text_classification/data/mokoron/test.csv',
                                  text_field='text_spellchecked',
                                  label_field='sentiment',
                                  vocab=vocab,
                                  noise_level=noise_level,
                                  max_text_length=seq_len)
    test_original_dataset = MokoronDataset('../text_classification/data/mokoron/test.csv',
                                           text_field='text_original',
                                           label_field='sentiment',
                                           vocab=vocab,
                                           noise_level=0,
                                           max_text_length=seq_len)

    train_dataloader = DataLoader(train_dataset, batch_size, True)
    val_dataloader = DataLoader(val_dataset, batch_size, True)
    val_original_dataloader = DataLoader(val_original_dataset, batch_size, True)
    test_dataloader = DataLoader(test_dataset, batch_size, True)
    test_original_dataloader = DataLoader(test_original_dataset, batch_size, True)

    logger.info('Building graph')
    graph = tf.Graph()

    with graph.as_default():
        ckpt = tf.train.get_checkpoint_state(rove_path)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
        rove_input = graph.get_tensor_by_name('input:0')
        rove_output = graph.get_tensor_by_name('target:0')
        tf.stop_gradient(rove_output)
        model = RNN(sequence_length=seq_len,
                    num_classes=2,
                    cell_type='sru',
                    embeddings_size=300,
                    hidden_size=rnn_size,
                    input_tensor=rove_output)

    logger.info('Starting training process')

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25))
    with tf.Session(config=config, graph=graph) as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)

        global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope('not_adam'):  # crutch
            if use_annealing:
                lr_op = tf.train.cosine_decay_restarts(lr, global_step, len(train_dataloader), t_mul=1.0)
                optimizer = tf.train.AdamOptimizer(lr_op)
            else:
                optimizer = tf.train.AdamOptimizer(lr)

            if use_gradclip:
                gradients = model.clip_grads(model.loss, gradclip_norm)

                train_op = optimizer.apply_gradients(
                    gradients, global_step=global_step
                )
            else:
                train_op = optimizer.minimize(model.loss, global_step=global_step)

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        sw = tf.summary.FileWriter(writer_name + '_train', sess.graph)
        sw_val = tf.summary.FileWriter(writer_name + '_val', sess.graph)
        sw_val_orig = tf.summary.FileWriter(writer_name + '_val_original', sess.graph)

        #
        # Training loop
        #

        exit_iteration_flag = False
        for epoch in range(epochs):
            if exit_iteration_flag:
                break

            logger.info(f'Epoch {epoch}')
            for i, (batch, label) in enumerate(train_dataloader):
                if i * (epoch+1) > max_iters:
                    logger.warning('Exited training loop due to max_iters rule')
                    exit_iteration_flag = True
                    break

                sess.run(tf.local_variables_initializer())
                step = sess.run(global_step)

                feed_dict = {
                    rove_input: batch,
                    model.labels: label,
                    model.dropout_prob: dropout
                }
                loss, summary, _ = sess.run([model.loss, model.summary, train_op], feed_dict=feed_dict)
                sw.add_summary(summary, step)
                experiment.set_step(step)
                experiment.log_metric('loss', loss)

                if step % 100 == 0:
                    evaluate(model, val_dataloader, sess, rove_input, sw_val, step)
                    evaluate(model, train_dataloader, sess, rove_input, sw, step)
                    evaluate(model, val_original_dataloader, sess, rove_input, sw_val_orig, step)

            # checkpoint
            os.makedirs(save_model_path, exist_ok=True)
            saver.save(sess, save_model_path)

            # evaluate
            logger.info('Model evaluation')
            train_metrics = model.evaluate(val_dataloader, sess, rove_input, frac=0.05)
            writer.add_scalar('f1_train', train_metrics['f1'], step)
            writer.add_scalar('accuracy_train', train_metrics['accuracy'], step)

            val_metrics = model.evaluate(val_dataloader, sess, rove_input, frac=0.25)
            writer.add_scalar('f1_val', val_metrics['f1'], step)
            writer.add_scalar('accuracy_val', val_metrics['accuracy'], step)
            experiment.log_epoch_end(epoch, step)

        logger.info('Training is finished')

        #
        # Test loop
        #
        logging.info('Test evaluation')

        # acc_test,f1_test,noise_level_test,model_type,noise_level_train,acc_train,f1_train
        results = []
        for _ in tqdm(range(10), leave=False):  # 10 times for statistics
            test_metrics = model.evaluate(test_dataloader, sess, rove_input)
            writer.add_scalar('f1_test', test_metrics['f1'], step)
            writer.add_scalar('accuracy_test', test_metrics['accuracy'], step)

            results.append({'acc_test': test_metrics['accuracy'],
                            'f1_test': test_metrics['f1'],
                            'noise_level_test': noise_level,
                            'model_type': 'rove_rnn',
                            'noise_level_train': noise_level})

        test_metrics = model.evaluate(test_original_dataloader, sess, rove_input)
        writer.add_scalar('f1_test', test_metrics['f1'], step)
        writer.add_scalar('accuracy_test', test_metrics['accuracy'], step)

        results.append({'acc_test': test_metrics['accuracy'],
                        'f1_test': test_metrics['f1'],
                        'noise_level_test': -1,
                        'model_type': 'rove_rnn',
                        'noise_level_train': noise_level})
        os.makedirs(os.path.dirname(save_results_path), exist_ok=True)
        if os.path.exists(save_results_path):
            old_results = pd.read_csv(save_results_path)
        else:
            old_results = pd.DataFrame()

        results_df = pd.concat([old_results, pd.DataFrame(results)], sort=False)
        results_df.to_csv(save_results_path, index=False)


def evaluate(model, dataloader, sess, rove_input, sw, step, iters=5):
    sess.run(tf.local_variables_initializer())
    for j, (batch, label) in enumerate(dataloader):
        if j >= iters:
            break

        feed_dict = {
            rove_input: batch,
            model.labels: label,
            model.dropout_prob: 0
        }
        sess.run(model.metrics_ops, feed_dict=feed_dict)
    summ = sess.run(model.metrics_summary)
    sw.add_summary(summ, step)
    return summ


def noise_experiment(**kwargs):
    noise_levels = np.concatenate([np.arange(0, 0.05, 0.01), np.arange(0.05, 0.2, 0.025)])
    for noise_level in noise_levels:
        train(noise_level=noise_level, **kwargs)


if __name__ == '__main__':
    noise_experiment(max_iters=2500)
