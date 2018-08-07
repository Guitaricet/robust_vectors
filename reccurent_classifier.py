import logging
from random import random, choice

import numpy as np
import pandas as pd
import tensorflow as tf
from pymystem3 import Mystem
from sklearn.metrics import f1_score, accuracy_score
from tensorboardX import SummaryWriter

from sru import SRUCell
from sample import RoVeSampler


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


CLIP_NORM = 0.1

# TODO: remove smiles from mokoron
# TODO: dataloader use rove vocab(?)

ALPHABET = ['<UNK>', '\n'] + [s for s in """ abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'’’/\|_@#$%ˆ&* ̃‘+-=<>()[]{}"""]


class IMDBDataset:
    ...


class MokoronDataset:
    """
    Dataset class for .csv-based datasets

    Alphabet should be the same as at RoVe train
    Can be used only in tf.session
    """
    noise_level = 0

    def __init__(self,
                 filepath,
                 text_field,
                 label_field,
                 alphabet=None,
                 max_text_length=128):
        self.alphabet = alphabet or ALPHABET
        self.text_field = text_field
        self.label_field = label_field
        self.data = pd.read_csv(filepath)
        assert self.label_field in self.data.columns
        self.label2int = {l: i for i, l in enumerate(sorted(self.data[self.label_field].unique()))}
        self.label_placeholder = np.zeros(len(self.label2int))
        self.max_text_length = max_text_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data.iloc[idx]
        text = line[self.text_field].lower()
        # for mokoron dataset we should remove smiles
        if ')' not in self.alphabet:
            text = [t for t in text if t not in ('(', ')')]
        label = self.label2int[line[self.label_field]]

        if self.noise_level > 0:
            text = self._noise_generator(text)

        return text, label

    def _noise_generator(self, string):
        noised = ""
        for c in string:
            if random() > self.noise_level:
                noised += c
            if random() < self.noise_level:
                noised += choice(self.alphabet)
        return noised


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
        return batch, labels

    def __iter__(self):
        return self

    def reset_batch_pointer(self):
        self._batch_pointer = 0
        if self.shuffle:
            self._shuffle()

    def _shuffle(self):
        np.random.shuffle(self._indices)


# based on github.com/roomylee/rnn-text-classification-tf
class RNN:
    def __init__(self, sequence_length, num_classes,
                 cell_type, embeddings_size, hidden_size):

        # Placeholders for input, output and dropout
        self.input_vectors = tf.placeholder(tf.float32, shape=[None, sequence_length, embeddings_size], name='input_vectors')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')
        self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

        labels_one_hot = tf.one_hot(self.labels, num_classes)
        text_length = self._length(self.input_vectors)

        # Recurrent Neural Network
        with tf.name_scope("rnn"):
            rnn_input = tf.layers.dense(self.input_vectors, hidden_size, name='linear')
            cell = self._get_cell(hidden_size, cell_type)
            self.h_outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                                                  inputs=rnn_input,
                                                  sequence_length=text_length,
                                                  dtype=tf.float32)

        self.last_output = self.h_outputs[:, -1, :]
        self.last_output = tf.layers.dropout(self.last_output, self.dropout_prob)

        self.logits = tf.layers.dense(self.last_output, num_classes,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=labels_one_hot)
        self.loss = tf.reduce_mean(losses)

        # Metrics
        self.labels_int = tf.argmax(labels_one_hot, axis=1)
        # self.accuracy = tf.metrics.accuracy(labels_int, self.predictions)
        # self.precision = tf.metrics.precision(labels_int, self.predictions)
        # self.recall = tf.metrics.recall(labels_int, self.predictions)
        # self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)

    @staticmethod
    def _get_cell(hidden_size, cell_type):
        if cell_type.lower() == "vanilla":
            return tf.nn.rnn_cell.BasicRNNCell(hidden_size)
        elif cell_type.lower() == "lstm":
            return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        elif cell_type.lower() == "gru":
            return tf.nn.rnn_cell.GRUCell(hidden_size)
        elif cell_type.lower() == "sru":
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
    def clip_grads(loss, variables=None):
        if variables is None:
            variables = tf.trainable_variables()
        grads = tf.gradients(loss, variables)
        clipped_grads, _ = tf.clip_by_global_norm(grads, CLIP_NORM)
        return zip(clipped_grads, variables)

    def evaluate(self, dataloader, sess, rove, pad, frac=1.0):
        predictions = []
        labels = []

        max_iters = int(len(dataloader) * frac)
        for i, (batch, label) in enumerate(dataloader):
            if i >= max_iters:
                break
            batch = rove.sample(batch, pad=pad)
            feed_dict = {
                self.input_vectors: batch,
                self.labels: label,
                self.dropout_prob: 0
            }
            preds, labs = sess.run([self.predictions, self.labels_int], feed_dict=feed_dict)
            predictions += list(preds)
            labels += list(labs)

        res = {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions)
        }
        return res


def train():
    epochs = 2
    batch_size = 64
    dropout = 0.5
    seq_len = 32

    rove_path = 'save/ruscorpora'
    rove_type = 'sru'

    writer = SummaryWriter(comment='_test')
    logger.info('Writer: ', list(writer.all_writers.keys())[0])

    logger.info('Preparing datasets')
    train_dataset = MokoronDataset('../text_classification/data/mokoron/train.csv',
                                   text_field='text_spellchecked',
                                   label_field='sentiment')
    val_dataset = MokoronDataset('../text_classification/data/mokoron/validation.csv',
                                 text_field='text_spellchecked',
                                 label_field='sentiment')

    train_dataloader = DataLoader(train_dataset, batch_size, True)
    val_dataloader = DataLoader(val_dataset, batch_size, True)

    # logger.info('Building model')

    # lth = tf.train.LoggingTensorHook({'tensor_to_log_name': tensor_to_log})

    logger.info('Starting training process')

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25))
    with tf.Session(config=config, graph=tf.Graph()) as sess:
        rove = RoVeSampler(rove_path, rove_type, sess)
        tf.stop_gradient(rove.model.target)

        model = RNN(sequence_length=seq_len, num_classes=2, cell_type='sru', embeddings_size=300, hidden_size=128)

        global_step = tf.train.get_or_create_global_step()
        lr_op = tf.train.cosine_decay_restarts(1e-2, global_step, len(train_dataloader), t_mul=1.0)

        optimizer = tf.train.AdamOptimizer(lr_op)
        gradients = model.clip_grads(model.loss)

        train_op = optimizer.apply_gradients(
            gradients, global_step=global_step
        )

        sess.run(tf.global_variables_initializer())

        step = sess.run(global_step)
        for epoch in range(epochs):
            logger.info('Epoch ', epoch)
            for i, (batch, label) in enumerate(train_dataloader):
                # if i > 100:
                #     break
                step = sess.run(global_step)

                batch = rove.sample(batch, pad=seq_len)

                feed_dict = {
                    model.input_vectors: batch,
                    model.labels: label,
                    model.dropout_prob: dropout
                }
                # summary, _ = sess.run([summary, train_op], feed_dict=feed_dict)
                # self.train_writer.add_summary(summary)
                loss, _ = sess.run([model.loss, train_op], feed_dict=feed_dict)
                writer.add_scalar('loss', loss, step)

            # saver = tf.train.Saver()
            # saver.save(sess, model_file_path)

            # evaluate
            logger.info('Evaluating the model')
            train_metrics = model.evaluate(val_dataloader, sess, rove, pad=seq_len, frac=.1)
            writer.add_scalar('f1_train', train_metrics['f1'], step)
            writer.add_scalar('accuracy_train', train_metrics['accuracy'], step)

            val_metrics = model.evaluate(val_dataloader, sess, rove, pad=seq_len)
            writer.add_scalar('f1_val', val_metrics['f1'], step)
            writer.add_scalar('accuracy_val', val_metrics['accuracy'], step)


if __name__ == '__main__':
    train()
