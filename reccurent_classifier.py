from random import random, choice

import numpy as np
import pandas as pd
import tensorflow as tf
from pymystem3 import Mystem
from sklearn.metrics import f1_score, accuracy_score
from tensorboardX import SummaryWriter

from sru import SRUCell
from sample import RoVeSampler

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
                 rove_path=None,
                 rove_type='sru',
                 max_text_len=128,
                 alphabet=None,
                 return_vectors=True):
        """
        :param filepath:
        :param text_field:
        :param label_field:
        :param rove_path:
        :param rove_type:
        :param max_text_len:
        :param alphabet:
        :param return_vectors: if True, return rove; else return noised text
        """
        self.alphabet = alphabet or ALPHABET
        self.text_field = text_field
        self.label_field = label_field
        self.data = pd.read_csv(filepath)
        assert self.label_field in self.data.columns
        self.label2int = {l: i for i, l in enumerate(sorted(self.data[self.label_field].unique()))}
        self.label_placeholder = np.zeros(len(self.label2int))
        self.max_text_len = max_text_len
        self.rove_path = rove_path
        self.rove_type = rove_type
        self.rove = None  # moved this to dataloader for batching
        self.return_vectors = return_vectors

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.rove is None and self.return_vectors:
            raise RuntimeError('Call .load_rove() first')

        line = self.data.iloc[idx]
        text = line[self.text_field].lower()
        # for mokoron dataset we should remove smiles
        if ')' not in self.alphabet:
            text = [t for t in text if t not in ('(', ')')]
        label = self.label2int[line[self.label_field]]

        if self.noise_level > 0:
            text = self._noise_generator(text)

        if self.return_vectors:
            text = self.rove.sample_multi(text, pad=self.max_text_len)

        return text, label

    def load_rove(self, sess):
        self.rove = RoVeSampler(self.rove_path, self.rove_type, sess)

    def _noise_generator(self, string):
        noised = ""
        for c in string:
            if random() > self.noise_level:
                noised += c
            if random() < self.noise_level:
                noised += choice(self.alphabet)
        return noised


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle, rove_path, rove_type='sru', max_text_len=128):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._batch_pointer = 0
        self._indices = list(range(len(dataset)))
        if shuffle:
            self._shuffle()

        self.rove = None
        self.rove_path = rove_path
        self.rove_type = rove_type
        self.max_text_len = max_text_len
        self.buffer = None  # for debug

    def load_rove(self, sess):
        self.rove = RoVeSampler(self.rove_path, self.rove_type, sess)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __next__(self):
        if self.rove is None:
            raise RuntimeError('Call .load_rove() first')
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
        self.buffer = batch
        batch = self.rove.sample_multi(batch, pad=self.max_text_len)
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

        self.logits = tf.layers.dense(self.last_output, num_classes, kernel_initializer=tf.contrib.layers.xavier_initializer())
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

    # Length of the sequence data
    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.reduce_sum(tf.abs(seq), axis=2))
        length = tf.reduce_sum(relevant, axis=1)
        length = tf.cast(length, tf.int32)
        return length

    # def model_fn(self, features, labels, mode, params):
    #     predict_output = {'values': self.logits}
    #     if mode == tf.estimator.ModeKeys.PREDICT:
    #         export_outputs = {
    #             'predictions': tf.estimator.export.PredictOutput(predict_output)
    #         }
    #         return tf.estimator.EstimatorSpec(
    #             mode=mode,
    #             predictions=predict_output,
    #             export_outputs=export_outputs)
    #
    #     global_step = tf.train.get_global_step()
    #     epoch_length = 164
    #
    #     lr_op = tf.train.cosine_decay_restarts(1e-3, global_step, epoch_length, t_mul=1.0)
    #     optimizer = tf.train.AdamOptimizer(lr_op)
    #     train_op = optimizer.minimize(clip_grads(self.loss), global_step=global_step)
    #     eval_metric_ops = {
    #         'accuracy': self.accuracy,
    #         'f1': self.f1
    #     }
    #
    #     estimator_spec = tf.estimator.EstimatorSpec(mode=mode,
    #                                                 loss=self.loss,
    #                                                 train_op=train_op,
    #                                                 eval_metric_ops=eval_metric_ops)
    #     return estimator_spec


def clip_grads(loss):
    params = tf.trainable_variables()
    grads = tf.gradients(loss, params)
    clipped_grads, _ = tf.clip_by_global_norm(grads, CLIP_NORM)
    return zip(clipped_grads, params)


def train():
    epochs = 2
    batch_size = 64
    dropout = 0.5

    writer = SummaryWriter(comment='_test')

    print('Preparing datasets')
    train_dataset = MokoronDataset('../text_classification/data/mokoron/train.csv',
                                   'text_spellchecked',
                                   'sentiment',
                                   return_vectors=False)
    val_dataset = MokoronDataset('../text_classification/data/mokoron/validation.csv',
                                 'text_spellchecked',
                                 'sentiment',
                                 return_vectors=False)

    train_dataloader = DataLoader(train_dataset, batch_size, True, 'save/ruscorpora')
    val_dataloader = DataLoader(val_dataset, batch_size, True, None)

    # print('Building model')

    # lth = tf.train.LoggingTensorHook({'tensor_to_log_name': tensor_to_log})

    print('Starting training process')
    step = 0
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25))
    with tf.Session(config=config, graph=tf.Graph()) as sess:
        # TODO: gradient-freeze rove
        train_dataloader.load_rove(sess)
        val_dataloader.rove = train_dataloader.rove

        model = RNN(sequence_length=128, num_classes=2, cell_type='sru', embeddings_size=300, hidden_size=128)

        global_step = tf.train.get_or_create_global_step()
        lr_op = tf.train.cosine_decay_restarts(1e-3, global_step, len(train_dataloader), t_mul=1.0)

        optimizer = tf.train.AdamOptimizer(lr_op)
        gradients = clip_grads(model.loss)

        train_op = optimizer.apply_gradients(
            gradients, global_step=global_step
        )

        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            print(f'Epoch {epoch}')
            for batch, label in train_dataloader:
                step += 1

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
            train_metrics = evaluate(model, val_dataloader, sess, frac=.1)
            writer.add_scalar('f1_train', train_metrics['f1'], step)
            writer.add_scalar('accuracy_train', train_metrics['accuracy'], step)

            val_metrics = evaluate(model, val_dataloader, sess)
            writer.add_scalar('f1_val', val_metrics['f1'], step)
            writer.add_scalar('accuracy_val', val_metrics['accuracy'], step)


def evaluate(model, dataloader, sess, frac=1.0):
    # evaluate
    predictions = []
    labels = []

    max_iters = len(dataloader) * frac
    for i, (batch, label) in enumerate(dataloader):
        if i >= max_iters:
            break
        feed_dict = {
            model.input_vectors: batch,
            model.labels: label,
            model.dropout_prob: 0
        }
        preds, labs = sess.run([model.predictions, model.labels_int], feed_dict=feed_dict)
        predictions += preds
        labels += labs

    res = {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions)
    }
    return res


if __name__ == '__main__':
    # pass
    # dataset = MokoronDataset('../text_classification/data/mokoron/train.csv',
    #                          'text_spellchecked',
    #                          'sentiment',
    #                          'save/ruscorpora',
    #                          return_vectors=False)
    # _ = dataset[4]
    # dataloader = DataLoader(dataset, 2, True, 'save/ruscorpora')
    #
    # config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25))
    #
    # model = RNN(sequence_length=128, num_classes=2, cell_type='sru', embeddings_size=300, hidden_size=128)
    #
    # with tf.Session(config=config, graph=tf.Graph()) as sess:
    #     dataloader.load_rove(sess)
    #     batch, label = next(dataloader)
    #     pass
    train()
