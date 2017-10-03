import tensorflow
import tensorflow as tf

from nn_utils import rnn_decoder
from utils import letters2vec
from nltk.tokenize import word_tokenize
import numpy as np
from tensorflow.python.ops import rnn_cell_impl
rnn_cell = rnn_cell_impl
rnn = tf.contrib.rnn


class Model:
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        elif args.model == 'biLSTM':
            # Forward direction cell
            cell_fn1 = rnn.BasicLSTMCell
            # Backward direction cell
            cell_fn2 = rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        # Attention this model is suitable for bidirectional LSTM (for test)
        cell = cell_fn(args.rnn_size, state_is_tuple=False)# is not necesery arg

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=False)

        self.input_data = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.letter_size])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        self.change = tf.placeholder(tf.bool, [args.batch_size])

        initial_state = tf.where(self.change, cell.zero_state(args.batch_size, tf.float32), self.initial_state)

        inputs = tf.split(self.input_data, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        with tf.variable_scope("input_linear"):
            linears = []
            for i in range(len(inputs)):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                linears.append((rnn_cell._linear(inputs[i], args.rnn_size, bias=True)))
        outputs, last_state = rnn_decoder(linears, initial_state, cell,
                                          # loop_function=loop if infer else None,
                                          scope="rnnlm")
        self.final_state = last_state

        loss1 = tf.constant(0.0)
        loss2 = tf.constant(0.0)
        final_vectors = []

        ones = tf.diag([1.] * args.batch_size)
        with tf.variable_scope("output_linear"):
            for i in range(len(outputs)):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output = rnn_cell._linear(outputs[i], args.w2v_size, bias=True)
                output = tf.nn.l2_normalize(output, 1)
                output = tf.nn.dropout(output, args.dropout_keep_prob)
                # negative sampling
                matrix = tf.matmul(output, output, transpose_b=True) - ones
                loss1 += tf.maximum(0.0, matrix)
                final_vectors.append(output)

        seq_slices = tf.reshape(tf.concat(final_vectors, 1), [args.batch_size, args.seq_length, args.w2v_size])
        seq_slices = tf.split(seq_slices, args.batch_size, 0)
        seq_slices = [tf.squeeze(input_, [0]) for input_ in seq_slices]
        with tf.variable_scope("additional_loss"):
            for i in range(len(seq_slices)):  # should be length of batch_size
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                seq_context = tf.nn.l2_normalize(seq_slices[i], 1)
                # context similarity
                matrix = tf.matmul(seq_context, seq_context, transpose_b=True)
                loss2 += 1. - matrix

        self.target = final_vectors[-1]
        self.cost = tf.reduce_sum(loss1) / args.batch_size / args.seq_length
        self.cost += tf.reduce_sum(loss2) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars,
                                                       aggregation_method=
                                                       tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N),
                                          args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, vocab, prime=' '):
        state = self.cell.zero_state(1, tf.float32).eval()
        tokens = word_tokenize(prime)
        targets = []
        for token in tokens:
            x = letters2vec(token, vocab).reshape((1, 1, -1))

            feed = {self.input_data: x, self.initial_state: state, self.change: np.zeros((1,))}
            [state, target] = sess.run([self.final_state, self.target], feed)
            targets.append(np.squeeze(target))

        return targets
