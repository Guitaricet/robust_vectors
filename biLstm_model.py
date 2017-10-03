import tensorflow as tf
import numpy as np
from nltk.tokenize import word_tokenize
from utils import letters2vec

rnn = tf.contrib.rnn


class BiLSTM:
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        cell_forw = []
        cell_back = []
        for _ in range(args.num_layers):
            # Forward direction cell
            cell_forw.append(rnn.BasicLSTMCell(args.rnn_size, forget_bias=1.0))
            # Backward direction cell
            cell_back.append(rnn.BasicLSTMCell(args.rnn_size, forget_bias=1.0))

        # Attention this model is suitable for bidirectional LSTM (for test)


        self.cell_fw = cell_forw[0]
        self.cell_bw = cell_back[0]

        self.input_data = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.letter_size])

        self.initial_state_fw = self.cell_fw.zero_state(args.batch_size, tf.float32)
        self.initial_state_bw = self.cell_bw.zero_state(args.batch_size, tf.float32)

        self.change = tf.placeholder(tf.bool, [args.batch_size])

        initial_state_fw = self.initial_state_fw  # tf.where(self.change, cell_fw.zero_state(args.batch_size, tf.float32), self.initial_state_fw)
        initial_state_bw = self.initial_state_bw  # tf.where(self.change, cell_bw.zero_state(args.batch_size, tf.float32), self.initial_state_bw)

        inputs = tf.split(self.input_data, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        with tf.variable_scope("input_linear"):
            fixed_size_vectors = []  # tf.Variable(tf.float32, [args.seq_length, args.rnn_size,args.letter_size])
            for i, _input in enumerate(inputs):
                if i > 0:
                    tf.get_variable_scope()
                full_vector = tf.contrib.layers.fully_connected(_input, args.rnn_size,
                                                                activation_fn=None)
                fixed_size_vectors.append(full_vector)

        fixed_input = tf.stack(fixed_size_vectors, axis=1)
        fixed_input = tf.reshape(fixed_input, [self.args.batch_size, self.args.seq_length, -1])

        output = fixed_input
        for n in range(self.args.num_layers):
            cell_fw = cell_forw[n]
            cell_bw = cell_back[n]

            _initial_state_fw = cell_fw.zero_state(self.args.batch_size, tf.float32)
            _initial_state_bw = cell_bw.zero_state(self.args.batch_size, tf.float32)

            (output_fw, output_bw), last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, output,
                                                                                 initial_state_fw=_initial_state_fw,
                                                                                 initial_state_bw=_initial_state_bw,
                                                                                 scope='BLSTM_' + str(n + 1),
                                                                                 dtype=tf.float32)

            output = tf.concat([output_fw, output_bw], axis=2)
        # (output_fw, output_bw), last_state = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw,
        #                                                                      fixed_input,
        #                                                                      initial_state_fw=initial_state_fw,
        #                                                                      initial_state_bw=initial_state_bw,
        #                                                                      scope="rnnlm",
        #                                                                      dtype=tf.float32)
        # outputs = tf.concat([output_fw, output_bw], axis=2)
        outputs = output
        self.final_state = last_state

        loss1 = tf.constant(0.0)
        loss2 = tf.constant(0.0)
        final_vectors = []

        ones = tf.diag([1.] * args.batch_size)
        outputs = tf.unstack(outputs, axis=1)

        with tf.variable_scope("output_linear"):
            for i in range(len(outputs)):
                if i > 0:
                    tf.get_variable_scope()  # why did we use reuse variables
                output = tf.contrib.layers.fully_connected(outputs[i], args.w2v_size,
                                                           activation_fn=None)
                output = tf.nn.l2_normalize(output, 1)
                output = tf.nn.dropout(output, args.dropout_keep_prob)
                # negative sampling ??TODO
                print(output.shape)

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
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars,
                                                       aggregation_method=
                                                       tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N),
                                          args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, vocab, prime=' '):
        self.initial_state_fw = tf.convert_to_tensor(self.cell_fw.zero_state(1, tf.float32))
        self.initial_state_bw = tf.convert_to_tensor(self.cell_bw.zero_state(1, tf.float32))
        state_fw = self.initial_state_fw.eval()
        state_bw = self.initial_state_bw.eval()
        tokens = word_tokenize(prime)
        targets = []
        for token in tokens:
            x = letters2vec(token, vocab).reshape((1, 1, -1))
            feed = {self.input_data: x,
                    self.initial_state_fw: state_fw,
                    self.initial_state_bw: state_bw,
                    self.change: np.zeros((1,))
                    }
            [last_state, target] = sess.run([self.final_state, self.target], feed)
            state_fw = last_state[0]
            state_bw = last_state[1]
            targets.append(np.squeeze(target))

        return targets


class StackedBiLstm:
    """
    This model differ from class bidirectional lstm
    The combined outs from forward and backward use in some layers.
    Use with num_layers = 2
    https://arxiv.org/abs/1303.5778
    """

    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        cell_forw = []
        cell_back = []
        for _ in range(args.num_layers):
            # Forward direction cell
            cell_forw.append(rnn.BasicLSTMCell(args.rnn_size, forget_bias=1.0))
            # Backward direction cell
            cell_back.append(rnn.BasicLSTMCell(args.rnn_size, forget_bias=1.0))

        self.cells_fw = cell_forw
        self.cells_bw = cell_back

        self.input_data = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.letter_size])

        initial_state_fw = []
        initial_state_bw = []

        for (cell_fw, cell_bw) in zip(cell_forw, cell_back):
            initial_state_fw.append(cell_fw.zero_state(args.batch_size, tf.float32))
            initial_state_bw.append(cell_bw.zero_state(args.batch_size, tf.float32))

        self.change = tf.placeholder(tf.bool, [args.batch_size])

        self.initial_state_fw = initial_state_fw
        self.initial_state_bw = initial_state_bw
        print(initial_state_bw)
        inputs = tf.split(self.input_data, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        with tf.variable_scope("input_linear"):
            fixed_size_vectors = []  # tf.Variable(tf.float32, [args.seq_length, args.rnn_size,args.letter_size])
            for i, _input in enumerate(inputs):
                if i > 0:
                    tf.get_variable_scope()
                full_vector = tf.contrib.layers.fully_connected(_input, args.rnn_size,
                                                                activation_fn=None)
                fixed_size_vectors.append(full_vector)

        fixed_input = fixed_size_vectors
        # fixed_input = np.reshape(fixed_input, [self.args.rnn_size, self.args.batch_size, self.args.seq_length])
        # (output_fw, output_bw), last_state = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw, output,
        #                                           initial_state_fw=_initial_state_fw,
        #                                           initial_state_bw=_initial_state_bw,
        #                                           scope='bilstm',
        #                                           dtype=tf.float32)

        (outputs, last_state_fw, last_state_bw) = tf.contrib.rnn.stack_bidirectional_rnn(
            self.cells_fw, self.cells_bw,
            fixed_input,
            initial_states_fw=self.initial_state_fw,
            initial_states_bw=self.initial_state_bw,
            scope="stack_biLSTM",
            dtype=tf.float32)

        self.last_fw = [last_state_fw[0], last_state_fw[1]]
        self.last_bw = [last_state_bw[0], last_state_bw[1]]

        last_state = (last_state_fw, last_state_bw)
        self.final_state = last_state

        loss1 = tf.constant(0.0)
        loss2 = tf.constant(0.0)
        final_vectors = []

        ones = tf.diag([1.] * args.batch_size)

        with tf.variable_scope("output_linear"):
            for out in outputs:
                if i > 0:
                    tf.get_variable_scope()
                    # why did we use reused variables

                output = tf.contrib.layers.fully_connected(out, args.w2v_size,
                                                           activation_fn=None)
                output = tf.nn.l2_normalize(output, 1)
                output = tf.nn.dropout(output, args.dropout_keep_prob)

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
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars,
                                                       aggregation_method=
                                                       tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N),
                                          args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, vocab, prime=' '):
        initial_state_fw = []
        initial_state_bw = []

        for (cell_fw, cell_bw) in zip(self.cells_fw, self.cells_bw):
            initial_state_fw.append(tf.convert_to_tensor(cell_fw.zero_state(1, tf.float32)))
            initial_state_bw.append(tf.convert_to_tensor(cell_bw.zero_state(1, tf.float32)))

        self.initial_state_bw = initial_state_bw
        self.initial_state_fw = initial_state_fw
        state_fw = np.array([initial_state_fw[0].eval(), initial_state_fw[1].eval()])
        state_bw = np.array([initial_state_bw[0].eval(), initial_state_bw[1].eval()])

        # for i_state_fw, i_state_bw in zip(self.initial_state_fw, self.initial_state_bw):
        #     state_fw.append(i_state_fw.eval())
        #     state_bw.append(i_state_bw.eval())
        tokens = word_tokenize(prime)
        targets = []
        for token in tokens:
            x = letters2vec(token, vocab).reshape((1, 1, -1))
            feed = {self.input_data: x,
                    self.initial_state_fw[0]: state_fw[0],
                    self.initial_state_fw[1]: state_fw[1],
                    self.initial_state_bw[0]: state_bw[0],
                    self.initial_state_bw[1]: state_bw[1],
                    self.change: np.zeros((1,))
                    }

            [last_state, target] = sess.run([self.final_state, self.target], feed)
            state_fw = last_state[0]
            state_bw = last_state[1]
            targets.append(np.squeeze(target))

        return targets