import tensorflow as tf
import numpy as np
from nltk.tokenize import word_tokenize
# from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import DropoutWrapper

from sru import SRUCell
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
        for i in range(args.num_layers):
            # Forward direction cell
            with tf.variable_scope("forward" + str(i)):
                if args.model == "biSRU":
                    cell_forw.append(SRUCell(args.rnn_size, state_is_tuple=False))
                else:
                    cell_forw.append(rnn.BasicLSTMCell(args.rnn_size, forget_bias=1.0))

            # Backward direction cell
            with tf.variable_scope("backward" + str(i)):
                if args.model == "biSRU":
                    cell_back.append(SRUCell(args.rnn_size, state_is_tuple=False))
                else:
                    cell_back.append(rnn.BasicLSTMCell(args.rnn_size, forget_bias=1.0))

        self.cell_fw = cell_forw[0]
        self.cell_bw = cell_back[0]

        self.input_data = tf.placeholder(tf.float32, [None, None, args.letter_size], name='input')
        input_shape = tf.shape(self.input_data)
        self.batch_size = input_shape[0]
        self.seq_length = input_shape[1]

        self.initial_state_fw = self.cell_fw.zero_state(self.batch_size, tf.float32)
        self.initial_state_bw = self.cell_bw.zero_state(self.batch_size, tf.float32)

        # self.change = tf.placeholder(tf.bool, [self.batch_size])

        inputs = tf.split(self.input_data, self.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        with tf.variable_scope("input_linear"):
            fixed_size_vectors = []  # tf.Variable(tf.float32, [args.seq_length, args.rnn_size,args.letter_size])
            for i, _input in enumerate(inputs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                full_vector = tf.contrib.layers.fully_connected(_input, args.rnn_size,
                                                                activation_fn=None, scope="input_fc")
                fixed_size_vectors.append(full_vector)

        fixed_input = tf.stack(fixed_size_vectors, axis=1)
        fixed_input = tf.reshape(fixed_input, [self.batch_size, self.seq_length, -1])

        output = fixed_input
        with tf.variable_scope("lstm"):
            for n in range(self.args.num_layers):
                cell_fw = cell_forw[n]
                cell_bw = cell_back[n]
                # print(output.shape)
                _initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
                _initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)

                (output_fw, output_bw), last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, output,
                                                                                     initial_state_fw=_initial_state_fw,
                                                                                     initial_state_bw=_initial_state_bw,
                                                                                     scope='BLSTM_' + str(n + 1),
                                                                                     dtype=tf.float32)
                output = output_fw

        output = tf.concat([output_fw, output_bw], axis=2)

        outputs = output
        self.final_state = last_state

        loss1 = tf.constant(0.0)
        loss2 = tf.constant(0.0)
        final_vectors = []

        ones = tf.diag([1.] * self.batch_size)
        outputs = tf.unstack(outputs, axis=1)  # 1 - is sequence dim

        with tf.variable_scope("output_linear"):
            for i in range(len(outputs)):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output = tf.contrib.layers.fully_connected(outputs[i], args.w2v_size,
                                                           activation_fn=None, scope="out_fc")
                output = tf.nn.l2_normalize(output, 1)
                output = tf.nn.dropout(output, args.dropout_keep_prob)
                # negative sampling
                matrix = tf.matmul(output, output, transpose_b=True) - ones
                loss1 += tf.maximum(0.0, matrix)
                final_vectors.append(output)

        self.target = tf.reshape(tf.concat(final_vectors, 1),
                                 [self.batch_size, self.seq_length, args.w2v_size],
                                 name='target')
        seq_slices = tf.split(self.target, self.batch_size, 0)
        seq_slices = [tf.squeeze(input_, [0]) for input_ in seq_slices]
        with tf.variable_scope("additional_loss"):
            for i in range(len(seq_slices)):  # should be length of batch_size
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                seq_context = tf.nn.l2_normalize(seq_slices[i], 1)
                # context similarity
                matrix = tf.matmul(seq_context, seq_context, transpose_b=True)
                loss2 += 1. - matrix

        self.cost = tf.reduce_sum(loss1) / self.batch_size / self.seq_length
        self.cost += tf.reduce_sum(loss2) / self.batch_size / self.seq_length
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # Validation eval : TODO add None size to placeholders
        self.valid_data = tf.placeholder(tf.float32, [1, 1, args.letter_size])

        self.valid_initial_state_fw = self.cell_fw.zero_state(1, tf.float32)
        self.valid_initial_state_bw = self.cell_bw.zero_state(1, tf.float32)

        valid_inputs = tf.split(self.valid_data, 1, 1)
        valid_inputs = [tf.squeeze(input_, [1]) for input_ in valid_inputs]

        with tf.variable_scope("valid_input"):
            valid_fixed_size_vectors = []  # tf.Variable(tf.float32, [args.seq_length, args.rnn_size,args.letter_size])
            for i, _input in enumerate(valid_inputs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                full_vector = tf.contrib.layers.fully_connected(_input, args.rnn_size,
                                                                activation_fn=None, scope="valid_in_fc")
                valid_fixed_size_vectors.append(full_vector)

        valid_fixed_input = tf.stack(valid_fixed_size_vectors, axis=1)
        valid_fixed_input = tf.reshape(valid_fixed_input, [1, 1, -1])

        valid_output = valid_fixed_input
        with tf.variable_scope("valid_lstm"):
            for n in range(self.args.num_layers):
                cell_fw = cell_forw[n]
                cell_bw = cell_back[n]

                _initial_state_fw = cell_fw.zero_state(1, tf.float32)
                _initial_state_bw = cell_bw.zero_state(1, tf.float32)

                (valid_output_fw, valid_output_bw), valid_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, valid_output,
                                                                                     initial_state_fw=_initial_state_fw,
                                                                                     initial_state_bw=_initial_state_bw,
                                                                                     scope='valid_BLSTM_' + str(n + 1),
                                                                                     dtype=tf.float32)
                valid_output = valid_output_fw

        valid_output = tf.concat([valid_output_fw, valid_output_bw], axis=2)

        valid_outputs = valid_output
        self.valid_state = valid_state

        ones = tf.diag([1.])
        valid_outputs = tf.unstack(valid_outputs, axis=1)
        valid_vectors = []
        with tf.variable_scope("valid_output"):
            for i in range(len(valid_outputs)):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output = tf.contrib.layers.fully_connected(valid_outputs[i], args.w2v_size,
                                                           activation_fn=None, scope="valid_out_fc")
                output = tf.nn.l2_normalize(output, 1)
                output = tf.nn.dropout(output, args.dropout_keep_prob)

                matrix = tf.matmul(output, output, transpose_b=True) - ones
                loss1 += tf.maximum(0.0, matrix)
                valid_vectors.append(output)

        self.valid_vector = valid_vectors[-1]

    def valid_run(self, sess, vocab, prime):

        self.valid_initial_state_fw = tf.convert_to_tensor(self.cell_fw.zero_state(1, tf.float32))
        self.valid_initial_state_bw = tf.convert_to_tensor(self.cell_bw.zero_state(1, tf.float32))
        state_fw = self.valid_initial_state_fw.eval()
        state_bw = self.valid_initial_state_bw.eval()
        tokens = word_tokenize(prime)
        targets = []

        for token in tokens:
            x = letters2vec(token, vocab).reshape((1, 1, -1))
            feed = {self.valid_data: x,
                    self.valid_initial_state_fw: state_fw,
                    self.valid_initial_state_bw: state_bw,
                    }
            [last_state, target] = sess.run([self.valid_state, self.valid_vector], feed)
            state_fw = last_state[0]
            state_bw = last_state[1]
            targets.append(np.squeeze(target))
        return targets

    def sample(self, sess, vocab, prime_batch, batch_size=1, pad=128):
        """
        :param sess: tf session
        :param vocab: char vocabulary
        :param prime_batch: list of strings

        :return: sequence of robust word vectors
        """
        self.initial_state_fw = tf.convert_to_tensor(self.cell_fw.zero_state(batch_size, tf.float32))
        self.initial_state_bw = tf.convert_to_tensor(self.cell_bw.zero_state(batch_size, tf.float32))

        max_seq = pad
        data = np.zeros((batch_size, max_seq, 7*len(vocab)))
        for i, _sent in enumerate(prime_batch):
            sent = word_tokenize(_sent)
            if len(sent) > max_seq:
                sent = sent[:max_seq]
            sent_vecs = []
            for t in sent:
                x = letters2vec(t, vocab).reshape((1, 1, -1))
                sent_vecs.append(x)

            data[i, :len(sent_vecs)] = sent_vecs

        feed = {
            self.input_data: data,
            self.initial_state_fw: self.initial_state_fw.eval(),
            self.initial_state_bw: self.initial_state_bw.eval()
        }
        target_vectors = sess.run(self.target, feed)
        return target_vectors


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

        inputs = tf.split(self.input_data, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        print(inputs)
        with tf.variable_scope("input_linear"):
            fixed_size_vectors = []  # tf.Variable(tf.float32, [args.seq_length, args.rnn_size,args.letter_size])
            for i, _input in enumerate(inputs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                full_vector = tf.contrib.layers.fully_connected(_input, args.rnn_size,
                                                                activation_fn=None)
                fixed_size_vectors.append(full_vector)

        fixed_input = fixed_size_vectors

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
                    tf.get_variable_scope().reuse_variables()

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
        # Validation
        self.valid_data = tf.placeholder(tf.float32, [1, 1, args.letter_size])
        valid_inputs = tf.split(self.valid_data, 1, 1)
        valid_inputs = [tf.squeeze(_input, [1]) for _input in valid_inputs]

        valid_initial_state_fw = []
        valid_initial_state_bw = []

        for (cell_fw, cell_bw) in zip(cell_forw, cell_back):
            valid_initial_state_fw.append(cell_fw.zero_state(1, tf.float32))
            valid_initial_state_bw.append(cell_bw.zero_state(1, tf.float32))

        with tf.variable_scope("validate_input"):
            validate_fixed_size_vectors = []
            for i, _input in enumerate(valid_inputs):
                if i > 0:
                    tf.get_variable_scope()
                full_vector = tf.contrib.layers.fully_connected(_input, args.rnn_size,
                                                                activation_fn=None)
                validate_fixed_size_vectors.append(full_vector)

        valid_input_fixed = validate_fixed_size_vectors
        self.valid_initial_state_fw = valid_initial_state_fw
        self.valid_initial_state_bw = valid_initial_state_bw

        (outputs_valid, last_state_v_fw, last_state_v_bw) = tf.contrib.rnn.stack_bidirectional_rnn(
            self.cells_fw, self.cells_bw,
            valid_input_fixed,
            initial_states_fw=self.valid_initial_state_fw,
            initial_states_bw=self.valid_initial_state_bw,
            scope="valid_stack_biLSTM",
            dtype=tf.float32)
        valid_vectors = []

        valid_last_state = (last_state_v_fw, last_state_v_bw)
        self.valid_state = valid_last_state

        ones = tf.diag([1.])

        with tf.variable_scope("valid_output"):
            for out in outputs_valid:
                if i > 0:
                    tf.get_variable_scope()

                output = tf.contrib.layers.fully_connected(out, args.w2v_size,
                                                           activation_fn=None)
                output = tf.nn.l2_normalize(output, 1)
                output = tf.nn.dropout(output, args.dropout_keep_prob)

                matrix = tf.matmul(output, output, transpose_b=True) - ones
                valid_vectors.append(output)

        self.valid_vector = valid_vectors

    def valid_run(self, sess, vocab, prime):
        valid_initial_state_fw = []
        valid_initial_state_bw = []

        for (cell_fw, cell_bw) in zip(self.cells_fw, self.cells_bw):
            valid_initial_state_fw.append(tf.convert_to_tensor(cell_fw.zero_state(1, tf.float32)))
            valid_initial_state_bw.append(tf.convert_to_tensor(cell_bw.zero_state(1, tf.float32)))

        self.valid_initial_state_bw = valid_initial_state_bw
        self.valid_initial_state_fw = valid_initial_state_fw
        state_fw = np.array([valid_initial_state_fw[0].eval(), valid_initial_state_fw[1].eval()])
        state_bw = np.array([valid_initial_state_bw[0].eval(), valid_initial_state_bw[1].eval()])

        tokens = word_tokenize(prime)
        targets = []
        for token in tokens:
            x = letters2vec(token, vocab).reshape((1, 1, -1))
            feed = {self.valid_data: x,
                    self.valid_initial_state_fw[0]: state_fw[0],
                    self.valid_initial_state_fw[1]: state_fw[1],
                    self.valid_initial_state_bw[0]: state_bw[0],
                    self.valid_initial_state_bw[1]: state_bw[1],
                    }
            [last_state, target] = sess.run([self.valid_state, self.valid_vector], feed)
            state_fw = last_state[0]
            state_bw = last_state[1]
            targets.append(np.squeeze(target))
        return targets

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
