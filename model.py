import tensorflow as tf

from nn_utils import rnn_decoder
from sru import SRUCell
from utils import letters2vec
from nltk.tokenize import word_tokenize
import numpy as np
from tensorflow.python.ops import rnn_cell_impl
rnn_cell = rnn_cell_impl
rnn = tf.contrib.rnn


# TODO rename
class Model:
    def __init__(self, args, infer=False, seq_len=128):
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
        elif args.model == 'sru':
            cell_fn = SRUCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        self.seq_len = seq_len
        #with tf.variable_scope("rnn", reuse=True):
        cell = cell_fn(args.rnn_size, state_is_tuple=False)  # is not necessary arg

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=False)

        self.input_data = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.letter_size])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        self.change = tf.placeholder(tf.bool, [args.batch_size])
        #initial_state = tf.where(self.change, cell.zero_state(args.batch_size, tf.float32), self.initial_state)
        initial_state = self.initial_state
        inputs = tf.split(self.input_data, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        with tf.variable_scope("input_linear"):
            linears = []
            # for i, _input in enumerate(inputs):
            #     if i > 0:
            #         tf.get_variable_scope()
                # full_vector = tf.contrib.layers.fully_connected(_input, args.rnn_size,
                #                                                 activation_fn=None)
                # linears.append(full_vector)

            for i in range(len(inputs)):
                reuse = None
                if i > 0:
                    reuse = True
                    tf.get_variable_scope().reuse_variables()
                linears.append(tf.layers.dense(inputs[i], args.rnn_size, reuse=reuse))

        #fixed_input = tf.stack(linears, axis=1)
        #fixed_input = tf.reshape(fixed_input, [self.args.batch_size, self.args.seq_length, -1])
        # with tf.variable_scope("rnn_scope"):
        #     tf.get_variable_scope().reuse_variables()
        # outputs, last_state = tf.nn.dynamic_rnn(cell, fixed_input,
        #                                        initial_state=initial_state,
        #                                        scope="rnnlm")

        outputs, last_state = rnn_decoder(linears, initial_state, cell,
                                          scope="rnnlm")
        print("Shape of the last state", last_state.shape)
        self.final_state = last_state

        loss1 = tf.constant(0.0)
        loss2 = tf.constant(0.0)
        final_vectors = []

        ones = tf.diag([1.] * args.batch_size)
        #outputs = tf.unstack(outputs, axis = 1)
        with tf.variable_scope("output_linear"):
            for i in range(len(outputs)):
                # if i > 0:
                #     tf.get_variable_scope()
                # output = tf.contrib.layers.fully_connected(outputs[i], args.w2v_size,
                #                                            activation_fn=None)
                #
                # print(output.shape)
                reuse = None
                if i > 0:
                    reuse = True
                    tf.get_variable_scope().reuse_variables()
                output = tf.layers.dense(outputs[i], args.w2v_size, reuse=reuse)
                # output = rnn_cell._linear(outputs[i], args.w2v_size, bias=True)

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

        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars,
                                                           aggregation_method=
                                                           tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N),
                                              args.grad_clip)
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        # Validation
        self.valid_data = tf.placeholder(tf.float32, [1, 1, args.letter_size])
        self.valid_initial_state = cell.zero_state(1, tf.float32)

        valid_initial_state = self.valid_initial_state

        valid_inputs = tf.split(self.valid_data, 1, 1)
        valid_inputs = [tf.squeeze(input_, [1]) for input_ in valid_inputs]

        with tf.variable_scope("input_valid"):
            valid_fixed_input = []
            for i, _input in enumerate(valid_inputs):
                reuse = None
                if i > 0:
                    reuse = True
                    tf.get_variable_scope().reuse_variables()
                valid_fixed_input.append(tf.layers.dense(valid_inputs[i], args.rnn_size, reuse=reuse))
                # valid_fixed_input.append((rnn_cell._linear(valid_inputs[i], args.rnn_size, bias=True)))

        # valid_fixed_input = tf.stack(valid_fixed_input, axis=1)
        # valid_fixed_input = tf.reshape(valid_fixed_input, [1, 1, args.rnn_size])

        valid_outputs, valid_last_state = rnn_decoder(valid_fixed_input, valid_initial_state, cell,
                                          scope="rnnlm")
        # valid_outputs, valid_last_state = tf.nn.dynamic_rnn(cell, valid_fixed_input,
        #                                        initial_state=valid_initial_state,
        #                                        scope="lst_valid")

        self.valid_state = valid_last_state

        valid_vectors = []

        valid_outputs = tf.unstack(valid_outputs, axis = 1)
        with tf.variable_scope("output_valid"):
            for i in range(len(valid_outputs)):
                # if i > 0:
                #     tf.get_variable_scope().reuse_variables()
                # output = tf.nn.l2_normalize(output, 1)
                # valid_vectors.append(output)
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output = tf.layers.dense(valid_outputs[i], args.w2v_size)
                # output = rnn_cell._linear(valid_outputs[i], args.w2v_size, bias=True)
                valid_vectors.append(output)
        self.valid_vector = valid_vectors[-1]

    def valid_run(self, sess, vocab, prime):
        state = self.cell.zero_state(1, tf.float32).eval()
        tokens = word_tokenize(prime)
        targets = []
        for token in tokens:
            x = letters2vec(token, vocab).reshape((1, 1, -1))
            feed = {self.valid_data: x,
                    self.valid_initial_state: state,
                    }
            [state, target] = sess.run([self.valid_state, self.valid_vector], feed)
            targets.append(np.squeeze(target))
        return targets

    # def sample(self, sess, vocab, prime=' '):
    #     state = self.cell.zero_state(1, tf.float32).eval()
    #     tokens = word_tokenize(prime)
    #     targets = []
    #     for token in tokens:
    #         x = letters2vec(token, vocab).reshape((1, 1, -1))
    #         feed = {self.input_data: x,
    #                 self.initial_state: state,
    #                 self.change: np.zeros((1,))
    #                 }
    #         [state, target] = sess.run([self.final_state, self.target], feed)
    #         targets.append(np.squeeze(target))
    #
    #     return targets

    def sample(self, sess, vocab, prime_batch=' ', batch_size=1):
        self.initial_state = tf.convert_to_tensor(self.cell.zero_state(batch_size, tf.float32))
        batch_tokenized = []
        for sent in prime_batch:
            batch_tokenized.append(word_tokenize(sent))
        max_seq = self.seq_len
        data = np.zeros((batch_size, max_seq, 7*len(vocab)))  # letter2vec encoding
        for i, sent in enumerate(batch_tokenized):
            if len(sent) > max_seq:
                sent = sent[:max_seq]
            sentence_vecs = []
            for t in sent:
                x = letters2vec(t, vocab).reshape((1, 1, -1))
                sentence_vecs.append(x)

            data[i, :len(sentence_vecs)] = sentence_vecs

        data = np.array(data.reshape(max_seq, batch_size, -1))
        state_fw = self.initial_state.eval()
        target_vectors = []

        for word_batch in data:
            feed = {self.input_data: np.expand_dims(word_batch, 1),
                    self.initial_state: state_fw,
                    self.change: np.zeros((batch_size,))
                    }
            [last_state, target] = sess.run([self.final_state, self.target], feed)
            state_fw = last_state[0]
            word_vec = target
            target_vectors.append(word_vec)
        target_vectors = np.array(target_vectors)
        return target_vectors.reshape(batch_size, max_seq, -1)


class ReccurentClassifier:
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
        elif args.model == 'sru':
            cell_fn = SRUCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size, state_is_tuple=False)

        self.cell = rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=False)

        self.input_data = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.letter_size])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        self.labels = tf.placeholder(tf.float32, [args.batch_size, 1])

        # Network
        outputs, _ = rnn_decoder(self.input_data, self.initial_state, self.cell, scope="cls_rnn")
        last_output = outputs[-1]
        last_output = tf.nn.dropout(last_output, args.dropout_keep_prob)
        pred = tf.layers.dense(last_output, 1)

        # Training-related graph
        loss = tf.losses.softmax_cross_entropy(self.labels, pred)
        train_op = tf.train.AdamOptimizer().minimize(loss)

    def fit(self, X_train, y_train, X_val, y_val):
        ...

    def evaluate(self, X, y):
        ...
