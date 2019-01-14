import tensorflow as tf
import numpy as np
from nltk.tokenize import word_tokenize
from utils import letters2vec

rnn = tf.contrib.rnn

class ConvLSTMModel:
    def __init__(self, args, infer=False):
        self.args = args
        self.num_channels = self.args.rnn_size
        print(args.seq_length)
        if infer:
            args.batch_size = 1
            args.seq_length = 8 #TODO  use as argument to inference

        self.input_data = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.letter_size])

        filters_size = [3, 5, 3]

        inputs = tf.split(self.input_data, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        with tf.variable_scope("input_linear"):
            fixed_size_vectors = []
            for i, _input in enumerate(inputs):
                if i > 0:
                    tf.get_variable_scope()
                full_vector = tf.contrib.layers.fully_connected(_input, args.rnn_size,
                                                                activation_fn=None)
                fixed_size_vectors.append(full_vector)

        fixed_input = tf.stack(fixed_size_vectors, axis=1)
        fixed_input = tf.reshape(fixed_input, [self.args.batch_size,1, self.args.seq_length, -1])
        # Layer 1
        with tf.variable_scope("cnn_1"):
            filter_shape = [1, filters_size[0], args.rnn_size, 512]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[512]), name="b")
            conv = tf.nn.conv2d(fixed_input, W, strides=[1, 1, 2, 1], padding="SAME", name="conv1")
            h1 = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

        # Layer2
        with tf.name_scope("cnn_2"):
            filter_shape = [1, filters_size[1], 512, 1024]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[1024]), name="b")
            conv2 = tf.nn.conv2d(h1, W, strides=[1, 1, 2, 1], padding="SAME", name="conv2")
            print(conv2.shape)
            h2 = tf.nn.relu(tf.nn.bias_add(conv2, b), name="relu2")

            print(h2.shape)

        with tf.name_scope("cnn_3"):
            filter_shape = [1, 2, 1024, 1024]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[1024]), name="b")
            conv3 = tf.nn.conv2d(h2, W, strides=[1, 1, 1, 1], padding="SAME", name="conv3")
            print(conv3.shape)
            h3 = tf.nn.relu(tf.nn.bias_add(conv3, b), name="relu3")
        print(h3.shape)

        inputs = tf.squeeze(h3)
        inputs = tf.reshape(inputs, [args.batch_size, args.seq_length, -1])
        attention_size = inputs.shape[-1] #?
        with tf.name_scope("attention"):

            initializer = tf.random_uniform_initializer()
            W_a = tf.get_variable("weights_a", [attention_size, attention_size], initializer=initializer)
            w = tf.get_variable("weights_w", [attention_size], initializer=initializer)
            print(inputs.shape)
            print(W_a.shape)
            v = tf.tanh(tf.einsum("aij,jk->aik", inputs, W_a))
            a = tf.nn.softmax(tf.einsum("aij,j->ai", v, w))
            print(a.shape)
        self.change = tf.placeholder(tf.bool, [args.batch_size])

        cell_fn = rnn.LSTMCell

        cell = cell_fn(args.rnn_size, state_is_tuple=False)# is not necesery arg
        self.cell = cell = rnn.MultiRNNCell([cell] * args.num_layers, state_is_tuple=False)
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        self.initial_state = a
        inputs = tf.reshape(inputs, [args.batch_size, args.seq_length, -1])
        print(inputs.shape)
        initial_state = self.initial_state #TODO add change to signal

        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs,
                                                initial_state=initial_state,
                                                scope="rnn_cnn")
        print(outputs.shape)
        print(last_state)
        self.final_state = last_state

        print("output shape {}".format(outputs.shape))

        loss1 = tf.constant(0.0)
        loss2 = tf.constant(0.0)
        final_vectors = []
        outputs = tf.unstack(outputs, axis = 1)

        ones = tf.diag([1.] * args.batch_size)

        outputs = tf.reshape(outputs, [args.batch_size, args.seq_length,  -1])
        outputs = tf.unstack(outputs, axis=1)
        with tf.variable_scope("output_linear"):
            for i in range(len(outputs)):
                if i > 0:
                    tf.get_variable_scope()
                output = tf.contrib.layers.fully_connected(outputs[i], args.w2v_size,
                                                           activation_fn=None)

                output = tf.nn.l2_normalize(output, 1)
                output = tf.nn.dropout(output, args.dropout_keep_prob)
                print(output.shape)
                # negative sampling
                matrix = tf.matmul(output, output, transpose_b=True) - ones
                loss1 += tf.maximum(0.0, matrix)
                final_vectors.append(output)

        self.target = final_vectors[-1]

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
        batch_size = 1
        self.valid_input_data = tf.placeholder(tf.float32, [batch_size, self.args.seq_length, self.args.letter_size])

        filters_size = [3, 3, 2]
        num_filters_per_size = 300

        inputs = tf.split(self.valid_input_data, self.args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        with tf.variable_scope("valid_input_linear"):
            fixed_size_vectors = []
            for i, _input in enumerate(inputs):
                if i > 0:
                    tf.get_variable_scope()
                full_vector = tf.contrib.layers.fully_connected(_input, self.num_channels,
                                                                activation_fn=None)
                fixed_size_vectors.append(full_vector)

        fixed_input = tf.stack(fixed_size_vectors, axis=1)
        fixed_input = tf.reshape(fixed_input, [batch_size, 1, self.args.seq_length, -1])
        # Layer 1
        with tf.variable_scope("valid_cnn_1"):
            filter_shape = [1, filters_size[0], self.num_channels, 512]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[512]), name="b")
            conv = tf.nn.conv2d(fixed_input, W, strides=[1, 1, 2, 1], padding="SAME", name="conv1valid")
            h1 = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

        # Layer2
        with tf.name_scope("valid_cnn_2"):
            filter_shape = [1, filters_size[1], 512, 1024]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[1024]), name="b")
            conv2 = tf.nn.conv2d(h1, W, strides=[1, 1, 2, 1], padding="SAME", name="conv2valid")
            h2 = tf.nn.relu(tf.nn.bias_add(conv2, b), name="relu2")

        with tf.name_scope("valid_cnn_3"):
            filter_shape = [1, filters_size[2], 1024, 1024]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[1024]), name="b")
            conv3 = tf.nn.conv2d(h2, W, strides=[1, 1, 1, 1], padding="SAME", name="conv3valid")
            h3 = tf.nn.relu(tf.nn.bias_add(conv3, b), name="relu3")

        inputs = tf.squeeze(h3)

        self.valid_initial_state = self.cell.zero_state(batch_size, tf.float32)

        inputs = tf.reshape(inputs, [batch_size, self.args.seq_length, -1])
        print(inputs.shape)

        valid_outputs, valid_state = tf.nn.dynamic_rnn(cell, inputs,
                                                       initial_state=self.valid_initial_state,
                                                       scope="valid_rnn_cnn")
        self.valid_state = valid_state
        final_vectors = []
        valid_outputs = tf.reshape(valid_outputs, [batch_size, self.args.seq_length, -1])
        valid_outputs = tf.unstack(valid_outputs, axis=1)
        with tf.variable_scope("valid_output_linear"):
            for i in range(len(valid_outputs)):
                if i > 0:
                    tf.get_variable_scope()
                output = tf.contrib.layers.fully_connected(valid_outputs[i], self.args.w2v_size,
                                                           activation_fn=None)
                output = tf.nn.l2_normalize(output, 1)
                output = tf.nn.dropout(output, self.args.dropout_keep_prob)
                final_vectors.append(output)

        self.valid_target = final_vectors[-1]

    def valid_run(self, sess, vocab, prime):
        tokens = word_tokenize(prime)
        valids = []  # np.zeros((len(tokens), self.args.w2v_size))
        word = np.zeros((len(tokens), self.args.letter_size))
        seq_l = self.args.seq_length
        for i, token in enumerate(tokens):
            x = letters2vec(token, vocab)
            word[i] = x

            if (((i % (seq_l - 1) == 0) and (i != 0)) or (i == (len(tokens) - 1))) and (i > seq_l - 2):
                fix_words = word[-seq_l:].reshape((1, seq_l, self.args.letter_size))

                feed = {
                        self.valid_input_data: fix_words,
                       }
                [target] = sess.run([self.valid_target], feed)
                valids.append(np.squeeze(target))
            if (i == (len(tokens) - 1)) and (len(tokens) < seq_l):
                word = np.append(word, np.zeros((seq_l - len(tokens), self.args.letter_size)))
                fix_words = word.reshape((1, seq_l, self.args.letter_size))
                feed = {
                        self.valid_input_data: fix_words,
                       }
                [target] = sess.run([self.valid_target], feed)
                return np.squeeze(target)
        return valids

    def sample(self, sess, vocab, prime=' '):
        tokens = word_tokenize(prime)
        # targets = np.zeros((len(tokens), self.args.w2v_size)) #? TODO remove punctuation?
        targets = []
        word = np.zeros((len(tokens), self.args.letter_size))
        seq_l = self.args.seq_length
        for i, token in enumerate(tokens):
            x = letters2vec(token, vocab)
            word[i] = x

            if (((i % (seq_l - 1) == 0) and (i != 0)) or (i == (len(tokens) - 1))) and (i > seq_l - 2):
                fix_words = word[-seq_l:].reshape((1, seq_l, self.args.letter_size))

                feed = {self.input_data: fix_words,
                    }
                [target] = sess.run([self.target], feed)
                targets.append(np.squeeze(target))
            if (i == (len(tokens) - 1)) and (len(tokens) < seq_l):
                word = np.append(word, np.zeros((seq_l - len(tokens), self.args.letter_size)))
                fix_words = word.reshape((1, seq_l, self.args.letter_size))
                feed = {self.input_data: fix_words,
                        }
                [target] = sess.run([self.target], feed)
                return np.squeeze(target)

        return targets