import tensorflow as tf
import numpy as np
from nltk.tokenize import word_tokenize
from utils import letters2vec


class Conv3LayerModel:
    def __init__(self, args, infer=False):
        self.args = args
        self.num_channels = self.args.rnn_size
        print(args.seq_length)
        if infer:
            args.batch_size = 1
            args.seq_length = 8 #TODO  use as argument to inference

        self.input_data = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.letter_size])

        filters_size = [3, 5, 3]
        num_filters_per_size = 300

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
            filter_shape = [1, filters_size[0], 256, 512]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[512]), name="b")
            conv = tf.nn.conv2d(fixed_input, W, strides=[1, 1, 2, 1], padding="SAME", name="conv1")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, 1, 2, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool1")
            print(pooled.shape)
        # Layer2
        with tf.name_scope("cnn_2"):
            filter_shape = [1, filters_size[1], 512, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv2 = tf.nn.conv2d(pooled, W, strides=[1, 1, 2, 1], padding="SAME", name="conv2")
            print(conv2.shape)
            h2 = tf.nn.relu(tf.nn.bias_add(conv2, b), name="relu2")

            print(h2.shape)

        with tf.name_scope("cnn_3"):
            filter_shape = [1, filters_size[2], 300, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv3 = tf.nn.conv2d(h2, W, strides=[1, 1, 1, 1], padding="SAME", name="conv3")
            print(conv3.shape)
            h3 = tf.nn.relu(tf.nn.bias_add(conv3, b), name="relu3")

        outputs = tf.squeeze(h3)
        print("output shape {}".format(outputs.shape))

        loss1 = tf.constant(0.0)
        loss2 = tf.constant(0.0)
        final_vectors = []

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

        self.target = final_vectors

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
        self._valid_model()

    def _valid_model(self):
        batch_size = 1
        self.valid_input_data = tf.placeholder(tf.float32, [batch_size, self.args.seq_length, self.args.letter_size])

        filters_size = [3, 3, 3]
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
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, 1, 2, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool1")
        # Layer2
        with tf.name_scope("valid_cnn_2"):
            filter_shape = [1, filters_size[1], 512, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv2 = tf.nn.conv2d(pooled, W, strides=[1, 1, 2, 1], padding="SAME", name="conv2valid")
            h2 = tf.nn.relu(tf.nn.bias_add(conv2, b), name="relu2")


        with tf.name_scope("valid_cnn_3"):
            filter_shape = [1, filters_size[2], 300, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv3 = tf.nn.conv2d(h2, W, strides=[1, 1, 1, 1], padding="SAME", name="conv3valid")
            h3 = tf.nn.relu(tf.nn.bias_add(conv3, b), name="relu3")

        outputs = tf.squeeze(h3)

        final_vectors = []
        outputs = tf.reshape(outputs, [batch_size, self.args.seq_length, -1])
        outputs = tf.unstack(outputs, axis=1)
        with tf.variable_scope("valid_output_linear"):
            for i in range(len(outputs)):
                if i > 0:
                    tf.get_variable_scope()
                output = tf.contrib.layers.fully_connected(outputs[i], self.args.w2v_size,
                                                           activation_fn=None)
                output = tf.nn.l2_normalize(output, 1)
                output = tf.nn.dropout(output, self.args.dropout_keep_prob)
                final_vectors.append(output)

        self.valid_target = final_vectors

    def valid_run(self, sess, vocab, prime):
        tokens = word_tokenize(prime)
        valids = np.zeros((len(tokens), self.args.w2v_size))
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
                valids[i - (seq_l- 1):i + 1] = (np.squeeze(target))
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
        targets = np.zeros((len(tokens), self.args.w2v_size)) #? TODO remove punctuation?
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
                targets[i - (seq_l- 1):i + 1] = (np.squeeze(target))
            if (i == (len(tokens) - 1)) and (len(tokens) < seq_l):
                word = np.append(word, np.zeros((seq_l - len(tokens), self.args.letter_size)))
                fix_words = word.reshape((1, seq_l, self.args.letter_size))
                feed = {
                    self.input_data: fix_words,
                        }
                [target] = sess.run([self.target], feed)
                return np.squeeze(target)

        return targets


class Conv1d3Layer:
    def __init__(self, args, infer=False):
        self.args = args
        self.num_channels = self.args.rnn_size
        print(args.seq_length)
        if infer:
            args.batch_size = 1
            args.seq_length = 8

        self.input_data = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.letter_size])

        filters_size = [3, 5, 3]
        num_filters_per_size = 400

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
        #fixed_input = tf.reshape(fixed_input, [self.args.batch_size,1, self.args.seq_length, -1])
        # Layer 1
        with tf.variable_scope("cnn_1"):
            filter_shape = [filters_size[0], 256, 512]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[512]), name="b")
            conv = tf.nn.conv1d(fixed_input, W, stride=2, padding="SAME", name="conv1")
            h1 = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

        # Layer2
        with tf.name_scope("cnn_2"):
            filter_shape = [filters_size[1], 512, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv2 = tf.nn.conv1d(h1, W, stride=2, padding="SAME", name="conv2")
            print(conv2.shape)
            h2 = tf.nn.relu(tf.nn.bias_add(conv2, b), name="relu2")

            print(h2.shape)

        with tf.name_scope("cnn_3"):
            filter_shape = [filters_size[2], num_filters_per_size, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv3 = tf.nn.conv1d(h2, W, stride=1, padding="SAME", name="conv3")
            print(conv3.shape)
            h3 = tf.nn.relu(tf.nn.bias_add(conv3, b), name="relu3")

        outputs = h3
        print("output shape {}".format(outputs.shape))

        loss1 = tf.constant(0.0)
        loss2 = tf.constant(0.0)
        final_vectors = []

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

        self.target = final_vectors

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
        self._valid_model()

    def _valid_model(self):
        batch_size = 1
        self.valid_input_data = tf.placeholder(tf.float32, [batch_size, self.args.seq_length, self.args.letter_size])

        filters_size = [3, 5, 3]
        num_filters_per_size = 400

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
        #fixed_input = tf.reshape(fixed_input, [batch_size, 1, self.args.seq_length, -1])
        # Layer 1
        with tf.variable_scope("valid_cnn_1"):
            filter_shape = [filters_size[0], self.num_channels, 512]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[512]), name="b")
            conv = tf.nn.conv1d(fixed_input, W, stride=2, padding="SAME", name="conv1valid")
            h1= tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

        # Layer2
        with tf.name_scope("valid_cnn_2"):
            filter_shape = [filters_size[1], 512, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv2 = tf.nn.conv1d(h1, W, stride=2, padding="SAME", name="conv2valid")
            h2 = tf.nn.relu(tf.nn.bias_add(conv2, b), name="relu2")


        with tf.name_scope("valid_cnn_3"):
            filter_shape = [filters_size[2], num_filters_per_size, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv3 = tf.nn.conv1d(h2, W, stride=1, padding="SAME", name="conv3valid")
            h3 = tf.nn.relu(tf.nn.bias_add(conv3, b), name="relu3")

        outputs = h3
        final_vectors = []
        outputs = tf.reshape(outputs, [batch_size, self.args.seq_length, -1])
        outputs = tf.unstack(outputs, axis=1)
        with tf.variable_scope("valid_output_linear"):
            for i in range(len(outputs)):
                if i > 0:
                    tf.get_variable_scope()
                output = tf.contrib.layers.fully_connected(outputs[i], self.args.w2v_size,
                                                           activation_fn=None)
                output = tf.nn.l2_normalize(output, 1)
                output = tf.nn.dropout(output, self.args.dropout_keep_prob)
                final_vectors.append(output)

        self.valid_target = final_vectors

    def valid_run(self, sess, vocab, prime):
        tokens = word_tokenize(prime)
        valids = np.zeros((len(tokens), self.args.w2v_size))
        word = np.zeros((len(tokens), 833))
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
                valids[i - (seq_l- 1):i + 1] = (np.squeeze(target))
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
        targets = np.zeros((len(tokens), self.args.w2v_size)) #? TODO remove punctuation?
        word = np.zeros((len(tokens), 833))
        seq_l = self.args.seq_length
        for i, token in enumerate(tokens):
            x = letters2vec(token, vocab)
            word[i] = x

            if (((i % (seq_l - 1) == 0) and (i != 0)) or (i == (len(tokens) - 1))) and (i > seq_l - 2):
                fix_words = word[-seq_l:].reshape((1, seq_l, self.args.letter_size))

                feed = {self.input_data: fix_words,
                    }
                [target] = sess.run([self.target], feed)
                targets[i - (seq_l- 1):i + 1] = (np.squeeze(target))
            if (i == (len(tokens) - 1)) and (len(tokens) < seq_l):
                word = np.append(word, np.zeros((seq_l - len(tokens), self.args.letter_size)))
                fix_words = word.reshape((1, seq_l, self.args.letter_size))
                feed = {self.input_data: fix_words,
                        }
                [target] = sess.run([self.target], feed)
                return np.squeeze(target)

        return targets

class Conv6LayerModel:
    def __init__(self, args, infer=False):
        self.args = args
        self.num_channels = self.args.rnn_size
        print(args.seq_length)
        if infer:
            args.batch_size = 1
            args.seq_length = 10 #TODO
        # Apropiate sequnce length
        self.input_data = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.letter_size])

        filters_size = [3, 3, 2, 3, 2, 3]
        num_filters_per_size = 300

        inputs = tf.split(self.input_data, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        print(inputs[0].shape)
        with tf.variable_scope("input_linear"):
            fixed_size_vectors = []
            for i, _input in enumerate(inputs):
                if i > 0:
                    tf.get_variable_scope()
                full_vector = tf.contrib.layers.fully_connected(_input, args.rnn_size,
                                                                activation_fn=None)
                fixed_size_vectors.append(full_vector)

        fixed_input = tf.stack(fixed_size_vectors, axis=1)
        fixed_input = tf.reshape(fixed_input, [self.args.batch_size, -1, self.args.seq_length])
        fixed_input = tf.unstack(fixed_input, axis=0)
        with tf.variable_scope("active_linear"):
            fixed_size_vectors = []
            for i, _input in enumerate(fixed_input):
                if i > 0:
                    tf.get_variable_scope()
                full_vector = tf.contrib.layers.fully_connected(_input, 50,
                                                                activation_fn=None)
                fixed_size_vectors.append(full_vector)
        fixed_input = tf.stack(fixed_size_vectors)
        fixed_input = tf.reshape(fixed_input, [self.args.batch_size,1, 50, -1])
        # Layer 1
        with tf.variable_scope("cnn_1"):  # TODO get back to conv1d
            filter_shape = [1, filters_size[0], 256, 1024]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[1024]), name="b")
            conv = tf.nn.conv2d(fixed_input, W, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, 1, 5, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool1")
        # Layer2
        with tf.name_scope("cnn_2"):
            filter_shape = [1, filters_size[1], 1024, 512]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[512]), name="b")
            conv2 = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv2")
            print(conv2.shape)
            h2 = tf.nn.relu(tf.nn.bias_add(conv2, b), name="relu2")
            pooled = tf.nn.max_pool(
                h2,
                ksize=[1, 1, 5, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool2")
        print(pooled.shape)

        with tf.name_scope("cnn_3"):
            filter_shape = [1, filters_size[2], 512, 512]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[512]), name="b")
            conv3 = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv3")
            print(conv3.shape)
            h3 = tf.nn.relu(tf.nn.bias_add(conv3, b), name="relu3")

        with tf.name_scope("cnn_4"):
            filter_shape = [1, filters_size[3], 512, 512]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[512]), name="b")
            conv4 = tf.nn.conv2d(h3, W, strides=[1, 1, 1, 1], padding="VALID", name="conv4")
            print(conv4.shape)
            h4 = tf.nn.relu(tf.nn.bias_add(conv4, b), name="relu4")
            pooled = tf.nn.max_pool(
                h4,
                ksize=[1, 1, 3, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool4")

        with tf.name_scope("cnn_5"):
            filter_shape = [1, filters_size[4], 512, 300]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[300]), name="b")
            conv5 = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv5")
            print(conv5.shape)
            h5 = tf.nn.relu(tf.nn.bias_add(conv5, b), name="relu5")
            pooled = tf.nn.max_pool(
                h5,
                ksize=[1, 1, 2, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool5")

        with tf.name_scope("cnn_6"):
            filter_shape = [1, filters_size[5], 300, 10*args.seq_length]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[10*args.seq_length]), name="b")
            conv6 = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv6")
            print(conv6.shape)
            h6 = tf.nn.relu(tf.nn.bias_add(conv6, b), name="relu6")
        outputs = tf.squeeze(h6)
        print("output shape {}".format(outputs.shape))

        loss1 = tf.constant(0.0)
        loss2 = tf.constant(0.0)
        final_vectors = []

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

        self.target = final_vectors

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
        self._valid_model()

    def _valid_model(self):
        batch_size = 1
        self.valid_input_data = tf.placeholder(tf.float32, [batch_size, self.args.seq_length, self.args.letter_size])

        filters_size = [3, 3, 3, 2, 3, 3]
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
        fixed_input = tf.reshape(fixed_input, [self.args.batch_size, -1, self.args.seq_length])
        fixed_input = tf.unstack(fixed_input, axis=0)
        with tf.variable_scope("valid_active_linear"):
            fixed_size_vectors = []
            for i, _input in enumerate(fixed_input):
                if i > 0:
                    tf.get_variable_scope()
                full_vector = tf.contrib.layers.fully_connected(_input, 50,
                                                                activation_fn=None)
                fixed_size_vectors.append(full_vector)
        fixed_input = tf.stack(fixed_size_vectors)
        fixed_input = tf.reshape(fixed_input, [batch_size, 1, 50, -1])
        # Layer 1
        with tf.variable_scope("valid_cnn_1"):
            filter_shape = [1, filters_size[0], self.num_channels, 1024]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[1024]), name="b")
            conv = tf.nn.conv2d(fixed_input, W, strides=[1, 1, 1, 1], padding="VALID", name="conv1valid")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, 1, 5, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool1")
        # Layer2
        with tf.name_scope("valid_cnn_2"):
            filter_shape = [1, filters_size[1], 1024, 512]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[512]), name="b")
            conv2 = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv2valid")
            h2 = tf.nn.relu(tf.nn.bias_add(conv2, b), name="relu2")
            pooled = tf.nn.max_pool(
                h2,
                ksize=[1, 1, 5, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool2")

        with tf.name_scope("valid_cnn_3"):
            filter_shape = [1, filters_size[2], 512, 512]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[512]), name="b")
            conv3 = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv3valid")
            h3 = tf.nn.relu(tf.nn.bias_add(conv3, b), name="relu3")

        with tf.name_scope("valid_cnn_4"):
            filter_shape = [1, filters_size[3], 512, 512]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[512]), name="b")
            conv4 = tf.nn.conv2d(h3, W, strides=[1, 1, 1, 1], padding="VALID", name="conv4valid")
            print(conv4.shape)
            h4 = tf.nn.relu(tf.nn.bias_add(conv4, b), name="relu4")
            pooled = tf.nn.max_pool(
                h4,
                ksize=[1, 1, 3, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="valid_pool4")

        with tf.name_scope("valid_cnn_5"):
            filter_shape = [1, filters_size[4], 512, 300]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[300]), name="b")
            conv5 = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv5valid")
            print(conv5.shape)
            h5 = tf.nn.relu(tf.nn.bias_add(conv5, b), name="relu5")
            pooled = tf.nn.max_pool(
                h5,
                ksize=[1, 1, 2, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="valid_pool5")

        with tf.name_scope("valid_cnn_6"):
            filter_shape = [1, filters_size[5], 300, 10*self.args.seq_length]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[10*self.args.seq_length]), name="b")
            conv6 = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv6valid")
            print(conv6.shape)
            h6 = tf.nn.relu(tf.nn.bias_add(conv6, b), name="relu6")

        outputs = tf.squeeze(h6)

        final_vectors = []
        outputs = tf.reshape(outputs, [batch_size, self.args.seq_length, -1])
        outputs = tf.unstack(outputs, axis=1)
        with tf.variable_scope("valid_output_linear"):
            for i in range(len(outputs)):
                if i > 0:
                    tf.get_variable_scope()
                output = tf.contrib.layers.fully_connected(outputs[i], self.args.w2v_size,
                                                           activation_fn=None)
                output = tf.nn.l2_normalize(output, 1)
                output = tf.nn.dropout(output, self.args.dropout_keep_prob)
                final_vectors.append(output)
        self.valid_target = final_vectors

    def valid_run(self, sess, vocab, prime):
        tokens = word_tokenize(prime)
        valids = np.zeros((len(tokens), self.args.w2v_size))
        word = np.zeros((len(tokens), 833))
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
                valids[i - (seq_l- 1):i + 1] = (np.squeeze(target))
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
        targets = np.zeros((len(tokens), self.args.w2v_size)) #? TODO remove punctuation?
        word = np.zeros((len(tokens), 833))
        seq_l = self.args.seq_length
        for i, token in enumerate(tokens):
            x = letters2vec(token, vocab)
            word[i] = x

            if (((i % (seq_l - 1) == 0) and (i != 0)) or (i == (len(tokens) - 1))) and (i > seq_l - 2):
                fix_words = word[-seq_l:].reshape((1, seq_l, self.args.letter_size))

                feed = {self.input_data: fix_words,
                    }
                [target] = sess.run([self.target], feed)
                targets[i - (seq_l- 1):i + 1] = (np.squeeze(target))
            if (i == (len(tokens) - 1)) and (len(tokens) < seq_l):
                word = np.append(word, np.zeros((seq_l - len(tokens), self.args.letter_size)))
                fix_words = word.reshape((1, seq_l, self.args.letter_size))
                feed = {self.input_data: fix_words,
                        }
                [target] = sess.run([self.target], feed)
                return np.squeeze(target)

        return targets