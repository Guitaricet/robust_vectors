import tensorflow as tf
import numpy as np
from nltk.tokenize import word_tokenize
from utils import letters2vec


class ConvModel:
    def __init__(self, args, infer=False):
        self.args = args
        print(args.seq_length)
        if infer:
            args.batch_size = 1
            args.seq_length = 10


        self.input_data = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.letter_size])

        filters_size = [3, 3, 3]
        num_filters_per_size = 300

        inputs = tf.split(self.input_data, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        print(self.input_data.shape)
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
        with tf.variable_scope("cnn_1"): # TODO get back to conv1d
            filter_shape = [1, filters_size[0], 256, 512]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[512]), name="b")
            conv = tf.nn.conv2d(fixed_input, W, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
            print(conv.shape)
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
            conv2 = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv2")
            print(conv2.shape)
            h2 = tf.nn.relu(tf.nn.bias_add(conv2, b), name="relu2")
            pooled = tf.nn.max_pool(
                h2,
                ksize=[1, 1, 2, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool2")
        print(pooled.shape)

        with tf.name_scope("cnn_3"):
            filter_shape = [1, filters_size[2], 300, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv3 = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv3")
            print(conv3.shape)
            h3 = tf.nn.relu(tf.nn.bias_add(conv3, b), name="relu3")

        outputs = tf.squeeze(h3)

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
            targets(np.squeeze(target))
        return targets

    def sample(self, sess, vocab, prime=' '):
        tokens = word_tokenize(prime)
        targets = np.zeros((len(tokens),self.args.w2v_size)) #? TODO remove punctuation?
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

