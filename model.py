import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
from utils import letters2vec
from pymorphy2.tokenizers import simple_word_tokenize


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
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.letter_size])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        inputs = tf.split(1, args.seq_length, self.input_data)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        with tf.variable_scope("input_linear"):
            linears = []
            for i in xrange(len(inputs)):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                linears.append((rnn_cell.linear(inputs[i], args.rnn_size, bias=True)))

        outputs, last_state = seq2seq.rnn_decoder(linears, self.initial_state, cell,
                                                  # loop_function=loop if infer else None,
                                                  scope='rnnlm')

        loss = tf.zeros([args.batch_size, args.seq_length])
        final_vectors = []

        self.indices = tf.zeros([args.batch_size], dtype=tf.int32)
        with tf.variable_scope("output_linear"):
            mask = tf.diag([-1.] * args.batch_size)
            for i in xrange(0, len(outputs)):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output = rnn_cell.linear(outputs[i], args.w2v_size, bias=True)
                output = tf.nn.l2_normalize(output, 1)
                if i > 0:
                    matrix = tf.matmul(output, output, transpose_b=True)
                    matrix = matrix + mask * matrix
                    loss += tf.log(1. + tf.exp(matrix))
                final_vectors.append(output)

        seq_slices = tf.reshape(tf.concat(1, final_vectors), [args.batch_size, args.seq_length, args.w2v_size])
        seq_slices = tf.split(0, args.batch_size, seq_slices)
        seq_slices = [tf.squeeze(input_, [0]) for input_ in seq_slices]
        with tf.variable_scope("additional loss"):
            mask = tf.diag([-1.] * args.seq_length)
            for i in xrange(0, len(seq_slices)):  # should be length of batch_size
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                    matrix = tf.matmul(seq_slices[i], seq_slices[i], transpose_b=True)
                    matrix = matrix + mask * matrix
                    loss += tf.log(1. + tf.exp(-matrix))

        self.targets = final_vectors
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars,
                                                       aggregation_method=
                                                       tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N),
                                          args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, vocab, prime='The '):
        state = self.cell.zero_state(1, tf.float32).eval()

        tokens = simple_word_tokenize(prime)
        targets = []
        for token in tokens:
            x = letters2vec(token, vocab)

            feed = {self.input: x, self.initial_state: state}
            [state, target] = sess.run([self.final_state, self.targets], feed)
            targets.append(target)

        return targets
