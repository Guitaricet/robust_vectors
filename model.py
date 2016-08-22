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

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        self.input = tf.zeros([1, args.letter_size])

        if not infer:
            with tf.device("/cpu:0"):
                self.embedding = tf.get_variable("letter_embedding", [args.word_vocab_size, args.letter_size])
                inputs = tf.split(1, args.seq_length,
                                  tf.stop_gradient(tf.nn.embedding_lookup(self.embedding, self.input_data)))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        else:
            inputs = [self.input]

        with tf.variable_scope("input_linear"):
            linears = []
            for i in xrange(len(inputs)):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                linears.append((rnn_cell.linear(inputs[i], args.rnn_size, bias=True)))

        outputs, last_state = seq2seq.rnn_decoder(linears, self.initial_state, cell,
                                                  # loop_function=loop if infer else None,
                                                  scope='rnnlm')

        loss = tf.constant(0.0)
        final_vectors = []

        self.indices = tf.zeros([args.batch_size], dtype=tf.int32)
        with tf.variable_scope("output_linear"):
            for i in xrange(0, len(outputs)):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output = rnn_cell.linear(outputs[i], args.w2v_size, bias=True)
                output = output / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(output), 1, keep_dims=True)), 1e-12)
                diff_linear = tf.gather(output, self.indices)
                if i > 0:
                    loss += tf.log(1. + tf.exp(-tf.matmul(output, final_vectors[-1], transpose_b=True)))
                    loss += tf.log(1. + tf.exp(tf.matmul(output, diff_linear, transpose_b=True)))
                final_vectors.append(output)

        self.targets = outputs
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
