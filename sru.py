import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.ops import variable_scope


class SRUCell(RNNCell):
    """Simple recurrent unit cell.
    The implementation of: https://arxiv.org/abs/1709.02755.
    """
    def __init__(self, num_units, state_is_tuple=True, activation=tf.nn.tanh, reuse=None):
        super(SRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._linear = None

        self.Wr = tf.Variable(self.init_matrix([self._num_units, self._num_units]))
        self.br = tf.Variable(self.init_matrix([self._num_units]))

        self.Wf = tf.Variable(self.init_matrix([self._num_units, self._num_units]))
        self.bf = tf.Variable(self.init_matrix([self._num_units]))

        self.U = tf.Variable(self.init_matrix([self._num_units, self._num_units]))

    @property
    def state_size(self):
        return  self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """
        f - forget gate
        r - reset gate
        c - final cell
        :param inputs:
        :param state:
        :param scope:
        :return: state, cell
        """
        with variable_scope.variable_scope(scope or type(self).__name__):
            if self._state_is_tuple:
                (c_prev, h_prev) = state
            else:
                c_prev = state

            f = tf.sigmoid(
                tf.matmul(inputs, self.Wf) + self.bf
            )

            r = tf.sigmoid(
                tf.matmul(inputs, self.Wr) + self.br
            )

            c = f * c_prev + (1.0 - f) * tf.matmul(inputs, self.U)

            hidden_state = r * self._activation(c) + (1.0 - r) * inputs

            if self._state_is_tuple:
                return hidden_state, LSTMStateTuple(c, hidden_state)
            else:
                return hidden_state, c

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)
