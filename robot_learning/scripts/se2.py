import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
from tensorflow.python.framework import ops

#def se2(x):
#    # x = [NxTx3]
#    T = tf.unstack(tf.shape(x))[1]
#    i = tf.constant(0, dtype=tf.int32)
#    tf.while_loop(

class SE2CompositeLayer(LayerRNNCell):
    def __init__(self, reuse=None, name=None):
        super(LayerRNNCell, self).__init__(_reuse=reuse, name=name)

        self._state_size = tensor_shape.TensorShape([3])       # Accumulated SE3 Matrix
        self._output_size = tensor_shape.TensorShape([3])   # xyz + q

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def build(self, inputs_shape):
        print(type(inputs_shape), inputs_shape)
        self.built = True

    def __call__(self, inputs, state,
            scope=None, *args, **kwargs):
        # returns output, state
        # state  = (x, y, h)
        # inputs = (dx, dy, dh)
        x, y, h = tf.unstack(state, axis=-1)
        dx, dy, dh = tf.unstack(inputs, axis=-1)
        c = tf.cos(h)
        s = tf.sin(h)

        dx_R = (c*dx - s*dy)
        dy_R = (s*dx + c*dy)

        x1 = x + dx_R
        y1 = y + dy_R
        h1 = h + dh

        s1 = tf.stack([x1,y1,h1], axis=-1)
        return s1, s1
