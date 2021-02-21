from tensorflow.keras import layers
import tensorflow as tf

SIZE = 3
PATH_TO_CSV = 'C:/Users/Hugues/Desktop/RLProject/sudoku.csv'
PATH_TO_NETWORK = 'C:/Users/Hugues/Desktop/RLProject/policy_network'


class SoftmaxMap(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(SoftmaxMap, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        e = tf.exp(x - tf.math.reduce_max(x, axis=self.axis, keepdims=True))
        s = tf.math.reduce_sum(e, axis=self.axis, keepdims=True)
        return e / s

    def get_output_shape_for(self, input_shape):
        return input_shape


class UnsolvableError(ValueError):
    pass
